# relation_processor.py
import torch
from models.HGNN_model import HGNN
from src.matrix_processor4 import (
    create_feature_matrix,
    create_hypergraph_structure,
    generate_normalized_laplacian
)
import json
import numpy as np
from scipy.sparse import csr_matrix
from collections import defaultdict


class RelationProcessor:
    """하이퍼그래프 기반 관계 분류 처리기"""

    CATEGORY_MAPPING = {
        1: "인과or상관성", 2: "변화&추세", 3: "시장및거래관계",
        4: "정책&제도", 5: "기업활동", 6: "금융상품&자산",
        7: "위험및위기", 8: "기술및혁신"
    }

    def __init__(self, model_path: str, pmi_path: str):
        # PMI 데이터와 키워드 초기화
        self.pmi_path = pmi_path
        with open(self.pmi_path, 'r', encoding='utf-8') as f:
            pmi_data = json.load(f)
        self.keywords = self._extract_keywords(pmi_data)
        self.keyword_to_idx = {kw: idx for idx, kw in enumerate(self.keywords)}

        # 행렬 구조 초기화
        self.feature_matrix = create_feature_matrix(pmi_path, self.keywords, self.keyword_to_idx)
        H, W = create_hypergraph_structure(pmi_path, self.keywords, self.keyword_to_idx)
        self.L = generate_normalized_laplacian(H, W)

        # HGNN 모델 초기화
        self.model = self._initialize_model(model_path)
        self.model.eval()

    def _extract_keywords(self, pmi_data: dict) -> list:
        """PMI 데이터에서 키워드 추출"""
        keywords = {kw for pairs in pmi_data.values()
                    for pair in pairs.keys()
                    for kw in pair.split(" | ")}
        return list(keywords)

    def _initialize_model(self, model_path: str) -> HGNN:
        """HGNN 모델 초기화"""
        model = HGNN(
            in_ch=self.feature_matrix.shape[1],
            n_class=len(self.CATEGORY_MAPPING),
            n_hid=64,
            dropout=0.5
        )
        model.load_state_dict(torch.load(model_path))
        return model

    def _convert_to_sparse_tensor(self, matrix: csr_matrix) -> torch.sparse.Tensor:
        """CSR 행렬을 PyTorch 희소 텐서로 변환"""
        coo = matrix.tocoo()
        indices = torch.from_numpy(np.vstack((coo.row, coo.col))).long()
        values = torch.from_numpy(coo.data).float()
        sparse_tensor = torch.sparse_coo_tensor(
            indices=indices,
            values=values,
            size=coo.shape
        )
        return sparse_tensor.coalesce()

    def classify_relations(self, graph_data: dict) -> dict:
        """그래프 데이터의 관계 분류"""
        L_sparse = self._convert_to_sparse_tensor(self.L)
        device = next(self.model.parameters()).device
        L_sparse = L_sparse.to(device)

        # 엣지 정보를 카테고리와 함께 확장
        enhanced_edges = []
        for edge in graph_data['edges']:
            keywords = edge['nodes']
            if len(keywords) != 2:
                continue

            indices = [self.keyword_to_idx.get(kw) for kw in keywords
                       if kw in self.keyword_to_idx]

            if len(indices) == 2:
                # 입력 텐서 준비
                input_tensor = torch.zeros(
                    (self.L.shape[0], self.feature_matrix.shape[1]),
                    dtype=torch.float32,
                    device=device
                )

                # 특징 벡터 계산
                mean_features = torch.from_numpy(
                    self.feature_matrix[indices].mean(axis=0)
                ).float().to(device)

                for idx in indices:
                    input_tensor[idx] = mean_features

                # 예측 수행
                with torch.no_grad():
                    output = self.model(input_tensor, L_sparse)
                    relevant_outputs = output[indices]
                    avg_prediction = torch.mean(relevant_outputs, dim=0)
                    category_idx = torch.argmax(avg_prediction).item() + 1
                    confidence = float(torch.softmax(avg_prediction, dim=0).max())

                edge_with_category = edge.copy()
                edge_with_category.update({
                    "category": self.CATEGORY_MAPPING.get(category_idx, "기타"),
                    "confidence": confidence
                })
                enhanced_edges.append(edge_with_category)
            else:
                edge_with_category = edge.copy()
                edge_with_category.update({
                    "category": "기타",
                    "confidence": 0.0
                })
                enhanced_edges.append(edge_with_category)

        # 원본 그래프 데이터 업데이트
        result = graph_data.copy()
        result['edges'] = enhanced_edges
        return result