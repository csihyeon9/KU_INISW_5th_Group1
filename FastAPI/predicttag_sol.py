# predicttag_sol.py
import sys
import os
from typing import Dict, List, Tuple, Set, Optional

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import torch
import numpy as np
from collections import defaultdict
from scipy.sparse import coo_matrix, csr_matrix
from models.HGNN_model import HGNN
from src.matrix_processor4 import create_feature_matrix, create_hypergraph_structure, generate_normalized_laplacian

class SingleArticleProcessor:
    """
    하이퍼그래프 기반 단일 기사 처리 및 관계 분석 시스템

    수학적 프레임워크:
    G = (V, E, W) where:
    - V: 키워드 정점 집합
    - E: 문서 관계 하이퍼엣지 집합
    - W: 가중치 텐서
    - H: 인시던스 행렬
    - L = D^{-1/2}HWH^TD^{-1/2}: 정규화된 라플라시안
    """

    CATEGORY_MAPPING = {
        1: "인과or상관성", 2: "변화&추세", 3: "시장및거래관계",
        4: "정책&제도", 5: "기업활동", 6: "금융상품&자산",
        7: "위험및위기", 8: "기술및혁신"
    }

    def __init__(self, model_path: str, pmi_path: str):
        """하이퍼그래프 처리기 초기화"""
        # 키워드와 특징 초기화
        self.pmi_path = pmi_path
        self.keywords = self._extract_keywords()
        self.keyword_to_idx = {kw: idx for idx, kw in enumerate(self.keywords)}

        # 행렬 구조 초기화
        self.feature_matrix = create_feature_matrix(pmi_path, self.keywords, self.keyword_to_idx)
        H, W = create_hypergraph_structure(pmi_path, self.keywords, self.keyword_to_idx)
        self.L = generate_normalized_laplacian(H, W)

        # HGNN 모델 초기화
        self.model = self._initialize_model(model_path)
        self.model.eval()

    def _extract_keywords(self) -> List[str]:
        """PMI 파일에서 키워드 집합 추출"""
        with open(self.pmi_path, 'r', encoding='utf-8') as f:
            pmi_data = json.load(f)
        keywords = {kw for pairs in pmi_data.values()
                    for pair in pairs.keys()
                    for kw in pair.split(" | ")}
        return list(keywords)

    def _initialize_model(self, model_path: str) -> HGNN:
        """HGNN 모델 로드 및 초기화"""
        model = HGNN(
            in_ch=self.feature_matrix.shape[1],
            n_class=len(self.CATEGORY_MAPPING),
            n_hid=64,
            dropout=0.5
        )
        model.load_state_dict(torch.load(model_path))
        return model

    def _convert_to_sparse_tensor(self, matrix: csr_matrix) -> torch.sparse.Tensor:
        """
        희소 행렬의 효율적인 PyTorch 텐서 변환

        Mathematical Framework:
        ----------------------
        Input: 희소 행렬 A ∈ R^{m×n}
        Output: 희소 텐서 T ∈ R^{m×n}

        Steps:
        1. CSR → COO 변환
        2. 인덱스 행렬 통합
        3. 텐서 생성
        """
        coo = matrix.tocoo()
        indices = np.vstack((coo.row, coo.col))
        indices = torch.from_numpy(indices).long()
        values = torch.from_numpy(coo.data).float()

        sparse_tensor = torch.sparse_coo_tensor(
            indices=indices,
            values=values,
            size=coo.shape
        )
        return sparse_tensor.coalesce()

    def process_article_relations(self, article_data: List[Dict]) -> Dict[str, List[Dict]]:
        """
        단일 기사의 관계 분석 및 카테고리 예측

        수학적 연산:
        1. 희소 텐서 변환: L_sparse ∈ R^{n×n}
        2. 특징 집계: X = mean(F_i), F_i ∈ R^d
        3. HGNN 순전파: Y = HGNN(X, L_sparse)
        """
        # 라플라시안 텐서 변환
        L_sparse = self._convert_to_sparse_tensor(self.L)
        device = next(self.model.parameters()).device
        L_sparse = L_sparse.to(device)

        predictions = defaultdict(list)

        for article in article_data:
            article_title = article.get('title', '')
            relations = article.get('relations', [])

            for relation in relations:
                keyword_pair = relation['keywords']
                indices = [self.keyword_to_idx.get(kw) for kw in keyword_pair
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

                    # 노드 특징 할당
                    for idx in indices:
                        input_tensor[idx] = mean_features

                    # 예측 수행
                    with torch.no_grad():
                        output = self.model(input_tensor, L_sparse)
                        relevant_outputs = output[indices]
                        avg_prediction = torch.mean(relevant_outputs, dim=0)
                        category_idx = torch.argmax(avg_prediction).item() + 1
                        confidence = float(torch.softmax(avg_prediction, dim=0).max())

                    predictions[article_title].append({
                        "verb": relation['verb'],
                        "keywords": keyword_pair,
                        "category": self.CATEGORY_MAPPING.get(category_idx, "Unknown"),
                        "confidence": confidence
                    })
                else:
                    predictions[article_title].append({
                        "verb": relation['verb'],
                        "keywords": keyword_pair,
                        "category": "Unknown",
                        "confidence": 0.0
                    })

        return dict(predictions)


def process_single_article(article_path: str, model_path: str, pmi_path: str, output_path: str = None):
    """단일 기사 처리를 위한 인터페이스 함수"""
    processor = SingleArticleProcessor(model_path, pmi_path)

    with open(article_path, 'r', encoding='utf-8') as f:
        article_data = json.load(f)

    results = processor.process_article_relations(article_data)

    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=4)

    return results


def main():
    """
    메인 테스트 함수

    실행 방법:
    1. 필요한 파일 경로 설정
    2. 단일 기사 처리
    3. 결과 저장 및 검증
    """
    try:
        # 1. 파일 경로 설정
        paths = {
            'article': 'data/predicttag/article_relations.json',  # 'data/predicttag/article_relations.json',
            'model': 'results/models/hgnn_model.pth',
            'pmi': 'data/pairwise_pmi_values3.json',
            'output': 'data/predicttag/pkm_example.json'
        }

        # 2. 입력 파일 존재 확인
        for key, path in paths.items():
            if not os.path.exists(path):
                raise FileNotFoundError(f"Required {key} file not found at: {path}")

        print("시작: 단일 기사 관계 분석")
        print("-" * 50)

        # 3. 기사 처리
        results = process_single_article(
            paths['article'],
            paths['model'],
            paths['pmi'],
            paths['output']
        )

        # 4. 결과 요약 출력
        total_relations = sum(len(relations) for relations in results.values())
        total_articles = len(results)

        print(f"\n처리 완료:")
        print(f"- 처리된 기사 수: {total_articles}")
        print(f"- 총 관계 수: {total_relations}")

        # 5. 카테고리별 통계
        category_stats = defaultdict(int)
        for article, relations in results.items():
            for relation in relations:
                category_stats[relation['category']] += 1

        print("\n카테고리별 통계:")
        for category, count in category_stats.items():
            print(f"- {category}: {count}개 ({(count / total_relations * 100):.1f}%)")

        print(f"\n결과 저장 위치: {paths['output']}")
        print("-" * 50)

    except Exception as e:
        print(f"오류 발생: {str(e)}")
        raise


if __name__ == "__main__":
    main()