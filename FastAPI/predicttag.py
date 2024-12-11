import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import torch
import numpy as np
from models.HGNN_model import HGNN
from src.matrix_processor4 import create_feature_matrix, create_hypergraph_structure, generate_normalized_laplacian
from scipy.sparse import coo_matrix

# 카테고리 매핑
CATEGORY_MAPPING = {
    1: "인과or상관성",
    2: "변화&추세",
    3: "시장및거래관계",
    4: "정책&제도",
    5: "기업활동",
    6: "금융상품&자산",
    7: "위험및위기",
    8: "기술및혁신",
    9: "기타"  # 평균보다 낮은 신뢰도를 가진 결과는 기타로 분류
}

def extract_keywords(pairwise_pmi_path):
    """키워드 목록 추출"""
    with open(pairwise_pmi_path, 'r', encoding='utf-8') as f:
        pmi_data = json.load(f)
    keywords = set()
    for pairs in pmi_data.values():
        for pair in pairs.keys():
            kw1, kw2 = pair.split(" | ")
            keywords.update([kw1, kw2])
    return list(keywords)

def load_hgnn_model(model_path, input_dim, hidden_dim, output_dim, dropout=0.5):
    model = HGNN(in_ch=input_dim, n_class=output_dim, n_hid=hidden_dim, dropout=dropout)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def predict_categories(article_path, model, keywords, keyword_to_idx, feature_matrix, L):
    # csr_matrix -> coo_matrix 변환
    L_coo = coo_matrix(L)

    # 희소 행렬을 PyTorch 희소 텐서로 변환
    L_sparse_tensor = torch.sparse_coo_tensor(
        indices=torch.tensor(np.vstack((L_coo.row, L_coo.col)), dtype=torch.long),
        values=torch.tensor(L_coo.data, dtype=torch.float32),
        size=torch.Size(L_coo.shape)
    ).coalesce()

    # JSON 파일 로드
    with open(article_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    predictions = {}
    for article, relations in data.items():
        predictions[article] = []
        for relation in relations:
            keyword_pair = relation['keywords']  # ["달러", "불확실성"] 형태의 키워드 쌍
            indices = [keyword_to_idx.get(kw) for kw in keyword_pair if kw in keyword_to_idx]

            if len(indices) == 2:  # 두 키워드가 모두 인덱싱되어 있는 경우
                # 모델 입력 준비
                input_vector = feature_matrix[indices].mean(axis=0)
                input_tensor = torch.zeros(L_sparse_tensor.shape[0], feature_matrix.shape[1], dtype=torch.float32)
                input_tensor[indices] = torch.tensor(input_vector, dtype=torch.float32)

                # 예측 수행
                with torch.no_grad():
                    output = model(input_tensor, L_sparse_tensor)
                    # 두 키워드에 대한 예측 평균 계산
                    avg_prediction = torch.mean(output[indices], dim=0)
                    predicted_category = torch.argmax(avg_prediction).item() + 1
                    category_name = CATEGORY_MAPPING.get(predicted_category, "기타")

                predictions[article].append({
                    "verb": relation['verb'],
                    "keywords": keyword_pair,
                    "category": category_name
                })
            else:
                # 키워드를 찾을 수 없는 경우 "기타"로 처리
                predictions[article].append({
                    "verb": relation['verb'],
                    "keywords": keyword_pair,
                    "category": "기타"
                })

    return predictions

def save_predictions(output_path, predictions):
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(predictions, f, ensure_ascii=False, indent=4)
    print(f"Predictions saved to {output_path}")

def main():
    article_path = 'predicttag/article_relations1.json'
    output_path = 'predicttag/predicted_categories.json'
    model_path = 'results/models/hgnn_model.pth'
    pairwise_pmi_path = "data/pairwise_pmi_values3.json"

    # 키워드 추출 및 전처리
    keywords = extract_keywords(pairwise_pmi_path)
    keyword_to_idx = {kw: idx for idx, kw in enumerate(keywords)}
    feature_matrix = create_feature_matrix(pairwise_pmi_path, keywords, keyword_to_idx)
    H, W = create_hypergraph_structure(pairwise_pmi_path, keywords, keyword_to_idx)
    L = generate_normalized_laplacian(H, W)

    # 모델 로드
    model = load_hgnn_model(model_path, feature_matrix.shape[1], hidden_dim=64, output_dim=8)

    # 예측 수행
    predictions = predict_categories(article_path, model, keywords, keyword_to_idx, feature_matrix, L)

    # 결과 저장
    save_predictions(output_path, predictions)

if __name__ == "__main__":
    main()
