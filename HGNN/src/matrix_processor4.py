import json
import numpy as np
import torch
from collections import defaultdict
from scipy import sparse

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

def create_feature_matrix(pairwise_pmi_path, keywords, keyword_to_idx):
    """
    노드 특징 행렬 생성 :
    노드 간의 관계 강도로 활용. 각 키워드 쌍의 PMI 값이 행렬의 
    해당 위치로 할당. 노드 간의 의미적 연관성을 수치화해서 표현.
    """
    num_keywords = len(keywords)
    feature_matrix = np.zeros((num_keywords, num_keywords))
    
    with open(pairwise_pmi_path, 'r', encoding='utf-8') as f:
        pmi_data = json.load(f)
    
    # PMI 값을 노드 특징으로 사용
    for category, pairs in pmi_data.items():
        for pair, pmi_value in pairs.items():
            kw1, kw2 = pair.split(" | ")
            if kw1 in keyword_to_idx and kw2 in keyword_to_idx:
                idx1, idx2 = keyword_to_idx[kw1], keyword_to_idx[kw2]
                feature_matrix[idx1, idx2] = pmi_value
                feature_matrix[idx2, idx1] = pmi_value
    
    print(f"Feature Matrix Shape: {feature_matrix.shape}")
    return feature_matrix

def create_hypergraph_structure(pairwise_pmi_path, keywords, keyword_to_idx):
    """
    하이퍼그래프 구조 생성 : 
    카테고리별로 하이퍼엣지를 생성. PMI 값을 엣지 가중치로 활용.
    연결 강도를 반영한 더 풍부한 그래프 구조를 만듦.
    """
    num_keywords = len(keywords)
    with open(pairwise_pmi_path, 'r', encoding='utf-8') as f:
        pmi_data = json.load(f)

    # 카테고리별 하이퍼엣지 생성
    hyperedges = defaultdict(list)
    edge_weights = defaultdict(float)
    
    for category, pairs in pmi_data.items():
        for pair, pmi_value in pairs.items():
            kw1, kw2 = pair.split(" | ")
            if kw1 in keyword_to_idx and kw2 in keyword_to_idx:
                edge_id = f"{category}_{kw1}_{kw2}"
                hyperedges[edge_id] = [keyword_to_idx[kw1], keyword_to_idx[kw2]]
                edge_weights[edge_id] = pmi_value

    # 인시던스 행렬 생성
    num_edges = len(hyperedges)
    H = np.zeros((num_keywords, num_edges))
    W = np.zeros(num_edges)  # 엣지 가중치 벡터

    for edge_idx, (edge_id, nodes) in enumerate(hyperedges.items()):
        for node_idx in nodes:
            H[node_idx, edge_idx] = 1
        W[edge_idx] = edge_weights[edge_id]

    print(f"Incidence Matrix Shape: {H.shape}")
    print(f"Edge Weights Shape: {W.shape}")
    return H, W

def generate_normalized_laplacian(H, W):
    """정규화된 라플라시안 행렬 생성 (희소 행렬 사용)"""
    H = sparse.csr_matrix(H)
    W = sparse.diags(W)
    
    # 노드 차수와 엣지 차수 계산
    DV = np.array(H.dot(W).sum(axis=1)).flatten()
    DE = np.array(H.sum(axis=0)).flatten()

    # 차수 행렬의 역행렬 계산
    DV_inv_sqrt = sparse.diags(np.power(DV, -0.5, where=DV > 0))
    DE_inv = sparse.diags(np.power(DE, -1, where=DE > 0))

    # 정규화된 라플라시안 계산
    L = DV_inv_sqrt.dot(H).dot(W).dot(DE_inv).dot(H.T).dot(DV_inv_sqrt)

    print(f"Laplacian Matrix Shape: {L.shape}")
    return L

def create_labels_from_data(pairwise_pmi_path, keyword_to_idx):
    """키워드별 레이블 생성
    : 각 키워드가 가장 빈번하게 등장하는 카테고리를 해당 키워드의
    레이블로 할당. """
    with open(pairwise_pmi_path, 'r', encoding='utf-8') as f:
        pmi_data = json.load(f)

    # 카테고리와 인덱스 매핑
    categories = list(pmi_data.keys())
    category_to_idx = {cat: idx for idx, cat in enumerate(categories)}
    
    # 각 키워드의 카테고리 빈도 계산
    keyword_categories = defaultdict(lambda: defaultdict(int))
    for category, pairs in pmi_data.items():
        for pair in pairs.keys():
            kw1, kw2 = pair.split(" | ")
            if kw1 in keyword_to_idx:
                keyword_categories[kw1][category] += 1
            if kw2 in keyword_to_idx:
                keyword_categories[kw2][category] += 1

    # 가장 빈번한 카테고리를 레이블로 할당
    labels = np.full(len(keyword_to_idx), -1)
    for keyword, categories_freq in keyword_categories.items():
        if keyword in keyword_to_idx:
            most_common_category = max(categories_freq.items(), key=lambda x: x[1])[0]
            labels[keyword_to_idx[keyword]] = category_to_idx[most_common_category]

    print(f"Labels Shape: {labels.shape}")
    print(f"Number of unique labels: {len(np.unique(labels))}")
    return torch.tensor(labels, dtype=torch.long)

def save_hgnn_data(output_path, feature_matrix, H, W, L, keyword_to_idx, labels):
    """HGNN 학습 데이터 저장"""

    # 희소 행렬을 밀집 텐서로 변환
    if sparse.issparse(L):
        L = torch.tensor(L.todense(), dtype = torch.float32)
    else:
        L = torch.tensor(L, dtype=torch.float32)
    torch.save({
        'X': torch.tensor(feature_matrix, dtype=torch.float32),
        'H': torch.tensor(H, dtype=torch.float32),
        'W': torch.tensor(W, dtype=torch.float32),
        'L': L,
        'keyword_to_idx': keyword_to_idx,
        'labels': labels
    }, output_path)
    print(f"HGNN 데이터가 {output_path}에 저장되었습니다.")

def main():
    pairwise_pmi_path = "data/pairwise_pmi_values3.json"
    output_path = "data/processed_data/hgnn_data4.pt"

    # 1. 키워드 추출 및 인덱싱
    keywords = extract_keywords(pairwise_pmi_path)
    keyword_to_idx = {kw: idx for idx, kw in enumerate(keywords)}

    # 2. 노드 특징 행렬 생성
    feature_matrix = create_feature_matrix(pairwise_pmi_path, keywords, keyword_to_idx)

    # 3. 하이퍼그래프 구조 생성
    H, W = create_hypergraph_structure(pairwise_pmi_path, keywords, keyword_to_idx)

    # 4. 정규화된 라플라시안 생성
    G = generate_normalized_laplacian(H, W)

    # 5. 레이블 생성
    labels = create_labels_from_data(pairwise_pmi_path, keyword_to_idx)

    # 6. 데이터 저장
    save_hgnn_data(output_path, feature_matrix, H, W, G, keyword_to_idx, labels)

if __name__ == "__main__":
    main()