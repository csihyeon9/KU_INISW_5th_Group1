import json
import numpy as np
import torch
from collections import defaultdict
from tqdm import tqdm

def create_pmi_matrix(pairwise_pmi_path, unique_keywords, max_features):
    """
    PMI 행렬 생성 함수
    """
    with open(pairwise_pmi_path, 'r', encoding='utf-8') as f:
        pmi_data = json.load(f)

    keyword_to_idx = {keyword: idx for idx, keyword in enumerate(unique_keywords[:max_features])}
    num_keywords = len(keyword_to_idx)

    pmi_matrix = np.zeros((num_keywords, num_keywords))
    for category, pairs in pmi_data.items():
        for pair_str, pmi_value in pairs.items():
            kw1, kw2 = pair_str.split(" | ")
            if kw1 in keyword_to_idx and kw2 in keyword_to_idx:
                idx1, idx2 = keyword_to_idx[kw1], keyword_to_idx[kw2]
                pmi_matrix[idx1, idx2] = pmi_value
                pmi_matrix[idx2, idx1] = pmi_value

    return pmi_matrix, keyword_to_idx

def create_hyperedges(pairwise_pmi_path):
    """
    하이퍼엣지 생성 함수
    """
    with open(pairwise_pmi_path, 'r', encoding='utf-8') as f:
        pmi_data = json.load(f)
    
    hyperedges = defaultdict(set)
    for category, pairs in pmi_data.items():
        for pair_str in pairs.keys():
            kw1, kw2 = pair_str.split(" | ")
            hyperedges[category].update([kw1, kw2])
    
    return hyperedges


def create_incidence_matrix(hyperedges, unique_keywords, keyword_to_idx):
    """
    하이퍼그래프 인시던스 행렬 생성 함수
    """
    num_keywords = len(unique_keywords)
    num_hyperedges = len(hyperedges)

    H = np.zeros((num_keywords, num_hyperedges))
    for edge_idx, (category, keywords) in enumerate(hyperedges.items()):
        for keyword in keywords:
            if keyword in keyword_to_idx:
                keyword_idx = keyword_to_idx[keyword]
                H[keyword_idx, edge_idx] = 1

    return H


def generate_G_from_H(H, hyperedges, pairwise_pmi_path):
    """
    하이퍼그래프 라플라시안 정규화된 행렬 G 생성
    """
    H = np.array(H)
    n_edge = H.shape[1]

    # 하이퍼엣지 가중치를 PMI 값으로 설정
    with open(pairwise_pmi_path, 'r', encoding='utf-8') as f:
        pmi_data = json.load(f)

    W = np.zeros(n_edge)
    for edge_idx, (category, _) in enumerate(hyperedges.items()):
        W[edge_idx] = np.mean([pmi_data[category][pair] for pair in pmi_data[category]])

    DV = np.sum(H * W, axis=1)  # 노드 차수
    DE = np.sum(H, axis=0)  # 하이퍼엣지 차수

    invDE = np.diag(np.power(DE, -1, where=DE > 0))
    DV2 = np.diag(np.power(DV, -0.5, where=DV > 0))
    W = np.diag(W)

    H = np.mat(H)
    HT = H.T

    G = DV2 @ H @ W @ invDE @ HT @ DV2
    return G


def create_labels(hyperedges, keyword_to_idx):
    """
    키워드에 대한 레이블 생성 함수
    - 각 키워드가 속한 동사 카테고리를 기반으로 레이블 생성
    
    Args:
        hyperedges (dict): 동사 카테고리와 관련 키워드 정보
        keyword_to_idx (dict): 키워드 -> 인덱스 매핑 정보
    
    Returns:
        labels (torch.Tensor): 키워드에 대한 레이블
    """
    num_keywords = len(keyword_to_idx)
    labels = np.full(num_keywords, -1)  # 기본값은 -1로 초기화 (카테고리 없음)

    # 각 카테고리별 키워드를 순회하며 레이블 할당
    for category_idx, (category, keywords) in enumerate(hyperedges.items()):
        for keyword in keywords:
            if keyword in keyword_to_idx:
                keyword_idx = keyword_to_idx[keyword]
                labels[keyword_idx] = category_idx

    return torch.tensor(labels, dtype=torch.long)

def save_data_for_hgnn(output_path, pmi_matrix, incidence_matrix, G, keyword_to_idx, labels):
    """
    HGNN 학습용 데이터 저장 함수
    """
    torch.save({
        'X': torch.tensor(pmi_matrix, dtype=torch.float32),
        'H': torch.tensor(incidence_matrix, dtype=torch.float32),
        'G': torch.tensor(G, dtype=torch.float32),
        'keyword_to_idx': keyword_to_idx,
        'labels': labels  # 레이블 추가
    }, output_path)
    print(f"HGNN 학습 데이터를 {output_path}에 저장했습니다.")

def main():
    """
    PMI 행렬(가중치 행렬), 하이퍼엣지, 인시던스 행렬 및 G 생성 후 저장
    """
    pairwise_pmi_path = "data/pairwise_pmi_values3.json"
    unique_keywords_path = "data/updated_unique_keywords.json"
    output_path = "data/processed_data/hgnn_data2.pt"

    max_features = 500

    # 유니크 키워드 로드
    with open(unique_keywords_path, 'r', encoding='utf-8') as f:
        unique_keywords = json.load(f)["unique_keywords"]

    # PMI 행렬 생성
    pmi_matrix, keyword_to_idx = create_pmi_matrix(pairwise_pmi_path, unique_keywords, max_features)
    
    # 하이퍼엣지 생성
    hyperedges = create_hyperedges(pairwise_pmi_path)
    
    # 인시던스 행렬 생성
    incidence_matrix = create_incidence_matrix(hyperedges, unique_keywords, keyword_to_idx)
    
    # 라플라시안 행렬 생성
    G = generate_G_from_H(incidence_matrix, hyperedges, pairwise_pmi_path)
    
    # 레이블 생성
    labels = create_labels(hyperedges, keyword_to_idx)

    # 데이터 저장
    save_data_for_hgnn(output_path, pmi_matrix, incidence_matrix, G, keyword_to_idx, labels)

if __name__ == "__main__":
    main()
