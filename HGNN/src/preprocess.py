import os
import sys
import json
import torch
from utils.graph_utils import create_hyperedges

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def preprocess_data(config):
    """
    데이터 전처리 함수
    :param config: 설정 딕셔너리
    """
    # PMI 값 로드
    with open(config['data']['pairwise_pmi_path'], 'r', encoding='utf-8') as f:
        pairwise_pmi = json.load(f)

    # single pmi 값은 아직 미사용
    with open(config['data']['verb_single_keyword_pmi_path'], 'r', encoding='utf-8') as f:
        verb_single_keyword_pmi = json.load(f)

    # 원본 데이터 로드
    with open(config['data']['raw_data_path'], 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 키워드와 동사 카테고리 추출
    keywords = set()
    verb_categories = set()
    for article in data:
        relations = article.get('relations', {})
        for category, relation_list in relations.items():
            verb_categories.add(category)
            for relation in relation_list:
                keywords.update([word for word in relation[:2] if not word.endswith(("하다", "되다", "치다", "다"))])

    keyword_to_idx = {kw: idx for idx, kw in enumerate(keywords)}
    category_to_idx = {cat: idx for idx, cat in enumerate(verb_categories)}

   # PMI 행렬 생성
    num_keywords = len(keywords)
    pmi_matrix = torch.zeros((num_keywords, num_keywords))  # num_keywords에 따라 크기 조정

    for category, pairs in pairwise_pmi.items():
        for pair_str, pmi_value in pairs.items():
            kw1, kw2 = pair_str.split(" | ")
            if kw1 in keyword_to_idx and kw2 in keyword_to_idx:
                idx1, idx2 = keyword_to_idx[kw1], keyword_to_idx[kw2]
                print(idx1, idx2)
                pmi_matrix[idx1, idx2] = pmi_matrix[idx2, idx1] = pmi_value

    # X를 PMI 행렬로 설정
    X = pmi_matrix 

    # X의 크기를 확인하고 필요에 따라 축소
    if X.shape[0] > config['model']['in_features']:
        X = X[:config['model']['in_features'], :config['model']['in_features']]  # 필요한 경우 축소

    # 하이퍼엣지 생성
    hyperedges = create_hyperedges(data, keyword_to_idx, category_to_idx)

    num_edges = len(hyperedges)
    H = torch.zeros((num_keywords, num_edges))
    
    for edge_idx, nodes in hyperedges.items():
        for node in nodes:
            if node in keyword_to_idx:
                H[keyword_to_idx[node], edge_idx] = pmi_matrix[keyword_to_idx[node]].sum()  # Sum of PMIs for all connections

    # 출력 디렉토리가 존재하는지 확인하고 생성
    output_dir = os.path.dirname(config['data']['processed_data_path'])
    os.makedirs(output_dir, exist_ok=True)  # 디렉토리가 없으면 생성

    # 데이터 저장
    torch.save({
        'X': pmi_matrix,
        'H': H,
        'keyword_to_idx': keyword_to_idx,
        'category_to_idx': category_to_idx,
        'raw_data' : data
    }, config['data']['processed_data_path'])

    # X와 H 각각의 텐서 값을 출력
    # X: 키워드 간의 PMI 값을 나타내는 정방행렬.
    # H: 키워드와 하이퍼엣지 간의 연결 관계를 나타내는 인접 행렬.

    print("X tensor (PMI 행렬) values:")
    print(X)
    
    print("H tensor (하이퍼엣지 인접 행렬) values:")
    print(H)


if __name__ == "__main__":
    from config.config import load_config
    config = load_config('config/config.yaml')
    preprocess_data(config)