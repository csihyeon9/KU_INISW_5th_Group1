# json_preprocessing.py
import json
import numpy as np
from gensim.models import Word2Vec
from sklearn.preprocessing import normalize

def load_json_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def preprocess_json_for_hgnn(file_path, embedding_size=128):
    data = load_json_data(file_path)

    # 상위 카테고리로 그룹화
    # def get_category(text):
    #     # 경제, AI, 금융, 정책 등 주요 카테고리로 분류
    #     categories = {
    #         '경제/금융': ['경제', '금융', '투자', '시장', '대출', '은행', '코인', '주식'],
    #         '정책/제도': ['정책', '제도', '규제', '지원', '개혁'],
    #         '산업/기술': ['산업', '기업', '생산', '기술', 'AI', '디지털', '반도체']
    #     }

    #     for cat, keywords in categories.items():
    #         if any(k in text for k in keywords):
    #             return cat
    #     return '기타'

    # relations = [get_category(item['relation_type']) for item in data]
    relations = [item['relation_type'] for item in data]
    keywords_lists = [item['keywords'] for item in data]

    # Word2Vec 학습
    keyword_model = Word2Vec(keywords_lists, vector_size=embedding_size, 
                           window=5, min_count=1, workers=4)

    # Feature 생성
    features = np.zeros((len(data), embedding_size))
    for i, keywords in enumerate(keywords_lists):
        keyword_vectors = [keyword_model.wv[word] for word in keywords if word in keyword_model.wv]
        if keyword_vectors:
            features[i] = np.mean(keyword_vectors, axis=0)

    features = normalize(features)

    # Label encoding
    unique_relations = list(set(relations))
    labels = np.array([unique_relations.index(r) for r in relations])

    # Stratified split
    from sklearn.model_selection import StratifiedShuffleSplit
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_idx, test_idx in sss.split(features, labels):
        return features.astype(np.float32), labels, train_idx, test_idx

def construct_hypergraph_from_json(features, K_neigs=[10], m_prob=1, is_probH=True):
    """
    Construct hypergraph from feature matrix using existing hypergraph_utils
    """
    import utils.hypergraph_utils as hgut

    H = hgut.construct_H_with_KNN(features, K_neigs=K_neigs,
                                 is_probH=is_probH, m_prob=m_prob)
    return H

def analyze_data_distribution(file_path):
    data = load_json_data(file_path)
    relations = [item['relation_type'] for item in data]
    unique_relations = set(relations)
    class_counts = {rel: relations.count(rel) for rel in unique_relations}

    print(f"Total samples: {len(relations)}")
    print(f"Number of classes: {len(unique_relations)}")
    print("\nClass distribution:")
    for rel, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"{rel}: {count}")

    return class_counts