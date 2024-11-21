import json
import numpy as np
from gensim.models import Word2Vec
from sklearn.preprocessing import normalize
from collections import Counter
from sklearn.model_selection import StratifiedShuffleSplit

def load_json_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

# def preprocess_json_for_hgnn(file_path, embedding_size=128, min_samples_per_class=2):
#     data = load_json_data(file_path)
    
#     # Count samples per class
#     relation_counts = Counter(item['relation_type'] for item in data)
    
#     # Filter out classes with insufficient samples
#     valid_relations = {rel for rel, count in relation_counts.items() if count >= min_samples_per_class}
    
#     # Filter data
#     filtered_data = [item for item in data if item['relation_type'] in valid_relations]
    
#     if not filtered_data:
#         raise ValueError("No classes with sufficient samples found after filtering")
    
#     relations = [item['relation_type'] for item in filtered_data]
#     keywords_lists = [item['keywords'] for item in filtered_data]

#     # Word2Vec training
#     keyword_model = Word2Vec(keywords_lists, vector_size=embedding_size, 
#                            window=5, min_count=1, workers=4)

#     # Feature creation
#     features = np.zeros((len(filtered_data), embedding_size))
#     for i, keywords in enumerate(keywords_lists):
#         keyword_vectors = [keyword_model.wv[word] for word in keywords if word in keyword_model.wv]
#         if keyword_vectors:
#             features[i] = np.mean(keyword_vectors, axis=0)

#     features = normalize(features)

#     # Label encoding
#     unique_relations = list(set(relations))
#     labels = np.array([unique_relations.index(r) for r in relations])

#     # Stratified split
#     from sklearn.model_selection import StratifiedShuffleSplit
#     sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    
#     for train_idx, test_idx in sss.split(features, labels):
#         return features.astype(np.float32), labels, train_idx, test_idx

def preprocess_json_for_hgnn(file_path, embedding_size=128, min_class_size=10):
    data = load_json_data(file_path)
    
    # Count and filter classes
    relation_counts = Counter(item['relation_type'] for item in data)
    valid_relations = {rel for rel, count in relation_counts.items() if count >= min_class_size}
    
    # Filter data to include only classes with sufficient samples
    filtered_data = [item for item in data if item['relation_type'] in valid_relations]
    
    relations = [item['relation_type'] for item in filtered_data]
    keywords_lists = [item['keywords'] for item in filtered_data]

    # Word2Vec training
    keyword_model = Word2Vec(keywords_lists, vector_size=embedding_size, 
                           window=5, min_count=1, workers=4)

    features = np.zeros((len(filtered_data), embedding_size))
    for i, keywords in enumerate(keywords_lists):
        keyword_vectors = [keyword_model.wv[word] for word in keywords if word in keyword_model.wv]
        if keyword_vectors:
            features[i] = np.mean(keyword_vectors, axis=0)

    features = normalize(features)

    unique_relations = list(set(relations))
    labels = np.array([unique_relations.index(r) for r in relations])

    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_idx, test_idx in sss.split(features, labels):
        return features.astype(np.float32), labels, train_idx, test_idx

def construct_hypergraph_from_json(features, K_neigs=[10], m_prob=1, is_probH=True):
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