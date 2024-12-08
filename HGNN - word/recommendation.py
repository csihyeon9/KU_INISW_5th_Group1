import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
from models.HGNN_model import HGNN
from config.config import load_config

def augment_data(X, labels):
    """SMOTE를 사용한 데이터 증강"""
    unique_labels, counts = np.unique(labels, return_counts=True)
    print("Class counts before augmentation:", dict(zip(unique_labels, counts)))

    # 가장 적은 샘플 수에 맞춰 k_neighbors 조정
    smote = SMOTE(k_neighbors=min(5, min(counts) - 1))

    try:
        X_resampled, y_resampled = smote.fit_resample(X, labels)
        print(f"Original data size: {X.shape}, Resampled data size: {X_resampled.shape}")
        return X_resampled, y_resampled
    except ValueError as e:
        print("SMOTE could not be applied:", str(e))
        print("Falling back to simple oversampling.")
        return simple_oversample(X, labels)

def simple_oversample(X, labels):
    """간단한 데이터 복제를 통한 증강"""
    unique_labels, counts = np.unique(labels, return_counts=True)
    max_count = max(counts)

    X_augmented = []
    labels_augmented = []

    for label in unique_labels:
        X_label = X[labels == label]
        count = len(X_label)

        # 소수 클래스 샘플 복제
        if count < max_count:
            indices = np.random.choice(len(X_label), max_count - count, replace=True)
            X_label_augmented = X_label[indices]
            X_augmented.append(X_label_augmented)
            labels_augmented.extend([label] * len(X_label_augmented))

        X_augmented.append(X_label)
        labels_augmented.extend([label] * count)

    X_augmented = np.vstack(X_augmented)
    labels_augmented = np.array(labels_augmented)
    print(f"Original data size: {X.shape}, Oversampled data size: {X_augmented.shape}")
    return X_augmented, labels_augmented

def visualize_embeddings(embeddings, labels=None):
    """임베딩 벡터를 2D로 시각화"""
    pca = PCA(n_components=2)
    reduced_embeddings = pca.fit_transform(embeddings)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=labels, cmap='jet', alpha=0.7)
    if labels is not None:
        plt.colorbar(scatter)
    plt.title("Embedding Visualization (PCA)")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.show()

def recommend_keywords(input_idx, idx_to_keyword, model, X, top_k=5):
    """입력 키워드 인덱스 기반으로 유사 키워드 추천"""
    if input_idx not in idx_to_keyword:
        raise ValueError(f"Index '{input_idx}'는 데이터에 존재하지 않습니다.")

    # 모델로부터 임베딩 계산
    with torch.no_grad():
        embeddings = model(X, X).cpu().numpy()

    # 임베딩 정규화
    scaler = MinMaxScaler()
    embeddings = scaler.fit_transform(embeddings)

    # PCA 시각화
    visualize_embeddings(embeddings)

    # 코사인 유사도 계산
    input_embedding = embeddings[input_idx].reshape(1, -1)
    similarities = cosine_similarity(input_embedding, embeddings).flatten()

    # 가장 유사한 키워드 상위 top_k 추출
    similar_indices = np.argsort(similarities)[::-1][1:top_k+1]  # 본인 제외
    recommendations = [(idx_to_keyword[idx], similarities[idx]) for idx in similar_indices]

    return recommendations


if __name__ == "__main__":
    config = load_config('config/config.yaml')
    model_path = "results/models/hgnn_model.pth"
    data_path = config['data']['processed_data_path']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 모델 및 데이터 로드
    model = HGNN(
        in_ch=config['model']['in_features'],
        n_hid=config['model']['hidden_features'],
        n_class=config['model']['out_features']
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    data = torch.load(data_path)
    X, labels, keyword_to_idx = data['X'].numpy(), data['labels'].numpy(), data['keyword_to_idx']

    # 데이터 증강
    X_resampled, labels_resampled = augment_data(X, labels)

    idx_to_keyword = {idx: keyword for keyword, idx in keyword_to_idx.items()}
    input_idx = int(input("추천을 원하시는 키워드의 인덱스를 입력하세요: ").strip())

    recommendations = recommend_keywords(input_idx, idx_to_keyword, model, torch.tensor(X_resampled), top_k=5)
    print(f"'{idx_to_keyword[input_idx]}'와 유사한 키워드 추천:")
    for keyword, similarity in recommendations:
        print(f" - {keyword}: 유사도 {similarity:.4f}")
