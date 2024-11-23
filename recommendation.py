import os
import json
import torch
import numpy as np
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity
from utils.hypergraph_utils import construct_H_with_keywords, generate_G_from_H
from datasets.data_helper import create_keyword_embeddings, preprocess_documents, load_json_data
from models import HGNNRecommender


class PersonalizedRecommender:
    """
    Hypergraph Neural Network(HGNN)를 기반으로 한 개인화 추천 시스템.
    저장된 모델을 불러와 새로운 문서를 추천합니다.
    """

    def __init__(self, model_path, documents, G):
        """
        개인화 추천 시스템 초기화.

        Args:
            model_path (str): 학습된 HGNNRecommender 모델이 저장된 경로.
            documents (list): 기존 문서들의 리스트.
            G (ndarray): 기존 문서의 하이퍼그래프 행렬.
        """
        self.documents = documents  # 기존 문서 목록
        self.G = G  # 하이퍼그래프 행렬
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 모델 로드
        self.model = HGNNRecommender(
            in_dim=128, hidden_dim=256, embedding_dim=128
        ).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

        # 문서 임베딩 생성
        self.keyword_model = create_keyword_embeddings(documents, embedding_size=128)
        features = preprocess_documents(documents, self.keyword_model, embedding_size=128)
        features = torch.FloatTensor(features).to(self.device)
        with torch.no_grad():
            self.document_embeddings = self.model(features, torch.FloatTensor(G).to(self.device)).cpu().numpy()

    def recommend(self, new_doc, top_k=5):
        """
        새 문서를 기준으로 가장 유사한 문서 top-k를 추천합니다.

        Args:
            new_doc (dict): 새 문서 정보.
            top_k (int): 추천할 문서 개수.

        Returns:
            list: 추천된 문서들의 정보(URL, 유사도, 키워드 포함).
        """
        # 새 문서를 기존 하이퍼그래프에 추가
        H_new = construct_H_with_keywords([new_doc] + self.documents)
        G_new = generate_G_from_H(H_new)

        # 새 문서와 기존 문서의 특징 결합
        features_existing = preprocess_documents(self.documents, self.keyword_model, embedding_size=128)
        features_new = preprocess_documents([new_doc], self.keyword_model, embedding_size=128)
        features_combined = np.vstack([features_existing, features_new])
        features_combined = torch.FloatTensor(features_combined).to(self.device)

        # 새 문서의 임베딩 생성
        with torch.no_grad():
            embeddings = self.model(features_combined, torch.FloatTensor(G_new).to(self.device))
        new_doc_embedding = embeddings[-1].cpu().numpy()

        # 새 문서와 기존 문서 간의 유사도 계산
        similarities = cosine_similarity([new_doc_embedding], self.document_embeddings)[0]
        top_indices = np.argsort(similarities)[-top_k:][::-1]

        recommendations = []
        for idx in top_indices:
            doc = self.documents[idx]
            recommendations.append({
                "url": doc['url'],
                "similarity": similarities[idx],
                "keywords": doc['keywords'],
                "relation_type": doc.get('relation_type', 'Unknown')
            })
        
        return recommendations

    def run(self, folder_path):
        """
        개인화 추천 실행: 새 문서를 불러와 추천 결과 출력.

        Args:
            folder_path (str): 새 문서가 저장된 폴더 경로.
        """
        file_path = os.path.join(folder_path, "New.json")
        with open(file_path, 'r', encoding='utf-8') as file:
            new_articles = json.load(file)

        for new_doc in new_articles:
            print(f"\nProcessing new article: {new_doc['url']}")
            recommendations = self.recommend(new_doc)

            print("\nTop recommendations:")
            for i, rec in enumerate(recommendations, 1):
                print(f"{i}. URL: {rec['url']}, Similarity: {rec['similarity']:.4f}, Keywords: {', '.join(rec['keywords'])}")


# 실행 예시
if __name__ == "__main__":
    # 기존 문서 데이터 및 하이퍼그래프 로드
    documents = load_json_data("./home/kuinisw5/data/EconFinNews_Raw.json")
    G = generate_G_from_H(construct_H_with_keywords(documents))

    # 개인화 추천 시스템 초기화 및 실행
    model_path = "./saves/hgnn_recommender_model.pth"  # train.py에서 저장한 모델 경로
    personalization_recommender = PersonalizedRecommender(model_path, documents, G)
    personalization_recommender.run("personalization_data")


# 이 추천 시스템은 HyperGNN을 사용한 train.py에서 저장한 saves/hgnn_recommender_model.pth를 이용하여
# 입력된 새 문서와 가장 유사한 문서를 추천합니다. 
# 유사도 계산, 하이퍼그래프 연결성, 키워드 기반 분석을 통해 사용자에게 풍부한 정보를 제공합니다. (코사인 유사도 기반)