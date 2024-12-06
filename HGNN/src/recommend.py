import torch
import torch.nn.functional as F
from scipy.spatial.distance import cosine

class KeywordRecommender:
    def __init__(self, model_path, data_path):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.data = torch.load(data_path)
        self.model = torch.load(model_path).to(self.device)
        
    def calculate_similarity(self, input_keyword, top_k=5):
        keywords = self.data['keywords']
        features = self.data['feature_matrix']
        
        input_idx = keywords.index(input_keyword)
        input_embedding = features[input_idx]
        
        similarities = []
        for i, keyword in enumerate(keywords):
            if keyword != input_keyword:
                similarity = 1 - cosine(input_embedding, features[i])
                similarities.append((keyword, similarity))
        
        return sorted(similarities, key=lambda x: x[1], reverse=True)[:top_k]
    
    def recommend_keywords(self, input_keyword, top_k=5, threshold=0.7):
        recommendations = self.calculate_similarity(input_keyword, top_k)
        return [rec for rec in recommendations if rec[1] >= threshold]

def main():
    recommender = KeywordRecommender(
        'experiments/models/hgnn_model.pth', 
        'data/processed_data/hgnn_input.pt'
    )
    
    input_keyword = '인플레이션'
    recommendations = recommender.recommend_keywords(input_keyword)
    print(f"Recommendations for '{input_keyword}':")
    for keyword, similarity in recommendations:
        print(f"{keyword}: {similarity}")

if __name__ == "__main__":
    main()