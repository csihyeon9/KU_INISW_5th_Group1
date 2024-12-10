# recommend.py
import torch
import yaml
import json
from typing import List, Dict

from src.models import KeywordHGNN, RecommendationModule
from src.data_processor import KeywordProcessor

def load_config(config_path: str = 'config/config.yaml') -> dict:
   with open(config_path, 'r') as f:
       return yaml.safe_load(f)

class ArticleRecommender:
    def __init__(self, model_path: str = 'results/models/best_model.pt'):
        # 설정 로드
        self.config = load_config()
        
        # 데이터 프로세서 초기화
        self.processor = KeywordProcessor(
            self.config['data']['unique_keywords_path'],
            self.config['data']['news_data_path']
        )
        self.processor.load_data()
        
        # 모델 로드
        self.model = KeywordHGNN(
            num_keywords=self.processor.matrix_size,
            hidden_dim=self.config['model']['hidden_dim'],
            num_layers=self.config['model']['num_layers'],
            dropout=self.config['model']['dropout']
        )
        
        checkpoint = torch.load(model_path, map_location='cpu')
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        # 추천 모듈 초기화 - config 제거
        self.recommender = RecommendationModule(
            self.model,
            self.processor.keyword2idx,
            self.processor.idx2keyword
        )
        
        # 뉴스 데이터 로드
        with open(self.config['data']['news_data_path'], 'r', encoding='utf-8') as f:
            self.news_data = json.load(f)
            
    def recommend(self, keywords: List[str], top_k: int = 5) -> List[Dict]:
        """키워드 기반 기사 및 키워드 추천"""
        # news_data를 직접 전달
        raw_recommendations = self.recommender.get_recommendations(
            keywords,
            top_k=top_k,
            news_data=self.news_data  # 뉴스 데이터 전달
        )

        # JSON 직렬화 가능한 형태로 변환
        recommendations = [
            {
                'keyword': rec['keyword'],
                'similarity': float(rec['similarity']),  # tensor나 numpy값을 float로 변환
                'articles': [
                    {
                        'title': article['title'],
                        'url': article['url'],
                        'date': article['date'],
                        'keywords': article['keywords']
                    }
                    for article in rec['articles']
                ]
            }
            for rec in raw_recommendations
        ]
        
        return recommendations

def main():
    recommender = ArticleRecommender()
    
    while True:
        print("\n키워드를 입력하세요 (쉼표로 구분, 종료하려면 q):")
        user_input = input().strip()
       
        if user_input.lower() == 'q':
            break
           
        keywords = [k.strip() for k in user_input.split(',')]
        print(f"\n입력한 키워드: {keywords}")
       
        recommendations = recommender.recommend(keywords)
       
        print("\n=== 추천 결과 ===")

        if len(recommendations) == 0:
            print("\n유효한 키워드가 아닙니다.")

        for i, rec in enumerate(recommendations, 1):
            print(f"\n{i}. 추천 키워드: {rec['keyword']}")
            print(f"   유사도: {rec['similarity']:.4f}")
           
            print("\n   관련 기사:")
            for j, article in enumerate(rec['articles'], 1):
                print(f"\n   {j}) 제목: {article['title']}")
                print(f"      URL: {article['url']}")
                print(f"      키워드: {', '.join(article['keywords'])}")
                print(f"      기사날짜: {article['date']}")
       
        print("\n" + "="*50)

if __name__ == '__main__':
   main()