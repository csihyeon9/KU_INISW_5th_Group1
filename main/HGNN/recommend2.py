# recommend2.py -> link prediction 기반 추천 시스템
import torch
import yaml
from pathlib import Path
import json
import torch.nn.functional as F
from typing import List, Dict

from src.ver2.models2 import KeywordHGNN
from src.ver2.trainer2 import Trainer
from src.ver2.data_processor2 import KeywordProcessor

def load_config(config_path: str = 'config/config.yaml') -> dict:
    """YAML 설정 파일 로드"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

class ArticleRecommender:
    def __init__(self, 
                 model_path: str = 'results/models/best_model.pt', 
                 config_path: str = 'config/config.yaml'):
        """
        링크 예측 기반 추천 시스템 초기화
        
        Args:
            model_path (str): 사전 학습된 모델 경로
            config_path (str): 설정 파일 경로
        """
        # 설정 로드
        self.config = load_config(config_path)
        
        # 데이터 프로세서 초기화
        self.processor = KeywordProcessor(
            self.config['data']['unique_keywords_path'],
            self.config['data']['news_data_path']
        )
        self.processor.load_data()
        
        # 모델 초기화 및 로드
        self.model = KeywordHGNN(
            num_keywords=self.processor.matrix_size,
            hidden_dim=self.config['model']['hidden_dim'],
            num_layers=self.config['model']['num_layers'],
            dropout=self.config['model']['dropout']
        )
        
        # 체크포인트에서 모델 상태 로드
        checkpoint = torch.load(model_path, map_location='cpu')
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        # 트레이너 초기화 (선택적)
        self.trainer = Trainer(
            model=self.model, 
            config=self.config, 
            keyword2idx=self.processor.keyword2idx,
            idx2keyword=self.processor.idx2keyword
        )
        
        # 뉴스 데이터 로드
        with open(self.config['data']['news_data_path'], 'r', encoding='utf-8') as f:
            self.news_data = json.load(f)
    
    def recommend(self, 
                  keywords: List[str], 
                  top_k: int = 5, 
                  similarity_threshold: float = 0.5) -> List[Dict]:
        """
        키워드 기반 링크 예측 추천
        
        Args:
            keywords (List[str]): 추천의 기준이 되는 키워드 리스트
            top_k (int): 추천할 키워드 개수
            similarity_threshold (float): 추천 필터링을 위한 유사도 임계값
        
        Returns:
            List[Dict]: 추천된 키워드와 관련 기사 정보
        """
        # 모델을 평가 모드로 설정
        self.model.eval()
        
        # 유효한 키워드 인덱스 선택
        query_indices = [
            self.processor.keyword2idx[k] 
            for k in keywords 
            if k in self.processor.keyword2idx
        ]
        
        if not query_indices:
            print("\n입력된 키워드 중 유효한 키워드가 없습니다.")
            return []
        
        # 다양한 추천 메서드 시도
        recommendations = []
        
        # 1. 트레이너의 키워드 추천 메서드 사용
        try:
            recommended_keywords = self.trainer.recommend_keywords(keywords, top_k)
        except Exception as e:
            print(f"트레이너 추천 메서드 실패: {e}")
            recommended_keywords = []
        
        # 2. 직접 링크 예측 (대체 방법)
        if not recommended_keywords:
            # 후보 키워드 생성
            candidate_indices = list(
                set(range(self.model.num_keywords)) - set(query_indices)
            )
            
            # 노드 쌍 생성
            query_tensor = torch.tensor(query_indices, dtype=torch.long)
            candidate_tensor = torch.tensor(candidate_indices, dtype=torch.long)
            node_pairs = torch.cartesian_prod(query_tensor, candidate_tensor)
            
            # 링크 예측 점수 계산
            with torch.no_grad():
                scores = self.model.predict_links(node_pairs)
            
            # Top-K 점수 인덱스 선택
            top_k_indices = scores.topk(top_k).indices
            
            # 추천 키워드 추출
            recommended_keywords = [
                self.processor.idx2keyword[node_pairs[idx, 1].item()] 
                for idx in top_k_indices
            ]
        
        # 관련 기사 추천
        for recommended_keyword in recommended_keywords:
            # 관련 기사 검색
            relevant_articles = [
                {
                    'title': article['title'],
                    'url': article.get('url', 'No URL'),
                    'keywords': article['all_keywords'],
                    'date': article.get('date', 'No date')
                }
                for article in self.news_data
                if recommended_keyword in article['all_keywords']
            ][:3]  # 최대 3개 기사
            
            recommendations.append({
                'keyword': recommended_keyword,
                'articles': relevant_articles
            })
        
        return recommendations


def main():
    """추천 시스템 실행"""
    # ArticleRecommender 초기화
    recommender = ArticleRecommender(
        model_path="results/models/best_model.pt",
        config_path="config/config.yaml"
    )
    
    while True:
        print("\n키워드를 입력하세요 (쉼표로 구분, 종료하려면 q):")
        user_input = input().strip()
        
        if user_input.lower() == 'q':
            break
        
        # 입력 키워드 처리
        keywords = [k.strip() for k in user_input.split(',')]
        print(f"\n입력한 키워드: {keywords}")
        
        # 링크 예측 기반 추천
        recommendations = recommender.recommend(keywords, top_k=5)
        
        # 결과 출력
        print("\n=== 추천 결과 ===")
        for i, rec in enumerate(recommendations, 1):
            print(f"\n{i}. 추천 키워드: {rec['keyword']}")
            
            print("\n   관련 기사:")
            for j, article in enumerate(rec['articles'], 1):
                print(f"   {j}) 제목: {article['title']}")
                print(f"      URL: {article['url']}")
                print(f"      키워드: {', '.join(article['keywords'])}")
                print(f"      날짜: {article['date']}")
        
        print("\n" + "=" * 50)

if __name__ == '__main__':
    main()