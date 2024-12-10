# models.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HypergraphConv
import math
from typing import List, Dict, Union

class KeywordHGNN(nn.Module):
    def __init__(self, num_keywords: int, hidden_dim: int = 64, num_layers: int = 3, dropout: float = 0.5):
        super().__init__()
        self.num_keywords = num_keywords
        self.hidden_dim = hidden_dim
        
        # 키워드 임베딩 레이어
        self.embedding = nn.Embedding(num_keywords, hidden_dim)
        
        # HypergraphConv 레이어들
        self.convs = nn.ModuleList()
        self.convs.append(HypergraphConv(hidden_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.convs.append(HypergraphConv(hidden_dim, hidden_dim))
            
        # Attention 레이어들
        self.attention_q = nn.Linear(hidden_dim, hidden_dim)
        self.attention_k = nn.Linear(hidden_dim, hidden_dim)
        self.attention_v = nn.Linear(hidden_dim, hidden_dim)
        
        # 드롭아웃
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, hyperedge_index: torch.Tensor, hyperedge_weight: torch.Tensor = None) -> torch.Tensor:
        # 초기 임베딩
        x = self.embedding.weight
        
        # HypergraphConv 레이어들 통과
        for conv in self.convs[:-1]:
            x = conv(x, hyperedge_index, hyperedge_weight)
            x = F.relu(x)
            x = self.dropout(x)
        
        # 마지막 레이어
        x = self.convs[-1](x, hyperedge_index, hyperedge_weight)
        
        return x
        
    def compute_attention(self, node_embeddings: torch.Tensor) -> torch.Tensor:
        """어텐션 메커니즘을 사용해 노드 임베딩 결합"""
        Q = self.attention_q(node_embeddings)
        K = self.attention_k(node_embeddings)
        V = self.attention_v(node_embeddings)
        
        # 스케일드 닷-프로덕트 어텐션
        attention_weights = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.hidden_dim)
        attention_weights = F.softmax(attention_weights, dim=-1)
        
        attended_values = torch.matmul(attention_weights, V)
        return attended_values.mean(dim=0)  # 평균 풀링
        
    def get_embedding(self, indices: torch.Tensor) -> torch.Tensor:
        """특정 인덱스의 임베딩 반환"""
        return self.embedding(indices)

class RecommendationModule:
    def __init__(self, model: KeywordHGNN, keyword2idx: dict, idx2keyword: dict):
        self.model = model
        self.keyword2idx = keyword2idx
        self.idx2keyword = idx2keyword
        self.device = next(model.parameters()).device
        
    def get_recommendations(self, 
                       query_keywords: List[str], 
                       top_k: int = 5,
                       news_data: List[Dict] = None) -> List[Dict]:
        """키워드 기반 추천"""
        self.model.eval()
        
        with torch.no_grad():
            # 쿼리 키워드 인덱스 변환
            query_indices = torch.tensor([
                self.keyword2idx[k] for k in query_keywords if k in self.keyword2idx
            ], device=self.device)
            
            if len(query_indices) == 0:
                return []
            
            # 쿼리 임베딩 계산
            query_embeddings = self.model.get_embedding(query_indices)
            query_vector = self.model.compute_attention(query_embeddings)
            
            # 모든 키워드와의 유사도 계산
            all_embeddings = self.model.embedding.weight
            similarities = F.cosine_similarity(
                query_vector.unsqueeze(0),
                all_embeddings
            )
            
            # Top-K 추천 결과 가져오기
            top_k_values, top_k_indices = similarities.topk(top_k + len(query_keywords))
            
            # 입력 키워드 제외하고 추천 결과 생성
            recommendations = []
            
            for idx, sim in zip(top_k_indices, top_k_values):
                keyword = self.idx2keyword[idx.item()]
                if keyword not in query_keywords:
                    rec = {
                        'keyword': keyword,
                        'similarity': sim.item()
                    }
                    
                    if news_data:
                        # 해당 키워드를 포함하는 기사들 찾기
                        relevant_articles = []
                        for article in news_data:
                            if keyword in article['all_keywords']:
                                relevant_articles.append({
                                    'title': article['title'],
                                    'url': article.get('url', 'No URL'),
                                    'keywords': article['all_keywords'],
                                    'date': article.get('date', 'No date')
                                })
                        rec['articles'] = relevant_articles[:3]  # 최대 3개 기사만
                    
                    recommendations.append(rec)
                    
                    if len(recommendations) >= top_k:
                        break
                    
            return recommendations
            
    def _generate_explanation(self, 
                            query_keywords: List[str], 
                            recommended_keyword: str, 
                            similarity: float) -> str:
        """추천 결과에 대한 설명 생성"""
        explanation = f"'{recommended_keyword}'(이)가 추천된 이유:\n"
        explanation += f"- 입력한 키워드 {query_keywords}와(과)의 유사도: {similarity:.3f}\n"
        
        # 추가적인 설명 로직을 구현할 수 있음
        # 예: 어떤 문맥에서 자주 함께 등장하는지 등
        
        return explanation