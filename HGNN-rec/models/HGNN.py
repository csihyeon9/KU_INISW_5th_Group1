# models/hgnn.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
from .layers import HGNN_conv

class HGNN(nn.Module):
    """기본 HGNN 모델
    
    하이퍼그래프 신경망의 기본 구조를 정의합니다.
    """
    
    def __init__(
        self,
        in_ch: int,
        n_class: int,
        n_hid: int,
        dropout: float = 0.5
    ):
        super(HGNN, self).__init__()
        self.dropout = dropout
        
        # 하이퍼그래프 컨볼루션 레이어
        self.hgc1 = HGNN_conv(in_ch, n_hid*2)
        self.hgc2 = HGNN_conv(n_hid*2, n_hid)
        self.hgc3 = HGNN_conv(n_hid, n_class)
        
        # 배치 정규화 레이어
        self.bn1 = nn.BatchNorm1d(n_hid*2)
        self.bn2 = nn.BatchNorm1d(n_hid)

    def forward(
        self,
        x: torch.Tensor,
        G: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            x: 입력 노드 특징
            G: 하이퍼그래프 라플라시안
            
        Returns:
            x: 변환된 노드 특징
        """
        x = F.relu(self.bn1(self.hgc1(x, G)))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.bn2(self.hgc2(x, G)))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.hgc3(x, G)
        return x

class HGNNRecommender(nn.Module):
    """경제 금융 뉴스 추천을 위한 HGNN 모델
    
    키워드 관계 학습과 추천을 위한 특화된 HGNN 모델입니다.
    """
    
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        embedding_dim: int,
        dropout: float = 0.5
    ):
        super(HGNNRecommender, self).__init__()
        self.embedding_dim = embedding_dim
        self.dropout = dropout
        
        # 하이퍼그래프 컨볼루션 레이어
        self.hgc1 = HGNN_conv(in_dim, hidden_dim*2)
        self.hgc2 = HGNN_conv(hidden_dim*2, hidden_dim)
        self.hgc3 = HGNN_conv(hidden_dim, embedding_dim)
        
        # 배치 정규화 레이어
        self.bn1 = nn.BatchNorm1d(hidden_dim*2)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        
        # 관계 임베딩 레이어 (추후 확장용)
        self.relation_embedding = nn.Parameter(torch.Tensor(30, embedding_dim))
        nn.init.xavier_uniform_(self.relation_embedding)

    def forward(self, x, G):
        # 첫 번째 레이어
        x = F.relu(self.bn1(self.hgc1(x, G)))
        x = F.dropout(x, self.dropout, training=self.training)
        
        # 두 번째 레이어
        x = F.relu(self.bn2(self.hgc2(x, G)))
        x = F.dropout(x, self.dropout, training=self.training)
        
        # 최종 레이어
        x = self.hgc3(x, G)
        
        # L2 정규화
        x = F.normalize(x, p=2, dim=1)
        
        return x  # 튜플이 아닌 단일 텐서 반환

    def get_recommendations(
        self,
        query_embedding: torch.Tensor,
        document_embeddings: torch.Tensor,
        top_k: int = 5
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """추천 문서 검색
        
        Args:
            query_embedding: 쿼리 문서 임베딩
            document_embeddings: 전체 문서 임베딩 행렬
            top_k: 추천 개수
            
        Returns:
            indices: 추천 문서 인덱스
            scores: 유사도 점수
        """
        # 코사인 유사도 계산
        similarities = F.cosine_similarity(
            query_embedding.unsqueeze(0),
            document_embeddings,
            dim=1
        )
        
        # Top-k 추천
        scores, indices = torch.topk(similarities, k=top_k)
        return indices, scores

    def compute_loss(self, embeddings, G_batch):
        """배치 단위 손실 계산
        
        Args:
            embeddings: 배치의 임베딩 (batch_size, embedding_dim)
            G_batch: 배치의 G 행렬 (batch_size, num_nodes)
        """
        batch_size = embeddings.size(0)
        
        # 배치 내 유사도 계산
        sim_matrix = torch.mm(embeddings, embeddings.t())  # (batch_size, batch_size)
        
        # 배치에 대한 G 행렬 처리
        G_batch_square = torch.mm(G_batch, G_batch.t())  # (batch_size, batch_size)
        
        # 포지티브/네거티브 마스크
        pos_mask = (G_batch_square > 0).float()
        neg_mask = (G_batch_square == 0).float()
        
        # 마진 기반 랭킹 손실
        margin = 0.3
        pos_scores = sim_matrix * pos_mask
        neg_scores = sim_matrix * neg_mask
        
        loss = torch.clamp(margin - pos_scores.unsqueeze(2) + neg_scores.unsqueeze(1), min=0)
        loss = loss.mean()
        
        return loss