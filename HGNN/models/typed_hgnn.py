# typed_hgnn.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional

class TypedHGNNConv(nn.Module):
    """관계 타입별 가중치를 가진 하이퍼그래프 컨볼루션 레이어"""
    def __init__(self, in_channels: int, out_channels: int, num_relation_types: int):
        super(TypedHGNNConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_relation_types = num_relation_types
        
        # 관계 타입별 가중치 행렬
        self.weights = nn.ParameterDict({
            f'weight_{i}': nn.Parameter(torch.Tensor(in_channels, out_channels))
            for i in range(num_relation_types)
        })
        
        # 관계 타입별 중요도
        self.relation_importance = nn.Parameter(torch.Tensor(num_relation_types))
        
        # 편향
        self.bias = nn.Parameter(torch.Tensor(out_channels))
        
        self.reset_parameters()
        
    def reset_parameters(self):
        """가중치 초기화"""
        for weight in self.weights.values():
            nn.init.xavier_uniform_(weight)
        nn.init.ones_(self.relation_importance)  # 초기에는 모든 관계를 동등하게 취급
        nn.init.zeros_(self.bias)
    
    def forward(self, X: torch.Tensor, H_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        순전파
        
        Args:
            X: 노드 특성 행렬 (N x F)
            H_dict: 관계 타입별 인시던스 행렬 딕셔너리
            
        Returns:
            torch.Tensor: 변환된 노드 특성
        """
        outputs = []
        
        # 각 관계 타입별로 처리
        for rel_idx, (rel_type, H) in enumerate(H_dict.items()):
            # 정규화된 라플라시안 계산
            D_v = torch.sum(H, dim=1)
            D_e = torch.sum(H, dim=0)
            
            D_v_invsqrt = torch.pow(D_v + 1e-8, -0.5)
            D_e_invsqrt = torch.pow(D_e + 1e-8, -0.5)
            
            D_v_invsqrt = torch.diag(D_v_invsqrt)
            D_e_invsqrt = torch.diag(D_e_invsqrt)
            
            # 메시지 전파
            theta = torch.matmul(torch.matmul(D_v_invsqrt, H), D_e_invsqrt)
            H_norm = torch.matmul(torch.matmul(theta, theta.t()), X)
            
            # 관계 타입별 변환
            out = torch.matmul(H_norm, self.weights[f'weight_{rel_idx}'])
            
            # 관계 중요도 적용
            out = out * F.sigmoid(self.relation_importance[rel_idx])
            
            outputs.append(out)
        
        # 모든 관계 타입의 결과 통합
        final_output = torch.sum(torch.stack(outputs), dim=0)
        return final_output + self.bias

class TypedHGNN(nn.Module):
    """관계 타입을 고려하는 HGNN 모델"""
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_relation_types: int,
        num_layers: int = 3,
        dropout: float = 0.2
    ):
        super(TypedHGNN, self).__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        
        # 레이어 크기 정의
        layer_sizes = [in_channels] + [hidden_channels] * (num_layers - 1) + [in_channels]
        
        # 컨볼루션 레이어
        self.convs = nn.ModuleList([
            TypedHGNNConv(
                layer_sizes[i],
                layer_sizes[i+1],
                num_relation_types
            )
            for i in range(num_layers)
        ])
        
        # 배치 정규화
        self.batch_norms = nn.ModuleList([
            nn.BatchNorm1d(size)
            for size in layer_sizes[1:-1]
        ])
        
        # 레이어 정규화
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(size)
            for size in layer_sizes[1:-1]
        ])
        
        # 관계 타입별 어텐션
        self.relation_attention = nn.Parameter(
            torch.ones(num_relation_types) / num_relation_types
        )
    
    def forward(
        self,
        X: torch.Tensor,
        H_dict: Dict[str, torch.Tensor],
        return_attention: bool = False
    ) -> torch.Tensor:
        """
        순전파
        
        Args:
            X: 입력 특성
            H_dict: 관계 타입별 인시던스 행렬
            return_attention: 관계 타입별 어텐션 점수 반환 여부
            
        Returns:
            torch.Tensor: 변환된 노드 표현
        """
        # 입력 저장 (잔차 연결용)
        identity = X
        
        # 관계 타입별 어텐션 정규화
        relation_weights = F.softmax(self.relation_attention, dim=0)
        
        # 가중치가 적용된 인시던스 행렬
        weighted_H_dict = {
            rel_type: H * relation_weights[i]
            for i, (rel_type, H) in enumerate(H_dict.items())
        }
        
        # 컨볼루션 레이어 통과
        for i in range(self.num_layers - 1):
            X = self.convs[i](X, weighted_H_dict)
            X = self.batch_norms[i](X)
            X = self.layer_norms[i](X)
            X = F.elu(X)
            X = F.dropout(X, p=self.dropout, training=self.training)
        
        # 마지막 레이어
        X = self.convs[-1](X, weighted_H_dict)
        
        # 잔차 연결
        if X.shape == identity.shape:
            X = X + identity
        
        if return_attention:
            return X, relation_weights
        return X

class RelationPredictor(nn.Module):
    """관계 타입 예측을 위한 모듈"""
    def __init__(self, embedding_dim: int, num_relation_types: int):
        super(RelationPredictor, self).__init__()
        
        self.predictor = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(embedding_dim, num_relation_types)
        )
    
    def forward(self, src_embed: torch.Tensor, dst_embed: torch.Tensor) -> torch.Tensor:
        """두 노드 임베딩으로부터 관계 타입 예측"""
        combined = torch.cat([src_embed, dst_embed], dim=-1)
        return self.predictor(combined)

class EnhancedHGNN(nn.Module):
    """관계 예측이 가능한 향상된 HGNN 모델"""
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_relation_types: int,
        num_layers: int = 3
    ):
        super(EnhancedHGNN, self).__init__()
        
        # 기본 HGNN
        self.hgnn = TypedHGNN(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            num_relation_types=num_relation_types,
            num_layers=num_layers
        )
        
        # 관계 예측기
        self.relation_predictor = RelationPredictor(
            embedding_dim=in_channels,
            num_relation_types=num_relation_types
        )
    
    def forward(
        self,
        X: torch.Tensor,
        H_dict: Dict[str, torch.Tensor],
        src_nodes: Optional[torch.Tensor] = None,
        dst_nodes: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        순전파
        
        Args:
            X: 노드 특성
            H_dict: 관계 타입별 인시던스 행렬
            src_nodes: 관계 예측을 위한 시작 노드 인덱스
            dst_nodes: 관계 예측을 위한 도착 노드 인덱스
            
        Returns:
            Dict: {
                'embeddings': 노드 임베딩,
                'relation_scores': 관계 타입 예측 점수 (옵션),
                'attention_weights': 관계 타입별 어텐션 가중치
            }
        """
        # 노드 임베딩 계산
        node_embeddings, attention_weights = self.hgnn(X, H_dict, return_attention=True)
        
        result = {
            'embeddings': node_embeddings,
            'attention_weights': attention_weights
        }
        
        # 관계 예측이 요청된 경우
        if src_nodes is not None and dst_nodes is not None:
            src_embeds = node_embeddings[src_nodes]
            dst_embeds = node_embeddings[dst_nodes]
            relation_scores = self.relation_predictor(src_embeds, dst_embeds)
            result['relation_scores'] = relation_scores
        
        return result