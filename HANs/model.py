# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple

class HypergraphAttention(nn.Module):
    """Hypergraph Attention Layer
    
    하이퍼엣지 단위로 attention을 계산하고, 
    엣지 내의 노드들 간의 집합적 상호작용을 모델링합니다.
    """
    def __init__(self, hidden_size: int, heads: int = 1, dropout: float = 0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.heads = heads
        self.head_dim = hidden_size // heads
        assert self.head_dim * heads == hidden_size, "hidden_size must be divisible by heads"
        
        # Edge attention parameters
        self.edge_attention = nn.Parameter(torch.Tensor(heads, self.head_dim, 1))
        
        # Node feature transformations
        self.W = nn.Linear(hidden_size, hidden_size)
        self.a = nn.Parameter(torch.Tensor(heads, 2 * self.head_dim))
        
        # Output transformation
        self.output_transform = nn.Linear(hidden_size, hidden_size)
        
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize parameters"""
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.edge_attention, gain=gain)
        nn.init.xavier_normal_(self.a, gain=gain)
        nn.init.xavier_normal_(self.W.weight, gain=gain)
        
    def edge_attention_pattern(self, edge_index: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """하이퍼엣지 단위의 attention pattern 계산"""
        num_nodes = x.size(0)
        x = self.W(x)  # Node feature transformation
        
        # Reshape for multi-head attention
        x = x.view(num_nodes, self.heads, self.head_dim)  # 수정: -1 대신 num_nodes 사용
        
        # For each hyperedge, compute attention between all nodes in the edge
        edge_weights = []
        unique_edges, edge_inverse = torch.unique(edge_index[1], return_inverse=True)
        
        for e in unique_edges:
            # Get nodes in current hyperedge
            nodes_in_edge = edge_index[0][edge_inverse == e]
            if len(nodes_in_edge) < 2:
                continue
                
            # Get features of nodes in edge
            edge_features = torch.index_select(x, 0, nodes_in_edge)  # (edge_size, heads, head_dim)
            
            # Compute pairwise attention scores within edge
            score = self._hyperedge_attention(edge_features)
            edge_weights.append((e, nodes_in_edge, score))
            
        return edge_weights

    def _hyperedge_attention(self, edge_features: torch.Tensor) -> torch.Tensor:
        """하이퍼엣지 내부의 attention 계산"""
        edge_size = edge_features.size(0)
        
        # Reshape edge features to match attention parameter dimension
        # edge_features: (edge_size, heads, head_dim) -> (heads, edge_size, head_dim)
        edge_features = edge_features.permute(1, 0, 2)
        
        # Compute attention logits
        # self.edge_attention: (heads, head_dim, 1)
        # Result: (heads, edge_size, 1)
        logits = torch.matmul(edge_features, self.edge_attention)
        
        # Normalize attention weights across nodes in edge
        attention = F.softmax(logits, dim=1)  # (heads, edge_size, 1)
        
        # Permute back to original shape: (edge_size, heads, 1)
        attention = attention.permute(1, 0, 2)
        
        return self.dropout(attention)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        num_nodes = x.size(0)
        
        # Compute edge attention patterns
        edge_attention_patterns = self.edge_attention_pattern(edge_index, x)
        
        # Aggregate messages for each node
        output = torch.zeros(num_nodes, x.size(1), device=x.device)  # 차원 명시적 지정
        attention_weights = torch.zeros(num_nodes, num_nodes, device=x.device)
        
        for edge_idx, nodes_in_edge, attention in edge_attention_patterns:
            # Update features for nodes in edge
            nodes_features = x[nodes_in_edge]  # (edge_size, hidden_size)
            
            # attention: (edge_size, heads, 1)
            # Reshape attention for broadcasting
            attention = attention.squeeze(-1)  # (edge_size, heads)
            attention = attention.mean(dim=1)  # (edge_size,)
            
            # 수정된 부분: expand 대신 unsqueeze 만 사용하여 [edge_size, 1]로 변환
            attention_scaled = attention.unsqueeze(1)  # (edge_size, 1)
            edge_contribution = attention_scaled * nodes_features  # (edge_size, hidden_size)
            
            # Update output features
            output[nodes_in_edge] += edge_contribution
            
            # Store attention weights for visualization/analysis
            for i, node_i in enumerate(nodes_in_edge):
                for j, node_j in enumerate(nodes_in_edge):
                    if node_i != node_j:  # Prevent self-loops
                        # 주의: attention[i]는 이제 스칼라이므로 .item() 사용 가능
                        attention_weights[node_i, node_j] += attention[i].item()
        
        # Normalize output
        edge_counts = torch.zeros(num_nodes, device=x.device)
        edge_counts.scatter_add_(0, edge_index[0], torch.ones_like(edge_index[0], dtype=torch.float))
        edge_counts = edge_counts.clamp(min=1.0)  # Avoid division by zero
        output = output / edge_counts.view(-1, 1)
        
        # Final transformation with normalization
        output = self.output_transform(output)
        output = nn.LayerNorm(output.size(-1)).to(output.device)(output)
        
        attention_info = {
            'attention_weights': attention_weights,
            'edge_patterns': edge_attention_patterns
        }
        
        return output, attention_info

class HGAN(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int = None,
        num_layers: int = 3,
        num_heads: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.hidden_features = hidden_features or in_features
        
        # Input projection
        self.input_proj = nn.Linear(in_features, hidden_features)
        
        # HGAN layers
        self.layers = nn.ModuleList([
            HypergraphAttention(
                hidden_size=hidden_features,
                heads=num_heads,
                dropout=dropout
            ) for _ in range(num_layers)
        ])
        
        # Layer normalization
        self.norms = nn.ModuleList([
            nn.LayerNorm(hidden_features)
            for _ in range(num_layers)
        ])
        
        # Output projection
        self.output_proj = nn.Linear(hidden_features, in_features)
        
        # Uncertainty estimation
        self.uncertainty = nn.Sequential(
            nn.Linear(hidden_features, hidden_features // 2),
            nn.ReLU(),
            nn.Linear(hidden_features // 2, 1),
            nn.Sigmoid()
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Initial projection
        h = self.input_proj(x)  # (num_nodes, hidden_features)
        
        # Store attention info from all layers
        attention_info = []
        
        # Process through HGAN layers
        for layer, norm in zip(self.layers, self.norms):
            # Apply attention
            layer_out, layer_attention = layer(h, edge_index)
            
            # Residual connection and normalization
            h = norm(h + self.dropout(layer_out))
            
            attention_info.append(layer_attention)
        
        # Output projection이 필요없음 (hidden representation 사용)
        # output = self.output_proj(h)
        
        return {
            'embeddings': h,  # hidden representation 반환
            'hidden_features': h,
            'attention_info': attention_info,
            'uncertainty': self.uncertainty(h)
        }

class HGANPredictor(nn.Module):
    """하이퍼그래프 관계 예측 모듈"""
    def __init__(self, hidden_size: int, num_relations: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, 1)
        )
    
    def forward(self, node_features_i: torch.Tensor, node_features_j: torch.Tensor) -> torch.Tensor:
        """
        Args:
            node_features_i: 첫 번째 노드의 특성 (batch_size, hidden_size)
            node_features_j: 두 번째 노드의 특성 (batch_size, hidden_size)
        """
        # 두 노드의 특성을 연결
        pair_features = torch.cat([node_features_i, node_features_j], dim=1)
        return self.mlp(pair_features)