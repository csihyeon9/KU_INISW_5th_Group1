import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HypergraphConv
from typing import List, Dict

# 1. KeywordHGNN 모델 정의
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
            
        # 드롭아웃
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, hyperedge_index: torch.Tensor, hyperedge_weight: torch.Tensor = None) -> torch.Tensor:
        x = self.embedding.weight  # 초기 임베딩
        for conv in self.convs[:-1]:
            x = conv(x, hyperedge_index, hyperedge_weight)
            x = F.relu(x)
            x = self.dropout(x)
        x = self.convs[-1](x, hyperedge_index, hyperedge_weight)
        return x  # 노드 임베딩 반환

    def predict_links(self, node_pairs: torch.Tensor) -> torch.Tensor:
        node1_emb = self.embedding(node_pairs[:, 0])
        node2_emb = self.embedding(node_pairs[:, 1])
        scores = (node1_emb * node2_emb).sum(dim=1)
        
        return torch.sigmoid(scores)  # 확률로 변환

# 2. LinkPredictor 클래스 정의
class LinkPredictor:
    def __init__(self, model: KeywordHGNN, learning_rate: float = 0.001, weight_decay: float = 0.0001):
        
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.criterion = nn.BCELoss()
    
    def train(self, hyperedge_index: torch.Tensor, num_epochs: int = 100, batch_size: int = 128):
        num_nodes = self.model.num_keywords
        for epoch in range(num_epochs):
            self.model.train()
            
            positive_edges = hyperedge_index.t()
            random_edges = torch.randint(0, num_nodes, positive_edges.shape, device=positive_edges.device)
            
            edges = torch.cat([positive_edges, random_edges], dim=0)
            
            labels = torch.cat([
                torch.ones(positive_edges.size(0), device=positive_edges.device),
                torch.zeros(random_edges.size(0), device=random_edges.device)
            ])
            perm = torch.randperm(edges.size(0))
            edges = edges[perm]
            labels = labels[perm]
            
            for i in range(0, edges.size(0), batch_size):
                batch_edges = edges[i:i + batch_size]
                batch_labels = labels[i:i + batch_size]
                
                predictions = self.model.predict_links(batch_edges)
                loss = self.criterion(predictions, batch_labels)
                
                self.optimizer.zero_grad()
                loss.backward()
                
                self.optimizer.step()
            
            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item():.4f}")

# 3. RecommendationModule 클래스 정의
class RecommendationModule:
    def __init__(self, model: KeywordHGNN, keyword2idx: dict, idx2keyword: dict):
        self.model = model
        self.keyword2idx = keyword2idx
        self.idx2keyword = idx2keyword
    
    def recommend_links(self, query_keywords: List[str], top_k: int = 5) -> List[str]:
        self.model.eval()
        query_indices = [self.keyword2idx[k] for k in query_keywords if k in self.keyword2idx]

        if not query_indices:
            print("No valid keywords found in query.")
            return []
        
        candidate_indices = set(range(self.model.num_keywords)) - set(query_indices)
        query_tensor = torch.tensor(query_indices, dtype=torch.long)
        candidate_tensor = torch.tensor(list(candidate_indices), dtype=torch.long)
        node_pairs = torch.cartesian_prod(query_tensor, candidate_tensor)
        
        with torch.no_grad():
            scores = self.model.predict_links(node_pairs)
        top_k_indices = scores.topk(top_k).indices
        recommendations = [self.idx2keyword[node_pairs[idx, 1].item()] for idx in top_k_indices]
        return recommendations

# 4. JSON 데이터에서 hyperedge_index 생성 함수
def create_hyperedge_index_from_json(json_path, keyword2idx):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    hyperedge_list = []
    for article in data:
        relations = article.get('relations', {})
        for category, relation_list in relations.items():
            hyperedge_id = int(category)
            for relation in relation_list:
                keywords = [keyword2idx[k] for k in relation if k in keyword2idx]
                for keyword in keywords:
                    hyperedge_list.append((keyword, hyperedge_id))
    return torch.tensor(hyperedge_list, dtype=torch.long).t()

