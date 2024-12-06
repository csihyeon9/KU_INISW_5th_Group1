import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HypergraphConv
from transformers import BertModel, BertTokenizer

class AdvancedHypergraphNetwork(nn.Module):
    def __init__(self, 
                 num_keywords, 
                 hidden_dim=128, 
                 num_layers=3, 
                 dropout=0.3,
                 bert_model='bert-base-multilingual-cased'):
        super().__init__()
        
        # BERT 임베딩 로드
        self.bert = BertModel.from_pretrained(bert_model)
        self.tokenizer = BertTokenizer.from_pretrained(bert_model)
        
        # 멀티헤드 셀프 어텐션
        self.multihead_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim, 
            num_heads=4, 
            dropout=dropout
        )
        
        # 하이퍼그래프 컨볼루션 레이어
        self.convs = nn.ModuleList([
            HypergraphConv(hidden_dim, hidden_dim, use_attention=True) 
            for _ in range(num_layers)
        ])
        
        # 하이퍼에지 가중치 학습 레이어
        self.hyperedge_weight_generator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        # 최종 분류/임베딩 레이어
        self.final_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.BatchNorm1d(hidden_dim//2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim//2, hidden_dim)
        )
        
    def forward(self, 
                keyword_indices, 
                hyperedge_index, 
                keyword_texts=None):
        # BERT 임베딩 (선택적)
        if keyword_texts is not None:
            bert_embeddings = []
            for text in keyword_texts:
                inputs = self.tokenizer(text, return_tensors='pt', truncation=True, max_length=32)
                with torch.no_grad():
                    outputs = self.bert(**inputs)
                    bert_embeddings.append(outputs.last_hidden_state.mean(dim=1))
            x = torch.stack(bert_embeddings)
        else:
            x = self.embedding(keyword_indices)
        
        # 멀티헤드 어텐션 적용
        x, _ = self.multihead_attention(x, x, x)
        
        # 하이퍼그래프 컨볼루션
        for conv in self.convs:
            # 동적 하이퍼에지 가중치 생성
            hyperedge_weights = self.hyperedge_weight_generator(x)
            x = conv(x, hyperedge_index, hyperedge_weights)
            x = F.relu(x)
        
        # 최종 임베딩 처리
        return self.final_layer(x)
    
    def compute_semantic_similarity(self, query_embeddings, candidate_embeddings):
        # 개선된 유사도 계산 (코사인 + 어텐션 기반)
        cos_sim = F.cosine_similarity(
            query_embeddings.unsqueeze(1), 
            candidate_embeddings.unsqueeze(0)
        )
        
        # 어텐션 기반 가중치 적용
        attention_weights = F.softmax(cos_sim, dim=-1)
        weighted_similarities = cos_sim * attention_weights
        
        return weighted_similarities

# 사용 예시
model = AdvancedHypergraphNetwork(num_keywords=1000)