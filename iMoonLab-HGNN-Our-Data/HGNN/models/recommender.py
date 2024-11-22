# recommender.py
import torch.nn as nn
import torch.nn.functional as F
from models import HGNN_conv

class HGNNRecommender(nn.Module):
    def __init__(self, in_dim, hidden_dim, embedding_dim):
        super(HGNNRecommender, self).__init__()
        self.embedding_dim = embedding_dim
        
        # 3-layer structure
        self.hgc1 = HGNN_conv(in_dim, hidden_dim*2)
        self.hgc2 = HGNN_conv(hidden_dim*2, hidden_dim)
        self.hgc3 = HGNN_conv(hidden_dim, embedding_dim)
        
        self.bn1 = nn.BatchNorm1d(hidden_dim*2)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        
        self.dropout = 0.5

    def forward(self, x, G):
        # Generate document embeddings
        x = F.relu(self.bn1(self.hgc1(x, G)))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.bn2(self.hgc2(x, G)))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.hgc3(x, G)
        
        # Normalize embeddings
        x = F.normalize(x, p=2, dim=1)
        return x