import torch
import torch.nn as nn
import torch.nn.functional as F
from models.layers import GraphConvolution, GraphAttentionLayer

# class HGNN(nn.Module):
#     """
#     하이퍼그래프 신경망 모델 클래스
#     """
#     def __init__(self, in_features, hidden_features, out_features, dropout):
#         """
#         :param in_features: 입력 특성 차원
#         :param hidden_features: 은닉층 특성 차원
#         :param out_features: 출력 특성 차원
#         :param dropout: 드롭아웃 비율
#         """
#         super(HGNN, self).__init__()
#         self.gc1 = GraphConvolution(in_features, hidden_features)
#         self.gc2 = GraphConvolution(hidden_features, out_features)
#         self.dropout = dropout

#     def forward(self, x, adj):
#         """
#         순전파 함수
#         :param x: 입력 특성
#         :param adj: 인접 행렬
#         :return: 로그 소프트맥스 출력
#         """
#         x = F.relu(self.gc1(x, adj))
#         x = F.dropout(x, self.dropout, training=self.training)
#         x = self.gc2(x, adj)
#         return F.log_softmax(x, dim=1)

class HGNN(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, dropout):
        super(HGNN, self).__init__()
        self.gat1 = GraphAttentionLayer(in_features, hidden_features)
        self.gat2 = GraphAttentionLayer(hidden_features, out_features)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gat1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gat2(x, adj)
        return F.log_softmax(x, dim=1)