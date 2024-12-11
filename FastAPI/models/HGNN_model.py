import torch
from torch import nn
from pathlib import Path
import torch.nn.functional as F
from models.layers import HGNN_conv  # HGNN 레이어 모듈 / # GraphConvolutionlayer, GraphAttentionLayer


class HGNN(nn.Module):
    def __init__(self, in_ch, n_class, n_hid, dropout=0.5):
        """
        in_ch: 입력 피처 차원
        n_hid: 은닉 레이어 차원
        n_class: 출력 클래스 개수
        dropout: 드롭아웃 비율
        """
        super(HGNN, self).__init__()
        self.hgc1 = HGNN_conv(in_ch, n_hid)  # 첫 번째 HGNN 레이어
        self.hgc2 = HGNN_conv(n_hid, n_class)  # 두 번째 HGNN 레이어
        self.dropout = dropout

    def forward(self, X, G_sparse):
        """
        Forward Pass
        x: 노드 피처 매트릭스 (N, F_in)
        G: 하이퍼그래프 라플라시안 (N, N)
        return: 분류 결과 (N, F_out)

        X: 노드 특징 행렬 (\( \text{num\_nodes}, \text{num\_features} \))
        G_sparse: 희소 라플라시안 행렬 (\( \text{num\_nodes}, \text{num\_nodes} \))
        """
        X = torch.sparse.mm(G_sparse, X)  # 희소 행렬 곱
        X = F.relu(self.hgc1(X, G_sparse))  # 첫 번째 레이어
        X = F.dropout(X, self.dropout, training=self.training)
        X = self.hgc2(X, G_sparse)  # 두 번째 레이어
        return X
    
class HGNN_basic(nn.Module):
    def __init__(self, in_ch, n_class, n_hid, dropout=0.5):
        """
        in_ch: 입력 피처 차원
        n_hid: 은닉 레이어 차원
        n_class: 출력 클래스 개수
        dropout: 드롭아웃 비율
        """
        super(HGNN, self).__init__()
        self.hgc1 = HGNN_conv(in_ch, n_hid)  # 첫 번째 HGNN 컨볼루션 레이어
        self.hgc2 = HGNN_conv(n_hid, n_class)  # 두 번째 HGNN 컨볼루션 레이어
        self.dropout = dropout

    def forward(self, x, H, G):
        """
        x: 입력 노드 특징 (dense matrix)
        H: 인시던스 행렬 (dense matrix)
        G_sparse: 희소 라플라시안 행렬 (sparse matrix)
        """
        # 첫 번째 컨볼루션
        x = F.relu(self.hgc1(x, G))  # 희소 행렬과의 연산
        x = F.dropout(x, self.dropout, training=self.training)

        # 두 번째 컨볼루션
        x = self.hgc2(x, G)
        return x