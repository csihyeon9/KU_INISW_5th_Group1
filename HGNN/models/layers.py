import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


class HGNN_conv(nn.Module):
    """
    HGNN_conv (HyperGraph Neural Network Convolution Layer)
    역할: 라플라시안 정규화 행렬 𝐺 x 입력 특성 𝑋를 이용해 특성을 업데이트.
    
    구성: PMI weight -> 입력 특성과 출력 특성을 매핑하는 학습 가능한 가중치 / bias: 선택적으로 추가되는 편향
    순전파 과정: 1) 입력 𝑋에 PMI 가중치 행렬을 곱함. 2) 𝐺와 곱하여 특성을 전파 3) 편향이 있으면 더함.
    """
    def __init__(self, in_ft, out_ft, bias=True):
        super(HGNN_conv, self).__init__()
        self.weight = Parameter(torch.Tensor(in_ft, out_ft))
        if bias:
            self.bias = Parameter(torch.Tensor(out_ft))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x: torch.Tensor, G: torch.Tensor):
        x = x.matmul(self.weight)
        if self.bias is not None:
            x = x + self.bias
        x = G.matmul(x)
        return x

class HGNN_fc(nn.Module):
    """
    HGNN_fc (Fully Connected Layer for HGNN)
    역할: HGNN에서 마지막에 사용되는 Fully Connected Layer -> 노드 또는 하이퍼엣지의 특성을 특정 클래스나 값으로 변환.
    구성: fc -> 입력 특성을 출력 크기로 변환하는 선형 계층.
    """
    def __init__(self, in_ch, out_ch):
        super(HGNN_fc, self).__init__()
        self.fc = nn.Linear(in_ch, out_ch)

    def forward(self, x):
        return self.fc(x)

class HGNN_embedding(nn.Module):
    """
    HGNN_embedding:
    역할: HGNN에서 두 개의 HGNN_conv 레이어를 쌓아, 입력 특성을 점진적으로 변환하고 학습 가능한 임베딩을 생성.
    구성: hgc1, hgc2 -> 두 개의 HGNN_conv 레이어.
    dropout: 과적합 방지를 위해 사용.
    순전파 과정: 1) HGNN_conv를 적용하고, 활성화 함수(ReLU)를 사용.
    드롭아웃을 적용해 과적합 방지.
    두 번째 HGNN_conv를 적용.
    """
    def __init__(self, in_ch, n_hid, dropout=0.5):
        super(HGNN_embedding, self).__init__()
        self.dropout = dropout
        self.hgc1 = HGNN_conv(in_ch, n_hid)
        self.hgc2 = HGNN_conv(n_hid, n_hid)

    def forward(self, x, G):
        x = F.relu(self.hgc1(x, G))
        x = F.dropout(x, self.dropout)
        x = F.relu(self.hgc2(x, G))
        return x

class GraphConvolutionlayer(nn.Module):
    """
    그래프 합성곱 레이어 클래스
    """
    def __init__(self, in_features, out_features, bias=True):
        """
        :param in_features: 입력 특성 차원
        :param out_features: 출력 특성 차원
        :param bias: 편향 사용 여부
        """
        super(GraphConvolutionlayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
    
    def reset_parameters(self):
        """
        가중치와 편향 초기화 함수
        """
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, input, adj):
        """
        순전파 함수
        :param input: 입력 특성
        :param adj: 인접 행렬
        :return: 그래프 합성곱 결과
        """
        if input.size(1) != self.weight.size(0):  # 입력 데이터와 가중치 차원이 다를 경우
            print(f"Reshaping weight matrix from {self.weight.size()} to match input size {input.size(1)}.")
            self.weight = nn.Parameter(torch.FloatTensor(input.size(1), self.out_features))
            self.reset_parameters()
        
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        self.a = nn.Parameter(torch.zeros(size=(2 * out_features, 1)))
        self.leakyrelu = nn.LeakyReLU(0.2)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W.data)
        nn.init.xavier_uniform_(self.a.data)

    def forward(self, h, adj):
        if h.size(1) != self.W.size(0):
            print(f"Reshaping W: {self.W.size()} to match h size: {h.size(1)}")
            self.W = nn.Parameter(torch.zeros(size=(h.size(1), self.out_features)))
            self.reset_parameters()

        Wh = torch.mm(h, self.W)  # (N, in_features) * (in_features, out_features)
        e = self.leakyrelu(torch.mm(Wh, Wh.T))  # (N, N)

        attention = F.softmax(e, dim=1)  # Normalize attention scores
        return torch.mm(attention, Wh)  # Weighted sum of features
