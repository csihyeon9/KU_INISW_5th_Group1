import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphConvolution(nn.Module):
    """
    그래프 합성곱 레이어 클래스
    """
    def __init__(self, in_features, out_features, bias=True):
        """
        :param in_features: 입력 특성 차원
        :param out_features: 출력 특성 차원
        :param bias: 편향 사용 여부
        """
        super(GraphConvolution, self).__init__()
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
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(GraphAttentionLayer, self).__init__()
        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        self.a = nn.Parameter(torch.zeros(size=(2 * out_features, 1)))  # (128, 1)
        self.leakyrelu = nn.LeakyReLU(0.2)
        
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W.data)
        nn.init.xavier_uniform_(self.a.data)

    def forward(self, h, adj):
        Wh = torch.mm(h, self.W)  # (N, out_features) -> (N, 64)
        N = Wh.size(0)

        # Compute attention coefficients
        a_input = self._prepare_attentional_mechanism_input(Wh)

        # Concatenate the two tensors for attention calculation
        e = self.leakyrelu(torch.matmul(a_input[0], self.a[:64]) + torch.matmul(a_input[1], self.a[64:]))  # (N*N, 128) * (128, 1)

        # Apply softmax to get attention weights
        attention = F.softmax(e.view(N, N), dim=1)
        
        # Return the output features
        return torch.matmul(attention, Wh)

    def _prepare_attentional_mechanism_input(self, Wh):
        N = Wh.size(0)
        
        # Prepare input for attention mechanism
        return Wh.repeat(1, N).view(N * N, -1), Wh.repeat(N, 1)