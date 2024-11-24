# models/layers.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from typing import Optional

class HGNN_conv(nn.Module):
    """하이퍼그래프 컨볼루션 레이어
    
    하이퍼그래프의 메시지 전달을 수행하는 기본 레이어입니다.
    향후 확장성을 고려하여 다양한 집계 함수를 지원할 수 있도록 설계되었습니다.
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        aggregate_fn: str = 'mean'
    ):
        super(HGNN_conv, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.aggregate_fn = aggregate_fn
        
        # 학습 가능한 가중치 행렬
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
            
        self.reset_parameters()

    def reset_parameters(self):
        """가중치 초기화"""
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(
        self,
        x: torch.Tensor,
        G: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            x: 입력 노드 특징 (batch_size, in_features)
            G: 하이퍼그래프 라플라시안 (batch_size, batch_size)
            
        Returns:
            output: 변환된 노드 특징 (batch_size, out_features)
        """
        # 특징 변환
        x = torch.matmul(x, self.weight)
        
        if self.bias is not None:
            x = x + self.bias
            
        # 배치 단위로 메시지 전달
        batch_size = x.size(0)
        output = torch.matmul(G[:batch_size, :batch_size], x)
        
        return output

    def extra_repr(self) -> str:
        """모델 출력용 문자열"""
        return f'in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}'
