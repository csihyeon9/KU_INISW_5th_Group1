import torch
from torch import nn
import torch.nn.functional as F
from models.layers import HGNN_conv

class HGNN(nn.Module):
    """
    HGNN 모델 정의
    - in_ch: 입력 피처 크기
    - n_hid: 히든 레이어 크기
    - n_class: 출력 클래스 수
    - dropout: 드롭아웃 비율
    """
    def __init__(self, in_ch, n_hid, n_class, dropout=0.5):
        super(HGNN, self).__init__()
        self.hgc1 = HGNN_conv(in_ch, n_hid)
        self.hgc2 = HGNN_conv(n_hid, n_class)
        self.dropout = dropout  # Dropout 비율 설정

    def forward(self, X, G_sparse):
        """
        순전파 정의
        - X: 입력 데이터 (특징 행렬)
        - G_sparse: 하이퍼그래프 희소 행렬
        """
        # 입력 크기와 희소 행렬 크기 일치 확인
        if G_sparse.shape[0] != X.shape[0]:
            raise ValueError(f"Mismatch between G_sparse ({G_sparse.shape[0]}) and X ({X.shape[0]})")

        # 첫 번째 레이어
        X = F.relu(self.hgc1(X, G_sparse))
        X = F.dropout(X, self.dropout, training=self.training)  # Dropout 적용

        # 두 번째 레이어
        X = self.hgc2(X, G_sparse)
        return X
