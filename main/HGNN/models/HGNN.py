# HGNN.py
from torch import nn
from models import HGNN_conv
import torch.nn.functional as F

class HGNN(nn.Module):
    def __init__(self, in_ch, n_class, n_hid, dropout=0.5):
        super(HGNN, self).__init__()
        self.dropout = dropout
        self.hgc1 = HGNN_conv(in_ch, n_hid*2)
        self.hgc2 = HGNN_conv(n_hid*2, n_hid)
        self.hgc3 = HGNN_conv(n_hid, n_class)

        self.bn1 = nn.BatchNorm1d(n_hid*2)
        self.bn2 = nn.BatchNorm1d(n_hid)
    
    def forward(self, x, G):
        x = F.relu(self.bn1(self.hgc1(x, G)))
        # x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.bn2(self.hgc2(x, G)))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.hgc3(x, G)
        return x
    