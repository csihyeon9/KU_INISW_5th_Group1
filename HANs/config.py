# config.py
from dataclasses import dataclass
from typing import Optional

@dataclass
class HGANConfig:
    """HGAN 모델 설정"""
    # 모델 파라미터
    embedding_dim: int = 64
    hidden_dim: int = 128
    num_layers: int = 3
    num_heads: int = 4
    dropout: float = 0.1
    
    # 학습 파라미터
    batch_size: int = 32
    num_epochs: int = 200
    learning_rate: float = 0.001
    weight_decay: float = 1e-5
    negative_ratio: float = 1.0
    early_stopping_patience: int = 10
    eval_every: int = 1
    
    # 데이터/시스템 관련
    data_path: Optional[str] = None
    device: Optional[str] = None
    seed: int = 42
    
    # 저장 관련
    save_dir: str = "results"
    model_name: str = "hgan_model"
    save_embeddings: bool = True