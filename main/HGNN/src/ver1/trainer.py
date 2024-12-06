import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from pathlib import Path
import logging
from typing import Tuple, Dict

class Trainer:
    def __init__(self, model: nn.Module, config: dict):
        self.model = model
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        # 옵티마이저 설정
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay']
        )
        
        # 체크포인트 디렉토리 생성
        self.save_dir = Path(config['training']['save_dir'])
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # 로깅 설정
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    def train_epoch(self, dataloader: DataLoader, epoch: int) -> float:
        """한 에포크 학습"""
        self.model.train()
        total_loss = 0
        num_batches = 0

        for batch_idx, (edges, labels) in enumerate(tqdm(dataloader, desc=f'Epoch {epoch}')):
            # 데이터를 GPU로 이동
            edges = edges.to(self.device)
            labels = labels.to(self.device)
            
            self.optimizer.zero_grad()
            
            # 링크 예측
            predictions = self.model.predict_links(edges)
            
            # Loss 계산
            loss = self._calculate_bce_loss(predictions, labels)
            
            # 역전파
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            # 학습 로그 출력
            if batch_idx % 10 == 0:
                self.logger.info(
                    f"Batch {batch_idx}, Loss: {loss.item():.4f}, "
                    f"Edges: {edges.shape[0]}"
                )
                
        return total_loss / max(num_batches, 1)
        
    def _calculate_bce_loss(self, 
                            predictions: torch.Tensor, 
                            labels: torch.Tensor) -> torch.Tensor:
        """Binary Cross-Entropy Loss 계산"""
        criterion = nn.BCELoss()
        return criterion(predictions, labels)
        
    def validate(self, dataloader: DataLoader) -> float:
        """검증 데이터에 대한 평가"""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for edges, labels in dataloader:
                edges = edges.to(self.device)
                labels = labels.to(self.device)
                
                predictions = self.model.predict_links(edges)
                loss = self._calculate_bce_loss(predictions, labels)
                total_loss += loss.item()
                num_batches += 1
                
        return total_loss / max(num_batches, 1)
        
    def save_checkpoint(self, 
                       epoch: int, 
                       loss: float,
                       is_best: bool = False) -> None:
        """체크포인트 저장"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
        }
        
        # 일반 체크포인트 저장
        if epoch % self.config['training']['checkpoint_interval'] == 0:
            path = self.save_dir / f'checkpoint_epoch_{epoch}.pt'
            torch.save(checkpoint, path)
            self.logger.info(f'Checkpoint saved: {path}')
        
        # 최고 성능 모델 저장
        if is_best:
            best_path = self.save_dir / 'best_model.pt'
            torch.save(checkpoint, best_path)
            self.logger.info(f'Best model saved: {best_path}')

    def evaluate_recommendations(self, dataloader: DataLoader, k: int = 5) -> Tuple[float, float]:
        """Precision@K와 Recall@K 계산"""
        self.model.eval()
        total_precision = 0
        total_recall = 0
        num_batches = 0
        
        with torch.no_grad():
            for edges, labels in dataloader:
                edges = edges.to(self.device)
                predictions = self.model.predict_links(edges)
                
                # Top-K 추천 계산
                _, top_k_indices = predictions.topk(k)
                recommended = set(top_k_indices.cpu().numpy())
                
                # 실제 연결된 노드들
                true_edges = set((edges[labels == 1].cpu().numpy()))
                
                if len(recommended) > 0:
                    precision = len(recommended & true_edges) / len(recommended)
                    total_precision += precision
                
                if len(true_edges) > 0:
                    recall = len(recommended & true_edges) / len(true_edges)
                    total_recall += recall
                
                num_batches += 1
        
        return (
            total_precision / max(num_batches, 1),
            total_recall / max(num_batches, 1)
        )
