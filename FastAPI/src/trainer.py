# src/trainer.py
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
        
        for batch_idx, (edge_index, edge_weight) in enumerate(tqdm(dataloader, desc=f'Epoch {epoch}')):
            # 데이터를 GPU로 이동
            edge_index = edge_index.to(self.device)
            edge_weight = edge_weight.to(self.device)
            
            self.optimizer.zero_grad()
            
            try:
                # 순전파
                output = self.model(edge_index, edge_weight)
                
                # Contrastive Loss 계산
                loss = self._calculate_contrastive_loss(output, edge_index)
                
                # 역전파
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
                
                # 메모리 사용량 모니터링 (10배치마다)
                if batch_idx % 10 == 0:
                    self.logger.info(
                        f'Batch {batch_idx}, Loss: {loss.item():.4f}, '
                        f'Edges: {edge_index.shape[1]}, '
                        f'Non-zero weights: {(edge_weight != 0).sum().item()}'
                    )
                    
            except RuntimeError as e:
                self.logger.error(f"Error in batch {batch_idx}: {str(e)}")
                self.logger.error(f"Edge index shape: {edge_index.shape}")
                self.logger.error(f"Edge weight shape: {edge_weight.shape}")
                raise e
                
        return total_loss / max(num_batches, 1)
        
    def _calculate_contrastive_loss(self, 
                                  embeddings: torch.Tensor, 
                                  edge_index: torch.Tensor,
                                  temperature: float = 0.1) -> torch.Tensor:
        """Contrastive Loss 계산 (희소 행렬 버전)"""
        # 정규화
        embeddings = nn.functional.normalize(embeddings, dim=1)
        
        # 연결된 노드 쌍만 사용 (희소 행렬의 non-zero 엔트리)
        src_nodes = edge_index[0]
        dst_nodes = edge_index[1]
        
        # Positive pairs
        pos_score = torch.sum(embeddings[src_nodes] * embeddings[dst_nodes], dim=1)
        pos_score = torch.exp(pos_score / temperature)
        
        # Negative pairs (배치 내에서만)
        # 메모리 효율을 위해 모든 쌍이 아닌 배치 내 다른 노드들만 사용
        batch_size = src_nodes.size(0)
        neg_pairs = torch.randint(0, embeddings.size(0), (batch_size, 5), device=self.device)
        neg_score = torch.sum(embeddings[src_nodes].unsqueeze(1) * embeddings[neg_pairs], dim=2)
        neg_score = torch.exp(neg_score / temperature).sum(dim=1)
        
        # Loss 계산
        loss = -torch.log(pos_score / (pos_score + neg_score)).mean()
        return loss
        
    def validate(self, dataloader: DataLoader) -> float:
        """검증 데이터에 대한 평가"""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for edge_index, edge_weight in dataloader:
                edge_index = edge_index.to(self.device)
                edge_weight = edge_weight.to(self.device)
                
                output = self.model(edge_index, edge_weight)
                loss = self._calculate_contrastive_loss(output, edge_index)
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
            best_path = self.save_dir / '(v1)hgnn_data.pt'
            torch.save(checkpoint, best_path)
            self.logger.info(f'Best model saved: {best_path}')

    def load_checkpoint(self, checkpoint_path: str) -> Dict:
        """체크포인트 로드"""
        if not Path(checkpoint_path).exists():
            self.logger.error(f"Checkpoint not found: {checkpoint_path}")
            raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}")
        
        try:
            # CPU로 먼저 로드
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            
            # 모델 가중치 로드
            self.model.load_state_dict(checkpoint['model_state_dict'])
            
            # 모델을 적절한 디바이스로 이동
            self.model = self.model.to(self.device)
            
            # 옵티마이저 상태 로드
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            # 옵티마이저 상태를 적절한 디바이스로 이동
            for state in self.optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(self.device)
            
            self.logger.info(f"Checkpoint loaded from {checkpoint_path}")
            self.logger.info(f"Resumed from epoch {checkpoint['epoch']}")
            
            return checkpoint
            
        except Exception as e:
            self.logger.error(f"Error loading checkpoint: {str(e)}")
            raise e
        
    def evaluate_recommendations(self, dataloader: DataLoader, k: int = 5) -> Tuple[float, float]:
        """Precision@K와 Recall@K 계산"""
        self.model.eval()
        total_precision = 0
        total_recall = 0
        num_batches = 0
        
        with torch.no_grad():
            for edge_index, edge_weight in dataloader:
                edge_index = edge_index.to(self.device)
                edge_weight = edge_weight.to(self.device)
                
                # 모델 출력
                embeddings = self.model(edge_index, edge_weight)
                
                # 각 노드에 대해 Top-K 추천
                similarities = torch.matmul(embeddings, embeddings.t())
                
                # 실제 연결된 노드들 (ground truth)
                true_edges = {(i.item(), j.item()) for i, j in edge_index.t()}
                
                # 각 노드에 대해 평가
                for node in range(embeddings.size(0)):
                    # 실제 연결된 노드들
                    true_neighbors = {j for i, j in true_edges if i == node}
                    
                    # Top-K 추천
                    _, top_k_indices = similarities[node].topk(k + 1)  # +1 for excluding self
                    recommended = set(top_k_indices.cpu().numpy()) - {node}
                    
                    # Precision과 Recall 계산
                    if len(recommended) > 0:
                        precision = len(recommended & true_neighbors) / len(recommended)
                        total_precision += precision
                    
                    if len(true_neighbors) > 0:
                        recall = len(recommended & true_neighbors) / len(true_neighbors)
                        total_recall += recall
                
                num_batches += 1
        
        return (
            total_precision / (num_batches * embeddings.size(0)),
            total_recall / (num_batches * embeddings.size(0))
        )
    
    def create_confusion_matrix(self, dataloader: DataLoader) -> np.ndarray:
        """추천 결과에 대한 혼동 행렬 생성"""
        self.model.eval()
        num_keywords = self.model.num_keywords
        confusion_matrix = np.zeros((num_keywords, num_keywords))
        
        with torch.no_grad():
            for edge_index, edge_weight in dataloader:
                edge_index = edge_index.to(self.device)
                edge_weight = edge_weight.to(self.device)
                
                # 모델 출력
                embeddings = self.model(edge_index, edge_weight)
                
                # 각 노드에 대한 추천 결과
                similarities = torch.matmul(embeddings, embeddings.t())
                
                # 실제 연결과 예측 결과 비교
                for i, j in edge_index.t():
                    i, j = i.item(), j.item()
                    predicted_score = similarities[i, j].item()
                    confusion_matrix[i, j] += predicted_score
        
        # 정규화
        row_sums = confusion_matrix.sum(axis=1, keepdims=True)
        confusion_matrix = np.where(row_sums > 0, confusion_matrix / row_sums, 0)
        
        return confusion_matrix