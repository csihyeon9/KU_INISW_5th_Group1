import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from pathlib import Path
import logging
from typing import Tuple, Dict, List

class Trainer:
    def __init__(self, model: nn.Module, config: dict, keyword2idx: dict = None, idx2keyword: dict = None):
        """
        Initialize trainer for Hypergraph Neural Network
        
        Args:
            model (nn.Module): The hypergraph neural network model
            config (dict): Training configuration
            keyword2idx (dict, optional): Mapping of keywords to indices
            idx2keyword (dict, optional): Mapping of indices to keywords
        """
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
        
        # Loss criterion
        self.criterion = nn.BCELoss()
        
        # 체크포인트 디렉토리 생성
        self.save_dir = Path(config['training']['save_dir'])
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # 로깅 설정
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # 키워드 매핑 (선택적)
        self.keyword2idx = keyword2idx
        self.idx2keyword = idx2keyword
        
    def prepare_training_data(self, 
                               hyperedge_index: torch.Tensor, 
                               num_negative_samples: int = 1) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        데이터 준비: 링크 예측을 위한 양성/음성 샘플 생성
        
        Args:
            hyperedge_index (torch.Tensor): 하이퍼그래프의 엣지 인덱스
            num_negative_samples (int): 각 양성 샘플당 생성할 음성 샘플 수
        
        Returns:
            Tuple of (edges, labels)
        """
        # 양성 샘플 (실제 연결된 노드 쌍)
        positive_edges = hyperedge_index.t()
        
        # 음성 샘플 (랜덤 노드 쌍 생성)
        num_nodes = self.model.num_keywords
        negative_edges = torch.randint(0, num_nodes, positive_edges.shape, device=hyperedge_index.device)
        
        # 레이블 생성: 1 (양성), 0 (음성)
        edges = torch.cat([positive_edges, negative_edges], dim=0)
        labels = torch.cat([
            torch.ones(positive_edges.size(0), device=hyperedge_index.device),
            torch.zeros(negative_edges.size(0), device=hyperedge_index.device)
        ])
        
        return edges, labels
        
    def train_epoch(self, 
                    hyperedge_index: torch.Tensor, 
                    epoch: int, 
                    batch_size: int = 128) -> float:
        """
        한 에포크 학습
        
        Args:
            hyperedge_index (torch.Tensor): 하이퍼그래프의 엣지 인덱스
            epoch (int): 현재 에포크 번호
            batch_size (int): 배치 크기
        
        Returns:
            float: 평균 손실
        """
        self.model.train()
        
        # 훈련 데이터 준비
        edges, labels = self.prepare_training_data(hyperedge_index)
        
        # 데이터 섞기
        perm = torch.randperm(edges.size(0))
        edges = edges[perm]
        labels = labels[perm]
        
        total_loss = 0
        num_batches = 0
        
        # 배치 학습
        for i in range(0, edges.size(0), batch_size):
            batch_edges = edges[i:i + batch_size]
            batch_labels = labels[i:i + batch_size]
            
            self.optimizer.zero_grad()
            
            # 링크 예측
            predictions = self.model.predict_links(batch_edges)
            
            # Loss 계산
            loss = self.criterion(predictions, batch_labels.float())
            
            # 역전파
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            # 학습 로그 출력
            if i % (batch_size * 10) == 0:
                self.logger.info(
                    f"Epoch {epoch}, Batch {i//batch_size}, "
                    f"Loss: {loss.item():.4f}, "
                    f"Edges: {batch_edges.shape[0]}"
                )
                
        return total_loss / max(num_batches, 1)
        
    def validate(self, 
                 hyperedge_index: torch.Tensor, 
                 batch_size: int = 128) -> float:
        """
        검증 데이터에 대한 평가
        
        Args:
            hyperedge_index (torch.Tensor): 하이퍼그래프의 엣지 인덱스
            batch_size (int): 배치 크기
        
        Returns:
            float: 평균 검증 손실
        """
        self.model.eval()
        
        # 검증 데이터 준비
        edges, labels = self.prepare_training_data(hyperedge_index)
        
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for i in range(0, edges.size(0), batch_size):
                batch_edges = edges[i:i + batch_size]
                batch_labels = labels[i:i + batch_size]
                
                predictions = self.model.predict_links(batch_edges)
                loss = self.criterion(predictions, batch_labels.float())
                
                total_loss += loss.item()
                num_batches += 1
                
        return total_loss / max(num_batches, 1)
        
    def save_checkpoint(self, 
                       epoch: int, 
                       loss: float,
                       is_best: bool = False) -> None:
        """
        체크포인트 저장
        
        Args:
            epoch (int): 현재 에포크 번호
            loss (float): 현재 손실
            is_best (bool): 최고 성능 모델 여부
        """
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

    def evaluate_recommendations(self, 
                                 hyperedge_index: torch.Tensor, 
                                 k: int = 5) -> Tuple[float, float]:
        """
        Precision@K와 Recall@K 계산
        
        Args:
            hyperedge_index (torch.Tensor): 하이퍼그래프의 엣지 인덱스
            k (int): Top-K 추천 개수
        
        Returns:
            Tuple of (precision, recall)
        """
        self.model.eval()
        total_precision = 0
        total_recall = 0
        num_batches = 0
        
        # 전체 데이터 준비
        edges, labels = self.prepare_training_data(hyperedge_index)
        
        with torch.no_grad():
            # 모든 후보 노드에 대한 링크 예측
            num_nodes = self.model.num_keywords
            candidate_nodes = torch.arange(num_nodes, device=hyperedge_index.device)
            
            # 쿼리 노드 (예: 첫 번째 노드)
            query_node = 0
            
            # 노드 쌍 생성
            node_pairs = torch.cartesian_prod(torch.tensor([query_node], device=hyperedge_index.device), candidate_nodes)
            
            # 연결 확률 예측
            scores = self.model.predict_links(node_pairs)
            
            # Top-K 추천
            _, top_k_indices = scores.topk(k)
            recommended = set(candidate_nodes[top_k_indices].cpu().numpy())
            
            # 실제 연결된 노드들
            true_edges = set(edges[labels == 1].cpu().numpy())
            
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

    # def recommend_keywords(self, 
    #                        query_keywords: List[str], 
    #                        top_k: int = 5) -> List[str]:
    #     """
    #     키워드 추천
        
    #     Args:
    #         query_keywords (List[str]): 추천의 기준이 되는 키워드 리스트
    #         top_k (int): 추천할 키워드 개수
        
    #     Returns:
    #         List[str]: 추천된 키워드 리스트
    #     """
    #     if not self.keyword2idx or not self.idx2keyword:
    #         raise ValueError("Keyword mappings are required for recommendations")
        
    #     self.model.eval()
        
    #     # 쿼리 키워드 인덱스
    #     query_indices = [self.keyword2idx[k] for k in query_keywords if k in self.keyword2idx]
        
    #     if not query_indices:
    #         return []
        
    #     # 후보 키워드 생성
    #     candidate_indices = set(range(self.model.num_keywords)) - set(query_indices)
        
    #     # 노드 쌍 생성
    #     query_tensor = torch.tensor(query_indices, dtype=torch.long, device=self.device)
    #     candidate_tensor = torch.tensor(list(candidate_indices), dtype=torch.long, device=self.device)
    #     node_pairs = torch.cartesian_prod(query_tensor, candidate_tensor)
        
    #     # 스코어 계산
    #     with torch.no_grad():
    #         scores = self.model.predict_links(node_pairs)
        
    #     # Top-K 추천
    #     top_k_indices = scores.topk(top_k).indices
    #     recommendations = [
    #         self.idx2keyword[node_pairs[idx, 1].item()]
    #         for idx in top_k_indices
    #     ]
        
    #     return recommendations