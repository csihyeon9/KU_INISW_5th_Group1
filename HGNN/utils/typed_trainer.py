import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from typing import Dict, List, Tuple, Optional
import numpy as np
from tqdm import tqdm

class TypedHGNNTrainer:
    def __init__(
        self,
        model: torch.nn.Module,
        learning_rate: float = 0.0002,
        weight_decay: float = 1e-4,
        device: Optional[str] = None
    ):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        
        # 옵티마이저 설정
        self.optimizer = AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.999)
        )
        
        # 학습률 스케줄러
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=False
        )
        
        self.best_loss = float('inf')
        self.best_model_state = None
        self.losses = {'total': [], 'reconstruction': [], 'relation': []}
        
    def _compute_loss(
        self,
        output: Dict[str, torch.Tensor],
        target: torch.Tensor,
        relation_labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """손실 함수 계산"""
        losses = {}
        
        # 재구성 손실
        reconstruction_loss = F.mse_loss(output['embeddings'], target)
        l1_loss = F.l1_loss(output['embeddings'], target)
        cosine_loss = 1 - F.cosine_similarity(output['embeddings'], target).mean()
        losses['reconstruction'] = reconstruction_loss + 0.1 * l1_loss + 0.1 * cosine_loss
        
        # 관계 예측 손실 (있는 경우)
        if 'relation_scores' in output and relation_labels is not None:
            relation_loss = F.cross_entropy(output['relation_scores'], relation_labels)
            losses['relation'] = relation_loss
        
        # 전체 손실
        total_loss = losses['reconstruction']
        if 'relation' in losses:
            total_loss = total_loss + losses['relation']
        
        losses['total'] = total_loss
        return losses
    
    def train_epoch(
        self,
        dataset,
        relation_samples: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None
    ) -> Dict[str, float]:
        """한 에폭 학습"""
        self.model.train()
        
        # 데이터 준비
        X = dataset.get_features().to(self.device)
        H_dict = {
            rel_type: H.to(self.device)
            for rel_type, H in dataset.get_typed_incidence_matrices().items()
        }
        
        # 관계 예측을 위한 샘플
        if relation_samples is not None:
            src_nodes, dst_nodes, rel_labels = [
                t.to(self.device) for t in relation_samples
            ]
        else:
            src_nodes = dst_nodes = rel_labels = None
        
        # 순전파
        self.optimizer.zero_grad()
        output = self.model(X, H_dict, src_nodes, dst_nodes)
        
        # 손실 계산
        losses = self._compute_loss(output, X, rel_labels)
        
        # 역전파
        losses['total'].backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
        self.optimizer.step()
        
        return {k: v.item() for k, v in losses.items()}
    
    def train(
        self,
        dataset,
        num_epochs: int = 150,
        batch_size: int = 32,
        eval_every: int = 10,
        early_stopping_patience: int = 15
    ):
        """전체 학습 과정"""
        print("\nTraining Started")
        print("-" * 50)
        print(f"{'Epoch':>5} {'Total':>10} {'Recon':>10} {'Relation':>10} {'LR':>10}")
        print("-" * 50)
        
        no_improvement = 0
        
        for epoch in range(num_epochs):
            # 관계 예측을 위한 샘플링
            relation_samples = self._sample_relations(dataset, batch_size)
            
            # 학습
            losses = self.train_epoch(dataset, relation_samples)
            
            # 손실 기록
            for k, v in losses.items():
                self.losses[k].append(v)
            
            # 학습률 조정
            self.scheduler.step(losses['total'])
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # 진행상황 출력
            if (epoch + 1) % eval_every == 0:
                print(f"{epoch+1:5d} {losses['total']:10.4f} "
                      f"{losses['reconstruction']:10.4f} "
                      f"{losses.get('relation', 0):10.4f} "
                      f"{current_lr:10.6f}")
            
            # 최고 성능 모델 저장
            if losses['total'] < self.best_loss:
                self.best_loss = losses['total']
                self.best_model_state = self.model.state_dict().copy()
                no_improvement = 0
            else:
                no_improvement += 1
            
            # Early stopping
            if no_improvement >= early_stopping_patience:
                print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                print(f"Best loss achieved: {self.best_loss:.4f}")
                break
        
        # 최고 성능 모델 복원
        self.model.load_state_dict(self.best_model_state)
        return self.losses
    
    def _sample_relations(
        self,
        dataset,
        batch_size: int
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """학습을 위한 관계 샘플링"""
        if not hasattr(dataset, 'get_relation_samples'):
            return None
            
        return dataset.get_relation_samples(batch_size)
    
    @torch.no_grad()
    def get_embeddings(self, dataset) -> np.ndarray:
        """학습된 임베딩 추출"""
        self.model.eval()
        
        X = dataset.get_features().to(self.device)
        H_dict = {
            rel_type: H.to(self.device)
            for rel_type, H in dataset.get_typed_incidence_matrices().items()
        }
        
        output = self.model(X, H_dict)
        return output['embeddings'].cpu().numpy()