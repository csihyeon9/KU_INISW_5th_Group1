# trainer.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import Dict, List, Optional, Any
import numpy as np

class HGANTrainer:
    """HGAN 모델 학습기"""
    
    def __init__(
        self,
        model: nn.Module,
        predictor: nn.Module,
        optimizer: Optional[optim.Optimizer] = None,
        device: Optional[str] = None,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5
    ):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.predictor = predictor.to(self.device)
        
        # 옵티마이저 설정
        self.optimizer = optimizer or optim.AdamW(
            list(model.parameters()) + list(predictor.parameters()),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # 학습률 스케줄러
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=False
        )
        
        self.reconstruction_criterion = nn.MSELoss()
        self.train_history = {
            'total_loss': [],
            'reconstruction_loss': [],
            'prediction_loss': [],
            'attention_entropy': []
        }
    
    def train_epoch(
        self,
        dataset: Any,
        batch_size: int,
        negative_ratio: float
    ) -> Dict[str, float]:
        self.model.train()
        self.predictor.train()
        
        # 데이터 준비
        x = dataset.embeddings.to(self.device)
        edge_index = dataset.edge_index.to(self.device)
        batch_data = dataset.sample_training_pairs(
            batch_size=batch_size,
            negative_ratio=negative_ratio
        )
        pos_pairs = batch_data['pos_pairs'].to(self.device)
        neg_pairs = batch_data['neg_pairs'].to(self.device)
        
        # Forward & Backward
        self.optimizer.zero_grad()
        losses = self._compute_losses(x, edge_index, pos_pairs, neg_pairs)
        losses['total_loss'].backward()
        
        # 그래디언트 클리핑
        torch.nn.utils.clip_grad_norm_(
            list(self.model.parameters()) + list(self.predictor.parameters()),
            max_norm=1.0
        )
        
        self.optimizer.step()
        
        return {k: v.item() for k, v in losses.items()}
    
    def _compute_losses(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        pos_pairs: torch.Tensor,
        neg_pairs: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        # Model forward pass
        model_output = self.model(x, edge_index)
        embeddings = model_output['embeddings']
        
        # 손실 계산
        reconstruction_loss = self.reconstruction_criterion(embeddings, x)
        prediction_loss = self._compute_prediction_loss(embeddings, pos_pairs, neg_pairs)
        attention_entropy = self._compute_attention_entropy(model_output['attention_info'])
        
        total_loss = reconstruction_loss + 0.5 * prediction_loss + 0.1 * attention_entropy
        
        return {
            'total_loss': total_loss,
            'reconstruction_loss': reconstruction_loss,
            'prediction_loss': prediction_loss,
            'attention_entropy': attention_entropy
        }

    def _compute_prediction_loss(
        self,
        embeddings: torch.Tensor,
        pos_pairs: torch.Tensor,
        neg_pairs: torch.Tensor
    ) -> torch.Tensor:
        # 특성 연결
        pos_features = torch.cat([
            embeddings[pos_pairs[:, 0]],
            embeddings[pos_pairs[:, 1]]
        ], dim=1)
        neg_features = torch.cat([
            embeddings[neg_pairs[:, 0]],
            embeddings[neg_pairs[:, 1]]
        ], dim=1)
        
        # 점수 계산
        pos_scores = self.predictor(pos_features).squeeze(-1)
        neg_scores = self.predictor(neg_features).squeeze(-1)
        
        # BCE Loss
        pos_loss = F.binary_cross_entropy_with_logits(
            pos_scores, 
            torch.ones_like(pos_scores)
        )
        neg_loss = F.binary_cross_entropy_with_logits(
            neg_scores, 
            torch.zeros_like(neg_scores)
        )
        
        return (pos_loss + neg_loss) / 2
    
    def _compute_attention_entropy(
        self,
        attention_info: List[Dict]
    ) -> torch.Tensor:
        """Attention 엔트로피 계산"""
        entropy = torch.tensor(0., device=self.device)
        
        for layer_info in attention_info:
            attention_weights = layer_info['attention_weights']
            layer_entropy = -(attention_weights * torch.log(attention_weights + 1e-10)).sum()
            entropy += layer_entropy
        
        return entropy / len(attention_info)
    
    def train(
        self,
        dataset: Any,
        num_epochs: int,
        batch_size: int = 32,
        negative_ratio: float = 1.0,
        eval_every: int = 1,
        early_stopping_patience: int = 10
    ) -> Dict[str, List[float]]:
        """전체 학습 과정"""
        best_loss = float('inf')
        patience_counter = 0
        
        print(f"Training on {self.device}")
        print(f"{'Epoch':>5} {'Total':>10} {'Recon':>10} {'Pred':>10} {'Entropy':>10}")
        print("-" * 55)
        
        for epoch in range(num_epochs):
            # 한 에폭 학습
            losses = self.train_epoch(
                dataset,
                batch_size=batch_size,
                negative_ratio=negative_ratio
            )
            
            # 학습률 조정
            self.scheduler.step(losses['total_loss'])
            
            # 진행상황 출력
            if (epoch + 1) % eval_every == 0:
                print(
                    f"{epoch+1:5d} "
                    f"{losses['total_loss']:10.4f} "
                    f"{losses['reconstruction_loss']:10.4f} "
                    f"{losses['prediction_loss']:10.4f} "
                    f"{losses['attention_entropy']:10.4f}"
                )
            
            # Early stopping 체크
            if losses['total_loss'] < best_loss:
                best_loss = losses['total_loss']
                best_state = {
                    'model': self.model.state_dict(),
                    'predictor': self.predictor.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'epoch': epoch
                }
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= early_stopping_patience:
                print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                break
        
        # 최적 모델 복원
        self.model.load_state_dict(best_state['model'])
        self.predictor.load_state_dict(best_state['predictor'])
        
        return self.train_history
    
    @torch.no_grad()
    def get_embeddings(self, dataset: Any) -> torch.Tensor:
        """학습된 임베딩 추출"""
        self.model.eval()
        x = dataset.embeddings.to(self.device)
        edge_index = dataset.edge_index.to(self.device)
        
        output = self.model(x, edge_index)
        return output['embeddings'].cpu()