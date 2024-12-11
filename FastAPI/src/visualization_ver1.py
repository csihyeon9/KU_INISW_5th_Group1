# src/visualization_ver1.py
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import List, Dict
import pandas as pd

class ModelEvaluationVisualizer:
    def __init__(self, config: dict):
        self.config = config
        self.vis_dir = Path(config['visualization']['save_dir'])
        self.vis_dir.mkdir(parents=True, exist_ok=True)
        
        # Set seaborn style
        sns.set_theme(style=config['visualization']['style'])
        
    def plot_training_history(self, 
                        train_losses: List[float], 
                        val_losses: List[float],
                        contrastive_losses: List[float] = None) -> None:
        """학습 과정에서의 손실 함수 변화를 시각화"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
        epochs = range(1, len(train_losses) + 1)
        
        # 첫 번째 서브플롯: Training & Validation Loss
        ax1.plot(epochs, train_losses, 'b-', label='Training Loss')
        ax1.plot(epochs, val_losses, 'r-', label='Validation Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 두 번째 서브플롯: Contrastive Loss
        if contrastive_losses:
            ax2.plot(epochs, contrastive_losses, 'g-', label='Contrastive Loss')
            ax2.set_title('Contrastive Loss')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Loss')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.vis_dir / 'training_history.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def plot_metrics(self, 
                    precision_values: Dict[str, List[float]], 
                    recall_values: Dict[str, List[float]]) -> None:
        """Precision@K와 Recall@K 메트릭을 시각화"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot Precision@K
        for k, values in precision_values.items():
            ax1.plot(range(1, len(values) + 1), values, marker='o', label=f'K={k}')
        ax1.set_title('Precision@K Over Time')
        ax1.set_xlabel('Evaluation Step')
        ax1.set_ylabel('Precision')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot Recall@K
        for k, values in recall_values.items():
            ax2.plot(range(1, len(values) + 1), values, marker='o', label=f'K={k}')
        ax2.set_title('Recall@K Over Time')
        ax2.set_xlabel('Evaluation Step')
        ax2.set_ylabel('Recall')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.vis_dir / 'recommendation_metrics.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def plot_confusion_matrix(self, matrix: np.ndarray, labels: List[str]) -> None:
        """혼동 행렬 시각화"""
        plt.figure(figsize=(10, 8))
        sns.heatmap(matrix, 
                   annot=True, 
                   fmt='.2f', 
                   xticklabels=labels,
                   yticklabels=labels)
        
        plt.title('Recommendation Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        
        plt.savefig(self.vis_dir / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()