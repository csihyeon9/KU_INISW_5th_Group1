import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List

class TrainingVisualizer:
    def __init__(self, config: dict):
        """
        학습 과정 시각화 도구 초기화
        - 저장 경로 생성 및 Seaborn 스타일 설정
        
        Args:
            config (dict): 설정 딕셔너리
        """
        self.config = config
        self.vis_dir = Path(config['visualization']['save_dir'])
        self.vis_dir.mkdir(parents=True, exist_ok=True)

        # Seaborn 스타일 설정
        sns.set_theme(style=config['visualization']['style'])

    def plot_training_history(self, train_losses: List[float], val_losses: List[float] = None) -> None:
        """
        학습 및 검증 손실 시각화
        
        Args:
            train_losses (List[float]): 각 epoch의 훈련 손실 값
            val_losses (List[float], optional): 각 epoch의 검증 손실 값 (기본값: None)
        """
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label='Training Loss', color='blue', linewidth=2)
        if val_losses is not None:
            plt.plot(val_losses, label='Validation Loss', color='red', linestyle='--', linewidth=2)
        
        plt.title('Training and Validation Loss', fontsize=16)
        plt.xlabel('Epoch', fontsize=14)
        plt.ylabel('Loss', fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(True)

        # 결과 저장
        save_path = self.vis_dir / 'training_history.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training history plot saved at {save_path}")
        plt.close()

    def plot_accuracy_history(self, train_accuracies: List[float], val_accuracies: List[float] = None) -> None:
        """
        학습 및 검증 정확도 시각화
        
        Args:
            train_accuracies (List[float]): 각 epoch의 훈련 정확도 값
            val_accuracies (List[float], optional): 각 epoch의 검증 정확도 값 (기본값: None)
        """
        plt.figure(figsize=(10, 6))
        plt.plot(train_accuracies, label='Training Accuracy', color='green', linewidth=2)
        if val_accuracies is not None:
            plt.plot(val_accuracies, label='Validation Accuracy', color='orange', linestyle='--', linewidth=2)
        
        plt.title('Training and Validation Accuracy', fontsize=16)
        plt.xlabel('Epoch', fontsize=14)
        plt.ylabel('Accuracy', fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(True)

        # 결과 저장
        save_path = self.vis_dir / 'accuracy_history.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Accuracy history plot saved at {save_path}")
        plt.close()
