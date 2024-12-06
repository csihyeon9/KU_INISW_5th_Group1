import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List

class ModelEvaluationVisualizer:
    def __init__(self, config: dict):
        self.config = config
        self.vis_dir = Path(config['visualization']['save_dir'])
        self.vis_dir.mkdir(parents=True, exist_ok=True)
        
        # Set seaborn style
        sns.set_theme(style=config['visualization']['style'])
        
    def plot_training_history(self, train_losses: List[float], val_losses: List[float]) -> None:
        """학습 과정에서의 손실 함수 변화를 시각화"""
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label='Training Loss', color='blue')
        plt.plot(val_losses, label='Validation Loss', color='red')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        plt.savefig(self.vis_dir / 'training_history.png', dpi=300, bbox_inches='tight')
        plt.close()