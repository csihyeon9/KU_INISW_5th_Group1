import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
import pandas as pd
import torch

class KeywordModelVisualizer:
    def __init__(self, config: dict):
        """
        키워드 추천 모델 시각화를 위한 초기화 메서드
        
        Args:
            config (dict): 시각화 설정을 포함하는 구성 사전
        """
        self.config = config
        self.vis_dir = Path(config['visualization']['save_dir'])
        self.vis_dir.mkdir(parents=True, exist_ok=True)
        
        # 시본 스타일 설정
        sns.set_theme(style=config['visualization'].get('style', 'whitegrid'))
        
    def plot_training_progress(self, 
                               train_losses: List[float], 
                               val_losses: List[float], 
                               precision_values: Optional[List[float]] = None, 
                               recall_values: Optional[List[float]] = None) -> None:
        """
        손실 및 추천 메트릭을 포함한 학습 진행 시각화
        
        Args:
            train_losses (List[float]): 에폭별 훈련 손실
            val_losses (List[float]): 에폭별 검증 손실
            precision_values (Optional[List[float]]): 정밀도 값 (옵션)
            recall_values (Optional[List[float]]): 재현율 값 (옵션)
        """
        # 서브플롯 레이아웃 결정
        if precision_values and recall_values:
            fig, axs = plt.subplots(2, 2, figsize=(16, 12))
        else:
            fig, axs = plt.subplots(1, 2, figsize=(16, 6))
        
        # 손실 그래프
        epochs = range(1, len(train_losses) + 1)
        axs[0].plot(epochs, train_losses, 'b-', label='훈련 손실')
        axs[0].plot(epochs, val_losses, 'r-', label='검증 손실')
        axs[0].set_title('훈련 및 검증 손실')
        axs[0].set_xlabel('에폭')
        axs[0].set_ylabel('손실')
        axs[0].legend()
        axs[0].grid(True, alpha=0.3)
        
        # 학습률 시각화 (구성에 있는 경우)
        lr = self.config['training'].get('learning_rate', None)
        if lr:
            axs[1].axhline(y=lr, color='g', linestyle='--', label='학습률')
            axs[1].set_title('학습률')
            axs[1].set_xlabel('에폭')
            axs[1].set_ylabel('학습률')
            axs[1].legend()
        
        # 데이터가 있는 경우 선택적 메트릭 그래프
        if precision_values and recall_values:
            axs[2].plot(epochs, precision_values, 'g-', label='정밀도@K')
            axs[2].set_title('정밀도@K')
            axs[2].set_xlabel('에폭')
            axs[2].set_ylabel('정밀도')
            axs[2].legend()
            
            axs[3].plot(epochs, recall_values, 'm-', label='재현율@K')
            axs[3].set_title('재현율@K')
            axs[3].set_xlabel('에폭')
            axs[3].set_ylabel('재현율')
            axs[3].legend()
        
        plt.tight_layout()
        plt.savefig(self.vis_dir / 'training_progress.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def visualize_keyword_embeddings(self, 
                                     model: torch.nn.Module, 
                                     keyword2idx: Dict[str, int], 
                                     idx2keyword: Dict[int, str], 
                                     top_n: int = 50) -> None:
        """
        차원 축소를 사용한 키워드 임베딩 시각화
        
        Args:
            model (torch.nn.Module): 학습된 키워드 임베딩 모델
            keyword2idx (Dict[str, int]): 키워드-인덱스 매핑
            idx2keyword (Dict[int, str]): 인덱스-키워드 매핑
            top_n (int): 시각화할 상위 키워드 수
        """
        from sklearn.manifold import TSNE
        
        # 임베딩 추출
        with torch.no_grad():
            embeddings = model.embedding.weight.cpu().numpy()
        
        # 빈도 또는 랜덤 선택으로 상위 키워드 선택
        top_keywords = list(keyword2idx.keys())[:top_n]
        top_indices = [keyword2idx[k] for k in top_keywords]
        
        # 차원 축소
        tsne = TSNE(n_components=2, random_state=42)
        reduced_embeddings = tsne.fit_transform(embeddings[top_indices])
        
        # 그래프 생성
        plt.figure(figsize=(15, 10))
        plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], alpha=0.7)
        
        # 포인트 주석 달기
        for i, keyword in enumerate(top_keywords):
            plt.annotate(keyword, 
                         (reduced_embeddings[i, 0], reduced_embeddings[i, 1]), 
                         alpha=0.7, 
                         fontsize=8)
        
        plt.title('키워드 임베딩 시각화')
        plt.xlabel('t-SNE 차원 1')
        plt.ylabel('t-SNE 차원 2')
        plt.tight_layout()
        plt.savefig(self.vis_dir / 'keyword_embeddings.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_link_prediction_distribution(self, 
                                          predictions: torch.Tensor, 
                                          labels: torch.Tensor) -> None:
        """
        링크 예측 확률 분포 시각화
        
        Args:
            predictions (torch.Tensor): 예측된 링크 확률
            labels (torch.Tensor): 실제 링크 레이블
        """
        plt.figure(figsize=(12, 6))
        
        # 양성 및 음성 샘플에 대한 예측 분포
        sns.histplot(predictions[labels == 1].cpu().numpy(), 
                     kde=True, 
                     color='green', 
                     alpha=0.5, 
                     label='양성 링크')
        sns.histplot(predictions[labels == 0].cpu().numpy(), 
                     kde=True, 
                     color='red', 
                     alpha=0.5, 
                     label='음성 링크')
        
        plt.title('링크 예측 확률 분포')
        plt.xlabel('예측된 링크 확률')
        plt.ylabel('빈도')
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.vis_dir / 'link_prediction_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_recommendation_performance(self, 
                                        recommended_keywords: List[str], 
                                        true_keywords: List[str]) -> None:
        """
        추천 성능 시각화
        
        Args:
            recommended_keywords (List[str]): 모델이 추천한 키워드
            true_keywords (List[str]): 실제 키워드
        """
        overlap = set(recommended_keywords) & set(true_keywords)
        
        plt.figure(figsize=(10, 6))
        plt.bar(['추천됨', '실제', '중복'], 
                [len(recommended_keywords), len(true_keywords), len(overlap)])
        plt.title('키워드 추천 성능')
        plt.ylabel('키워드 수')
        plt.tight_layout()
        plt.savefig(self.vis_dir / 'recommendation_performance.png', dpi=300, bbox_inches='tight')
        plt.close()