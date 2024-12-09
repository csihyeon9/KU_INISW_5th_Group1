import matplotlib.pyplot as plt
import seaborn as sns
import os
from typing import List, Dict
import numpy as np

class TrainingVisualizer:
   def __init__(self, config: Dict):
       """
       설정 파일을 사용하여 시각화 설정을 초기화합니다.
       
       Args:
           config: 시각화 매개변수가 포함된 설정 딕셔너리
       """
       self.config = config
       self.save_dir = config['visualization']['save_dir']
       os.makedirs(self.save_dir, exist_ok=True)
       
       # 설정에서 시각화 스타일 설정
       self.style = config['visualization']['style']
       sns.set_style(self.style)
       
       # 평가 간격 설정
       self.eval_interval = config['visualization']['metrics']['eval_interval']

   def plot_training_history(self, train_losses: List[float], val_losses: List[float], 
                           train_accuracies: List[float], val_accuracies: List[float]):
       """
       설정된 설정을 사용하여 학습 기록을 시각화하고 저장합니다.
       
       Args:
           train_losses: 학습 손실값 리스트
           val_losses: 검증 손실값 리스트
           train_accuracies: 학습 정확도 리스트
           val_accuracies: 검증 정확도 리스트
       """
       epochs = range(1, len(train_losses) + 1)
       
       # 두 개의 서브플롯이 있는 그림 생성
       fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
       
       # 학습 및 검증 손실 그래프
       self._plot_metric(ax1, epochs, train_losses, val_losses, 
                        '시간에 따른 손실값', '손실', '학습', '검증')
       
       # 학습 및 검증 정확도 그래프
       self._plot_metric(ax2, epochs, train_accuracies, val_accuracies,
                        '시간에 따른 정확도', '정확도', '학습', '검증')
       
       # 지정된 경우 평가 간격 마커 추가
       if self.eval_interval > 1:
           self._add_evaluation_markers(ax1, epochs, train_losses)
           self._add_evaluation_markers(ax2, epochs, train_accuracies)
       
       # 레이아웃 조정 및 저장
       plt.tight_layout()
       save_path = os.path.join(self.save_dir, 'training_metrics.png')
       plt.savefig(save_path, dpi=300, bbox_inches='tight')
       plt.close()
       
       # 요약 통계 생성
       self._save_training_summary(train_losses, val_losses, 
                                 train_accuracies, val_accuracies)

   def _plot_metric(self, ax, epochs, train_metric, val_metric, 
                   title, ylabel, train_label, val_label):
       """개별 메트릭을 플롯하는 헬퍼 메서드"""
       ax.plot(epochs, train_metric, 'b-', label=train_label)
       ax.plot(epochs, val_metric, 'r-', label=val_label)
       ax.set_title(title)
       ax.set_xlabel('에포크')
       ax.set_ylabel(ylabel)
       ax.legend()
       ax.grid(True)

   def _add_evaluation_markers(self, ax, epochs, metric):
       """평가 간격에 대한 마커 추가"""
       eval_epochs = [e for e in epochs if e % self.eval_interval == 0]
       eval_values = [metric[e-1] for e in eval_epochs]
       ax.plot(eval_epochs, eval_values, 'go', label='평가 지점')

   def _save_training_summary(self, train_losses, val_losses, 
                            train_accuracies, val_accuracies):
       """학습 메트릭의 요약을 생성하고 저장"""
       summary_path = os.path.join(self.save_dir, 'training_summary.txt')
       
       with open(summary_path, 'w', encoding='utf-8') as f:
           f.write("학습 요약\n")
           f.write("================\n\n")
           
           f.write("최종 메트릭:\n")
           f.write(f"학습 손실: {train_losses[-1]:.4f}\n")
           f.write(f"검증 손실: {val_losses[-1]:.4f}\n")
           f.write(f"학습 정확도: {train_accuracies[-1]:.4f}\n")
           f.write(f"검증 정확도: {val_accuracies[-1]:.4f}\n\n")
           
           f.write("최고 성능:\n")
           f.write(f"최저 학습 손실: {min(train_losses):.4f}\n")
           f.write(f"최저 검증 손실: {min(val_losses):.4f}\n")
           f.write(f"최고 학습 정확도: {max(train_accuracies):.4f}\n")
           f.write(f"최고 검증 정확도: {max(val_accuracies):.4f}\n")