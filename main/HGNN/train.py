# train.py
import torch
from torch.utils.data import DataLoader
import yaml
from scipy import sparse
import numpy as np
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
from typing import List, Dict

from src.data_processor import KeywordProcessor
from src.models import KeywordHGNN
from src.trainer import Trainer
from src.visualization import ModelEvaluationVisualizer

def load_config(config_path: str = 'config/config.yaml') -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

class SparseMatrixDataset:
    def __init__(self, matrices, keyword_processor: KeywordProcessor):
        self.matrices = matrices
        self.num_nodes = keyword_processor.matrix_size
    
    def __len__(self):
        return len(self.matrices)
    
    def __getitem__(self, idx):
        matrix = self.matrices[idx]
        coo = matrix.tocoo()
        
        # 유효한 인덱스만 선택
        mask = (coo.row < self.num_nodes) & (coo.col < self.num_nodes)
        row = coo.row[mask]
        col = coo.col[mask]
        data = coo.data[mask]
        
        indices = torch.from_numpy(np.vstack((row, col)))
        values = torch.from_numpy(data)
        
        return indices.long(), values.float()

def collate_sparse_batch(batch):
    """배치 내의 희소 행렬들을 하나의 큰 희소 행렬로 결합"""
    indices_list = []
    values_list = []
    
    for indices, values in batch:
        if indices.size(1) > 0:  # 빈 텐서가 아닌 경우만
            indices_list.append(indices)
            values_list.append(values)
    
    if not indices_list:  # 모든 텐서가 비어있는 경우
        return torch.zeros((2, 0)), torch.zeros(0)
    
    # 모든 인덱스와 값을 결합
    indices = torch.cat(indices_list, dim=1)
    values = torch.cat(values_list, dim=0)
    
    # indices[1]의 최대값에 맞춰서 values의 크기를 조정
    max_index = indices[1].max().item() + 1
    if max_index > len(values):
        new_values = torch.zeros(max_index)
        new_values[:len(values)] = values
        values = new_values
    
    return indices, values

def main():
    # 설정 로드
    config = load_config()
    
    # 결과 저장 디렉토리 생성
    save_dir = Path(config['training']['save_dir'])
    save_dir.mkdir(parents=True, exist_ok=True)

    # 로그 파일 생성
    log_file = save_dir / 'training_log.txt'
    with open(log_file, 'w') as f:
        f.write("Epoch\tTrain Loss\tVal Loss\tContrastive Loss")
        for k in config['visualization']['metrics']['k_values']:
            f.write(f"\tPrecision@{k}\tRecall@{k}")
        f.write("\n")
    
    # 데이터 처리
    processor = KeywordProcessor(
        config['data']['unique_keywords_path'],
        config['data']['news_data_path']
    )
    processor.load_data()
    
    # 희소 행렬 생성
    matrices = processor.process_all_articles()
    print(f"\nCreated {len(matrices)} sparse matrices")
    
    # 데이터셋 생성
    dataset = SparseMatrixDataset(matrices, processor)
    
    # config에서 분할 비율 가져오기
    split_ratio = config['data']['split_ratio']
    train_size = int(split_ratio['train'] * len(dataset))
    val_size = int(split_ratio['val'] * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    # 데이터 분할
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size]
    )
    
    # 데이터 로더 생성
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['training']['batch_size'],
        shuffle=True,
        collate_fn=collate_sparse_batch
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        collate_fn=collate_sparse_batch
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['training']['batch_size'],
        collate_fn=collate_sparse_batch
    )
    
    # 모델 초기화
    model = KeywordHGNN(
        num_keywords=processor.matrix_size,
        hidden_dim=config['model']['hidden_dim'],
        num_layers=config['model']['num_layers'],
        dropout=config['model']['dropout']
    )
    
    # 트레이너 초기화
    trainer = Trainer(model, config)
    
    # 시각화 도구 초기화
    visualizer = ModelEvaluationVisualizer(config)
    
    # 학습 히스토리 저장소
    train_losses = []
    val_losses = []
    contrastive_losses = []
    precision_values = {str(k): [] for k in config['visualization']['metrics']['k_values']}
    recall_values = {str(k): [] for k in config['visualization']['metrics']['k_values']}
    
    # 학습 시작
    print("\nStarting training...")
    best_val_loss = float('inf')
    
    for epoch in range(config['training']['num_epochs']):
        # 학습
        train_loss = trainer.train_epoch(train_loader, epoch)
        train_losses.append(train_loss)
        
        # 검증
        val_loss = trainer.validate(val_loader)
        val_losses.append(val_loss)
        
        # Contrastive Loss 추적
        try:
            contrastive_loss = trainer._calculate_contrastive_loss(
                model.embedding.weight,
                next(iter(train_loader))[0]
            ).item()
            contrastive_losses.append(contrastive_loss)
        except:
            contrastive_losses.append(0.0)
        
        # 주기적으로 추천 성능 평가
        if epoch % config['visualization']['metrics']['eval_interval'] == 0:
            for k in config['visualization']['metrics']['k_values']:
                try:
                    precision, recall = trainer.evaluate_recommendations(
                        test_loader, k=k
                    )
                    precision_values[str(k)].append(precision)
                    recall_values[str(k)].append(recall)
                except:
                    precision_values[str(k)].append(0.0)
                    recall_values[str(k)].append(0.0)
        
        # 체크포인트 저장
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            
        trainer.save_checkpoint(epoch, val_loss, is_best)
        
        print(f'Epoch {epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}')
        
        # 학습 과정 시각화
        try:
            visualizer.plot_training_history(train_losses, val_losses, contrastive_losses)
            if epoch % config['visualization']['metrics']['eval_interval'] == 0:
                visualizer.plot_metrics(precision_values, recall_values)
        except Exception as e:
            print(f"Visualization error: {e}")

        with open(log_file, 'a') as f:
            f.write(f"{epoch}\t{train_loss:.4f}\t{val_loss:.4f}\t{contrastive_loss:.4f}")
            if epoch % config['visualization']['metrics']['eval_interval'] == 0:
                for k in config['visualization']['metrics']['k_values']:
                    p = precision_values[str(k)][-1] if precision_values[str(k)] else 0
                    r = recall_values[str(k)][-1] if recall_values[str(k)] else 0
                    f.write(f"\t{p:.4f}\t{r:.4f}")
            f.write("\n")
    
    print('Training completed!')

if __name__ == '__main__':
    main()