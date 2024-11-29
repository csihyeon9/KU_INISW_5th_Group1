# train.py
import torch
from torch.utils.data import DataLoader
import yaml
from scipy import sparse
import numpy as np
from pathlib import Path

from src.data_processor import KeywordProcessor
from src.models import KeywordHGNN
from src.trainer import Trainer

def load_config(config_path: str = 'config/config.yaml') -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

class SparseMatrixDataset:
    def __init__(self, matrices, num_nodes=7812):
        self.matrices = matrices
        self.num_nodes = num_nodes
    
    def __len__(self):
        return len(self.matrices)
    
    def __getitem__(self, idx):
        # 희소 행렬을 PyTorch tensor로 변환
        matrix = self.matrices[idx]
        
        # 엣지 인덱스 추출
        coo = matrix.tocoo()
        
        # 유효한 인덱스만 선택
        mask = (coo.row < self.num_nodes) & (coo.col < self.num_nodes)
        row = coo.row[mask]
        col = coo.col[mask]
        data = coo.data[mask]
        
        indices = torch.from_numpy(np.vstack((row, col)))
        values = torch.from_numpy(data)
        
        return indices.long(), values.float()

def main():
    # 설정 로드
    config = load_config()
    
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
    dataset = SparseMatrixDataset(matrices)
    
    # 학습/검증 분할
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    def collate_sparse_batch(batch):
        """배치 내의 희소 행렬들을 하나의 큰 희소 행렬로 결합"""
        cumsum = 0
        indices_list = []
        values_list = []
        
        num_nodes = 7812  # unique_keywords의 크기
        
        # 각 샘플의 인덱스를 오프셋을 적용하여 결합
        for indices, values in batch:
            # 유효한 인덱스만 선택
            mask = (indices[0] < num_nodes) & (indices[1] < num_nodes)
            valid_indices = indices[:, mask]
            valid_values = values[mask]
            
            indices_list.append(valid_indices)
            values_list.append(valid_values)
            cumsum += 1
        
        # 모든 인덱스와 값을 결합
        indices = torch.cat(indices_list, dim=1)
        values = torch.cat(values_list, dim=0)
        
        return indices, values

    # 데이터 로더 생성 부분 수정
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
    
    # 모델 초기화
    model = KeywordHGNN(
        num_keywords=processor.matrix_size,
        hidden_dim=config['model']['hidden_dim'],
        num_layers=config['model']['num_layers'],
        dropout=config['model']['dropout']
    )
    
    # 트레이너 초기화
    trainer = Trainer(model, config)
    
    # 학습 시작
    print("\nStarting training...")
    best_val_loss = float('inf')
    
    for epoch in range(config['training']['num_epochs']):
        # 학습
        train_loss = trainer.train_epoch(train_loader, epoch)
        
        # 검증
        val_loss = trainer.validate(val_loader)
        
        # 체크포인트 저장
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
        
        trainer.save_checkpoint(epoch, val_loss, is_best)
        
        print(f'Epoch {epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}')
        
        # 첫 번째 배치의 메모리 사용량 출력 (디버깅용)
        for batch_idx, (indices, values) in enumerate(train_loader):
            print(f"Batch {batch_idx} memory usage:")
            print(f"Indices shape: {indices.shape}, Values shape: {values.shape}")
            print(f"Non-zero elements: {len(values)}")
            break
    
    print('Training completed!')

if __name__ == '__main__':
    main()