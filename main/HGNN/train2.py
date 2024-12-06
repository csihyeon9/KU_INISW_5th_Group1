# train2.py -> grid search 기법 추가

import itertools
import torch
from torch.utils.data import DataLoader
import yaml
from scipy import sparse
import numpy as np
from pathlib import Path
import logging
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.ver2.data_processor2 import KeywordProcessor
from src.ver2.models2 import KeywordHGNN
from src.ver2.trainer2 import Trainer
from src.ver2.visualization2 import ModelEvaluationVisualizer

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

def setup_detailed_logging(save_dir, params):
    """상세한 로깅을 위한 설정"""
    # 하이퍼파라미터별 로그 디렉토리 생성
    log_dir = Path(save_dir) / 'grid_search_logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # 파라미터 조합을 반영한 로그 파일명 생성
    param_str = '_'.join([f"{k}_{v}" for k, v in params.items()])
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f'grid_search_{param_str}_{timestamp}.log'
    
    # 로깅 핸들러 설정
    logger = logging.getLogger(param_str)
    logger.setLevel(logging.INFO)
    
    # 파일 핸들러 추가
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s: %(message)s')
    file_handler.setFormatter(formatter)
    
    # 기존 핸들러 제거 (중복 방지)
    logger.handlers.clear()
    logger.addHandler(file_handler)
    
    return logger, log_file

def grid_search_hgnn(config_path='config/config.yaml'):
    """HGNN를 위한 고급 그리드 서치 수행"""
    # 기본 구성 파일 로드
    with open(config_path, 'r') as f:
        base_config = yaml.safe_load(f)
    
    # 하이퍼파라미터 탐색 공간 정의
    param_grid = {
        'hidden_dim': [64, 128],
        'num_layers': [2, 3, 4],
        'dropout': [0.1, 0.2, 0.3],
        'learning_rate': [0.001, 0.0005, 0.0001],
        'batch_size': [64, 128, 256]
    }
    
    # 결과 추적을 위한 데이터 프레임
    all_results = []
    
    # 모든 파라미터 조합 생성
    param_combinations = [
        dict(zip(param_grid.keys(), v)) 
        for v in itertools.product(*param_grid.values())
    ]
    
    # 데이터 전처리 (그리드 서치 전 한 번만 수행)
    processor = KeywordProcessor(
        base_config['data']['unique_keywords_path'],
        base_config['data']['news_data_path']
    )
    processor.load_data()
    matrices = processor.process_all_articles()
    
    dataset = SparseMatrixDataset(matrices, processor)
    
    # 데이터 분할
    split_ratio = base_config['data']['split_ratio']
    train_size = int(split_ratio['train'] * len(dataset))
    val_size = int(split_ratio['val'] * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size]
    )
    
    # 최적 구성 초기화
    best_config = None
    best_val_loss = float('inf')
    
    # 그리드 서치 수행
    for idx, params in enumerate(param_combinations, 1):
        # 상세 로깅 설정
        logger, log_file = setup_detailed_logging(
            base_config['training']['save_dir'], 
            params
        )
        
        logger.info(f"그리드 서치 반복 {idx}/{len(param_combinations)}")
        logger.info(f"현재 파라미터: {params}")
        
        # 현재 파라미터로 구성 업데이트
        config = base_config.copy()
        config['model']['hidden_dim'] = params['hidden_dim']
        config['model']['num_layers'] = params['num_layers']
        config['model']['dropout'] = params['dropout']
        config['training']['learning_rate'] = params['learning_rate']
        config['training']['batch_size'] = params['batch_size']
        
        # 데이터 로더 생성
        train_loader = DataLoader(
            train_dataset, 
            batch_size=params['batch_size'],
            shuffle=True,
            collate_fn=collate_sparse_batch
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=params['batch_size'],
            collate_fn=collate_sparse_batch
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=params['batch_size'],
            collate_fn=collate_sparse_batch
        )
        
        # 시각화 도구 초기화
        visualizer = ModelEvaluationVisualizer(config)
        
        # 모델 및 트레이너 초기화
        model = KeywordHGNN(
            num_keywords=processor.matrix_size,
            hidden_dim=params['hidden_dim'],
            num_layers=params['num_layers'],
            dropout=params['dropout']
        )
        
        trainer = Trainer(model, config)
        
        # 학습 히스토리 저장소
        train_losses = []
        val_losses = []
        contrastive_losses = []
        precision_values = {str(k): [] for k in config['visualization']['metrics']['k_values']}
        recall_values = {str(k): [] for k in config['visualization']['metrics']['k_values']}
        
        # 학습 시작
        best_epoch_val_loss = float('inf')
        
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
            except Exception as e:
                logger.warning(f"Contrastive Loss 계산 중 오류: {e}")
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
                    except Exception as e:
                        logger.warning(f"추천 성능 평가 중 오류: {e}")
                        precision_values[str(k)].append(0.0)
                        recall_values[str(k)].append(0.0)
            
            # 체크포인트 저장
            is_best = val_loss < best_epoch_val_loss
            if is_best:
                best_epoch_val_loss = val_loss
            
            trainer.save_checkpoint(epoch, val_loss, is_best)
            
            logger.info(f'Epoch {epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}')
        
        # 학습 과정 시각화
        try:
            # 손실 곡선 시각화
            visualizer.plot_training_history(train_losses, val_losses, contrastive_losses)
            
            # 메트릭 시각화
            visualizer.plot_metrics(precision_values, recall_values)
        except Exception as e:
            logger.error(f"시각화 중 오류: {e}")
        
        # 결과 추적
        result = params.copy()
        result.update({
            'final_train_loss': train_losses[-1],
            'final_val_loss': val_losses[-1],
            'final_contrastive_loss': contrastive_losses[-1],
            'precision_values': precision_values,
            'recall_values': recall_values
        })
        all_results.append(result)
        
        # 최적 구성 업데이트
        if val_losses[-1] < best_val_loss:
            best_val_loss = val_losses[-1]
            best_config = result
        
        # GPU 메모리 정리
        torch.cuda.empty_cache()
    
    # 전체 결과 저장
    results_df = pd.DataFrame(all_results)
    results_file = Path(base_config['training']['save_dir']) / 'grid_search_comprehensive_results.csv'
    results_df.to_csv(results_file, index=False)
    
    # 최적 하이퍼파라미터 시각화
    try:
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=results_df, x='hidden_dim', y='final_val_loss', hue='num_layers')
        plt.title('하이퍼파라미터에 따른 검증 손실')
        plt.xlabel('은닉층 차원')
        plt.ylabel('최종 검증 손실')
        plt.savefig(Path(base_config['training']['save_dir']) / 'hyperparameter_performance.png')
        plt.close()
    except Exception as e:
        print(f"하이퍼파라미터 성능 시각화 중 오류: {e}")
    
    print('그리드 서치 완료!')
    print(f'최적 하이퍼파라미터: {best_config}')
    print(f'결과 저장 위치: {results_file}')
    
    return best_config

def main():
    # 최적 하이퍼파라미터 찾기
    best_hyperparams = grid_search_hgnn()

if __name__ == '__main__':
    main()