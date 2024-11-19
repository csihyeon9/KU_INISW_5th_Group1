# main.py
import os
import torch
import numpy as np
import random
from pathlib import Path
import json
from datetime import datetime
import argparse
import logging
from typing import Dict

from config import HGANConfig
from model import HGAN, HGANPredictor
from dataset import HGANDataset
from trainer import HGANTrainer
from evaluator import HGANEvaluator

def set_seed(seed: int):
    """재현성을 위한 시드 설정"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

def setup_logging(save_dir: Path) -> logging.Logger:
    """로깅 설정"""
    log_file = save_dir / 'training.log'
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def setup_experiment_dir(config: HGANConfig) -> Path:
    """실험 결과 저장 디렉토리 설정"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = Path(config.save_dir) / f"{config.model_name}_{timestamp}"
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 설정 저장
    with open(save_dir / 'config.json', 'w') as f:
        json.dump(vars(config), f, indent=2)
    
    return save_dir

def parse_args():
    """커맨드 라인 인자 파싱"""
    parser = argparse.ArgumentParser(description='HGAN for Financial Keywords Analysis')
    
    # 데이터 관련
    parser.add_argument('--data_path', type=str, required=True,
                      help='Path to the processed data file')
    parser.add_argument('--embedding_dim', type=int, default=64,
                      help='Embedding dimension')
    
    # 모델 관련
    parser.add_argument('--device', type=str, default=None,
                      help='Device to use (cuda/cpu)')
    parser.add_argument('--batch_size', type=int, default=32,
                      help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=200,
                      help='Number of training epochs')
    parser.add_argument('--hidden_dim', type=int, default=128,
                      help='Hidden dimension size')
    
    return parser.parse_args()

def save_results(save_dir: Path, results: Dict, embeddings: torch.Tensor):
    """결과 저장"""
    # 결과 저장
    with open(save_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # 임베딩 저장
    torch.save(embeddings, save_dir / 'embeddings.pt')

def main():
    """메인 실행 함수"""
    # 인자 파싱 및 설정
    args = parse_args()
    config = HGANConfig()
    for k, v in vars(args).items():
        if v is not None:
            setattr(config, k, v)
    
    # 시드 설정
    set_seed(config.seed)
    
    # 실험 디렉토리 설정
    save_dir = setup_experiment_dir(config)
    logger = setup_logging(save_dir)
    
    logger.info("Loading data...")
    try:
        # 데이터 로드
        with open(config.data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 데이터셋 생성
        dataset = HGANDataset(
            keywords=data['keywords'],
            relations=data['relations'],
            embeddings=torch.tensor(data['embeddings'], dtype=torch.float),
            relation_types=data.get('relation_types')
        )
        
        logger.info(f"Loaded {len(dataset)} keywords and {len(dataset.hyperedges)} hyperedges")
        
        # 모델 초기화
        model = HGAN(
            in_features=config.embedding_dim,  # 64
            hidden_features=config.hidden_dim,  # 128
            num_layers=config.num_layers,
            dropout=config.dropout
        )
        
        # predictor는 model의 hidden_features를 기반으로 초기화
        predictor = HGANPredictor(
            hidden_size=config.hidden_dim,  # 128 (concatenate하면 256이 됨)
            num_relations=len(dataset.relation_types)
        )
        
        # 학습기 및 평가기 초기화
        trainer = HGANTrainer(
            model=model,
            predictor=predictor,
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
            device=config.device
        )
        
        evaluator = HGANEvaluator(device=config.device)
        
        # 학습 실행
        logger.info("Starting training...")
        train_history = trainer.train(
            dataset=dataset,
            num_epochs=config.num_epochs,
            batch_size=config.batch_size,
            negative_ratio=config.negative_ratio,
            eval_every=config.eval_every,
            early_stopping_patience=config.early_stopping_patience
        )
        
        # 모델 평가
        logger.info("Evaluating model...")
        metrics = evaluator.evaluate(model, predictor, dataset)
        
        # 결과 저장
        results = {
            'train_history': train_history,
            'metrics': metrics
        }
        
        if config.save_embeddings:
            embeddings = trainer.get_embeddings(dataset)
            save_results(save_dir, results, embeddings)
        
        logger.info(f"Experiment completed. Results saved to {save_dir}")
        
        # 예시 키워드 추천
        logger.info("\nExample keyword recommendations:")
        example_keywords = random.sample(dataset.keywords, 3)
        for keyword in example_keywords:
            recommendations = evaluator.recommend_similar_keywords(
                query_keyword=keyword,
                dataset=dataset,
                model=model,
                k=5
            )
            
            logger.info(f"\nRecommendations for '{keyword}':")
            for rec_keyword, similarity, uncertainty in recommendations:
                logger.info(f"  - {rec_keyword:<20} (similarity: {similarity:.3f}, uncertainty: {uncertainty:.3f})")
    
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()