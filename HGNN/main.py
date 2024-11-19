# main.py
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import argparse
from pathlib import Path
import torch
import json
import yaml
from datetime import datetime

from models.typed_hgnn import EnhancedHGNN
from data.typed_dataset import TypedHypergraphDataset
from data.sample_data import create_financial_data, load_custom_data
from utils.typed_trainer import TypedHGNNTrainer
from utils.typed_evaluator import TypedHGNNEvaluator
from utils.typed_visualizer import TypedHGNNVisualizer

def parse_args():
    parser = argparse.ArgumentParser(description='Enhanced HGNN for Keyword Analysis')
    
    # 데이터 관련
    parser.add_argument('--data_path', type=str, default=None,
                        help='Path to custom data file (JSON format)')
    parser.add_argument('--embedding_dim', type=int, default=64,
                        help='Initial embedding dimension')
    
    # 모델 관련
    parser.add_argument('--hidden_dim', type=int, default=128,
                        help='Hidden layer dimension')
    parser.add_argument('--num_layers', type=int, default=3,
                        help='Number of HGNN layers')
    
    # 학습 관련
    parser.add_argument('--epochs', type=int, default=150,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for relation prediction')
    parser.add_argument('--lr', type=float, default=0.0002,
                        help='Initial learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay for optimizer')
    
    # 시스템 관련
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (cuda/cpu)')
    parser.add_argument('--save_dir', type=str, default='results',
                        help='Directory to save results')
    
    # 실험 관련
    parser.add_argument('--experiment_name', type=str, default=None,
                        help='Name of the experiment')
    parser.add_argument('--save_model', action='store_true',
                        help='Whether to save the trained model')
    
    return parser.parse_args()

def setup_experiment_dir(args):
    """실험 결과 저장을 위한 디렉토리 설정"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_name = args.experiment_name or f'experiment_{timestamp}'
    
    save_dir = Path(args.save_dir) / exp_name
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 설정 저장
    config = vars(args)
    with open(save_dir / 'config.yaml', 'w') as f:
        yaml.dump(config, f)
    
    return save_dir

def save_results(save_dir: Path, results: dict, metrics: dict):
    """실험 결과 저장"""
    # 메트릭 저장
    with open(save_dir / 'metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # 임베딩 저장
    if 'embeddings' in results:
        np.save(save_dir / 'embeddings.npy', results['embeddings'])

def main():
    args = parse_args()
    save_dir = setup_experiment_dir(args)
    
    print("\n=== Enhanced HGNN for Financial Keyword Analysis ===\n")
    
    # 1. 데이터 준비
    print("1. 데이터 준비 중...")
    if args.data_path:
        keywords, relations, embeddings = load_custom_data(args.data_path)
    else:
        keywords, relations, embeddings = create_financial_data(
            embedding_dim=args.embedding_dim
        )
    
    dataset = TypedHypergraphDataset(keywords, relations, embeddings)
    num_relation_types = len(dataset.relation_types)
    print(f"  - 키워드 수: {len(keywords)}")
    print(f"  - 관계 타입 수: {num_relation_types}")
    print(f"  - 전체 관계 수: {len(relations)}")
    
    # 2. 모델 초기화
    print("\n2. 모델 초기화...")
    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    model = EnhancedHGNN(
        in_channels=embeddings.shape[1],
        hidden_channels=args.hidden_dim,
        num_relation_types=num_relation_types,
        num_layers=args.num_layers
    )
    print(f"  - Device: {device}")
    print(f"  - Model parameters: {sum(p.numel() for p in model.parameters())}")
    
    # 3. 학습기, 평가기, 시각화기 초기화
    trainer = TypedHGNNTrainer(
        model,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        device=device
    )
    
    evaluator = TypedHGNNEvaluator(device=device)
    visualizer = TypedHGNNVisualizer()
    
    # 4. 모델 학습
    print("\n3. 모델 학습 중...")
    losses = trainer.train(
        dataset,
        num_epochs=args.epochs,
        batch_size=args.batch_size
    )
    
    # 학습 과정 시각화
    visualizer.plot_training_history(losses)
    
    # 5. 모델 평가
    print("\n4. 모델 평가 중...")
    metrics = evaluator.evaluate(model, dataset)
    print("\n=== 모델 평가 결과 ===")
    print(f"재구성 손실: {metrics['reconstruction_loss']:.4f}")
    print(f"관계 예측 AUC: {metrics['relation_prediction_auc']:.4f}")
    print(f"관계 보존성 점수: {metrics['relation_preservation']:.4f}")
    
    # 관계 타입별 메트릭 시각화
    visualizer.plot_relation_type_distribution(metrics['type_wise_metrics'])
    
    # 6. 임베딩 추출 및 분석
    print("\n5. 임베딩 추출 및 분석...")
    final_embeddings = trainer.get_embeddings(dataset)
    
    # 유사도 네트워크 시각화
    visualizer.plot_typed_similarity_network(
        keywords,
        final_embeddings,
        dataset.relation_types
    )
    
    # 키워드 클러스터링 시각화
    visualizer.plot_keyword_clusters(
        keywords,
        final_embeddings
    )
    
    # 7. 중요 키워드 유사도 분석
    print("\n6. 키워드 유사도 분석...")
    query_keywords = ["통화정책", "주식시장", "포트폴리오", "금리", "중앙은행"]
    
    print("\n=== 키워드 유사도 분석 결과 ===")
    for query_keyword in query_keywords:
        print(f"\n'{query_keyword}'와 유사한 키워드들:")
        query_idx = keywords.index(query_keyword)
        similar_keywords = evaluator.find_similar_keywords(
            dataset, query_idx, final_embeddings
        )
        
        for idx, similarity in similar_keywords:
            print(f"  - {keywords[idx]:<12} : {similarity:.4f}")
    
    # 8. 결과 저장
    if args.save_model:
        results = {
            'embeddings': final_embeddings,
            'model_state': model.state_dict(),
            'keywords': keywords,
            'relation_types': dataset.relation_types
        }
        save_results(save_dir, results, metrics)
        print(f"\n결과가 {save_dir}에 저장되었습니다.")
    
    print("\n=== 분석 완료 ===")
    
    return dataset, model, final_embeddings

if __name__ == "__main__":
    dataset, model, embeddings = main()