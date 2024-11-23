import os
import torch
import torch.nn.functional as F
import pprint as pp
from config import get_config
from datasets.data_helper import load_json_data, preprocess_json_for_hgnn, construct_hypergraph_from_json, analyze_data_distribution
from utils.hypergraph_utils import generate_G_from_H
from models import HGNNRecommender
from torch.optim.lr_scheduler import CosineAnnealingLR

# 환경 변수 설정
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# 설정 파일 로드
config = get_config('config/config.yaml')
data_dir = config['econfinnews_raw'] if config['on_dataset'] == 'EconFinNews' else Exception("Data not prepared yet!")

# 데이터 전처리
fts, lbls, idx_train, idx_test = preprocess_json_for_hgnn(data_dir)

# 하이퍼그래프 생성
H = construct_hypergraph_from_json(fts)
G = generate_G_from_H(H)

# PyTorch 텐서 변환
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
fts = torch.Tensor(fts).to(device)
lbls = torch.Tensor(lbls).squeeze().long().to(device)
G = torch.Tensor(G).to(device)
idx_train = torch.Tensor(idx_train).long().to(device)
idx_test = torch.Tensor(idx_test).long().to(device)

def _main():
    print(f"\nClassification on {config['on_dataset']} dataset!!! class number: {lbls.max().item() + 1}")
    print('\nConfiguration:')
    pp.pprint(config)
    print()

    # HGNNRecommender 모델 초기화
    model = HGNNRecommender(
        in_dim=fts.shape[1],
        hidden_dim=config['n_hid'],
        embedding_dim=fts.shape[1]
    ).to(device)

    # 옵티마이저 및 스케줄러 설정
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    scheduler = CosineAnnealingLR(optimizer, T_max=config['max_epoch'])

    # 모델 학습
    model.train()
    for epoch in range(config['max_epoch']):
        optimizer.zero_grad()
        outputs = model(fts, G)

        loss = F.cross_entropy(outputs[idx_train], lbls[idx_train])
        loss.backward()
        optimizer.step()
        scheduler.step()

        if (epoch + 1) % config['print_freq'] == 0:
            print(f"Epoch {epoch + 1}/{config['max_epoch']}, Loss: {loss.item():.4f}")

    # 학습 완료 후 모델 저장
    model_save_path = "./saves/hgnn_recommender_model.pth"
    torch.save(model.state_dict(), model_save_path)
    print(f"\nModel saved to {model_save_path}")

if __name__ == '__main__':
    analyze_data_distribution(data_dir)
    _main()
