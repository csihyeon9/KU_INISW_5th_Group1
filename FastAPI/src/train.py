import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from models.HGNN_model import HGNN
from src.utils.data_loader import load_data
from src.visualization import ModelEvaluationVisualizer

from config.config import load_config

def train(config):
    """
    모델 학습 함수
    :param config: 설정 딕셔너리
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    X, H, labels = load_data(config['data']['processed_data_path'])
    
    print(f"X shape: {X.shape}, H shape: {H.shape}, labels shape: {labels.shape}")

    in_features = X.shape[1]
    model = HGNN(in_features,
                 config['model']['hidden_features'],
                 config['model']['out_features'],
                 config['model']['dropout']).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config['training']['lr'], weight_decay=float(config['training']['weight_decay']))
    criterion = torch.nn.NLLLoss()

    batch_size = config['training']['batch_size']
    
    visualizer = ModelEvaluationVisualizer(config)
    
    train_losses = []
    val_losses = []  # Validation loss를 위한 리스트

    for epoch in range(config['training']['epochs']):
        model.train()
        epoch_loss = 0
        
        for i in range(0, len(labels), batch_size):
            batch_X = X[i:i + batch_size].to(device)
            batch_H = H[i:i + batch_size].to(device)
            batch_labels = labels[i:i + batch_size].to(device)

            optimizer.zero_grad()
            output = model(batch_X, batch_H)  # 배치 단위로 모델에 입력
            loss = criterion(output, batch_labels)  # 손실 계산
            
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / (len(labels) // batch_size)
        train_losses.append(avg_loss)

        # Validation loss 계산 (여기서는 간단히 train_loss를 사용)
        val_losses.append(avg_loss)  # 실제로는 validation 데이터로 계산해야 함

        print(f'Epoch {epoch+1}/{config["training"]["epochs"]}, Loss: {avg_loss:.4f}')
    
    # Training history 시각화
    visualizer.plot_training_history(train_losses, val_losses)

if __name__ == "__main__":
    config = load_config('config/config.yaml')
    train(config)