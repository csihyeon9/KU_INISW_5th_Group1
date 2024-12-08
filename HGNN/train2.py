import sys
import os
import torch
import torch.sparse
from models.HGNN_model import HGNN
from config.config import load_config
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from src.visualization import TrainingVisualizer

def train_hgnn(config):
    """
    HGNN 모델 학습 함수
    """
    # GPU 또는 CPU 선택
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 데이터 로드
    data = torch.load(config['data']['processed_data_path'])  # hgnn_data2.pt 파일 경로
    X, H, G, labels = data['X'], data['H'], data['G'], data['labels']
    X, H, G, labels = X.to(device), H.to(device), G.to(device), labels.to(device)

    # G를 희소 행렬로 변환
    G_sparse = torch.sparse_coo_tensor(
        indices=torch.nonzero(G, as_tuple=True),
        values=G[G != 0],
        size=G.shape,
        device=device
    )

    # 모델 초기화
    model = HGNN(
        in_ch=X.shape[1],  # 입력 특징 크기 (PMI 행렬의 열 개수)
        n_class=config['model']['out_features'],  # 출력 클래스 개수 (동사 카테고리 수)
        n_hid=config['model']['hidden_features'],  # 숨겨진 레이어 크기
        dropout=config['model']['dropout']  # 드롭아웃 비율
    ).to(device)

    # 옵티마이저 및 손실 함수 초기화
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config['training']['lr'],
        weight_decay=config['training']['weight_decay']
    )
    criterion = torch.nn.CrossEntropyLoss()

    # 시각화 도구 초기화
    visualizer = TrainingVisualizer(config)

    # 학습 기록
    train_losses = []
    accuracies = []

    # 학습 루프
    epochs = config['training']['epochs']
    batch_size = config['training']['batch_size']
    num_batches = max(1, len(labels) // batch_size)
    checkpoint_interval = config['training'].get('checkpoint_interval', 10)

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        y_true, y_pred = [], []

        for i in tqdm(range(0, len(labels), batch_size), desc=f"Epoch {epoch + 1}/{epochs}"):
            # 배치 데이터 준비
            batch_X = X[i:i + batch_size]
            batch_labels = labels[i:i + batch_size]

            optimizer.zero_grad()
            output = model(batch_X, H, G_sparse)  # 모델 forward
            loss = criterion(output, batch_labels)  # 손실 계산
            
            loss.backward()
            optimizer.step()

            # 손실 및 예측값 저장
            epoch_loss += loss.item()
            y_true.extend(batch_labels.cpu().numpy())                 # 모든 배치의 실제 레이블값 : batch_labels
            y_pred.extend(torch.argmax(output, dim=1).cpu().numpy())  # 모델의 출력값 : torch.argmax(output, dim=1)

        # 학습 손실 및 정확도 기록
        avg_loss = epoch_loss / num_batches
        accuracy = accuracy_score(y_true, y_pred)
        train_losses.append(avg_loss)
        accuracies.append(accuracy)
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")

        # Checkpoint 저장
        if (epoch + 1) % checkpoint_interval == 0:
            checkpoint_path = os.path.join(config['training']['save_dir'], f"hgnn_checkpoint_epoch_{epoch + 1}.pth")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Checkpoint saved at {checkpoint_path}")

    # 학습 곡선 시각화
    plot_save_dir = config['visualization']['save_dir']
    os.makedirs(plot_save_dir, exist_ok=True)
    visualizer.plot_training_history(train_losses, save_dir=plot_save_dir)
    visualizer.plot_accuracy_history(accuracies, save_dir=plot_save_dir)

    # 최종 모델 저장
    final_model_path = os.path.join(config['training']['save_dir'], "hgnn_model.pth")
    torch.save(model.state_dict(), final_model_path)
    print(f"Final HGNN model saved at {final_model_path}")

if __name__ == "__main__":
    # 설정 로드
    config = load_config('config/config.yaml')
    train_hgnn(config)
