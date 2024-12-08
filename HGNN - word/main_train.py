import torch
import os
from models.HGNN_model import HGNN
from config.config import load_config
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from src.visualization import TrainingVisualizer

def save_model(model, save_path):
    """학습된 모델 저장"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)  # 디렉토리 생성
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

def train_hgnn(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 데이터 로드
    data = torch.load(config['data']['processed_data_path'])
    X, H, G, labels = data['X'], data['H'], data['G'], data['labels']
    X, H, G, labels = X.to(device), H.to(device), G.to(device), labels.to(device)

    # 데이터셋 분할 (Train / Validation)
    from sklearn.model_selection import train_test_split
    train_indices, val_indices = train_test_split(range(len(labels)), test_size=0.2, random_state=42)

    # G_sparse를 밀집 행렬로 변환
    G_dense = G.to_dense()

    # 모델 초기화
    model = HGNN(
        in_ch=X.shape[1],
        n_hid=config['model']['hidden_features'],
        n_class=config['model']['out_features']
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config['training']['lr'],
        weight_decay=config['training']['weight_decay']
    )
    criterion = torch.nn.CrossEntropyLoss()

    train_losses = []
    val_losses = []  # Validation 손실 리스트 정의
    accuracies = []

    for epoch in range(config['training']['epochs']):
        model.train()
        epoch_loss = 0
        y_true, y_pred = [], []

        # Training loop
        for i in tqdm(range(0, len(train_indices), config['training']['batch_size']), desc=f"Epoch {epoch + 1}/{config['training']['epochs']}"):
            batch_indices = train_indices[i:i + config['training']['batch_size']]
            batch_X = X[batch_indices]
            batch_labels = labels[batch_indices]

            G_batch_dense = G_dense[batch_indices, :][:, batch_indices]
            G_batch_sparse = G_batch_dense.to_sparse()

            optimizer.zero_grad()
            output = model(batch_X, G_batch_sparse)
            loss = criterion(output, batch_labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            y_true.extend(batch_labels.cpu().numpy())
            y_pred.extend(torch.argmax(output, dim=1).cpu().numpy())

        # Training 손실 및 정확도 기록
        train_losses.append(epoch_loss / len(train_indices))
        train_accuracy = accuracy_score(y_true, y_pred)
        accuracies.append(train_accuracy)
        print(f"Epoch {epoch + 1}/{config['training']['epochs']} - Train Loss: {epoch_loss / len(train_indices):.4f}, Train Accuracy: {train_accuracy:.4f}")

        # Validation loop
        model.eval()
        val_loss = 0
        val_true, val_pred = [], []
        with torch.no_grad():
            for i in range(0, len(val_indices), config['training']['batch_size']):
                batch_indices = val_indices[i:i + config['training']['batch_size']]
                batch_X = X[batch_indices]
                batch_labels = labels[batch_indices]

                G_batch_dense = G_dense[batch_indices, :][:, batch_indices]
                G_batch_sparse = G_batch_dense.to_sparse()

                output = model(batch_X, G_batch_sparse)
                loss = criterion(output, batch_labels)

                val_loss += loss.item()
                val_true.extend(batch_labels.cpu().numpy())
                val_pred.extend(torch.argmax(output, dim=1).cpu().numpy())

        val_losses.append(val_loss / len(val_indices))  # Validation 손실 기록
        val_accuracy = accuracy_score(val_true, val_pred)
        print(f"Epoch {epoch + 1}/{config['training']['epochs']} - Validation Loss: {val_loss / len(val_indices):.4f}, Validation Accuracy: {val_accuracy:.4f}")

        # 에포크별 체크포인트 저장
        if (epoch + 1) % config['training']['checkpoint_interval'] == 0:
            checkpoint_path = os.path.join(config['training']['save_dir'], f"hgnn_checkpoint_epoch_{epoch + 1}.pth")
            save_model(model, checkpoint_path)

    # 최종 모델 저장
    final_model_path = os.path.join(config['training']['save_dir'], "hgnn_model.pth")
    save_model(model, final_model_path)

    # 학습 및 검증 손실 시각화
    visualizer = TrainingVisualizer(config)
    visualizer.plot_training_history(train_losses, val_losses)


if __name__ == "__main__":
    config = load_config('config/config.yaml')
    train_hgnn(config)
