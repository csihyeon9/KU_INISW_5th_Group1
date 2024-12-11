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
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

def train_hgnn(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 데이터 로드
    data = torch.load(config['data']['processed_data_path'])
    X, H, W, L, labels = data['X'], data['H'], data['W'], data['L'], data['labels']
    X, H, W, L, labels = X.to(device), H.to(device), W.to(device), L.to(device), labels.to(device)

    # 데이터셋 분할
    train_indices, val_indices = train_test_split(range(len(labels)), test_size=0.2, random_state=42)

    # 모델 초기화
    model = HGNN(
        in_ch=X.shape[1],
        n_hid=config['model']['hidden_features'],
        n_class=len(torch.unique(labels)),
        dropout=config['model']['dropout']
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config['training']['lr'],
        weight_decay=config['training']['weight_decay']
    )
    criterion = torch.nn.CrossEntropyLoss(reduction='sum')  # 합계로 reduction 방식 변경

    train_losses, val_losses, train_accuracies, val_accuracies = [], [], [], []

    for epoch in range(config['training']['epochs']):
        # Training Phase
        model.train()
        total_train_loss = 0
        all_train_preds = []
        all_train_labels = []

        for i in range(0, len(train_indices), config['training']['batch_size']):
            batch_indices = train_indices[i:i + config['training']['batch_size']]
            batch_X = X[batch_indices]
            batch_labels = labels[batch_indices]
            batch_L = L[batch_indices][:, batch_indices].to_sparse()

            optimizer.zero_grad()
            outputs = model(batch_X, batch_L)
            loss = criterion(outputs, batch_labels)
            
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            all_train_preds.extend(outputs.argmax(dim=1).cpu().numpy())
            all_train_labels.extend(batch_labels.cpu().numpy())

        # 전체 training 데이터에 대한 평균 손실과 정확도 계산
        avg_train_loss = total_train_loss / len(train_indices)
        train_accuracy = accuracy_score(all_train_labels, all_train_preds)
        
        train_losses.append(avg_train_loss)
        train_accuracies.append(train_accuracy)

        # Validation Phase
        model.eval()
        total_val_loss = 0
        all_val_preds = []
        all_val_labels = []

        with torch.no_grad():
            for i in range(0, len(val_indices), config['training']['batch_size']):
                batch_indices = val_indices[i:i + config['training']['batch_size']]
                batch_X = X[batch_indices]
                batch_labels = labels[batch_indices]
                batch_L = L[batch_indices][:, batch_indices].to_sparse()

                outputs = model(batch_X, batch_L)
                loss = criterion(outputs, batch_labels)

                total_val_loss += loss.item()
                all_val_preds.extend(outputs.argmax(dim=1).cpu().numpy())
                all_val_labels.extend(batch_labels.cpu().numpy())

        # 전체 validation 데이터에 대한 평균 손실과 정확도 계산
        avg_val_loss = total_val_loss / len(val_indices)
        val_accuracy = accuracy_score(all_val_labels, all_val_preds)

        val_losses.append(avg_val_loss)
        val_accuracies.append(val_accuracy)

        print(f"Epoch {epoch + 1}/{config['training']['epochs']}")
        print(f"Train Loss: {avg_train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")
        print(f"Val Loss: {avg_val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

        if (epoch + 1) % config['training']['checkpoint_interval'] == 0:
            checkpoint_path = os.path.join(
                config['training']['save_dir'], 
                f"hgnn_checkpoint_epoch_{epoch + 1}.pth"
            )
            save_model(model, checkpoint_path)

    final_model_path = os.path.join(config['training']['save_dir'], "hgnn_model.pth")
    save_model(model, final_model_path)

    # 학습 과정 시각화를 위해 정확도 데이터도 전달
    visualizer = TrainingVisualizer(config)
    visualizer.plot_training_history(train_losses, val_losses, train_accuracies, val_accuracies)

if __name__ == "__main__":
    config = load_config('config/config.yaml')
    train_hgnn(config)