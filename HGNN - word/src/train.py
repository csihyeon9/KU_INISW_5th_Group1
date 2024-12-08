import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from models.HGNN_model import HGNN
from src.utils.data_loader import load_data
from src.visualization import TrainingVisualizer
from config.config import load_config


def train(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 데이터 로드
    print("Loading data...")
    data = torch.load(config['data']['processed_data_path'])
    X, H, G, labels = data['X'], data['H'], data['G'], data['labels']

    # 데이터를 GPU로 이동
    X, H, G, labels = X.to(device), H.to(device), G.to(device), labels.to(device)

    # Train-Test Split
    print("Splitting data into train and validation sets...")
    train_indices, val_indices = train_test_split(
        range(len(labels)),
        test_size=config['training'].get('validation_split', 0.2),
        random_state=42
    )

    # 학습 및 검증 데이터 분리
    X_train, X_val = X[train_indices], X[val_indices]
    labels_train, labels_val = labels[train_indices], labels[val_indices]

    # G 행렬 분리
    G_train = G[train_indices, :][:, train_indices].to_sparse()
    G_val = G[val_indices, :][:, val_indices].to_sparse()

    # 모델 초기화
    print("Initializing the model...")
    model = HGNN(
        in_ch=X.shape[1],
        n_hid=config['model']['hidden_features'],
        n_class=config['model']['out_features'],
        dropout=config['model']['dropout']
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config['training']['lr'],
        weight_decay=config['training']['weight_decay']
    )
    criterion = torch.nn.CrossEntropyLoss()

    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    # 학습 루프
    print("Starting training...")
    for epoch in range(config['training']['epochs']):
        model.train()
        train_loss = 0
        y_true, y_pred = [], []

        # Training Loop
        for i in range(0, len(train_indices), config['training']['batch_size']):
            batch_indices = train_indices[i:i + config['training']['batch_size']]
            batch_X = X_train[batch_indices]
            batch_labels = labels_train[batch_indices]
            G_batch = G_train[batch_indices, :][:, batch_indices]

            optimizer.zero_grad()
            output = model(batch_X, G_batch)  # 모델에 G 전달
            loss = criterion(output, batch_labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            y_true.extend(batch_labels.cpu().numpy())
            y_pred.extend(torch.argmax(output, dim=1).cpu().numpy())

        train_losses.append(train_loss / len(train_indices))
        train_accuracies.append(accuracy_score(y_true, y_pred))

        # Validation Loop
        model.eval()
        val_loss = 0
        y_true, y_pred = [], []
        with torch.no_grad():
            for i in range(0, len(val_indices), config['training']['batch_size']):
                batch_indices = val_indices[i:i + config['training']['batch_size']]
                batch_X = X_val[batch_indices]
                batch_labels = labels_val[batch_indices]
                G_batch = G_val[batch_indices, :][:, batch_indices]

                output = model(batch_X, G_batch)  # 모델에 G 전달
                loss = criterion(output, batch_labels)
                val_loss += loss.item()
                y_true.extend(batch_labels.cpu().numpy())
                y_pred.extend(torch.argmax(output, dim=1).cpu().numpy())

        val_losses.append(val_loss / len(val_indices))
        val_accuracies.append(accuracy_score(y_true, y_pred))

        print(f"Epoch {epoch + 1}/{config['training']['epochs']} - "
              f"Train Loss: {train_losses[-1]:.4f}, Train Acc: {train_accuracies[-1]:.4f} - "
              f"Val Loss: {val_losses[-1]:.4f}, Val Acc: {val_accuracies[-1]:.4f}")

        # 체크포인트 저장
        if (epoch + 1) % config['training']['checkpoint_interval'] == 0:
            checkpoint_path = f"{config['training']['save_dir']}/checkpoint_epoch_{epoch + 1}.pth"
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Checkpoint saved at {checkpoint_path}")

    # 학습 및 검증 결과 시각화
    print("Visualizing training results...")
    visualizer = TrainingVisualizer(config)
    visualizer.plot_training_history(train_losses, val_losses)
    visualizer.plot_accuracy_history(train_accuracies, val_accuracies)

    # 최종 모델 저장
    final_model_path = f"{config['training']['save_dir']}/final_model.pth"
    torch.save(model.state_dict(), final_model_path)
    print(f"Final model saved at {final_model_path}")


if __name__ == "__main__":
    config = load_config('config/config.yaml')
    train(config)
