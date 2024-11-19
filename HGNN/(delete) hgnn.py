# hgnn.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
import random
from torch.optim.lr_scheduler import ReduceLROnPlateau

class HGNNConv(nn.Module):
    """하이퍼그래프 컨볼루션 레이어"""
    def __init__(self, in_channels, out_channels):
        super(HGNNConv, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(in_channels, out_channels))
        self.bias = nn.Parameter(torch.Tensor(out_channels))
        self.reset_parameters()
        
    def reset_parameters(self):
        """가중치 초기화"""
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)
    
    def forward(self, X, H):
        """
        Args:
            X: 노드 특성 행렬
            H: 하이퍼그래프 인시던스 행렬
        """
        D_v = torch.sum(H, dim=1)
        D_e = torch.sum(H, dim=0)
        
        D_v_invsqrt = torch.pow(D_v + 1e-8, -0.5)
        D_e_invsqrt = torch.pow(D_e + 1e-8, -0.5)
        
        D_v_invsqrt = torch.diag(D_v_invsqrt)
        D_e_invsqrt = torch.diag(D_e_invsqrt)
        
        theta = torch.matmul(torch.matmul(D_v_invsqrt, H), D_e_invsqrt)
        H_norm = torch.matmul(torch.matmul(theta, theta.t()), X)
        
        out = torch.matmul(H_norm, self.weight) + self.bias
        return out

class HGNN(nn.Module):
    """하이퍼그래프 신경망"""
    def __init__(self, in_channels, hidden_channels):
        super(HGNN, self).__init__()
        # 컨볼루션 레이어
        self.conv1 = HGNNConv(in_channels, hidden_channels * 2)
        self.conv2 = HGNNConv(hidden_channels * 2, hidden_channels)
        self.conv3 = HGNNConv(hidden_channels, hidden_channels // 2)
        self.conv4 = HGNNConv(hidden_channels // 2, in_channels)
        
        # 정규화 레이어
        self.batch_norms = nn.ModuleList([
            nn.BatchNorm1d(hidden_channels * 2),
            nn.BatchNorm1d(hidden_channels),
            nn.BatchNorm1d(hidden_channels // 2)
        ])
        
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_channels * 2),
            nn.LayerNorm(hidden_channels),
            nn.LayerNorm(hidden_channels // 2)
        ])
        
        self.dropout = 0.2
        
    def forward(self, X, H):
        # 잔차 연결을 위한 원본 입력 저장
        identity = X
        
        # 컨볼루션 레이어 통과
        for i, (conv, batch_norm, layer_norm) in enumerate(zip(
            [self.conv1, self.conv2, self.conv3],
            self.batch_norms,
            self.layer_norms
        )):
            X = conv(X, H)
            X = batch_norm(X)
            X = layer_norm(X)
            X = F.elu(X)
            X = F.dropout(X, p=self.dropout, training=self.training)
        
        # 출력 레이어
        X = self.conv4(X, H)
        
        # 잔차 연결 (입출력 차원이 같을 때만)
        if X.shape == identity.shape:
            X = X + identity
        
        return X

class HypergraphDataset:
    """하이퍼그래프 데이터셋"""
    def __init__(self, keywords, relations, embeddings):
        self.keywords = keywords
        self.keyword_to_idx = {word: idx for idx, word in enumerate(keywords)}
        self.relations = relations
        self.embeddings = embeddings
        self.hyperedges = self._create_hyperedges()
    
    def _create_hyperedges(self):
        hyperedges = []
        for relation in self.relations:
            edge = {self.keyword_to_idx[k] for k in relation if k in self.keyword_to_idx}
            if edge:
                hyperedges.append(edge)
        return hyperedges
    
    def get_incidence_matrix(self):
        num_nodes = len(self.keywords)
        num_edges = len(self.hyperedges)
        
        H = torch.zeros((num_nodes, num_edges))
        for edge_idx, edge in enumerate(self.hyperedges):
            for node_idx in edge:
                H[node_idx, edge_idx] = 1
        return H
    
    def get_features(self):
        return torch.tensor(self.embeddings, dtype=torch.float)

class KeywordHGNN:
    def __init__(self, in_channels, hidden_channels):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = HGNN(in_channels, hidden_channels).to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=0.0001,
            weight_decay=1e-4,
            betas=(0.9, 0.999)
        )
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=False  # 경고 메시지 제거
        )
        self.early_stopping = EarlyStopping(patience=15, min_delta=1e-4)  # patience 증가
        self.losses = []
        self.best_loss = float('inf')

    def train(self, dataset, epochs=150):
        """모델 학습"""
        self.model.train()
        X = dataset.get_features().to(self.device)
        H = dataset.get_incidence_matrix().to(self.device)
        no_improvement = 0
        
        print("Epoch     Loss      Best Loss    LR")
        print("-" * 40)
        
        for epoch in range(epochs):
            self.optimizer.zero_grad()
            
            output = self.model(X, H)
            loss = self._compute_loss(output, X)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
            self.optimizer.step()
            
            current_lr = self.optimizer.param_groups[0]['lr']
            self.losses.append(loss.item())
            self.scheduler.step(loss)
            
            if loss.item() < self.best_loss:
                self.best_loss = loss.item()
                no_improvement = 0
            else:
                no_improvement += 1
            
            if (epoch + 1) % 10 == 0:
                print(f"[{epoch+1:03d}/{epochs:03d}] {loss.item():.4f}    {self.best_loss:.4f}    {current_lr:.6f}")
            
            # Early stopping 체크
            if no_improvement >= self.early_stopping.patience:
                print(f'\nEarly stopping triggered after {epoch+1} epochs')
                print(f'Best loss achieved: {self.best_loss:.4f}')
                break
    
    def _compute_loss(self, output, target):
        """손실 함수 계산"""
        mse_loss = F.mse_loss(output, target)
        l1_loss = F.l1_loss(output, target)
        cosine_loss = 1 - F.cosine_similarity(output, target).mean()
        return mse_loss + 0.1 * l1_loss + 0.1 * cosine_loss
    
    def evaluate(self, dataset):
        """모델 평가"""
        self.model.eval()
        with torch.no_grad():
            X = dataset.get_features().to(self.device)
            H = dataset.get_incidence_matrix().to(self.device)
            output = self.model(X, H)
            embeddings = output.cpu().numpy()
            
            metrics = {
                'mse_loss': F.mse_loss(output, X).item(),
                'link_prediction_auc': self._evaluate_link_prediction(dataset, embeddings),
                'relation_preservation': self._evaluate_relation_preservation(dataset, embeddings),
            }
            
            return metrics, embeddings
    
    def _evaluate_link_prediction(self, dataset, embeddings):
        """링크 예측 성능 평가"""
        pos_pairs = []
        for edge in dataset.hyperedges:
            edge = list(edge)
            for i in range(len(edge)):
                for j in range(i+1, len(edge)):
                    pos_pairs.append((edge[i], edge[j]))
        
        neg_pairs = []
        num_nodes = len(dataset.keywords)
        while len(neg_pairs) < len(pos_pairs):
            i, j = random.randint(0, num_nodes-1), random.randint(0, num_nodes-1)
            if i != j and (i, j) not in pos_pairs:
                neg_pairs.append((i, j))
        
        true_labels = [1] * len(pos_pairs) + [0] * len(neg_pairs)
        similarities = []
        
        for i, j in pos_pairs + neg_pairs:
            sim = np.dot(embeddings[i], embeddings[j]) / \
                (np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j]))
            similarities.append(sim)
        
        return roc_auc_score(true_labels, similarities)
    
    def _evaluate_relation_preservation(self, dataset, embeddings):
        """관계 보존성 평가"""
        scores = []
        for edge in dataset.hyperedges:
            edge = list(edge)
            if len(edge) < 2:
                continue
            
            edge_embeddings = embeddings[edge]
            similarities = []
            for i in range(len(edge)):
                for j in range(i+1, len(edge)):
                    sim = np.dot(edge_embeddings[i], edge_embeddings[j]) / \
                          (np.linalg.norm(edge_embeddings[i]) * np.linalg.norm(edge_embeddings[j]))
                    similarities.append(sim)
            scores.append(np.mean(similarities))
        
        return np.mean(scores)
    
    def find_similar_keywords(self, dataset, query_idx, embeddings, k=5):
        """유사 키워드 검색"""
        query_embedding = embeddings[query_idx]
        similarities = []
        
        for idx, embedding in enumerate(embeddings):
            if idx != query_idx:
                similarity = np.dot(query_embedding, embedding) / \
                           (np.linalg.norm(query_embedding) * np.linalg.norm(embedding))
                similarities.append((idx, similarity))
        
        return sorted(similarities, key=lambda x: x[1], reverse=True)[:k]

class EarlyStopping:
    """Early stopping 로직"""
    def __init__(self, patience=15, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
    
    def step(self, loss):
        if self.best_loss is None:
            self.best_loss = loss
            return False
            
        if loss < self.best_loss - self.min_delta:
            self.best_loss = loss
            self.counter = 0
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            self.early_stop = True
            return True
            
        return False
    
def create_sample_data():
    """경제/금융 도메인의 샘플 데이터 생성"""
    keywords = [
        # 금융 시장/상품
        "주식시장", "채권시장", "외환시장", "파생상품", "암호화폐",
        # 경제 지표
        "GDP", "물가상승률", "실업률", "금리", "환율",
        # 투자/분석
        "기술적분석", "기본적분석", "포트폴리오", "자산배분", "위험관리",
        # 금융기관
        "중앙은행", "시중은행", "증권사", "보험사", "자산운용사",
        # 경제이론/정책
        "통화정책", "재정정책", "케인스이론", "통화주의", "행동경제학"
    ]
    
    relations = [
        # 금융시장 관련 그룹
        ["주식시장", "기술적분석", "기본적분석", "증권사"],
        ["채권시장", "금리", "중앙은행", "통화정책"],
        ["외환시장", "환율", "중앙은행", "통화정책"],
        ["파생상품", "위험관리", "포트폴리오", "자산배분"],
        ["암호화폐", "기술적분석", "위험관리"],
        
        # 경제정책 관련 그룹
        ["중앙은행", "통화정책", "금리", "물가상승률"],
        ["통화정책", "물가상승률", "통화주의", "케인스이론"],
        ["재정정책", "GDP", "실업률", "케인스이론"],
        
        # 투자분석 관련 그룹
        ["기술적분석", "주식시장", "암호화폐", "외환시장"],
        ["기본적분석", "GDP", "물가상승률", "환율"],
        ["포트폴리오", "자산배분", "위험관리", "자산운용사"],
        
        # 금융기관 관련 그룹
        ["중앙은행", "시중은행", "통화정책", "금리"],
        ["증권사", "자산운용사", "포트폴리오", "주식시장"],
        ["보험사", "위험관리", "자산운용사", "채권시장"],
        
        # 경제이론 관련 그룹
        ["케인스이론", "재정정책", "실업률", "GDP"],
        ["통화주의", "통화정책", "물가상승률", "중앙은행"],
        ["행동경제학", "기술적분석", "포트폴리오", "위험관리"],
        
        # 거시경제 지표 관련 그룹
        ["GDP", "실업률", "물가상승률", "환율"],
        ["금리", "채권시장", "시중은행", "통화정책"],
        ["환율", "외환시장", "중앙은행", "물가상승률"]
    ]
    
    # 임베딩 생성 및 정규화
    np.random.seed(42)
    embeddings = np.random.randn(len(keywords), 64)
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    
    return keywords, relations, embeddings

def visualize_training(losses):
    """학습 과정 시각화"""
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(losses) + 1), losses, 'b-', linewidth=1)
    plt.title('Training Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()

def main():
    """메인 실행 함수"""
    print("\n=== 경제/금융 도메인 HGNN 키워드 분석 시작 ===\n")
    
    # 1. 데이터 준비
    print("1. 데이터 준비 중...")
    keywords, relations, embeddings = create_sample_data()
    dataset = HypergraphDataset(keywords, relations, embeddings)
    
    # 2. 모델 초기화
    print("\n2. 모델 초기화...")
    embedding_dim = embeddings.shape[1]
    hgnn = KeywordHGNN(in_channels=embedding_dim, hidden_channels=128)
    
    # 3. 모델 학습
    print("\n3. 모델 학습 중...")
    hgnn.train(dataset, epochs=150)
    
    # 4. 모델 평가
    print("\n4. 모델 평가 중...")
    metrics, learned_embeddings = hgnn.evaluate(dataset)
    
    print("\n=== 모델 평가 결과 ===")
    print(f"MSE Loss: {metrics['mse_loss']:.4f}")
    print(f"링크 예측 AUC: {metrics['link_prediction_auc']:.4f}")
    print(f"관계 보존성 점수: {metrics['relation_preservation']:.4f}")
    
    # 5. 학습 과정 시각화
    print("\n5. 학습 과정 시각화...")
    visualize_training(hgnn.losses)
    
    # 6. 키워드 유사도 분석
    print("\n6. 키워드 유사도 분석...")
    query_keywords = ["통화정책", "주식시장", "포트폴리오", "금리", "중앙은행"]
    
    print("\n=== 키워드 유사도 분석 결과 ===")
    for query_keyword in query_keywords:
        print(f"\n'{query_keyword}'와 유사한 키워드들:")
        query_idx = keywords.index(query_keyword)
        similar_keywords = hgnn.find_similar_keywords(dataset, query_idx, learned_embeddings)
        
        for idx, similarity in similar_keywords:
            print(f"  - {keywords[idx]:<12} : {similarity:.4f}")
    
    print("\n=== 분석 완료 ===")
    
    return dataset, hgnn, learned_embeddings

if __name__ == "__main__":
    dataset, hgnn, embeddings = main()