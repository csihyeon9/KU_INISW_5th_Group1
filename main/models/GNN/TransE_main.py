import json
import torch
import torch.nn.functional as F
from torch_geometric.utils import negative_sampling
import os
import random
from sklearn.metrics import roc_auc_score, accuracy_score
from typing import List, Tuple, Dict, Optional
from pyvis.network import Network
import matplotlib.pyplot as plt
import webbrowser

class TransE(torch.nn.Module):
    def __init__(self, num_entities: int, num_relations: int, embedding_dim: int, margin: float = 1):
        super().__init__()
        self.entity_embeddings = torch.nn.Embedding(num_entities, embedding_dim)
        self.relation_embeddings = torch.nn.Embedding(num_relations, embedding_dim)
        self.margin = margin
        
        # Embeddings 초기화 및 정규화
        torch.nn.init.xavier_uniform_(self.entity_embeddings.weight.data)
        torch.nn.init.xavier_uniform_(self.relation_embeddings.weight.data)
        self.relation_embeddings.weight.data = F.normalize(self.relation_embeddings.weight.data, p=2, dim=1)

    def forward(self, pos_edge_index: torch.Tensor, edge_type: torch.Tensor, neg_edge_index: torch.Tensor):
        pos_head_embeds = self.entity_embeddings(pos_edge_index[0])
        pos_tail_embeds = self.entity_embeddings(pos_edge_index[1])
        pos_rel_embeds = self.relation_embeddings(edge_type)
        
        neg_head_embeds = self.entity_embeddings(neg_edge_index[0])
        neg_tail_embeds = self.entity_embeddings(neg_edge_index[1])
        
        pos_score = torch.norm(pos_head_embeds + pos_rel_embeds - pos_tail_embeds, p=2, dim=1)
        neg_score = torch.norm(neg_head_embeds + pos_rel_embeds - neg_tail_embeds, p=2, dim=1)
        
        return pos_score, neg_score

    def get_entity_embeddings(self):
        return self.entity_embeddings.weight.data

    def get_relation_embeddings(self):
        return self.relation_embeddings.weight.data

class KnowledgeGraphProcessor:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.entity_to_idx: Dict[str, int] = {}
        self.idx_to_entity: Dict[int, str] = {}
        self.relation_to_idx: Dict[str, int] = {}
        self.idx_to_relation: Dict[int, str] = {}
        self.model: Optional[TransE] = None
        self.training_log = []
        
    def load_and_process_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        with open(self.file_path, 'r', encoding='utf8') as f:
            dataset = json.load(f)
           
        edges = []
        for node in dataset['nodes']:
            if 'id' in node and 'word' in node:
                self.entity_to_idx[node['id']] = len(self.entity_to_idx)
                self.idx_to_entity[self.entity_to_idx[node['id']]] = node['word']
        
        for edge in dataset['edges']:
            source, target = edge['source'], edge['target']
            if source in self.entity_to_idx and target in self.entity_to_idx:
                source_idx = self.entity_to_idx[source]
                target_idx = self.entity_to_idx[target]
                
                if edge['relation_description'] not in self.relation_to_idx:
                    rel_idx = len(self.relation_to_idx)
                    self.relation_to_idx[edge['relation_description']] = rel_idx
                    self.idx_to_relation[rel_idx] = edge['relation_description']
                
                rel_idx = self.relation_to_idx[edge['relation_description']]
                edges.append((source_idx, target_idx, rel_idx))
        
        edge_index = torch.tensor([(e[0], e[1]) for e in edges], dtype=torch.long).t()
        edge_types = torch.tensor([e[2] for e in edges], dtype=torch.long)
        
        return edge_index, edge_types

    def split_data(self, edge_index: torch.Tensor, edge_types: torch.Tensor, test_ratio=0.2):
        num_edges = edge_index.size(1)
        test_size = int(num_edges * test_ratio)
        
        # Shuffle edges
        perm = torch.randperm(num_edges)
        edge_index = edge_index[:, perm]
        edge_types = edge_types[perm]
        
        # Split into train and test
        train_edge_index = edge_index[:, :-test_size]
        train_edge_types = edge_types[:-test_size]
        test_edge_index = edge_index[:, -test_size:]
        test_edge_types = edge_types[-test_size:]
        
        return train_edge_index, train_edge_types, test_edge_index, test_edge_types

    def train_model(self, edge_index: torch.Tensor, edge_types: torch.Tensor, embedding_dim: int, n_epochs: int, batch_size: int, learning_rate: float):
        num_entities = len(self.entity_to_idx)
        num_relations = len(self.relation_to_idx)
        self.model = TransE(num_entities, num_relations, embedding_dim)
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        
        print("학습 시작...")
        for epoch in range(1, n_epochs + 1):
            self.model.train()
            optimizer.zero_grad()
            
            num_edges = edge_index.size(1)
            indices = torch.randperm(num_edges)
            
            total_loss = 0
            for start_idx in range(0, num_edges, batch_size):
                batch_indices = indices[start_idx:start_idx + batch_size]
                batch_edge_index = edge_index[:, batch_indices]
                batch_edge_types = edge_types[batch_indices]
                
                # Negative sampling (returns only the edge index)
                neg_edge_index = negative_sampling(edge_index=batch_edge_index, num_nodes=num_entities)
                
                # Forward pass with positive and negative edges
                pos_score, neg_score = self.model(batch_edge_index, batch_edge_types, neg_edge_index)
                
                # Loss 계산
                loss = F.margin_ranking_loss(pos_score, neg_score, torch.ones_like(pos_score), margin=self.model.margin)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            avg_loss = total_loss / (num_edges // batch_size)
            self.training_log.append(avg_loss)  # epoch 별 loss 로그 저장
            if epoch % 10 == 0:
                print(f'Epoch {epoch:03d}, Average Loss: {avg_loss:.4f}')

        print("학습 완료")
    
    def save_model(self, path: str):
        if not self.model:
            raise ValueError("저장할 모델이 없습니다.")
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'entity_to_idx': self.entity_to_idx,
            'idx_to_entity': self.idx_to_entity,
            'relation_to_idx': self.relation_to_idx,
            'idx_to_relation': self.idx_to_relation
        }, path)
        print(f"모델이 {path}에 저장되었습니다.")

    def load_model(self, path: str, embedding_dim: int = 100):
        if not os.path.exists(path):
            raise FileNotFoundError(f"모델 파일이 존재하지 않습니다: {path}")
            
        checkpoint = torch.load(path)
        self.entity_to_idx = checkpoint['entity_to_idx']
        self.idx_to_entity = checkpoint['idx_to_entity']
        self.relation_to_idx = checkpoint['relation_to_idx']
        self.idx_to_relation = checkpoint['idx_to_relation']
        
        self.model = TransE(len(self.entity_to_idx), len(self.relation_to_idx), embedding_dim)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"모델이 {path}에서 로드되었습니다.")

    def save_training_log(self, path="training_log.json"):
        with open(path, "w") as f:
            json.dump(self.training_log, f)
        print(f"훈련 로그가 {path}에 저장되었습니다.")

    def plot_training_log(self, path="training_loss_plot.png"):
        plt.figure()
        plt.plot(range(1, len(self.training_log) + 1), self.training_log, marker='o', color='b')
        plt.xlabel("Epoch")
        plt.ylabel("Average Loss")
        plt.title("Training Loss over Epochs")
        plt.savefig(path)
        print(f"학습 손실 그래프가 {path}에 저장되었습니다.")
        plt.show()

    def evaluate_link_prediction(self, edge_index: torch.Tensor, edge_types: torch.Tensor, test_edge_index: torch.Tensor, test_edge_types: torch.Tensor):
        print("링크 예측 평가 중...")
        self.model.eval()
        
        with torch.no_grad():
            # Compute scores for test edges
            pos_head_embeds = self.model.entity_embeddings(test_edge_index[0])
            pos_tail_embeds = self.model.entity_embeddings(test_edge_index[1])
            pos_rel_embeds = self.model.relation_embeddings(test_edge_types)
            
            pos_scores = torch.norm(pos_head_embeds + pos_rel_embeds - pos_tail_embeds, p=2, dim=1)
            
            # negative samples 생성
            neg_edge_index = negative_sampling(edge_index=edge_index, num_nodes=len(self.entity_to_idx), num_neg_samples=test_edge_index.size(1))
            neg_head_embeds = self.model.entity_embeddings(neg_edge_index[0])
            neg_tail_embeds = self.model.entity_embeddings(neg_edge_index[1])
            neg_scores = torch.norm(neg_head_embeds + pos_rel_embeds - neg_tail_embeds, p=2, dim=1)
            
            # Compute metrics 
            labels = torch.cat([torch.ones(pos_scores.size(0)), torch.zeros(neg_scores.size(0))])
            scores = torch.cat([pos_scores, neg_scores])
            roc_auc = roc_auc_score(labels.cpu().numpy(), -scores.cpu().numpy())
            
            hits1 = (pos_scores < neg_scores).float().mean()
            print(f'ROC-AUC: {roc_auc:.4f}, Hits@1: {hits1:.4f}')
    
    def predict_relations(self, new_keyword: str, top_k: int = 5) -> List[Dict]:
        """새로운 키워드와 기존 노드들 간의 관계 예측"""
        predictions = []
        is_new_keyword_in_graph = new_keyword in self.entity_to_idx  # 그래프에 키워드가 있는지 확인

        if is_new_keyword_in_graph:
            print(f"{new_keyword}는 지식 그래프에 이미 존재합니다.")
            temp_idx = self.entity_to_idx[new_keyword]
            temp_embedding = self.model.get_entity_embeddings()[temp_idx]
        else:
            print(f"새로운 키워드 인지: {new_keyword}(이/가) 지식 그래프에 없습니다. 임시 임베딩을 적용합니다.")
            # 새로운 키워드의 임시 임베딩 생성
            temp_embedding = torch.mean(self.model.get_entity_embeddings(), dim=0)

        # 예측 수행
        with torch.no_grad():
            for entity_idx in range(len(self.idx_to_entity)):
                if is_new_keyword_in_graph and entity_idx == temp_idx:
                    continue

                entity_embed = self.model.get_entity_embeddings()[entity_idx]
                for rel_idx in range(len(self.idx_to_relation)):
                    rel_embed = self.model.get_relation_embeddings()[rel_idx]

                    forward_score = torch.norm(temp_embedding + rel_embed - entity_embed, p=2).item()
                    backward_score = torch.norm(entity_embed + rel_embed - temp_embedding, p=2).item()
                    score = min(forward_score, backward_score)
                    direction = "forward" if forward_score < backward_score else "backward"

                    predictions.append({
                        'entity': self.idx_to_entity[entity_idx],
                        'relation': self.idx_to_relation[rel_idx],
                        'score': score,
                        'direction': direction
                    })

        predictions.sort(key=lambda x: x['score'])
        return predictions[:top_k]


    def visualize_knowledge_graph(self, edge_index, edge_types, output_file="knowledge_graph.html"):
        net = Network(height="600px", width="100%", bgcolor="#222222", font_color="white")

        # Add nodes
        for idx, entity in self.idx_to_entity.items():
            net.add_node(idx, label=entity)

        # Add edges
        for i in range(edge_index.size(1)):
            source = edge_index[0, i].item()
            target = edge_index[1, i].item()
            relation = self.idx_to_relation[edge_types[i].item()]
            net.add_edge(source, target, label=relation)

         # 그래프를 HTML로 저장
        net.save_graph(output_file)
        abs_path = os.path.abspath(output_file)
        print(f"지식 그래프가 {abs_path}에 저장되었습니다.")

        # Chrome 브라우저에서 파일 열기
        chrome_path = 'C:/Program Files/Google/Chrome/Application/chrome.exe %s'
        webbrowser.get(chrome_path).open("file://" + abs_path)

def main():
    # 경로 설정
    DATA_PATH = 'data/gnn_enhanced_700_dict.json'
    MODEL_PATH = "models/GNN_Trained_Models/TransE_model4.pt"
    EMBEDDING_DIM = 100
    N_EPOCHS = 100
    BATCH_SIZE = 1024
    LEARNING_RATE = 0.01

    # KnowledgeGraphProcessor 초기화
    kgp = KnowledgeGraphProcessor(DATA_PATH)

    # 데이터 로드 및 분할
    edge_index, edge_types = kgp.load_and_process_data()
    train_edge_index, train_edge_types, test_edge_index, test_edge_types = kgp.split_data(edge_index, edge_types)

    # 모델이 이미 존재하는지 확인
    if os.path.exists(MODEL_PATH):
        print("저장된 모델을 로드합니다.")
        kgp.load_model(MODEL_PATH, embedding_dim=EMBEDDING_DIM)

        # 재학습 여부 확인
        retrain = input("모델을 재학습하시겠습니까? (y/n): ").strip().lower()
        if retrain == 'y':
            print("모델을 재학습합니다.")
            # 모델 학습
            kgp.train_model(train_edge_index, train_edge_types, embedding_dim=EMBEDDING_DIM, 
                            n_epochs=N_EPOCHS, batch_size=BATCH_SIZE, learning_rate=LEARNING_RATE)
            kgp.save_model(MODEL_PATH)
            print("재학습된 모델이 저장되었습니다.")
            
            # 재학습 후 로그 및 손실 그래프 저장
            kgp.save_training_log("training_log.json")
            kgp.plot_training_log("training_loss_plot.png")
    else:
        print("새로운 모델을 처음부터 학습합니다.")
        kgp.train_model(train_edge_index, train_edge_types, embedding_dim=EMBEDDING_DIM, 
                        n_epochs=N_EPOCHS, batch_size=BATCH_SIZE, learning_rate=LEARNING_RATE)
        kgp.save_model(MODEL_PATH)
        print("모델이 저장되었습니다.")
        
        # 첫 학습 후 로그 및 손실 그래프 저장
        kgp.save_training_log("training_log.json")
        kgp.plot_training_log("training_loss_plot.png")

    # 링크 예측 평가 (모델이 학습된 후 실행 가능)
    print("\n링크 예측 평가를 진행합니다.")
    kgp.evaluate_link_prediction(train_edge_index, train_edge_types, test_edge_index, test_edge_types)

    # 지식 그래프 시각화 (학습된 모델 기반으로 시각화)
    print("\n지식 그래프를 시각화합니다.")
    kgp.visualize_knowledge_graph(edge_index, edge_types)

    # 새로운 키워드에 대한 관계 예측
    new_keyword = input("새로운 키워드 입력: ").strip()
    predictions = kgp.predict_relations(new_keyword, top_k=5)
    print(f"\n{new_keyword}와 관련된 상위 5개의 예측 관계:")
    for pred in predictions:
        print(f"{new_keyword} - {pred['relation']} -> {pred['entity']} (Score: {pred['score']:.4f})")

if __name__ == '__main__':
    main()
