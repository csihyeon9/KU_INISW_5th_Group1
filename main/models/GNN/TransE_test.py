import json
import torch
import torch.nn.functional as F
from torch_geometric.utils import negative_sampling
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
from typing import Dict, List, Tuple, Optional

from pyvis.network import Network
import networkx as nx
import plotly.graph_objects as go
from IPython.display import display, HTML
import webbrowser

class TransE(torch.nn.Module):
    def __init__(self, num_entities: int, num_relations: int, embedding_dim: int, margin: float = 1):
        super().__init__()
        self.entity_embeddings = torch.nn.Embedding(num_entities, embedding_dim)
        self.relation_embeddings = torch.nn.Embedding(num_relations, embedding_dim)
        self.margin = margin
        
        # 임베딩 실행
        torch.nn.init.xavier_uniform_(self.entity_embeddings.weight.data)
        torch.nn.init.xavier_uniform_(self.relation_embeddings.weight.data)
        
        # 임베딩 정규화
        self.relation_embeddings.weight.data = F.normalize(
            self.relation_embeddings.weight.data, p=2, dim=1
        )

    def forward(self, pos_edge_index: torch.Tensor, neg_edge_index: torch.Tensor, 
                edge_type: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        pos_head_embeds = self.entity_embeddings(pos_edge_index[0])
        pos_tail_embeds = self.entity_embeddings(pos_edge_index[1])
        rel_embeds = self.relation_embeddings(edge_type)
        
        neg_head_embeds = self.entity_embeddings(neg_edge_index[0])
        neg_tail_embeds = self.entity_embeddings(neg_edge_index[1])
        
        pos_score = torch.norm(pos_head_embeds + rel_embeds - pos_tail_embeds, p=2, dim=1)
        neg_score = torch.norm(neg_head_embeds + rel_embeds - neg_tail_embeds, p=2, dim=1)
        
        return pos_score, neg_score

    def get_entity_embeddings(self) -> torch.Tensor:
        return self.entity_embeddings.weight.data

    def get_relation_embeddings(self) -> torch.Tensor:
        return self.relation_embeddings.weight.data

class KnowledgeGraphProcessor:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.entity_to_idx: Dict[str, int] = {}
        self.idx_to_entity: Dict[int, str] = {}
        self.relation_to_idx: Dict[str, int] = {}
        self.idx_to_relation: Dict[int, str] = {}
        self.relation_descriptions: Dict[int, str] = {}
        self.tfidf_vectorizer = TfidfVectorizer()
        self.model: Optional[TransE] = None
        
    def load_and_process_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """데이터 로드 및 전처리"""
        with open(self.file_path, 'r', encoding='utf8') as f:
            dataset = json.load(f)
           
        for node in dataset['nodes']:
            if 'id' in node and 'word' in node:
                self.entity_to_idx[node['id']] = len(self.entity_to_idx)
                self.idx_to_entity[self.entity_to_idx[node['id']]] = node['word']
        
        if not self.entity_to_idx:
            raise ValueError("유효한 노드가 없습니다")
        
        # Edge 데이터 처리
        edge_index = []
        edge_types = []
        descriptions = []
        
        if 'edges' not in dataset:
            raise ValueError("데이터셋에 'edges' 필드가 없습니다")
            
        for edge in dataset['edges']:
            if all(k in edge for k in ['source', 'target', 'relation_description']):
                if edge['source'] in self.entity_to_idx and edge['target'] in self.entity_to_idx:
                    source_idx = self.entity_to_idx[edge['source']]
                    target_idx = self.entity_to_idx[edge['target']]
                    
                    if edge['relation_description'] not in self.relation_to_idx:
                        rel_idx = len(self.relation_to_idx)
                        self.relation_to_idx[edge['relation_description']] = rel_idx
                        self.idx_to_relation[rel_idx] = edge['relation_description']
                        self.relation_descriptions[rel_idx] = edge['relation_description']
                    
                    rel_idx = self.relation_to_idx[edge['relation_description']]
                    
                    edge_index.append([source_idx, target_idx])
                    edge_types.append(rel_idx)
                    descriptions.append(edge['relation_description'])
        
        if not edge_index:
            raise ValueError("유효한 엣지가 없습니다")
        
        # TF-IDF 벡터화
        self.tfidf_vectorizer.fit(descriptions)
        
        edge_index = torch.tensor(edge_index).t().contiguous()
        edge_types = torch.tensor(edge_types)
        
        print(f"로드됨: {len(self.entity_to_idx)} 엔티티, {len(self.relation_to_idx)} 관계 유형")
        print(f"전체 엣지 수: {len(edge_types)}")
        
        print("Edge index shape:", edge_index.shape)
        print("Edge types shape:", edge_types.shape)

        return edge_index, edge_types

    def predict_relations(self, new_keyword: str, top_k: int = 5) -> List[Dict]:
        """새로운 키워드와 기존 노드들 간의 관계 예측"""
            
        if new_keyword not in self.entity_to_idx:
            print(f"새로운 키워드 인지: {new_keyword}(이/가) 지식 그래프에 없습니다. 임베딩을 적용합니다.")
            temp_idx = len(self.entity_to_idx)
            with torch.no_grad():
                temp_embedding = torch.mean(self.model.get_entity_embeddings(), dim=0)
        else:
            temp_idx = self.entity_to_idx[new_keyword]
            temp_embedding = self.model.get_entity_embeddings()[temp_idx]
        
        predictions = []
        
        with torch.no_grad():
            for entity_idx in range(len(self.idx_to_entity)):
                if entity_idx == temp_idx:
                    continue
                    
                entity_embed = self.model.get_entity_embeddings()[entity_idx]
                
                for rel_idx in range(len(self.idx_to_relation)):
                    rel_embed = self.model.get_relation_embeddings()[rel_idx]
                    
                    forward_score = torch.norm(
                        temp_embedding + rel_embed - entity_embed, p=2
                    ).item()
                    
                    backward_score = torch.norm(
                        entity_embed + rel_embed - temp_embedding, p=2
                    ).item()
                    
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

    def train_model(self, embedding_dim: int = 100, n_epochs: int = 200, 
               batch_size: int = 1024, learning_rate: float = 0.01) -> TransE:
        """모델 학습"""

        # 데이터 로드 및 전처리
        edge_index, edge_types = self.load_and_process_data()

        # edge_index 및 edge_types 유효성 검사
        if edge_index.size(1) == 0 or edge_types.size(0) == 0:
            raise ValueError("유효한 edge_index 또는 edge_types가 없습니다. 데이터셋을 확인하세요.")

        # 모델 초기화
        self.model = TransE(
            num_entities=len(self.entity_to_idx),
            num_relations=len(self.relation_to_idx),
            embedding_dim=embedding_dim
        )

        # Optimizer 설정
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        
        print("학습 시작...")
        for epoch in range(1, n_epochs + 1):
            self.model.train()
            optimizer.zero_grad()
            
            # 배치 단위로 처리
            num_edges = edge_index.size(1)
            indices = torch.randperm(num_edges)
            
            total_loss = 0
            for start_idx in range(0, num_edges, batch_size):
                # 배치 인덱스 범위 확인
                batch_indices = indices[start_idx:start_idx + batch_size]
                if batch_indices.max() >= num_edges:
                    print("Warning: batch_indices가 edge_index의 크기를 초과합니다.")
                    continue
                
                # 배치 데이터 준비
                batch_edge_index = edge_index[:, batch_indices]
                batch_edge_types = edge_types[batch_indices]
                
                # Negative sampling 호출 전 크기 확인
                try:
                    neg_edge_index = negative_sampling(
                        edge_index=batch_edge_index,
                        num_nodes=len(self.entity_to_idx)
                    )
                except Exception as e:
                    print(f"Negative sampling 오류 발생: {str(e)}")
                    continue  # 오류 발생 시 해당 배치는 건너뜁니다.
                
                # Forward pass
                pos_score, neg_score = self.model(
                    batch_edge_index, neg_edge_index, batch_edge_types
                )
                
                # Loss 계산
                loss = F.margin_ranking_loss(
                    neg_score,
                    pos_score,
                    torch.ones_like(pos_score),
                    margin=self.model.margin
                )
                
                # Backpropagation
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            # Embedding 정규화
            with torch.no_grad():
                self.model.entity_embeddings.weight.data = F.normalize(
                    self.model.entity_embeddings.weight.data, p=2, dim=1
                )
                self.model.relation_embeddings.weight.data = F.normalize(
                    self.model.relation_embeddings.weight.data, p=2, dim=1
                )
            
            # 10 epochs마다 평균 손실 출력
            if epoch % 10 == 0:
                avg_loss = total_loss / max(1, (num_edges // batch_size))  # 0으로 나누는 경우 방지
                print(f'Epoch {epoch:03d}, Average Loss: {avg_loss:.4f}')
        
        print("학습 완료")
        return self.model

    def save_model(self, save_path: str):
        """모델 저장"""
        if not self.model:
            raise ValueError("저장할 모델이 없습니다")
            
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'entity_to_idx': self.entity_to_idx,
            'idx_to_entity': self.idx_to_entity,
            'relation_to_idx': self.relation_to_idx,
            'idx_to_relation': self.idx_to_relation,
            'relation_descriptions': self.relation_descriptions,
        }, save_path)
        
    def load_model(self, load_path: str, embedding_dim: int = 100):
        """모델 로드"""
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {load_path}")
            
        checkpoint = torch.load(load_path)
        
        self.entity_to_idx = checkpoint['entity_to_idx']
        self.idx_to_entity = checkpoint['idx_to_entity']
        self.relation_to_idx = checkpoint['relation_to_idx']
        self.idx_to_relation = checkpoint['idx_to_relation']
        self.relation_descriptions = checkpoint['relation_descriptions']
        
        self.model = TransE(
            num_entities=len(self.entity_to_idx),
            num_relations=len(self.relation_to_idx),
            embedding_dim=embedding_dim
        )
        self.model.load_state_dict(checkpoint['model_state_dict'])


    def visualize_knowledge_graph(self, output_file="knowledge_graph.html"):
        """
        Visualizes the knowledge graph using PyVis with custom options based on node embeddings.
        """
        if self.model is None:
            raise ValueError("No trained model available. Train the model before visualization.")

        # Extract embeddings from the trained model
        entity_embeddings = self.model.get_entity_embeddings().detach().numpy()

        # Initialize PyVis network
        net = Network(notebook=False, directed=True, cdn_resources='remote')

        # Add nodes with embeddings as node attributes
        for entity_idx, entity_name in self.idx_to_entity.items():
            net.add_node(entity_idx, label=entity_name, physics=True, mass=len(entity_embeddings[entity_idx]), color="#87CEEB")

        # Add edges with relationship descriptions
        edge_index, edge_types = self.load_and_process_data()
        for i, (source, target) in enumerate(edge_index.t().tolist()):
            if source < 1051 and target < 1051:  # 1051번째 노드 이전의 노드들만 연결
                if source in self.idx_to_entity and target in self.idx_to_entity:
                    rel_idx = edge_types[i].item()
                    relation_name = self.idx_to_relation[rel_idx]
                    net.add_edge(source, target, title=relation_name)

        # Set PyVis options
        net.set_options("""
        var options = {
        "nodes": {
            "font": {
            "size": 16,
            "face": "arial"
            }
        },
        "edges": {
            "arrows": {
            "to": {
                "enabled": true,
                "scaleFactor": 0.5
            }
            },
            "smooth": false
        },
        "physics": {
            "enabled": true,
            "barnesHut": {
            "gravitationalConstant": -2000,
            "springLength": 100
            }
        }
        }
        """)

        # Save the visualization to an HTML file
        net.save_graph(output_file)
        print(f"Knowledge graph saved to {output_file}. Open this file in a browser to view the visualization.")

def main():
    # 설정
    DATA_PATH = 'data/gnn_enhanced_700_dict.json'
    MODEL_PATH = "models/KG_model3.pt"
    EMBEDDING_DIM = 100
    N_EPOCHS = 200
    BATCH_SIZE = 1024
    LEARNING_RATE = 0.01

    # KnowledgeGraphProcessor 초기화
    processor = KnowledgeGraphProcessor(DATA_PATH)
    
    # 모델 학습
    print("새로운 모델을 처음부터 학습합니다...")
    processor.train_model(embedding_dim=EMBEDDING_DIM, n_epochs=N_EPOCHS,
                          batch_size=BATCH_SIZE, learning_rate=LEARNING_RATE)
    
    # 모델을 지정된 경로에 저장
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    processor.save_model(MODEL_PATH)
    processor.visualize_knowledge_graph()
    print(f"학습된 모델이 {MODEL_PATH}에 저장되었습니다.")
    

    # 새로운 데이터셋 or 기존 모델의 재학습을 위한 코드
    retrain = input("모델을 재학습하시겠습니까? (y/n): ").strip().lower()
    if retrain == 'y':
        print("모델을 재학습합니다...")
        processor.load_model(MODEL_PATH, embedding_dim=EMBEDDING_DIM)
        processor.train_model(embedding_dim=EMBEDDING_DIM, n_epochs=N_EPOCHS,
                              batch_size=BATCH_SIZE, learning_rate=LEARNING_RATE)
        processor.save_model(MODEL_PATH)
        print(f"재학습된 모델이 {MODEL_PATH}에 저장되었습니다.")


if __name__ == "__main__":
    main()