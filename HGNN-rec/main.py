# main.py
import torch
import logging
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import logging

from config.config import Config
from data.data_helper import NewsDataProcessor, EconomicNewsDataset, ExplanationGenerator
from utils.hypergraph_utils import HypergraphBuilder
from models.hgnn import HGNNRecommender

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.WARNING)
logging.getLogger().setLevel(logging.ERROR)

class EconomicNewsRecommender:
    """경제 금융 뉴스 추천 시스템
    
    HGNN을 활용하여 다양한 관점의 경제 금융 뉴스를 추천하고
    관계 기반의 설명을 제공하는 시스템입니다.
    """
    
    def __init__(self, config_path: str = 'config/config.yaml'):
        self.config = Config(config_path).config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.data_processor = NewsDataProcessor(self.config)
        self.hypergraph_builder = HypergraphBuilder(self.config['hypergraph'])
        self.model = None
        self.document_embeddings = None
        self.documents = None

    def prepare_data(self, data_path: str) -> Tuple[EconomicNewsDataset, Dict, Dict]:
        documents = self.data_processor.load_data(data_path)
        
        # 어휘 구축
        keyword_to_idx, relation_to_idx = self.data_processor.build_vocabularies(documents)
        
        from data.data_helper import ExplanationGenerator
        self.explanation_generator = ExplanationGenerator(
            idx_to_keyword={v: k for k, v in keyword_to_idx.items()},
            idx_to_relation={v: k for k, v in relation_to_idx.items()}
        )

        # 특징 벡터 생성
        features = self.data_processor.create_document_features(
            documents,
            self.config['model']['in_dim']
        )
        
        # 레이블 생성
        labels = np.array([
            relation_to_idx[doc['relation_type']]
            for doc in documents
        ])
        
        # 하이퍼그래프 구성
        H, W = self.hypergraph_builder.construct_H_from_documents(
            documents,
            keyword_to_idx,
            relation_to_idx
        )
        
        # 라플라시안 행렬 생성
        G = self.hypergraph_builder.generate_G_from_H(H, W)
        
        # 데이터셋 생성
        dataset = EconomicNewsDataset(features, labels, G)
        
        return dataset, keyword_to_idx, relation_to_idx

    def train(self, dataset: EconomicNewsDataset, documents: List[Dict], validation_split: float = 0.2):
        """
        Args:
            dataset: 학습 데이터셋
            documents: 원본 문서 리스트
        """
        logger.info("Starting model training...")
        
        # 모델 초기화
        self.model = HGNNRecommender(
            in_dim=self.config['model']['in_dim'],
            hidden_dim=self.config['model']['hidden_dim'],
            embedding_dim=self.config['model']['embedding_dim'],
            dropout=self.config['model']['dropout']
        ).to(self.device)
        
        # 데이터 분할
        dataset_size = len(dataset)
        indices = list(range(dataset_size))
        split = int(np.floor(validation_split * dataset_size))
        train_indices, val_indices = indices[split:], indices[:split]
        
        def custom_collate(batch):
            features = torch.stack([item[0] for item in batch])
            # G 행렬은 배치별로 필요한 부분만 추출
            G_batch = torch.stack([item[1] for item in batch])
            labels = torch.stack([item[2] for item in batch])
            return features, G_batch, labels

        train_loader = DataLoader(
            dataset,
            batch_size=32,
            sampler=train_indices,
            drop_last=True,
            num_workers=0
        )

        val_loader = DataLoader(
            dataset,
            batch_size=32,
            sampler=val_indices,
            drop_last=True,
            num_workers=0
        )
        
        # 옵티마이저 및 스케줄러 설정
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config['training']['learning_rate'],
            weight_decay=self.config['training']['weight_decay']
        )
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=self.config['training']['scheduler']['milestones'],
            gamma=self.config['training']['scheduler']['gamma']
        )
        
        # 학습 루프
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.config['training']['max_epochs']):
            # 학습
            train_loss = self._train_epoch(train_loader, optimizer)
            
            # 검증
            val_loss = self._validate_epoch(val_loader)
            
            # 스케줄러 업데이트
            scheduler.step()
            
            # 얼리 스토핑
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                self._save_checkpoint(epoch, val_loss)
            else:
                patience_counter += 1
                
            if patience_counter >= self.config['training']['early_stopping']['patience']:
                logger.info(f"Early stopping at epoch {epoch}")
                break
            
            logger.info(
                f"Epoch {epoch}: Train Loss = {train_loss:.4f}, "
                f"Val Loss = {val_loss:.4f}"
            )
        
        # 학습 완료 후 전체 문서 임베딩 생성
        self.documents = documents  # 원본 문서 저장
        self.model.eval()
        with torch.no_grad():
            features = dataset.features.to(self.device)
            G = dataset.G.to(self.device)
            self.document_embeddings = self.model(features, G).cpu()

    def recommend(self, query_document: Dict, top_k: int = 5) -> List[Dict]:
        """새로운 문서에 대한 추천"""
        if self.document_embeddings is None:
            raise ValueError("Model must be trained first!")

        # 쿼리 문서 임베딩 생성
        query_features = self.data_processor.create_document_features(
            [query_document],
            self.config['model']['embedding_dim']
        )
        query_tensor = torch.FloatTensor(query_features).to(self.device)
        
        # 모델을 통한 임베딩 변환
        self.model.eval()
        with torch.no_grad():
            query_embedding = self.model(
                query_tensor,
                torch.eye(1).to(self.device)  # 단일 문서용 G 행렬
            )
        
        # 유사도 계산 및 추천
        similarities = F.cosine_similarity(
            query_embedding,
            self.document_embeddings
        )
        
        # Top-k 추천
        top_indices = similarities.topk(top_k).indices.cpu().numpy()
        
        recommendations = []
        for idx in top_indices:
            doc = self.documents[idx]
            
            if self.explanation_generator:  # explanation_generator가 있을 때만 설명 생성
                explanation = self.explanation_generator.generate_explanation(
                    query_document['keywords'],
                    doc['keywords'],
                    query_document['relation_type'],
                    doc['relation_type']
                )
            else:
                explanation = "설명을 생성할 수 없습니다."
            
            recommendations.append({
                'url': doc['url'],
                'title': doc.get('title', ''),
                'keywords': doc['keywords'],
                'relation_type': doc['relation_type'],
                'similarity_score': float(similarities[idx]),
                'explanation': explanation
            })
        
        return recommendations

    def _train_epoch(self, train_loader: DataLoader, optimizer: optim.Optimizer) -> float:
        self.model.train()
        total_loss = 0
        
        for batch in tqdm(train_loader, desc="Training"):
            features, G_batch, _ = batch
            features = features.to(self.device)
            G_batch = G_batch.to(self.device)
            
            optimizer.zero_grad()
            
            # 순전파
            embeddings = self.model(features, G_batch)
            
            # 손실 계산
            loss = self.model.compute_loss(embeddings, G_batch)
            
            # 역전파
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(train_loader)


    def _validate_epoch(
        self,
        val_loader: DataLoader
    ) -> float:
        """한 에폭 검증
        
        Args:
            val_loader: 검증 데이터 로더
            
        Returns:
            epoch_loss: 에폭 평균 손실
        """
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in val_loader:
                features, G, _ = batch
                features = features.to(self.device)
                G = G.to(self.device)
                
                embeddings = self.model(features, G)
                loss = self.model.compute_loss(embeddings, G)
                
                total_loss += loss.item()
        
        return total_loss / len(val_loader)

    def _save_checkpoint(
        self,
        epoch: int,
        val_loss: float
    ):
        """체크포인트 저장
        
        Args:
            epoch: 현재 에폭
            val_loss: 검증 손실
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'val_loss': val_loss,
            'config': self.config
        }
        
        save_path = Path('checkpoints')
        save_path.mkdir(exist_ok=True)
        
        torch.save(
            checkpoint,
            save_path / f'model_epoch_{epoch}_loss_{val_loss:.4f}.pt'
        )

def main():
    # 추천 시스템 초기화
    recommender = EconomicNewsRecommender()
    
    # 데이터 준비
    dataset, keyword_to_idx, relation_to_idx = recommender.prepare_data(
        recommender.config['econfinnews_raw']
    )
    
    # 원본 문서 로드
    documents = recommender.data_processor.load_data(recommender.config['econfinnews_raw'])
    
    # 모델 학습 (documents 추가)
    recommender.train(dataset, documents)
    
    # 예시 추천
    query_document = {
        "url": "example_url",
        "title": "금리 인상이 소비에 미치는 영향",
        "relation_type": "금융 시장",
        "keywords": ["금리인상", "소비위축", "통화정책"]
    }
    
    recommendations = recommender.recommend(query_document)
    
    # 결과 출력
    for i, rec in enumerate(recommendations, 1):
        print(f"\n--------------------------------------------------\n")
        print(f"추천 {i} (유사도: {rec['similarity_score']:.4f})")
        print(f"제목: {rec['title']}")
        print(f"URL: {rec['url']}")
        print(f"관계: {rec['relation_type']}")
        print(f"키워드: {', '.join(rec['keywords'])}\n")
        print(f"추천 이유: \n{rec['explanation']}")

if __name__ == '__main__':
    main()