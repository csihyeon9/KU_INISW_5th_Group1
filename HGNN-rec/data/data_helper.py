# data/data_helper.py
import json
import numpy as np
from typing import List, Dict, Tuple, Set
from collections import Counter
from gensim.models import Word2Vec
import logging
from scipy.sparse import csr_matrix
import torch
from torch.utils.data import Dataset
from pathlib import Path

logging.basicConfig(level=logging.WARNING)
logging.getLogger().setLevel(logging.ERROR)

class NewsDataProcessor:
    """뉴스 데이터 처리 및 준비를 위한 클래스
    
    경제 금융 뉴스 데이터를 하이퍼그래프 학습에 적합한 형태로 변환합니다.
    향후 확장성을 고려한 모듈화된 구조를 가집니다.
    """
    
    def __init__(self, config: dict):
        """
        Args:
            config: 데이터 처리 관련 설정
        """
        self.config = config
        self.keyword_to_idx = {}
        self.idx_to_keyword = {}
        self.relation_to_idx = {}
        self.idx_to_relation = {}
        self.word2vec_model = None
        
    def load_data(self, file_path: str) -> List[Dict]:
        """JSON 형식의 뉴스 데이터 로드
        
        Args:
            file_path: 데이터 파일 경로
            
        Returns:
            documents: 문서 리스트
        """
        # logger.info(f"Loading data from {file_path}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 필터링: 키워드가 있는 문서만 선택
            documents = [doc for doc in data if doc.get('keywords') and len(doc['keywords']) > 0]
            
            # logger.info(f"Loaded {len(documents)} valid documents")
            return documents
            
        except Exception as e:
            # logger.error(f"Error loading data: {str(e)}")
            raise

    def build_vocabularies(self, documents: List[Dict]) -> Tuple[Dict, Dict]:
        """키워드와 관계 유형에 대한 인덱스 매핑 생성
        
        Args:
            documents: 문서 리스트
            
        Returns:
            keyword_to_idx: 키워드 -> 인덱스 매핑
            relation_to_idx: 관계 유형 -> 인덱스 매핑
        """
        # 키워드 및 관계 유형 수집
        keywords = set()
        relations = set()
        
        for doc in documents:
            keywords.update(doc['keywords'])
            relations.add(doc['relation_type'])
        
        # 인덱스 매핑 생성
        self.keyword_to_idx = {k: i for i, k in enumerate(sorted(keywords))}
        self.idx_to_keyword = {i: k for k, i in self.keyword_to_idx.items()}
        
        self.relation_to_idx = {r: i for i, r in enumerate(sorted(relations))}
        self.idx_to_relation = {i: r for r, i in self.relation_to_idx.items()}
        
        # logger.info(f"Built vocabularies: {len(keywords)} keywords, {len(relations)} relation types")
        
        return self.keyword_to_idx, self.relation_to_idx

    def create_keyword_embeddings(
        self,
        documents: List[Dict],
        embedding_dim: int = 128
    ) -> Word2Vec:
        """Word2Vec을 사용하여 키워드 임베딩 생성
        
        Args:
            documents: 문서 리스트
            embedding_dim: 임베딩 차원
            
        Returns:
            word2vec_model: 학습된 Word2Vec 모델
        """
        # 학습 데이터 준비
        keyword_sequences = [doc['keywords'] for doc in documents]
        
        # Word2Vec 모델 학습
        self.word2vec_model = Word2Vec(
            sentences=keyword_sequences,
            vector_size=embedding_dim,
            window=5,
            min_count=1,
            workers=4
        )
        
        # logger.info(f"Created keyword embeddings with dimension {embedding_dim}")
        return self.word2vec_model

    def create_document_features(
        self,
        documents: List[Dict],
        embedding_dim: int = 128
    ) -> np.ndarray:
        """문서별 특징 벡터 생성
        
        Args:
            documents: 문서 리스트
            embedding_dim: 임베딩 차원
            
        Returns:
            features: 문서 특징 행렬 (n_documents x embedding_dim)
        """
        if self.word2vec_model is None:
            self.create_keyword_embeddings(documents, embedding_dim)
        
        n_docs = len(documents)
        features = np.zeros((n_docs, embedding_dim))
        
        for i, doc in enumerate(documents):
            keyword_vectors = [
                self.word2vec_model.wv[keyword]
                for keyword in doc['keywords']
                if keyword in self.word2vec_model.wv
            ]
            
            if keyword_vectors:
                features[i] = np.mean(keyword_vectors, axis=0)
        
        # L2 정규화
        features_norm = np.linalg.norm(features, axis=1, keepdims=True)
        features_norm[features_norm == 0] = 1
        features = features / features_norm
        
        return features

class EconomicNewsDataset(Dataset):
    def __init__(self, features, labels, G):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)
        # 전체 G 행렬을 sparse 형태로 저장
        self.G = torch.FloatTensor(G.todense())
        self.num_nodes = self.G.shape[0]

    def __getitem__(self, idx):
        # 배치에 해당하는 G 행렬만 반환
        return self.features[idx], self.G[idx], self.labels[idx]

    def __len__(self):
        return len(self.features)

def analyze_data_distribution(documents: List[Dict]) -> Dict:
    """데이터 분포 분석
    
    Args:
        documents: 문서 리스트
        
    Returns:
        stats: 데이터 통계 정보
    """
    # 관계 유형별 분포
    relation_counts = Counter(doc['relation_type'] for doc in documents)
    
    # 키워드 통계
    keyword_counts = Counter()
    keywords_per_doc = []
    for doc in documents:
        keyword_counts.update(doc['keywords'])
        keywords_per_doc.append(len(doc['keywords']))
    
    stats = {
        'total_documents': len(documents),
        'unique_relations': len(relation_counts),
        'relation_distribution': dict(relation_counts),
        'unique_keywords': len(keyword_counts),
        'avg_keywords_per_doc': np.mean(keywords_per_doc),
        'max_keywords_per_doc': max(keywords_per_doc),
        'min_keywords_per_doc': min(keywords_per_doc),
        'top_keywords': dict(keyword_counts.most_common(20))
    }
    
    # logger.info("Data distribution analysis completed")
    return stats

class ExplanationGenerator:
    """추천 결과에 대한 설명 생성 클래스"""
    
    def __init__(self, idx_to_keyword: Dict[int, str], idx_to_relation: Dict[int, str]):
        self.idx_to_keyword = idx_to_keyword
        self.idx_to_relation = idx_to_relation
        self._load_templates()

    def _load_templates(self):
        """설명 템플릿 로드"""
        self.templates = {
            # 기본 템플릿
            'default': "{source_relation}에서 다룬\n{source_keywords}이(가),\n다른 {target_relation} 관점에서 {target_keywords}와(과) 관련되어 있습니다.",
            
            # 관계 전환별 특화 템플릿
            'relation_specific': {
                ('금융 시장', '거시 경제'): "금융시장의 {source_keywords}가 거시경제 측면에서 {target_keywords}에 영향을 미치는 사례입니다.",
                ('금융 시장', '규제와 법률'): "{source_keywords}에 대한 규제가 {target_keywords}에 미치는 영향을 다룹니다.",
                # 다른 관계 조합에 대한 템플릿 추가 가능
            }
        }

    def generate_explanation(
        self,
        source_keywords: List[str],
        target_keywords: List[str],
        source_relation: str,
        target_relation: str
    ) -> str:
        """설명 문장 생성
        
        Args:
            source_keywords: 원본 키워드 리스트
            target_keywords: 추천된 키워드 리스트
            source_relation: 원본 관계 유형
            target_relation: 추천된 관계 유형
            
        Returns:
            explanation: 생성된 설명 문장
        """
        # 관계 조합에 특화된 템플릿이 있는지 확인
        relation_pair = (source_relation, target_relation)
        template = self.templates['relation_specific'].get(
            relation_pair,
            self.templates['default']
        )
        
        # 설명 생성
        explanation = template.format(
            source_keywords=', '.join(source_keywords),
            # source_keywords=', '.join(source_keywords[:2]),  # 주요 키워드 2개만 사용
            target_keywords=', '.join(target_keywords),
            # target_keywords=', '.join(target_keywords[:2]),
            source_relation=source_relation,
            target_relation=target_relation
        )
        
        return explanation