# dataset.py
import torch
from torch.utils.data import Dataset
import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict

class HGANDataset(Dataset):
    """HGAN을 위한 금융 하이퍼그래프 데이터셋"""
    
    def __init__(
        self,
        keywords: List[str],
        relations: List[Dict],
        embeddings: torch.Tensor,
        relation_types: Optional[List[str]] = None
    ):
        """
        Args:
            keywords: 키워드 리스트
            relations: 관계 정보 리스트. 각 관계는 다음 정보를 포함:
                - keywords: List[str] - 관련 키워드들
                - type: str - 관계 유형
                - weight: float - 관계 가중치 (선택)
                - attributes: Dict - 추가 속성 (선택)
            embeddings: 키워드 임베딩 텐서 (num_keywords, embedding_dim)
            relation_types: 가능한 관계 유형 리스트 (선택)
        """
        self.keywords = keywords
        self.keyword_to_idx = {word: idx for idx, word in enumerate(keywords)}
        self.num_keywords = len(keywords)
        
        # 임베딩 설정
        self.embeddings = embeddings
        self.embedding_dim = embeddings.size(1)
        
        # 관계 유형 설정
        self.relation_types = relation_types or sorted(list(set(
            rel["type"] for rel in relations
        )))
        self.relation_type_to_idx = {
            rtype: idx for idx, rtype in enumerate(self.relation_types)
        }
        self.num_relation_types = len(self.relation_types)
        
        # 하이퍼그래프 구조 생성
        self.hyperedges, self.edge_types = self._create_hypergraph_structure(relations)
        self.edge_index = self._create_edge_index()
        
        # 시간 정보 처리 (있는 경우)
        self.temporal_edges = self._process_temporal_info(relations)
        
        # 노드별 이웃 정보 캐싱
        self.node_neighbors = self._cache_node_neighbors()
    
    def _create_hypergraph_structure(
        self,
        relations: List[Dict]
    ) -> Tuple[List[Set[int]], List[int]]:
        """하이퍼그래프 구조 생성"""
        hyperedges = []
        edge_types = []
        
        for rel in relations:
            # 키워드를 노드 인덱스로 변환
            nodes = {
                self.keyword_to_idx[k]
                for k in rel["keywords"]
                if k in self.keyword_to_idx
            }
            
            if len(nodes) > 1:  # 최소 2개 노드가 있는 경우만 추가
                hyperedges.append(nodes)
                edge_types.append(self.relation_type_to_idx[rel["type"]])
        
        return hyperedges, edge_types
    
    def _create_edge_index(self) -> torch.Tensor:
        """COO 형식의 엣지 인덱스 생성"""
        edge_index = []
        
        for edge_idx, edge in enumerate(self.hyperedges):
            # 엣지 내의 모든 노드 쌍에 대해 연결 추가
            nodes = list(edge)
            for i in range(len(nodes)):
                for j in range(len(nodes)):
                    if i != j:
                        edge_index.append([nodes[i], nodes[j]])
        
        if not edge_index:  # 엣지가 없는 경우
            return torch.zeros((2, 0), dtype=torch.long)
        
        # long 타입으로 변환하여 반환
        return torch.tensor(edge_index, dtype=torch.long).t()
    
    def _process_temporal_info(self, relations: List[Dict]) -> Optional[Dict]:
        """시간 정보 처리 (있는 경우)"""
        if not any("timestamp" in rel for rel in relations):
            return None
            
        temporal_info = {
            "edge_timestamps": [],
            "temporal_order": []
        }
        
        for idx, rel in enumerate(relations):
            if "timestamp" in rel:
                temporal_info["edge_timestamps"].append(rel["timestamp"])
            else:
                temporal_info["edge_timestamps"].append(float('inf'))
        
        # 시간순 정렬 인덱스
        temporal_info["temporal_order"] = np.argsort(temporal_info["edge_timestamps"])
        
        return temporal_info
    
    def _cache_node_neighbors(self) -> Dict[int, Set[int]]:
        """노드별 이웃 노드 캐싱"""
        neighbors = defaultdict(set)
        
        for edge in self.hyperedges:
            for node in edge:
                neighbors[node].update(edge - {node})
        
        return neighbors
    
    def get_edges_between(self, node_i: int, node_j: int) -> List[int]:
        """두 노드를 포함하는 하이퍼엣지 찾기"""
        return [
            idx for idx, edge in enumerate(self.hyperedges)
            if node_i in edge and node_j in edge
        ]
    
    def get_edge_index(self) -> torch.Tensor:
        """엣지 인덱스 텐서 반환"""
        return self.edge_index.long()

    def sample_training_pairs(
        self,
        batch_size: int,
        negative_ratio: float = 1.0
    ) -> Dict[str, torch.Tensor]:
        """학습을 위한 노드 쌍 샘플링"""
        # 긍정 샘플 생성
        pos_pairs = []
        for edge in self.hyperedges:
            nodes = list(edge)
            for i in range(len(nodes)):
                for j in range(i + 1, len(nodes)):
                    pos_pairs.append([nodes[i], nodes[j]])
        
        # 배치 크기에 맞게 샘플링
        if len(pos_pairs) > batch_size:
            indices = np.random.choice(len(pos_pairs), batch_size, replace=False)
            pos_pairs = [pos_pairs[i] for i in indices]
        
        # 부정 샘플 생성
        neg_pairs = []
        num_neg = int(batch_size * negative_ratio)
        
        while len(neg_pairs) < num_neg:
            i = np.random.randint(self.num_keywords)
            j = np.random.randint(self.num_keywords)
            if i != j and [i, j] not in pos_pairs:
                neg_pairs.append([i, j])
        
        # 텐서로 변환 (long 타입으로 지정)
        return {
            'pos_pairs': torch.tensor(pos_pairs, dtype=torch.long),
            'neg_pairs': torch.tensor(neg_pairs, dtype=torch.long)
        }
    
    def get_relation_type_mask(self) -> torch.Tensor:
        """관계 유형별 마스크 생성"""
        mask = torch.zeros(len(self.hyperedges), self.num_relation_types)
        for edge_idx, edge_type in enumerate(self.edge_types):
            mask[edge_idx, edge_type] = 1
        return mask
    
    def __len__(self) -> int:
        return self.num_keywords
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.embeddings[idx]