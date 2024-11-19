# typed_dataset.py
import torch
import numpy as np
from typing import Dict, List, Set, Tuple, Any, Optional

class TypedHypergraphDataset:
    def __init__(
        self,
        keywords: List[str],
        relations: List[Dict[str, Any]],
        embeddings: np.ndarray
    ):
        """
        Args:
            keywords: 키워드 리스트
            relations: 관계 정보 리스트
                [
                    {
                        "keywords": ["키워드1", "키워드2", ...],
                        "relation_type": "INFLUENCE",
                        "weight": 1.0,
                        "attributes": {...}  # 선택적
                    },
                    ...
                ]
            embeddings: 키워드 임베딩 행렬 (num_keywords x embedding_dim)
        """
        self.keywords = keywords
        self.keyword_to_idx = {word: idx for idx, word in enumerate(keywords)}
        self.relations = relations
        self.embeddings = embeddings
        
        # 관계 타입 추출
        self.relation_types = sorted(list(set(
            rel["relation_type"] for rel in relations
        )))
        self.relation_type_to_idx = {
            t: idx for idx, t in enumerate(self.relation_types)
        }
        
        # 하이퍼엣지 생성
        self.hyperedges = self._create_hyperedges()
        
    def _create_hyperedges(self) -> List[Dict[str, Any]]:
        """하이퍼엣지 생성"""
        hyperedges = []
        
        for relation in self.relations:
            # 키워드 인덱스 변환
            nodes = {
                self.keyword_to_idx[k]
                for k in relation["keywords"]
                if k in self.keyword_to_idx
            }
            
            if nodes:  # 유효한 노드가 있는 경우만 추가
                edge = {
                    "nodes": nodes,
                    "type": relation["relation_type"],
                    "weight": relation.get("weight", 1.0),
                    "attributes": relation.get("attributes", {})
                }
                hyperedges.append(edge)
        
        return hyperedges
    
    def get_features(self) -> torch.Tensor:
        """노드 특성 행렬 반환"""
        return torch.tensor(self.embeddings, dtype=torch.float)
    
    def get_typed_incidence_matrices(self) -> Dict[str, torch.Tensor]:
        """관계 타입별 인시던스 행렬 반환"""
        matrices = {}
        
        for rel_type in self.relation_types:
            # 해당 타입의 엣지만 선택
            type_edges = [
                e for e in self.hyperedges
                if e["type"] == rel_type
            ]
            
            # 인시던스 행렬 생성
            H = torch.zeros((len(self.keywords), len(type_edges)))
            
            for edge_idx, edge in enumerate(type_edges):
                for node_idx in edge["nodes"]:
                    H[node_idx, edge_idx] = edge["weight"]
            
            matrices[rel_type] = H
        
        return matrices
    
    def get_relation_samples(
        self,
        batch_size: int,
        negative_ratio: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """학습을 위한 관계 샘플링"""
        # 긍정 샘플 생성
        pos_samples = []
        for edge in self.hyperedges:
            nodes = list(edge["nodes"])
            if len(nodes) >= 2:
                for i in range(len(nodes)):
                    for j in range(i + 1, len(nodes)):
                        pos_samples.append((
                            nodes[i],
                            nodes[j],
                            self.relation_type_to_idx[edge["type"]]
                        ))
        
        # 배치 크기에 맞게 샘플링
        if len(pos_samples) > batch_size:
            indices = np.random.choice(
                len(pos_samples),
                batch_size,
                replace=False
            )
            pos_samples = [pos_samples[i] for i in indices]
        
        # 텐서로 변환
        src_nodes = torch.tensor([s[0] for s in pos_samples])
        dst_nodes = torch.tensor([s[1] for s in pos_samples])
        rel_types = torch.tensor([s[2] for s in pos_samples])
        
        return src_nodes, dst_nodes, rel_types
    
    def get_test_samples(
        self,
        test_ratio: float = 0.2
    ) -> Tuple[List[Tuple[int, int, int]], List[Tuple[int, int, int]]]:
        """테스트를 위한 샘플 생성"""
        # 모든 가능한 엣지 쌍 생성
        all_pairs = []
        for edge in self.hyperedges:
            nodes = list(edge["nodes"])
            if len(nodes) >= 2:
                for i in range(len(nodes)):
                    for j in range(i + 1, len(nodes)):
                        all_pairs.append((
                            nodes[i],
                            nodes[j],
                            self.relation_type_to_idx[edge["type"]]
                        ))
        
        # 테스트 셋 분리
        num_test = int(len(all_pairs) * test_ratio)
        test_idx = np.random.choice(
            len(all_pairs),
            num_test,
            replace=False
        )
        test_mask = np.zeros(len(all_pairs), dtype=bool)
        test_mask[test_idx] = True
        
        test_pairs = [p for i, p in enumerate(all_pairs) if test_mask[i]]
        train_pairs = [p for i, p in enumerate(all_pairs) if not test_mask[i]]
        
        return train_pairs, test_pairs
    
    def __len__(self) -> int:
        return len(self.keywords)