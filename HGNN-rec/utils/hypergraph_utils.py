# utils/hypergraph_utils.py
import numpy as np
from typing import List, Dict, Union, Tuple, Optional
from scipy import sparse
from scipy.sparse import csr_matrix, diags
import logging
from scipy.sparse import csr_matrix

logging.basicConfig(level=logging.WARNING)
logging.getLogger().setLevel(logging.ERROR)

class HypergraphBuilder:
    def __init__(self, config: dict):
        self.config = config
        self.construction_method = config.get('construction_method', 'natural')
        self.edge_weight_method = config.get('edge_weight_method', 'basic')
        self.k_neigs = config.get('k_neigs', [10])
        self.m_prob = config.get('m_prob', 1.0)
        self.is_probH = config.get('is_probH', True)

    def construct_H_from_documents(self, documents, keyword_to_idx, relation_to_idx):
        # 메모리 효율적인 하이퍼그래프 구성
        row_indices = []
        col_indices = []
        values = []
        
        for edge_idx, doc in enumerate(documents):
            keywords = [kw for kw in doc['keywords'] if kw in keyword_to_idx]
            if not keywords:
                continue
                
            node_indices = [keyword_to_idx[kw] for kw in keywords]
            row_indices.extend(node_indices)
            col_indices.extend([edge_idx] * len(node_indices))
            values.extend([1.0 / len(node_indices)] * len(node_indices))
        
        n_nodes = len(keyword_to_idx)
        n_edges = len(documents)
        
        H = csr_matrix(
            (values, (row_indices, col_indices)),
            shape=(n_nodes, n_edges)
        )
        
        W = np.ones(n_edges)
        
        return H, W

    def generate_G_from_H(self, H, W=None):
        """메모리 효율적인 라플라시안 행렬 생성"""
        if W is None:
            W = np.ones(H.shape[1])
        
        # 희소 행렬로 계산
        D_v = np.array(H.sum(axis=1)).flatten()
        D_e = np.array(H.sum(axis=0)).flatten()
        
        # 0으로 나누는 것을 방지
        D_v_invsqrt = np.power(D_v, -0.5, where=D_v!=0)
        D_v_invsqrt[np.isinf(D_v_invsqrt)] = 0
        
        D_e_invsqrt = np.power(D_e, -0.5, where=D_e!=0)
        D_e_invsqrt[np.isinf(D_e_invsqrt)] = 0
        
        # 희소 대각 행렬
        D_v_invsqrt_mat = sparse.diags(D_v_invsqrt)
        D_e_invsqrt_mat = sparse.diags(D_e_invsqrt)
        W_mat = sparse.diags(W)
        
        # G 계산
        G = D_v_invsqrt_mat @ H @ W_mat @ D_e_invsqrt_mat @ D_e_invsqrt_mat @ H.T @ D_v_invsqrt_mat
        
        return G.tocsr()

    @staticmethod
    def compute_edge_similarity(edge1_keywords: List[str], edge2_keywords: List[str]) -> float:
        """두 하이퍼에지 간의 유사도 계산"""
        set1 = set(edge1_keywords)
        set2 = set(edge2_keywords)
        
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        
        return intersection / union if union > 0 else 0.0

    # Legacy support: KNN 기반 하이퍼그래프 구성
    def construct_H_with_KNN(self, 
                           feat_mat: np.ndarray, 
                           k_neigs: Optional[List[int]] = None
    ) -> np.ndarray:
        """KNN 기반 하이퍼그래프 구성 (기존 코드와의 호환성 유지)"""
        if k_neigs is None:
            k_neigs = self.k_neigs
            
        dist_mat = self._calculate_distance(feat_mat)
        H = None
        
        for k in k_neigs:
            H_k = self._construct_H_from_distance(dist_mat, k)
            H = self._hyperedge_concat(H, H_k)
            
        return H

    def _calculate_distance(self, feat_mat: np.ndarray) -> np.ndarray:
        """특징 행렬로부터 거리 행렬 계산"""
        feat_mat = np.mat(feat_mat)
        dist_mat = np.zeros((feat_mat.shape[0], feat_mat.shape[0]))
        
        for i in range(feat_mat.shape[0]):
            for j in range(i + 1, feat_mat.shape[0]):
                dist = np.linalg.norm(feat_mat[i] - feat_mat[j])
                dist_mat[i, j] = dist_mat[j, i] = dist
                
        return dist_mat

    def _construct_H_from_distance(self, 
                                 dist_mat: np.ndarray, 
                                 k_neig: int
    ) -> np.ndarray:
        """거리 행렬로부터 하이퍼그래프 인시던스 행렬 구성"""
        H = np.zeros((dist_mat.shape[0], dist_mat.shape[0]))
        
        for i in range(dist_mat.shape[0]):
            idx = np.argsort(dist_mat[i])[1:k_neig+1]
            for j in idx:
                if self.is_probH:
                    H[i, j] = np.exp(-dist_mat[i, j] / self.m_prob)
                else:
                    H[i, j] = 1.0
                    
        return H

    @staticmethod
    def _hyperedge_concat(H1: Optional[np.ndarray], 
                         H2: np.ndarray
    ) -> np.ndarray:
        """하이퍼에지 연결"""
        if H1 is None:
            return H2
        return np.hstack((H1, H2))