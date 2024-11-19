# evaluator.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

class HGANEvaluator:
    """HGAN 모델 평가기"""
    
    def __init__(self, device: Optional[str] = None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    
    @torch.no_grad()
    def evaluate(
        self,
        model: nn.Module,
        predictor: nn.Module,
        dataset: Any
    ) -> Dict[str, float]:
        """모델 성능 평가"""
        model.eval()
        predictor.eval()
        
        # 데이터 준비
        x = dataset.embeddings.to(self.device)
        edge_index = dataset.edge_index.to(self.device)
        
        # 임베딩 계산
        output = model(x, edge_index)
        embeddings = output['embeddings']
        
        # 다양한 메트릭 계산
        metrics = {}
        
        # 1. 재구성 성능
        metrics['reconstruction_error'] = F.mse_loss(embeddings, x).item()
        
        # 2. 링크 예측 성능
        link_metrics = self._evaluate_link_prediction(
            predictor, embeddings, dataset
        )
        metrics.update(link_metrics)
        
        # 3. 군집화 품질
        clustering_metrics = self._evaluate_clustering(embeddings, dataset)
        metrics.update(clustering_metrics)
        
        # 4. 관계 보존성
        preservation_metrics = self._evaluate_relation_preservation(
            embeddings, dataset
        )
        metrics.update(preservation_metrics)
        
        return metrics
    
    def _evaluate_link_prediction(
        self,
        predictor: nn.Module,
        embeddings: torch.Tensor,
        dataset: Any,
        num_test_samples: int = 1000
    ) -> Dict[str, float]:
        """링크 예측 성능 평가"""
        # 테스트 샘플 생성
        test_pos_pairs = []
        test_neg_pairs = []
        
        # 긍정 샘플
        for edge in dataset.hyperedges:
            edge = list(edge)
            for i in range(len(edge)):
                for j in range(i+1, len(edge)):
                    test_pos_pairs.append([edge[i], edge[j]])
        
        # 부정 샘플
        while len(test_neg_pairs) < len(test_pos_pairs):
            i = np.random.randint(dataset.num_keywords)
            j = np.random.randint(dataset.num_keywords)
            if i != j and [i, j] not in test_pos_pairs:
                test_neg_pairs.append([i, j])
        
        # 예측 수행
        pos_scores = predictor(
            embeddings[torch.tensor(test_pos_pairs)[:,0]],
            embeddings[torch.tensor(test_pos_pairs)[:,1]]
        )
        neg_scores = predictor(
            embeddings[torch.tensor(test_neg_pairs)[:,0]],
            embeddings[torch.tensor(test_neg_pairs)[:,1]]
        )
        
        # AUC-ROC 계산
        labels = torch.cat([
            torch.ones(len(test_pos_pairs)),
            torch.zeros(len(test_neg_pairs))
        ])
        scores = torch.cat([pos_scores, neg_scores]).cpu().numpy()
        
        from sklearn.metrics import roc_auc_score, average_precision_score
        return {
            'link_prediction_auc': roc_auc_score(labels, scores),
            'link_prediction_ap': average_precision_score(labels, scores)
        }
    
    def _evaluate_clustering(
        self,
        embeddings: torch.Tensor,
        dataset: Any
    ) -> Dict[str, float]:
        """군집화 품질 평가"""
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score
        
        # K-means 클러스터링
        embeddings_np = embeddings.cpu().numpy()
        n_clusters = min(len(dataset.relation_types), len(embeddings_np) - 1)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(embeddings_np)
        
        return {
            'silhouette_score': silhouette_score(embeddings_np, clusters),
            'inertia': kmeans.inertia_
        }
    
    def _evaluate_relation_preservation(
        self,
        embeddings: torch.Tensor,
        dataset: Any
    ) -> Dict[str, float]:
        """관계 보존성 평가"""
        preservation_scores = []
        relation_type_scores = defaultdict(list)
        
        for edge_idx, (edge, edge_type) in enumerate(zip(dataset.hyperedges, dataset.edge_types)):
            edge = list(edge)
            if len(edge) < 2:
                continue
            
            # 엣지 내 노드들 간의 평균 유사도 계산
            edge_embeddings = embeddings[edge]
            similarities = []
            
            for i in range(len(edge)):
                for j in range(i+1, len(edge)):
                    sim = F.cosine_similarity(
                        edge_embeddings[i].unsqueeze(0),
                        edge_embeddings[j].unsqueeze(0)
                    ).item()
                    similarities.append(sim)
            
            avg_sim = np.mean(similarities)
            preservation_scores.append(avg_sim)
            relation_type_scores[edge_type].append(avg_sim)
        
        # 전체 및 관계 타입별 보존성 점수 계산
        metrics = {
            'overall_preservation': np.mean(preservation_scores),
            'preservation_std': np.std(preservation_scores)
        }
        
        # 관계 타입별 점수
        for rtype, scores in relation_type_scores.items():
            type_name = dataset.relation_types[rtype]
            metrics[f'preservation_{type_name}'] = np.mean(scores)
        
        return metrics
    
    def recommend_similar_keywords(
        self,
        query_keyword: str,
        dataset: Any,
        model: nn.Module,
        k: int = 5
    ) -> List[Tuple[str, float, float]]:
        """유사 키워드 추천"""
        model.eval()
        
        # 쿼리 키워드의 인덱스 찾기
        if query_keyword not in dataset.keyword_to_idx:
            raise ValueError(f"Keyword '{query_keyword}' not found in dataset")
        
        query_idx = dataset.keyword_to_idx[query_keyword]
        
        # 임베딩 계산
        x = dataset.embeddings.to(self.device)
        edge_index = dataset.edge_index.to(self.device)
        
        output = model(x, edge_index)
        embeddings = output['embeddings']
        uncertainties = output['uncertainty']
        
        # 유사도 계산
        query_embedding = embeddings[query_idx]
        similarities = F.cosine_similarity(
            query_embedding.unsqueeze(0),
            embeddings
        )
        
        # Top-k 유사 키워드 선택 (자기 자신 제외)
        similarities[query_idx] = -float('inf')
        top_k_values, top_k_indices = similarities.topk(k)
        
        # 결과 포맷팅
        recommendations = []
        for idx, (sim, uncertainty) in enumerate(zip(
            top_k_values,
            uncertainties[top_k_indices]
        )):
            keyword = dataset.keywords[top_k_indices[idx]]
            recommendations.append((
                keyword,
                sim.item(),
                uncertainty.item()
            ))
        
        return recommendations