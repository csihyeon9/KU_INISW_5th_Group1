# typed_evaluator.py
import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import roc_auc_score, precision_recall_curve, average_precision_score

class TypedHGNNEvaluator:
    def __init__(self, device: Optional[str] = None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    
    @torch.no_grad()
    def evaluate(
        self,
        model: torch.nn.Module,
        dataset
    ) -> Dict[str, float]:
        """
        모델 평가
        
        Returns:
            Dict: {
                'reconstruction_loss': 재구성 손실,
                'relation_prediction_auc': 관계 예측 AUC,
                'relation_prediction_ap': 관계 예측 AP,
                'relation_preservation': 관계 보존성 점수,
                'type_wise_metrics': 관계 타입별 메트릭
            }
        """
        model.eval()
        
        # 데이터 준비
        X = dataset.get_features().to(self.device)
        H_dict = {
            rel_type: H.to(self.device)
            for rel_type, H in dataset.get_typed_incidence_matrices().items()
        }
        
        # 임베딩 계산
        output = model(X, H_dict)
        embeddings = output['embeddings'].cpu().numpy()
        
        # 기본 메트릭
        metrics = {
            'reconstruction_loss': self._compute_reconstruction_loss(output['embeddings'], X),
            'relation_preservation': self._evaluate_relation_preservation(dataset, embeddings)
        }
        
        # 관계 예측 성능
        relation_metrics = self._evaluate_relation_prediction(
            model, dataset, embeddings
        )
        metrics.update(relation_metrics)
        
        # 관계 타입별 메트릭
        type_wise_metrics = self._evaluate_by_relation_type(
            dataset, embeddings
        )
        metrics['type_wise_metrics'] = type_wise_metrics
        
        return metrics
    
    def _compute_reconstruction_loss(
        self,
        output: torch.Tensor,
        target: torch.Tensor
    ) -> float:
        """재구성 손실 계산"""
        return torch.nn.functional.mse_loss(output, target).item()
    
    def _evaluate_relation_prediction(
        self,
        model: torch.nn.Module,
        dataset,
        embeddings: np.ndarray
    ) -> Dict[str, float]:
        """관계 예측 성능 평가"""
        # 테스트 데이터 준비
        pos_samples, neg_samples = dataset.get_test_samples()
        
        # 긍정 샘플 점수 계산
        pos_scores = self._compute_relation_scores(
            model, embeddings, pos_samples
        )
        
        # 부정 샘플 점수 계산
        neg_scores = self._compute_relation_scores(
            model, embeddings, neg_samples
        )
        
        # 라벨과 점수 준비
        y_true = np.concatenate([np.ones(len(pos_scores)), np.zeros(len(neg_scores))])
        y_score = np.concatenate([pos_scores, neg_scores])
        
        return {
            'relation_prediction_auc': roc_auc_score(y_true, y_score),
            'relation_prediction_ap': average_precision_score(y_true, y_score)
        }
    
    def _compute_relation_scores(
        self,
        model: torch.nn.Module,
        embeddings: np.ndarray,
        samples: List[Tuple[int, int, int]]
    ) -> np.ndarray:
        """관계 예측 점수 계산"""
        src_nodes = torch.tensor([s[0] for s in samples]).to(self.device)
        dst_nodes = torch.tensor([s[1] for s in samples]).to(self.device)
        rel_types = torch.tensor([s[2] for s in samples]).to(self.device)
        
        src_embeds = torch.tensor(embeddings[src_nodes]).to(self.device)
        dst_embeds = torch.tensor(embeddings[dst_nodes]).to(self.device)
        
        scores = model.relation_predictor(src_embeds, dst_embeds)
        scores = torch.softmax(scores, dim=1)
        
        # 각 샘플의 실제 관계 타입에 대한 점수 추출
        scores = scores[torch.arange(len(rel_types)), rel_types]
        return scores.cpu().numpy()
    
    def _evaluate_relation_preservation(
        self,
        dataset,
        embeddings: np.ndarray
    ) -> float:
        """관계 보존성 평가"""
        scores = []
        for edge in dataset.hyperedges:
            nodes = list(edge['nodes'])
            if len(nodes) < 2:
                continue
            
            edge_embeddings = embeddings[nodes]
            similarities = []
            
            for i in range(len(nodes)):
                for j in range(i+1, len(nodes)):
                    sim = self._cosine_similarity(
                        edge_embeddings[i],
                        edge_embeddings[j]
                    )
                    similarities.append(sim)
            
            # 엣지 가중치 반영
            score = np.mean(similarities) * edge['weight']
            scores.append(score)
        
        return np.mean(scores)
    
    def _evaluate_by_relation_type(
        self,
        dataset,
        embeddings: np.ndarray
    ) -> Dict[str, Dict[str, float]]:
        """관계 타입별 성능 평가"""
        metrics = {}
        
        for rel_type in dataset.relation_types:
            # 해당 타입의 엣지만 선택
            edges = [e for e in dataset.hyperedges if e['type'] == rel_type]
            
            if not edges:
                continue
            
            # 관계 보존성 계산
            preservation_scores = []
            for edge in edges:
                nodes = list(edge['nodes'])
                if len(nodes) < 2:
                    continue
                
                edge_embeddings = embeddings[nodes]
                similarities = []
                
                for i in range(len(nodes)):
                    for j in range(i+1, len(nodes)):
                        sim = self._cosine_similarity(
                            edge_embeddings[i],
                            edge_embeddings[j]
                        )
                        similarities.append(sim)
                
                score = np.mean(similarities) * edge['weight']
                preservation_scores.append(score)
            
            metrics[rel_type] = {
                'preservation': np.mean(preservation_scores),
                'count': len(edges)
            }
        
        return metrics
    
    @staticmethod
    def _cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
        """코사인 유사도 계산"""
        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    
    def find_similar_keywords(
        self,
        dataset,
        query_idx: int,
        embeddings: np.ndarray,
        k: int = 5
    ) -> List[Tuple[int, float]]:
        """유사 키워드 검색"""
        query_embedding = embeddings[query_idx]
        similarities = []
        
        for idx, embedding in enumerate(embeddings):
            if idx != query_idx:
                similarity = self._cosine_similarity(query_embedding, embedding)
                similarities.append((idx, similarity))
        
        return sorted(similarities, key=lambda x: x[1], reverse=True)[:k]