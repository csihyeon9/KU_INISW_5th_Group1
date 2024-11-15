from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import numpy as np

@dataclass
class Relation:
    """관계 정보를 담는 클래스"""
    source: str
    target: str
    relation_type: str
    weight: float = 1.0
    attributes: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.attributes is None:
            self.attributes = {}

@dataclass
class MultiRelation:
    """다중 키워드 간의 관계를 담는 클래스"""
    keywords: List[str]
    relation_type: str
    direction: Optional[List[str]] = None  # 방향성이 있는 경우
    weight: float = 1.0
    attributes: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.attributes is None:
            self.attributes = {}

def create_financial_relations() -> List[MultiRelation]:
    """
    경제/금융 도메인의 관계 데이터 생성
    
    관계 타입:
    - INFLUENCE: 영향을 미치는 관계
    - DETERMINE: 결정하는 관계
    - ANALYZE: 분석하는 관계
    - TRADE: 거래/운용하는 관계
    - REGULATE: 규제/감독하는 관계
    - BELONG: 소속/포함되는 관계
    """
    relations = [
        # 정책 결정 관계
        MultiRelation(
            keywords=["중앙은행", "금리", "통화정책"],
            relation_type="DETERMINE",
            direction=["중앙은행", "금리"],
            attributes={
                "strength": "strong",
                "frequency": "monthly",
                "description": "중앙은행의 기준금리 결정"
            }
        ),
        
        # 시장 영향 관계
        MultiRelation(
            keywords=["금리", "채권시장", "주식시장"],
            relation_type="INFLUENCE",
            direction=["금리", "채권시장"],
            attributes={
                "strength": "strong",
                "correlation": "negative",
                "description": "금리 변동이 채권 가격에 미치는 영향"
            }
        ),
        
        # 분석 관계
        MultiRelation(
            keywords=["기술적분석", "주식시장", "거래량"],
            relation_type="ANALYZE",
            attributes={
                "method": "technical",
                "timeframe": "short-term",
                "description": "차트와 거래량 기반 분석"
            }
        ),
        
        # 규제 관계
        MultiRelation(
            keywords=["중앙은행", "시중은행", "금융감독"],
            relation_type="REGULATE",
            direction=["중앙은행", "시중은행"],
            attributes={
                "scope": "comprehensive",
                "authority": "legal",
                "description": "은행권 건전성 규제"
            }
        ),
        
        # 거래/운용 관계
        MultiRelation(
            keywords=["자산운용사", "포트폴리오", "위험관리"],
            relation_type="TRADE",
            attributes={
                "style": "active",
                "risk_level": "moderate",
                "description": "전문적 자산 운용"
            }
        ),
        
        # 경제지표 영향 관계
        MultiRelation(
            keywords=["GDP", "물가상승률", "실업률"],
            relation_type="INFLUENCE",
            attributes={
                "scope": "macro",
                "correlation": "complex",
                "description": "주요 거시경제지표 간 상호작용"
            }
        ),
        
        # 이론적 관계
        MultiRelation(
            keywords=["케인스이론", "재정정책", "총수요"],
            relation_type="BELONG",
            attributes={
                "field": "macroeconomics",
                "school": "keynesian",
                "description": "케인지안 경제학의 정책적 함의"
            }
        )
    ]
    
    return relations

class TypedHypergraphDataset:
    """관계 타입이 있는 하이퍼그래프 데이터셋"""
    def __init__(self, 
                 keywords: List[str], 
                 relations: List[MultiRelation],
                 embeddings: np.ndarray):
        self.keywords = keywords
        self.keyword_to_idx = {word: idx for idx, word in enumerate(keywords)}
        self.relations = relations
        self.embeddings = embeddings
        self.relation_types = self._extract_relation_types()
        self.hyperedges = self._create_hyperedges()
        
    def _extract_relation_types(self) -> List[str]:
        """고유한 관계 타입 추출"""
        return list(set(rel.relation_type for rel in self.relations))
    
    def _create_hyperedges(self) -> List[Dict]:
        """관계 타입이 포함된 하이퍼엣지 생성"""
        hyperedges = []
        for relation in self.relations:
            edge = {
                'nodes': set(self.keyword_to_idx[k] for k in relation.keywords),
                'type': relation.relation_type,
                'weight': relation.weight,
                'direction': relation.direction,
                'attributes': relation.attributes
            }
            hyperedges.append(edge)
        return hyperedges
    
    def get_incidence_matrix(self, relation_type: Optional[str] = None) -> torch.Tensor:
        """
        인시던스 행렬 생성
        relation_type이 지정되면 해당 타입의 관계만 포함
        """
        num_nodes = len(self.keywords)
        if relation_type:
            edges = [e for e in self.hyperedges if e['type'] == relation_type]
        else:
            edges = self.hyperedges
            
        num_edges = len(edges)
        H = torch.zeros((num_nodes, num_edges))
        
        for edge_idx, edge in enumerate(edges):
            for node_idx in edge['nodes']:
                H[node_idx, edge_idx] = edge['weight']
        
        return H
    
    def get_features(self) -> torch.Tensor:
        return torch.tensor(self.embeddings, dtype=torch.float)
    
    def get_relation_subgraph(self, relation_type: str) -> 'TypedHypergraphDataset':
        """특정 관계 타입의 서브그래프 추출"""
        relevant_relations = [r for r in self.relations if r.relation_type == relation_type]
        return TypedHypergraphDataset(self.keywords, relevant_relations, self.embeddings)