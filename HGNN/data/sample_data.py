import json
import numpy as np
from typing import List, Dict, Tuple, Any

def create_financial_data(embedding_dim: int = 64, seed: int = 42) -> Tuple[List[str], List[Dict[str, Any]], np.ndarray]:
    """
    경제/금융 도메인의 샘플 데이터 생성
    
    Args:
        embedding_dim: 임베딩 차원
        seed: 랜덤 시드
        
    Returns:
        Tuple[List[str], List[Dict], np.ndarray]: (키워드 리스트, 관계 리스트, 임베딩 행렬)
    """
    # 키워드 정의
    keywords = [
        # 금융 시장/상품
        "주식시장", "채권시장", "외환시장", "파생상품", "암호화폐",
        # 경제 지표
        "GDP", "물가상승률", "실업률", "금리", "환율",
        # 투자/분석
        "기술적분석", "기본적분석", "포트폴리오", "자산배분", "위험관리",
        # 금융기관
        "중앙은행", "시중은행", "증권사", "보험사", "자산운용사",
        # 경제이론/정책
        "통화정책", "재정정책", "케인스이론", "통화주의", "행동경제학"
    ]
    
    # 관계 정의
    relations = [
        # 금융시장 관련 그룹
        {
            "keywords": ["주식시장", "기술적분석", "기본적분석", "증권사"],
            "relation_type": "ANALYSIS",
            "weight": 1.0,
            "attributes": {"domain": "stock_market", "timeframe": "daily"}
        },
        {
            "keywords": ["채권시장", "금리", "중앙은행", "통화정책"],
            "relation_type": "INFLUENCE",
            "weight": 1.0,
            "attributes": {"correlation": "negative", "significance": "high"}
        },
        {
            "keywords": ["외환시장", "환율", "중앙은행", "통화정책"],
            "relation_type": "INFLUENCE",
            "weight": 1.0,
            "attributes": {"market": "forex", "scale": "international"}
        },
        
        # 정책 관련 그룹
        {
            "keywords": ["중앙은행", "통화정책", "금리", "물가상승률"],
            "relation_type": "CONTROL",
            "weight": 1.0,
            "attributes": {"mechanism": "monetary_policy", "frequency": "regular"}
        },
        {
            "keywords": ["재정정책", "GDP", "실업률", "케인스이론"],
            "relation_type": "THEORY",
            "weight": 0.8,
            "attributes": {"school": "keynesian", "focus": "fiscal_policy"}
        },
        
        # 투자분석 관련 그룹
        {
            "keywords": ["포트폴리오", "자산배분", "위험관리", "자산운용사"],
            "relation_type": "MANAGEMENT",
            "weight": 1.0,
            "attributes": {"approach": "systematic", "risk_level": "moderate"}
        },
        {
            "keywords": ["기술적분석", "주식시장", "암호화폐", "외환시장"],
            "relation_type": "ANALYSIS",
            "weight": 0.9,
            "attributes": {"method": "technical", "timeframe": "short_term"}
        },
        
        # 거시경제 관련 그룹
        {
            "keywords": ["GDP", "물가상승률", "실업률", "환율"],
            "relation_type": "CORRELATION",
            "weight": 0.9,
            "attributes": {"scope": "macro", "frequency": "quarterly"}
        },
        {
            "keywords": ["통화주의", "통화정책", "물가상승률", "중앙은행"],
            "relation_type": "THEORY",
            "weight": 0.8,
            "attributes": {"school": "monetarist", "focus": "money_supply"}
        }
    ]
    
    # 임베딩 생성
    np.random.seed(seed)
    embeddings = np.random.randn(len(keywords), embedding_dim)
    # L2 정규화
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    
    return keywords, relations, embeddings

def load_custom_data(data_path: str) -> Tuple[List[str], List[Dict[str, Any]], np.ndarray]:
    """
    사용자 정의 데이터 파일 로드
    
    Args:
        data_path: JSON 데이터 파일 경로
        
    Returns:
        Tuple[List[str], List[Dict], np.ndarray]: (키워드 리스트, 관계 리스트, 임베딩 행렬)
        
    데이터 파일 형식 (JSON):
    {
        "keywords": ["키워드1", "키워드2", ...],
        "relations": [
            {
                "keywords": ["키워드1", "키워드2", ...],
                "relation_type": "TYPE1",
                "weight": 1.0,
                "attributes": {...}  # 선택적
            },
            ...
        ],
        "embeddings": [[0.1, 0.2, ...], ...]  # 선택적
    }
    """
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    keywords = data['keywords']
    relations = data['relations']
    
    # 임베딩이 제공되지 않은 경우 랜덤 생성
    if 'embeddings' in data:
        embeddings = np.array(data['embeddings'])
    else:
        embeddings = np.random.randn(len(keywords), 64)
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    
    return keywords, relations, embeddings

def save_data(
    data_path: str,
    keywords: List[str],
    relations: List[Dict[str, Any]],
    embeddings: np.ndarray
):
    """데이터를 JSON 파일로 저장"""
    data = {
        "keywords": keywords,
        "relations": relations,
        "embeddings": embeddings.tolist()
    }
    
    with open(data_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)