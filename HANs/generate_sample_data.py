# generate_sample_data.py
import numpy as np
import json
from pathlib import Path

def generate_sample_data(
    keyword_file: str,
    embedding_dim: int = 64,
    num_relations: int = 200,
    save_path: str = "data/processed/financial_data.json"
):
    """키워드 리스트 기반 샘플 데이터셋 생성"""
    
    # 키워드 리스트 로드
    with open(keyword_file, 'r', encoding='utf-8') as f:
        keywords = json.load(f)
    
    # 관계 유형 정의 (한국 금융시장 특화)
    relation_types = [
        "정책영향",        # 정부/중앙은행 정책이 시장에 미치는 영향
        "연계성",          # 지표/수치 간의 연계성
        "인과관계",        # 한 지표가 다른 지표에 미치는 영향
        "구성관계",        # 상위지표와 하위지표의 관계
        "동행성",          # 함께 움직이는 지표들
        "선행성",          # 선행지표와 후행지표의 관계
        "대체성",          # 대체 가능한 지표들의 관계
        "보완성"           # 서로 보완하는 지표들의 관계
    ]
    
    # 임베딩 생성
    np.random.seed(42)
    embeddings = np.random.randn(len(keywords), embedding_dim)
    # L2 정규화
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    
    # 관계 생성
    relations = []
    for _ in range(num_relations):
        # 랜덤하게 2-4개의 키워드 선택
        num_keywords_in_relation = np.random.randint(2, 5)
        related_keywords = np.random.choice(
            keywords,
            size=num_keywords_in_relation,
            replace=False
        ).tolist()
        
        # 관계 유형 선택 (키워드 특성에 따라)
        if any('지수' in k for k in related_keywords):
            rel_type = np.random.choice(["연계성", "동행성", "선행성"])
        elif any('비율' in k for k in related_keywords):
            rel_type = np.random.choice(["인과관계", "구성관계"])
        elif any(('정책' in k or '규제' in k) for k in related_keywords):
            rel_type = "정책영향"
        else:
            rel_type = np.random.choice(relation_types)
        
        relation = {
            "keywords": related_keywords,
            "type": rel_type,
            "weight": round(np.random.uniform(0.5, 1.0), 2),
            "attributes": {
                "confidence": np.random.choice(["high", "medium", "low"]),
                "timeframe": np.random.choice(["short-term", "mid-term", "long-term"])
            }
        }
        relations.append(relation)
    
    # 데이터셋 저장
    data = {
        "keywords": keywords,
        "relations": relations,
        "embeddings": embeddings.tolist(),
        "relation_types": relation_types
    }
    
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"Generated sample dataset with:")
    print(f"- {len(keywords)} keywords")
    print(f"- {len(relations)} relations")
    print(f"- {len(relation_types)} relation types")
    print(f"Saved to: {save_path}")

if __name__ == "__main__":
    generate_sample_data(
        keyword_file="data/raw/data_only_keywords.json",
        embedding_dim=64,
        num_relations=200
    )