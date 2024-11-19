# preprocess_data.py
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.preprocessing import normalize
import json
from pathlib import Path
from tqdm import tqdm

class FinancialDataPreprocessor:
    def __init__(
        self,
        pretrained_model_name: str = "klue/bert-base",
        max_length: int = 64,
        device: str = None
    ):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)
        self.model = AutoModel.from_pretrained(pretrained_model_name).to(self.device)
        self.max_length = max_length
    
    def process_raw_data(
        self,
        keywords: List[str],
        relations: List[Dict],
        save_path: str
    ) -> None:
        """원시 데이터 전처리"""
        
        # 1. 키워드 임베딩 생성
        embeddings = self._generate_embeddings(keywords)
        
        # 2. 관계 정보 검증 및 정제
        cleaned_relations = self._clean_relations(relations, keywords)
        
        # 3. 관계 유형 추출 및 표준화
        relation_types = self._standardize_relation_types(cleaned_relations)
        
        # 4. 데이터 저장
        processed_data = {
            "keywords": keywords,
            "relations": cleaned_relations,
            "embeddings": embeddings.tolist(),
            "relation_types": relation_types
        }
        
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(processed_data, f, ensure_ascii=False, indent=2)
    
    @torch.no_grad()
    def _generate_embeddings(self, keywords: List[str]) -> np.ndarray:
        """BERT를 사용한 키워드 임베딩 생성"""
        embeddings = []
        
        for keyword in tqdm(keywords, desc="Generating embeddings"):
            # 토크나이징
            inputs = self.tokenizer(
                keyword,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            ).to(self.device)
            
            # BERT 임베딩 추출
            outputs = self.model(**inputs)
            embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            embeddings.append(embedding[0])
        
        # L2 정규화
        embeddings = normalize(np.array(embeddings))
        return embeddings
    
    def _clean_relations(
        self,
        relations: List[Dict],
        valid_keywords: List[str]
    ) -> List[Dict]:
        """관계 정보 검증 및 정제"""
        cleaned_relations = []
        valid_keyword_set = set(valid_keywords)
        
        for relation in relations:
            # 유효한 키워드만 포함
            valid_relation_keywords = [
                k for k in relation["keywords"]
                if k in valid_keyword_set
            ]
            
            # 최소 2개 이상의 유효한 키워드가 있는 경우만 포함
            if len(valid_relation_keywords) >= 2:
                cleaned_relation = {
                    "keywords": valid_relation_keywords,
                    "type": relation["type"],
                    "weight": relation.get("weight", 1.0)
                }
                
                # 추가 속성이 있는 경우 포함
                if "attributes" in relation:
                    cleaned_relation["attributes"] = relation["attributes"]
                
                cleaned_relations.append(cleaned_relation)
        
        return cleaned_relations
    
    def _standardize_relation_types(self, relations: List[Dict]) -> List[str]:
        """관계 유형 표준화"""
        # 관계 유형 수집 및 표준화
        relation_types = set()
        for relation in relations:
            rtype = relation["type"].upper().strip()
            relation_types.add(rtype)
            relation["type"] = rtype  # 표준화된 유형으로 업데이트
        
        return sorted(list(relation_types))

def main():
    # 예시 사용법
    preprocessor = FinancialDataPreprocessor()
    
    # 원시 데이터 로드
    with open('data/raw/raw_financial_data.json', 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
    
    # 전처리 실행
    preprocessor.process_raw_data(
        keywords=raw_data['keywords'],
        relations=raw_data['relations'],
        save_path='data/processed/processed_financial_data.json'
    )

if __name__ == "__main__":
    main()