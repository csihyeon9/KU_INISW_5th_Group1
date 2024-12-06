# data_processor2.py -> 동사 카테고리 정보를 추가하여 processing 할 python 모듈
import json
import numpy as np
from scipy import sparse
import torch
from typing import List, Dict, Set, Tuple
from pathlib import Path
from collections import defaultdict
import math
from tqdm import tqdm

class KeywordProcessor:
    def __init__(self, unique_keywords_path: str, news_data_path: str):
        self.unique_keywords_path = unique_keywords_path
        self.news_data_path = news_data_path
        self.keyword2idx: Dict[str, int] = {}
        self.idx2keyword: Dict[int, str] = {}
        self.keyword_counts: Dict[str, int] = defaultdict(int)
        self.pair_counts: Dict[Tuple[str, str], int] = defaultdict(int)

        self.verb_categories: Set[int] = set()  # 동사 카테고리 매핑 결과를 저장할 set
        self.total_articles = 0
        
    def load_data(self) -> None:
        # 고유 키워드 로드
        with open(self.unique_keywords_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            # 정렬하여 일관된 순서 보장하고, 중복 제거
            unique_keywords = sorted(list(set(data['unique_keywords'])))
        
        print(f"Total unique keywords in file: {len(unique_keywords)}")
        
        # 매트릭스 크기 설정
        self.matrix_size = len(unique_keywords)
        
        # 키워드 인덱스 매핑 생성 - 연속적인 인덱스 보장
        for idx, keyword in enumerate(unique_keywords):
            self.keyword2idx[keyword] = idx
            self.idx2keyword[idx] = keyword
            
        print(f"\nTotal mapped keywords: {len(self.keyword2idx)}")

        # 뉴스 데이터 로드
        print("\nLoading news data...")
        with open(self.news_data_path, 'r', encoding='utf-8') as f:
            news_data = json.load(f)
        self.total_articles = len(news_data)
        print(f"Total articles loaded: {self.total_articles}")
        
        # 키워드 출현 빈도 계산 및 동사 카테고리 저장
        print("\nProcessing articles and extracting verb categories...")
        for article in tqdm(news_data):
            # unique_keywords에 있는 키워드만 필터링
            valid_keywords = [k for k in article['all_keywords'] if k in self.keyword2idx]
            
            # 개별 키워드 카운트
            for keyword in valid_keywords:
                self.keyword_counts[keyword] += 1
            
            # 키워드 쌍 카운트
            for i in range(len(valid_keywords)):
                for j in range(i + 1, len(valid_keywords)):
                    pair = tuple(sorted([valid_keywords[i], valid_keywords[j]]))
                    self.pair_counts[pair] += 1
            
            # 동사 카테고리 정보 추출
            if 'relations' in article:
                for verb_category, relations in article['relations'].items():
                    self.verb_categories.add(int(verb_category))
        
        print(f"\nVerb categories extracted: {self.verb_categories}")
        print(f"Total unique keywords found in articles: {len(self.keyword_counts)}")
        print(f"Total keyword pairs found: {len(self.pair_counts)}")

    def save_processed_data(self, output_path: str) -> None:
        """처리된 데이터 저장"""
        data = {
            'keyword2idx': self.keyword2idx,
            'idx2keyword': self.idx2keyword,
            'keyword_counts': dict(self.keyword_counts),
            'pair_counts': {str(k): v for k, v in self.pair_counts.items()},
            'verb_categories': list(self.verb_categories),  # Save verb categories
            'total_articles': self.total_articles
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            
        print(f"Processed data saved to {output_path}")
