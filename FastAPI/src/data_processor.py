# src/data_processor.py
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
        # print(f"힐스 index: {self.keyword2idx.get('힐스', 'Not found')}")
        # print(f"BC카드 index: {self.keyword2idx.get('BC카드', 'Not found')}")

        # 뉴스 데이터 로드
        print("\nLoading news data...")
        with open(self.news_data_path, 'r', encoding='utf-8') as f:
            news_data = json.load(f)
        self.total_articles = len(news_data)
        print(f"Total articles loaded: {self.total_articles}")
        
        # 키워드 출현 빈도 계산
        print("\nCalculating keyword frequencies...")
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
        
        print(f"\nTotal unique keywords found in articles: {len(self.keyword_counts)}")
        print(f"Total keyword pairs found: {len(self.pair_counts)}")
        
    def calculate_pmi(self, keyword1: str, keyword2: str) -> float:
        """두 키워드 간의 PMI 계산"""
        if keyword1 == keyword2:
            return 0.0
            
        pair = tuple(sorted([keyword1, keyword2]))
        
        # 개별 확률
        p_x = self.keyword_counts[keyword1] / self.total_articles
        p_y = self.keyword_counts[keyword2] / self.total_articles
        
        # 동시 출현 확률
        p_xy = self.pair_counts[pair] / self.total_articles
        
        # PMI 계산
        if p_xy == 0:
            return 0.0
        
        return math.log(p_xy / (p_x * p_y))
        
    def create_article_matrix(self, keywords: List[str]) -> sparse.csr_matrix:
        """하나의 기사에 대한 PMI 매트릭스 생성 (희소 행렬 버전)"""
        # 희소 행렬용 데이터 저장소
        rows = []
        cols = []
        data = []
        
        # unique_keywords에 있는 키워드만 필터링
        valid_keywords = [k for k in keywords if k in self.keyword2idx]
        
        # PMI 계산 및 희소 행렬 데이터 생성
        for i in range(len(valid_keywords)):
            for j in range(i + 1, len(valid_keywords)):
                idx1 = self.keyword2idx[valid_keywords[i]]
                idx2 = self.keyword2idx[valid_keywords[j]]
                pmi_value = self.calculate_pmi(valid_keywords[i], valid_keywords[j])
                
                if pmi_value != 0:  # 0이 아닌 값만 저장
                    rows.extend([idx1, idx2])
                    cols.extend([idx2, idx1])
                    data.extend([pmi_value, pmi_value])
        
        # 희소 행렬 생성
        return sparse.csr_matrix((data, (rows, cols)), 
                            shape=(self.matrix_size, self.matrix_size))

    def process_all_articles(self) -> List[sparse.csr_matrix]:
        """모든 기사에 대한 희소 행렬 리스트 생성"""
        print("Creating sparse matrices for all articles...")
        
        with open(self.news_data_path, 'r', encoding='utf-8') as f:
            news_data = json.load(f)
        
        matrices = []
        for article in tqdm(news_data):
            matrix = self.create_article_matrix(article['all_keywords'])
            matrices.append(matrix)
        
        print(f"\nSuccessfully processed {len(matrices)} articles")
        return matrices

    def save_processed_data(self, output_path: str) -> None:
        """처리된 데이터 저장"""
        data = {
            'keyword2idx': self.keyword2idx,
            'idx2keyword': self.idx2keyword,
            'keyword_counts': dict(self.keyword_counts),
            'pair_counts': {str(k): v for k, v in self.pair_counts.items()},
            'total_articles': self.total_articles
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            
        print(f"Processed data saved to {output_path}")