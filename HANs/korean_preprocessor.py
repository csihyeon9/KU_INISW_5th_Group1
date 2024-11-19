# import pandas as pd
# import numpy as np
# from typing import List, Dict, Tuple, Set
# import torch
# from transformers import AutoTokenizer, AutoModel
# from sklearn.preprocessing import normalize
# import json
# from pathlib import Path
# from tqdm import tqdm
# import re
# from konlpy.tag import Mecab  # 한국어 형태소 분석기

# class KoreanFinancialPreprocessor:
#     def __init__(
#         self,
#         pretrained_model_name: str = "klue/bert-base",  # 한국어 BERT 모델
#         max_length: int = 64,
#         device: str = None
#     ):
#         self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
#         self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)
#         self.model = AutoModel.from_pretrained(pretrained_model_name).to(self.device)
#         self.max_length = max_length
#         self.mecab = Mecab()  # 형태소 분석기 초기화
        
#         # 한국 경제/금융 뉴스에 특화된 관계 유형 정의
#         self.relation_types = [
#             # 정책 관련 관계
#             "정책영향",        # 정부/중앙은행 정책이 시장에 미치는 영향
#             "규제대상",        # 금융 규제의 대상이 되는 관계
#             "감독조치",        # 금융감독 관련 조치
            
#             # 시장 영향력 관계
#             "가격영향",        # 가격 변동에 영향을 주는 관계
#             "수급관계",        # 수요와 공급 관련 관계
#             "투자심리",        # 투자 심리에 영향을 주는 관계
            
#             # 산업/기업 관계
#             "산업연관",        # 산업 간 연관 관계
#             "기업실적",        # 기업 실적 관련 관계
#             "경쟁구도",        # 기업/산업 간 경쟁 관계
            
#             # 거시경제 관계
#             "경기동행",        # 경기와 함께 움직이는 관계
#             "경기선행",        # 경기에 선행하는 관계
#             "경기후행",        # 경기에 후행하는 관계
            
#             # 리스크 관계
#             "시장리스크",      # 시장 위험 관련 관계
#             "신용리스크",      # 신용 위험 관련 관계
#             "규제리스크",      # 규제 위험 관련 관계
            
#             # 대외 관계
#             "수출입영향",      # 수출입 관련 영향 관계
#             "환율영향",        # 환율 변동 영향 관계
#             "대외의존",        # 대외 의존성 관계
            
#             # 구조적 관계
#             "가치사슬",        # 가치 사슬 관계
#             "소유구조",        # 소유/지배 구조 관계
            
#             # 파급 관계
#             "수익성영향",      # 수익성에 미치는 영향
#             "비용구조",        # 비용 구조 관련 관계
#             "자금흐름"         # 자금 흐름 관련 관계
#         ]
    
#     def preprocess_news_articles(
#         self,
#         articles: List[Dict],
#         save_path: str
#     ) -> None:
#         """뉴스 기사 전처리"""
        
#         # 1. 키워드 추출
#         keywords = self._extract_keywords(articles)
        
#         # 2. 관계 추출
#         relations = self._extract_relations(articles, keywords)
        
#         # 3. 임베딩 생성
#         embeddings = self._generate_embeddings(keywords)
        
#         # 4. 데이터 저장
#         processed_data = {
#             "keywords": list(keywords),
#             "relations": relations,
#             "embeddings": embeddings.tolist(),
#             "relation_types": self.relation_types
#         }
        
#         save_path = Path(save_path)
#         save_path.parent.mkdir(parents=True, exist_ok=True)
        
#         with open(save_path, 'w', encoding='utf-8') as f:
#             json.dump(processed_data, f, ensure_ascii=False, indent=2)
    
#     def _extract_keywords(self, articles: List[Dict]) -> Set[str]:
#         """뉴스 기사에서 경제/금융 키워드 추출"""
#         keywords = set()
        
#         for article in tqdm(articles, desc="Extracting keywords"):
#             # 형태소 분석
#             morphs = self.mecab.pos(article['content'])
            
#             # 명사 추출 (NNG: 일반명사, NNP: 고유명사)
#             nouns = [word for word, pos in morphs if pos in ['NNG', 'NNP']]
            
#             # 복합명사 처리 (예: "기준금리", "물가상승률")
#             compound_nouns = self._extract_compound_nouns(article['content'])
            
#             keywords.update(nouns + compound_nouns)
        
#         # 금융/경제 도메인 필터링
#         filtered_keywords = self._filter_financial_keywords(keywords)
        
#         return filtered_keywords
    
#     def _extract_compound_nouns(self, text: str) -> List[str]:
#         """복합명사 추출"""
#         # 금융/경제 복합명사 패턴
#         patterns = [
#             r'[가-힣]+(금리|물가|지수|시장|정책|공시|공급|수요|실적|전망)',
#             r'[가-힣]+(주가|채권|환율|증시|펀드|주식|채용|실업|성장)',
#             r'[가-힣]+(가격|구조|지표|동향|추세|전망|예측|분석|조정)'
#         ]
        
#         compound_nouns = []
#         for pattern in patterns:
#             matches = re.finditer(pattern, text)
#             compound_nouns.extend([m.group() for m in matches])
        
#         return compound_nouns
    
#     def _filter_financial_keywords(self, keywords: Set[str]) -> Set[str]:
#         """금융/경제 관련 키워드 필터링"""
#         # 금융/경제 도메인 사전 (실제 구현시 더 광범위한 사전 필요)
#         financial_terms = {
#             '금리', '환율', '주가', '물가', '증시', '펀드', '채권', '주식',
#             '통화', '정책', '은행', '보험', '투자', '시장', '경기', '실적',
#             '성장', '수출', '수입', '무역', '산업', '기업', '소비', '생산',
#             '고용', '실업', '임금', '부동산', '원자재', '에너지'
#         }
        
#         return {k for k in keywords if any(
#             term in k for term in financial_terms
#         )}
    
#     def _extract_relations(
#         self,
#         articles: List[Dict],
#         keywords: Set[str]
#     ) -> List[Dict]:
#         """뉴스 기사에서 관계 추출"""
#         relations = []
        
#         for article in tqdm(articles, desc="Extracting relations"):
#             # 문장별 처리
#             sentences = self._split_sentences(article['content'])
            
#             for sentence in sentences:
#                 # 문장 내 키워드 찾기
#                 sentence_keywords = [
#                     k for k in keywords
#                     if k in sentence
#                 ]
                
#                 if len(sentence_keywords) >= 2:
#                     # 관계 유형 판별
#                     relation_type = self._determine_relation_type(
#                         sentence,
#                         sentence_keywords
#                     )
                    
#                     if relation_type:
#                         relation = {
#                             "keywords": sentence_keywords,
#                             "type": relation_type,
#                             "weight": 1.0,
#                             "source": {
#                                 "article_id": article.get('id'),
#                                 "date": article.get('date'),
#                                 "sentence": sentence
#                             }
#                         }
#                         relations.append(relation)
        
#         return relations
    
#     def _split_sentences(self, text: str) -> List[str]:
#         """문장 분리"""
#         # 기본적인 문장 구분
#         sentences = re.split(r'[.!?]\s+', text)
#         return [s.strip() for s in sentences if s.strip()]
    
#     def _determine_relation_type(
#         self,
#         sentence: str,
#         keywords: List[str]
#     ) -> Optional[str]:
#         """문장 내 키워드 간 관계 유형 판별"""
#         # 관계 유형 판별을 위한 패턴
#         patterns = {
#             "정책영향": r'(정책|조치|결정).*(영향|효과)',
#             "가격영향": r'(가격|시세).*(상승|하락|변동)',
#             "산업연관": r'(산업|업종).*(연관|관련|영향)',
#             "수급관계": r'(수요|공급|수급).*(영향|변화|전망)',
#             "경기동행": r'(경기|景氣).*(함께|동행|같이)',
#             # ... 더 많은 패턴 추가 가능
#         }
        
#         for rel_type, pattern in patterns.items():
#             if re.search(pattern, sentence):
#                 return rel_type
        
#         # 기본값으로 가장 적절한 관계 유형 반환
#         return self._fallback_relation_type(sentence, keywords)
    
#     def _fallback_relation_type(
#         self,
#         sentence: str,
#         keywords: List[str]
#     ) -> str:
#         """기본 관계 유형 결정"""
#         # 간단한 규칙 기반 판별
#         if any(w in sentence for w in ['정책', '규제', '감독']):
#             return "정책영향"
#         elif any(w in sentence for w in ['시장', '가격', '시세']):
#             return "가격영향"
#         elif any(w in sentence for w in ['산업', '업종', '분야']):
#             return "산업연관"
#         # 기본값
#         return "산업연관"