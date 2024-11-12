from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from ekonlpy import Mecab as EKoMecab
from eunjeon import Mecab as EunjeonMecab
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from typing import List, Tuple, Dict
import logging
import multiprocessing
import json
import re
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

class HybridEconomicTermExtractor:
    def __init__(self, vector_size: int = 100, window: int = 5, min_count: int = 2,
                 epochs: int = 300, seed_terms: List[str] = None):
        """
        Doc2Vec과 KF-DeBERTa를 결합한 하이브리드 경제 용어 추출기
        
        Args:
            vector_size: Doc2Vec 벡터 차원 수
            window: Doc2Vec 컨텍스트 윈도우 크기
            min_count: 최소 단어 출현 빈도
            epochs: Doc2Vec 학습 반복 횟수
            seed_terms: 초기 경제 용어 시드 리스트
        """
        # Mecab 초기화
        self.mecab = EKoMecab()
        self.mecab2 = EunjeonMecab()
        
        # Doc2Vec 관련 설정
        self.doc2vec_model = None
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.epochs = epochs
        
        # KF-DeBERTa 모델 초기화
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained("kakaobank/kf-deberta-base")
        self.deberta_model = AutoModel.from_pretrained("kakaobank/kf-deberta-base").to(self.device)
        
        # 시드 용어 로드
        with open('data_only_words.json', 'r', encoding='utf-8') as f:
            self.seed_terms = json.load(f)
        
        # 로깅 설정
        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
        
        # 시드 용어의 임베딩 캐시
        self.seed_term_embeddings = {}
        
    def preprocess_text(self, text: str) -> List[str]:
        """텍스트 전처리 및 형태소 분석"""
        morphs = self.mecab.pos(text)
        valid_tags = ['NNG', 'NNP', 'SL']
        
        words = []
        skip_previous_noun = False

        for i, (word, tag) in enumerate(morphs):
            if tag.startswith('VV'):
                if skip_previous_noun:
                    words.pop()
                skip_previous_noun = False
            elif tag in valid_tags and len(word) > 1:
                words.append(word)
                skip_previous_noun = True
            else:
                skip_previous_noun = False

        unique_words = list(set(words))
        return unique_words

    def get_deberta_embedding(self, text: str) -> torch.Tensor:
        """KF-DeBERTa를 사용하여 텍스트의 임베딩 추출"""
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.deberta_model(**inputs)
            # [CLS] 토큰의 임베딩 사용
            embedding = outputs.last_hidden_state[:, 0, :]
        
        return embedding.cpu()

    def prepare_training_data(self, articles: List[str]) -> List[TaggedDocument]:
        """Doc2Vec 학습 데이터 준비"""
        tagged_data = []
        for idx, article in enumerate(articles):
            words = self.preprocess_text(article)
            tagged_data.append(TaggedDocument(words=words, tags=[f'doc_{idx}']))
        return tagged_data

    def train_model(self, articles: List[str]):
        """Doc2Vec 모델 학습 및 시드 용어 임베딩 준비"""
        # Doc2Vec 모델 학습
        tagged_data = self.prepare_training_data(articles)
        
        self.doc2vec_model = Doc2Vec(
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            epochs=self.epochs,
            workers=multiprocessing.cpu_count()
        )
        
        self.doc2vec_model.build_vocab(tagged_data)
        self.doc2vec_model.train(
            tagged_data,
            total_examples=self.doc2vec_model.corpus_count,
            epochs=self.doc2vec_model.epochs
        )
        
        # 시드 용어의 DeBERTa 임베딩 미리 계산
        print("시드 용어의 DeBERTa 임베딩 계산 중...")
        for term in tqdm(self.seed_terms):
            self.seed_term_embeddings[term] = self.get_deberta_embedding(term)

    def calculate_hybrid_similarity(self, word: str, seed_term: str) -> float:
        """Doc2Vec과 DeBERTa 유사도를 결합하여 계산"""
        # Doc2Vec 유사도 계산
        doc2vec_sim = 0.0
        if word in self.doc2vec_model.wv and seed_term in self.doc2vec_model.wv:
            doc2vec_sim = self.doc2vec_model.wv.similarity(word, seed_term)
        
        # DeBERTa 유사도 계산
        word_embedding = self.get_deberta_embedding(word)
        seed_embedding = self.seed_term_embeddings[seed_term]
        deberta_sim = torch.cosine_similarity(word_embedding, seed_embedding).item()
        
        # 가중치를 적용한 하이브리드 유사도 계산 (여기서는 동일 가중치 사용)
        hybrid_sim = (doc2vec_sim + deberta_sim) / 2
        
        return hybrid_sim

    def extract_economic_terms(self, text: str, threshold: float = 0.5) -> Dict[str, float]:
        """하이브리드 방식으로 경제 용어 추출"""
        if self.doc2vec_model is None:
            raise ValueError("모델이 학습되지 않았습니다.")
        
        words = self.preprocess_text(text)
        economic_terms = {}
        
        for word in words:
            max_similarity = 0
            for seed_term in self.seed_terms:
                similarity = self.calculate_hybrid_similarity(word, seed_term)
                max_similarity = max(max_similarity, similarity)
            
            if max_similarity > threshold:
                economic_terms[word] = max_similarity
        
        return dict(sorted(economic_terms.items(), key=lambda x: x[1], reverse=True))

    def analyze_article(self, article: str, threshold: float = 0.5) -> Dict:
        """기사 분석"""
        economic_terms = self.extract_economic_terms(article, threshold)
        
        context_analysis = {}
        for term in economic_terms:
            similar_terms = []
            # Doc2Vec 유사어 검색
            if term in self.doc2vec_model.wv:
                similar_terms = self.doc2vec_model.wv.most_similar(term, topn=5)
            
            # DeBERTa 기반 문맥 분석 추가
            term_embedding = self.get_deberta_embedding(term)
            context_words = self.preprocess_text(article)
            
            deberta_similar_terms = []
            for context_word in context_words:
                if context_word != term:
                    context_embedding = self.get_deberta_embedding(context_word)
                    similarity = torch.cosine_similarity(term_embedding, context_embedding).item()
                    deberta_similar_terms.append((context_word, similarity))
            
            # 상위 5개 DeBERTa 유사어 선택
            deberta_similar_terms = sorted(deberta_similar_terms, key=lambda x: x[1], reverse=True)[:5]
            
            # Doc2Vec과 DeBERTa 결과 통합
            context_analysis[term] = {
                'doc2vec_similar': similar_terms,
                'deberta_similar': deberta_similar_terms
            }
        
        return {
            'economic_terms': economic_terms,
            'context_analysis': context_analysis
        }
    
    def load_training_articles(self, file_path: str) -> List[str]:
        """학습 데이터 로드"""
        try:
            df = pd.read_csv(file_path, usecols=['title', 'main'], encoding='utf-8')
        except UnicodeDecodeError:
            df = pd.read_csv(file_path, usecols=['title', 'main'], encoding='euc-kr')

        def clean_text(text):
            text = re.sub(r'\b\w+\.(com|co\.kr)\b', '', text)
            text = re.sub(r'\b\d{2,3}[-.\)]?\d{3,4}[-.]\d{4}\b', '', text)
            return text

        df['main'] = df['main'].apply(clean_text)
        training_articles = df.apply(lambda row: f"{row['title']} {row['main']}", axis=1).tolist()
        return training_articles

def main():
    # 추출기 초기화 및 학습
    file_path = 'article_df.csv'
    extractor = HybridEconomicTermExtractor()
    
    # 데이터 로드 및 모델 학습
    print("학습 데이터 로드 중...")
    training_articles = extractor.load_training_articles(file_path)
    
    print("모델 학습 중...")
    extractor.train_model(training_articles)
    
    # 테스트 기사
    test_article = """
    대출받을 예정인 30대 직장인 A씨는 매일 뱅크샐러드 애플리케이션(앱)에 들어가 대출금리 쿠폰을 받은 뒤, 쿠폰을 합쳐 ‘강화’하고 있다. 대출금리를 0.1% 할인해주는 쿠폰을 다른 0.1% 쿠폰과 합쳐 0.2% 쿠폰으로 만드는 식이다. 대출금리 할인쿠폰 랭킹을 보면 어떤 사람이 높은 쿠폰을 만들 수 있는지 확인할 수 있기 때문에 경쟁심리도 생겼다.

주요 핀테크 업체들이 온라인 대출 서비스 확대에 나서고 있다. 시중은행부터 제2금융권까지 한 번에 빠르게 조회할 수 있고, 비대면으로 대출받을 수 있어 편리하다. 디지털에 익숙한 MZ세대를 중심으로 관련 서비스를 이용하고 있지만, 재테크 카페를 중심으로 입소문이 나면서 많은 고객이 이용하고 있다.

10일 금융권에 따르면 핀테크 업체들은 대출 고객을 확보하기 위해 다양한 마케팅을 펼치고 있는 것으로 나타났다. 이들 업체들은 여러 은행의 대출을 한번에 조회해주고, 수수료를 받고 있는데 매출에서 큰 비중을 차지하는 것으로 분석됐다. 뱅크샐러드 대출 쿠폰은 뱅크샐러드에서 대출 실행 시 금리를 할인해주는 쿠폰으로, 고객은 보유한 쿠폰의 할인율만큼 금리를 낮출 수 있다. 특히 쿠폰에 ‘강화’ 기능의 게임 요소를 추가해 재미를 더했다. 친구에게 쿠폰을 1장 보내면 2장 받을 수 있어 활발히 공유가 이뤄지기도 한다.

뱅크샐러드는 대출 쿠폰 출시 1년 만에 대출 중개 건수가 629% 상승한 것으로 집계했다. 대출 쿠폰에서 만들어진 가장 높은 할인율은 2.8%로 쿠폰을 가장 많이 강화한 사람의 강화 횟수는 총 1266회였다. 대출 쿠폰이 적용된 가장 큰 대출 금액은 8억4500만원으로 집계됐다. 최대 캐시백 금액은 201만원으로 쿠폰으로 이만큼을 아낄 수 있었다는 뜻이다.

대출을 받으면 1년 치 이자를 한 번에 받을 수 있는 행사도 있다. 토스도 대출받기 이자 지원 이벤트를 펼치고 있다. 토스 애플리케이션(앱)을 통해 신용대출 혹은 마이너스 통장 대출받을 경우 매달 추첨을 통해 최대 5000만원까지 1년 치 이자를 받을 수 있다. 카카오페이도 대출받은 고객 5명을 추첨해 최대 1억원 대출금에 대한 1년 치 이자금을 카카오페이포인트로 지급한다. 대출 한도만 조회를 완료해도 카카오페이포인트를 최대 10만포인트까지 증정하는 ‘럭키머니건’ 이벤트도 진행 중이다.

네이버페이는 신용대출비교 서비스에서 첫 달 최대 15만원을 지원해주는 행사를 진행하고 있다. 네이버페이를 통해 대출받을 경우 대출 금리가 연 10% 이상이면 최대 10만원까지 4%를 적립해주고, 금리가 10% 미만이면 최대 5만원까지 0.2% 적립해준다. 신용점수를 확인하면 최대 3% 이자지원금을 5만원까지 받을 수 있다.
    """
    
    print("테스트 기사 분석 중...")
    result = extractor.analyze_article(test_article)
    
    # 결과 출력
    print("\n=== 추출된 경제 용어 ===")
    for term, score in result['economic_terms'].items():
        print(f"{term}: {score:.4f}")
    
    print("\n=== 용어 컨텍스트 분석 ===")
    for term, context in result['context_analysis'].items():
        print(f"\n{term}의 관련 용어:")
        print("Doc2Vec 유사어:")
        for word, sim in context['doc2vec_similar']:
            print(f"  - {word}: {sim:.4f}")
        print("DeBERTa 유사어:")
        for word, sim in context['deberta_similar']:
            print(f"  - {word}: {sim:.4f}")
    
    return result

if __name__ == "__main__":
    result = main()