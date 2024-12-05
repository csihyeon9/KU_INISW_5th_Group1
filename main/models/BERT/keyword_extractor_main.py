# deberta, keybert, tf-idf 결합한 키워드 추출 모듈 

from sklearn.feature_extraction.text import TfidfVectorizer
from typing import List, Dict, Optional
from bs4 import BeautifulSoup
from kiwipiepy import Kiwi
from sentence_transformers import util
from transformers import AutoTokenizer, AutoModel
from keybert import KeyBERT

import json, re, os
import pickle, torch
import numpy as np


class KeywordExtractor:
    def __init__(self, article_path: str, dictionary_path: str):
        """
        경제/금융 키워드 추출기 초기화

        Args:
            article_path (str): 기사 HTML 파일 경로
            dictionary_path (str): 경제/금융 용어 사전 JSON 파일 경로
        """
        self.article_path = article_path # 전달될 기사
        self.dictionary_path = dictionary_path
        self.embedding_cache_path = "financial_terms_embeddings.pkl"

        self.tokenizer = AutoTokenizer.from_pretrained("upskyy/kf-deberta-multitask")
        self.model = AutoModel.from_pretrained("upskyy/kf-deberta-multitask")

        self.kiwi = Kiwi()
        self.keybert_model = KeyBERT("multi-qa-mpnet-base-cos-v1")

        self.financial_terms = self.load_financial_dictionary()
        self.financial_terms_embeddings = self.load_or_create_embeddings()

    def load_financial_dictionary(self) -> List[str]:
        """
        경제/금융 용어 사전 로드
        """
        try:
            with open(self.dictionary_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
            terms = data.get('terms', [])
            cleaned_terms = list(set(re.sub(r'\(.*?\)', '', term).strip() for term in terms))
            print(f"총 {len(cleaned_terms)}개의 경제/금융 용어 로드 완료")
            return cleaned_terms
        except (json.JSONDecodeError, IOError) as e:
            print(f"사전 로딩 중 오류 발생: {e}")
            return []

    def load_or_create_embeddings(self) -> torch.Tensor:
        """
        기존 임베딩 캐시 로드 또는 새로 생성
        """
        if os.path.exists(self.embedding_cache_path):
            with open(self.embedding_cache_path, 'rb') as f:
                cached_data = pickle.load(f)
                if set(cached_data['terms']) == set(self.financial_terms):
                    print("기존 임베딩 캐시 사용")
                    return cached_data['embeddings']

        print("임베딩 생성 중...")
        embeddings = self.create_embeddings(self.financial_terms)
        with open(self.embedding_cache_path, 'wb') as f:
            pickle.dump({'terms': self.financial_terms, 'embeddings': embeddings}, f)
        return embeddings

    def extract_article_text(self) -> str:
        """
        기사 본문 텍스트 추출
        """
        try:
            with open(self.article_path, 'r', encoding='utf-8') as file:
                soup = BeautifulSoup(file.read(), 'html.parser')
            title = soup.title.string if soup.title else ""
            content = soup.find(id="dic_area").get_text(strip=True) if soup.find(id="dic_area") else ""
            return f"{title} {content}"
        except Exception as e:
            print(f"텍스트 추출 중 오류 발생: {e}")
            return ""

    def extract_nouns(self, text: str) -> List[str]:
        """
        Kiwi 형태소 분석기로 명사 추출
        """
        result = self.kiwi.analyze(text)
        return [token[0] for token in result[0][0] if token[1].startswith('N')]

    def create_embeddings(self, terms: List[str], batch_size: int = 128) -> torch.Tensor:
        """
        텍스트 임베딩 생성
        """
        if not terms:
            return torch.empty(0, self.model.config.hidden_size)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)

        embeddings = []
        for i in range(0, len(terms), batch_size):
            batch_terms = terms[i:i + batch_size]
            inputs = self.tokenizer(batch_terms, return_tensors="pt", padding=True, truncation=True).to(device)
            with torch.no_grad():
                outputs = self.model(**inputs)
            embeddings.append(outputs.last_hidden_state.mean(dim=1).cpu())
        torch.cuda.empty_cache()
        return torch.cat(embeddings)

    def extract_financial_keywords(
        self, 
        top_n: int = 10, 
        similarity_threshold: float = 0.6, 
        methods: Optional[List[str]] = None
    ) -> Dict[str, List[str]]:
        """
        경제/금융 키워드 추출

        Args:
            top_n (int, optional): 추출할 상위 키워드 개수
            similarity_threshold (float, optional): 유사도 임계값
            methods (Optional[List[str]], optional): 사용할 키워드 추출 방법

        Returns:
            Dict[str, List[str]]: 다양한 방법으로 추출된 키워드
        """
        methods = methods or ['kf-deberta', 'keybert', 'tfidf']
        
        # 기사 텍스트 및 명사 추출
        article_text = self.extract_article_text()
        nouns = self.extract_nouns(article_text)
        noun_embeddings = self.create_embeddings(nouns)

        result = {}

        if 'kf-deberta' in methods:
            result['kf-deberta_keywords'] = self._extract_similarity_keywords(
                noun_embeddings, nouns, top_n, similarity_threshold
            )

        if 'keybert' in methods:
            keybert_keywords = [
                kw[0] for kw in self.keybert_model.extract_keywords(
                    article_text, top_n=top_n, stop_words=None
                )
            ]
            # KeyBERT 키워드에서 명사만 추출
            result['keybert_keywords'] = [kw for kw in self.extract_nouns(' '.join(keybert_keywords)) if len(kw) > 1]

        if 'tfidf' in methods:
            result['tfidf_keywords'] = self._extract_tfidf_keywords(article_text, top_n)

        # 세 가지 방법에서 중복되는 키워드 제외
        all_keywords = set(result.get('kf-deberta_keywords', [])) | set(result.get('keybert_keywords', [])) | set(result.get('tfidf_keywords', []))
        
        # 최종 결과로 반환
        return {
            'all_keywords': list(all_keywords),
            **result
        }


    def _extract_similarity_keywords(self, noun_embeddings: torch.Tensor, nouns: List[str], top_n: int, similarity_threshold: float) -> List[str]:
        """
        유사도 기반 키워드 추출
        """
        if noun_embeddings.dim() == 1:
            noun_embeddings = noun_embeddings.unsqueeze(0)
        if self.financial_terms_embeddings.dim() == 1:
            self.financial_terms_embeddings = self.financial_terms_embeddings.unsqueeze(0)

        similar_keywords = []
        for i, noun_embedding in enumerate(noun_embeddings):
            scores = util.pytorch_cos_sim(noun_embedding.unsqueeze(0), self.financial_terms_embeddings)[0]
            # 용어의 길이가 1개 이상인 단어들에 대해서만 필터링
            if max(scores).item() > similarity_threshold and len(nouns[i]) > 1:
                similar_keywords.append(nouns[i])

        return sorted(similar_keywords, key=lambda x: len(x), reverse=True)[:top_n]

    def _extract_tfidf_keywords(self, article_text: str, top_n: int) -> List[str]:
        """
        TF-IDF 기반 키워드 추출
        """
        nouns = self.extract_nouns(article_text)
        filtered_nouns = [noun for noun in nouns if len(noun) > 1]
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform([' '.join(filtered_nouns)])
        terms = vectorizer.get_feature_names_out()
        scores = tfidf_matrix.toarray()[0]
        indices = np.argsort(scores)[::-1][:top_n]
        return [terms[idx] for idx in indices]


def format_keywords_output(keywords_dict: Dict[str, List[str]]) -> str:
    """
    Args:
        keywords_dict (dict): 키워드 추출 결과 딕셔너리

    Returns:
        str: 포맷된 키워드 출력 문자열
    """
    output = ["\n추출된 키워드:"]
    for method, keywords in keywords_dict.items():
        if method == "all_keywords":
            output.append("\n- 전체 키워드:")
        else:
            output.append(f"\n- {method.replace('_', ' ').capitalize()} 키워드:")
        unique_keywords = sorted(set(keywords))  # 중복 제거 및 정렬
        output.append(", ".join(unique_keywords))
    return "\n".join(output)

def main():
    """
    경제/금융 키워드 추출 메인 함수
    """
    # 기사와 사전 경로 설정
    article_path = "data/articles/sample1.html" # <- 요기서 새로운 기사 경로 입력하면 됩니당
    dictionary_path = "data/econ_dictionary.json" # <- 우리 3000개 데이터 + a (더 늘려야 할 듯)

    # 키워드 추출기 초기화
    extractor = KeywordExtractor(article_path, dictionary_path)

    # 키워드 추출
    keywords = extractor.extract_financial_keywords()

    # 결과 포맷팅 및 출력
    formatted_output = format_keywords_output(keywords)
    print(formatted_output)

if __name__ == '__main__':
    main()