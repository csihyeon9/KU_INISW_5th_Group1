import os
import json
import re
import pickle
import numpy as np
import torch

from typing import List, Dict, Optional
from kiwipiepy import Kiwi
from sentence_transformers import util
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoTokenizer, AutoModel
from keybert import KeyBERT

class KeywordExtractor:
    def __init__(self, json_folder_path: str, dictionary_path: str):
        """
        경제/금융 키워드 추출기 초기화
        """
        self.json_folder_path = json_folder_path
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

    def extract_article_text_per_article(self, json_file_path: str) -> List[Dict[str, str]]:
        """
        JSON 파일에서 각 기사의 제목과 내용을 별도로 추출
        """
        try:
            with open(json_file_path, 'r', encoding='utf-8') as file:
                articles = json.load(file)

            if isinstance(articles, list):
                return [{"title": article.get("title", ""), "content": article.get("content", "")} for article in articles]
            return []
        except Exception as e:
            print(f"기사별 텍스트 추출 중 오류 발생 ({json_file_path}): {e}")
            return []

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

    def extract_keywords_for_articles(self, json_file_path: str, output_folder: str):
        """
        JSON 파일 내 각 기사별로 키워드를 추출하고 JSON으로 저장
        """
        # 기사를 개별적으로 추출
        articles = self.extract_article_text_per_article(json_file_path)

        # 결과를 저장할 리스트
        results = []

        for i, article in enumerate(articles):
            article_text = f"{article['title']} {article['content']}"
            nouns = self.extract_nouns(article_text)
            noun_embeddings = self.create_embeddings(nouns)

            # 각 방법별 키워드 추출
            kf_deberta_keywords = self._extract_similarity_keywords(noun_embeddings, nouns, top_n=10, similarity_threshold=0.6)
            keybert_keywords = [
                kw[0] for kw in self.keybert_model.extract_keywords(article_text, top_n=10, stop_words=None)
            ]
            keybert_keywords = [kw for kw in self.extract_nouns(' '.join(keybert_keywords)) if len(kw) > 1]
            tfidf_keywords = self._extract_tfidf_keywords(article_text, top_n=10)

            all_keywords = list(set(kf_deberta_keywords + keybert_keywords + tfidf_keywords))

            # 각 기사별 키워드 저장
            results.append({
                "article_id": i,
                "title": article["title"],
                # "kf_deberta_keywords": kf_deberta_keywords,
                # "keybert_keywords": keybert_keywords,
                # "tfidf_keywords": tfidf_keywords,
                "all_keywords": all_keywords
            })

        # JSON으로 저장
        os.makedirs(output_folder, exist_ok=True)
        output_file_path = os.path.join(output_folder, os.path.basename(json_file_path).replace('.json', '_keywords.json'))

        with open(output_file_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=4)

        print(f"기사별 키워드가 {output_file_path}에 저장되었습니다.")

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

def main():
    """
    경제/금융 키워드 추출 메인 함수
    """
    # JSON 파일들이 있는 폴더와 사전 경로 설정
    json_folder_path = "json_data"
    dictionary_path = "data/econ_dictionary.json"
    output_folder = "output_keywords"

    # 키워드 추출기 초기화
    extractor = KeywordExtractor(json_folder_path, dictionary_path)

    # JSON 파일 처리
    json_files = [f for f in os.listdir(json_folder_path) if f.endswith('.json')]
    for json_file in json_files:
        json_file_path = os.path.join(json_folder_path, json_file)
        extractor.extract_keywords_for_articles(json_file_path, output_folder)

if __name__ == '__main__':
    main()
