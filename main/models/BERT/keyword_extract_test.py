import os
import json
import re
import pickle
import numpy as np
import torch
from typing import List, Dict
from kiwipiepy import Kiwi
from sentence_transformers import util
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoTokenizer, AutoModel
from keybert import KeyBERT
from tqdm import tqdm


# In[ ]:


class KeywordExtractor:
    def __init__(self, dictionary_path: str):
        """
        경제/금융 키워드 추출기 초기화
        """
        self.embedding_cache_path = "financial_terms_embeddings.pkl"

        # GPU 설정
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 모델 및 토크나이저 초기화
        self.tokenizer = AutoTokenizer.from_pretrained("upskyy/kf-deberta-multitask")
        self.model = AutoModel.from_pretrained("upskyy/kf-deberta-multitask").to(self.device)

        self.kiwi = Kiwi()
        self.keybert_model = KeyBERT("multi-qa-mpnet-base-cos-v1")

        self.financial_terms = self.load_financial_dictionary(dictionary_path)
        self.financial_terms_embeddings = self.load_or_create_embeddings()

    def load_financial_dictionary(self, dictionary_path: str) -> List[str]:
        """
        경제/금융 용어 사전 로드
        """
        try:
            with open(dictionary_path, 'r', encoding='utf-8') as file:
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
        Pickle 임베딩 캐시 로드 또는 생성
        """
        if os.path.exists(self.embedding_cache_path):
            with open(self.embedding_cache_path, 'rb') as f:
                cached_data = pickle.load(f)
                if set(cached_data['terms']) == set(self.financial_terms):
                    print("기존 임베딩 캐시 사용")
                    return torch.tensor(cached_data['embeddings'])

        print("임베딩 생성 중...")
        embeddings = self.create_embeddings(self.financial_terms)
        
        with open(self.embedding_cache_path, 'wb') as f:
            pickle.dump({
                'terms': self.financial_terms, 
                'embeddings': embeddings.numpy()
            }, f)
        return embeddings

    def create_embeddings(self, terms: List[str], batch_size: int = 128) -> torch.Tensor:
        """
        GPU 임베딩 생성
        """
        if not terms:
            return torch.empty(0, self.model.config.hidden_size)

        embeddings = []
        for i in range(0, len(terms), batch_size):
            batch_terms = terms[i:i + batch_size]
            inputs = self.tokenizer(batch_terms, return_tensors="pt", padding=True, truncation=True).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                batch_embeddings = outputs.last_hidden_state.mean(dim=1).cpu()
            
            embeddings.append(batch_embeddings)
        
        return torch.cat(embeddings)

    def extract_nouns(self, text: str) -> List[str]:
        """
        Kiwi 형태소 분석기로 명사 추출
        """
        result = self.kiwi.analyze(text)
        return [token[0] for token in result[0][0] if token[1].startswith('N')]

    def extract_keywords_from_context(self, json_file_path: str, batch_size: int = 100) -> List[Dict]:
        """
        JSON 파일에서 키워드 추출
        """
        with open(json_file_path, 'r', encoding='utf-8') as file:
            articles = json.load(file)

        valid_articles = [article for article in articles if article.get("content", "").strip()]
        print(f"총 {len(articles)}개의 기사 중 {len(valid_articles)}개의 유효한 content가 있습니다.")

        results = []
        for i in tqdm(range(0, len(valid_articles), batch_size), desc="기사 처리 중"):
            batch_articles = valid_articles[i:i+batch_size]
            batch_results = []

            for article in batch_articles:
                title = article.get("title", "unknown")
                content = article.get("content", "")
                date = article.get("time", "unknown")

                # 명사 추출
                content_nouns = self.extract_nouns(content)
                content_noun_embeddings = self.create_embeddings(content_nouns)

                # 키워드 추출 방법들
                kf_deberta_keywords = self._extract_similarity_keywords(
                    content_noun_embeddings, content_nouns, top_n=10, similarity_threshold=0.6
                )
                
                keybert_keywords = [
                    kw[0] for kw in self.keybert_model.extract_keywords(content, top_n=10, stop_words=None)
                ]
                keybert_keywords = [kw for kw in self.extract_nouns(' '.join(keybert_keywords)) if len(kw) > 1]
                
                tfidf_keywords = self._extract_tfidf_keywords(content, top_n=10)

                # 키워드 통합
                all_keywords = list(set(kf_deberta_keywords + keybert_keywords + tfidf_keywords))

                batch_results.append({
                    "title": title,
                    "content": content,
                    "date": date,
                    "all_keywords": all_keywords
                })

            results.extend(batch_results)

            # 배치별 샘플 로그 출력
            if batch_results:
                sample_result = batch_results[0]
                print(f"\n샘플 결과 (배치 {i//batch_size + 1}):")
                print(f"  제목: {sample_result['title']}")
                print(f"  날짜: {sample_result['date']}")
                print(f"  키워드: {sample_result['all_keywords']}")

        print("\n모든 기사가 처리되었습니다.")
        return results

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
    # JSON 파일과 사전 경로 설정
    json_file_path = "/home/jhk/Desktop/hankyung_financial-market_pages_101_to_200.json"
    dictionary_path = "/home/jhk/Desktop/econ_dictionary.json"
    output_folder = "/home/jhk/Desktop/output_keywords"

    # 키워드 추출기 초기화
    extractor = KeywordExtractor(dictionary_path)

    # 키워드 추출 실행
    results = extractor.extract_keywords_from_context(json_file_path, batch_size=100)

    # 최종 결과 저장
    output_file_path = os.path.join(output_folder, "final_keywords.json")
    os.makedirs(output_folder, exist_ok=True)
    
    with open(output_file_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    print(f"최종 키워드 결과가 {output_file_path}에 저장되었습니다.")

if __name__ == "__main__":
    main()




