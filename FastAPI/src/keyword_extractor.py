from sklearn.feature_extraction.text import TfidfVectorizer
from typing import List, Dict, Optional, Union
from kiwipiepy import Kiwi
from sentence_transformers import util
from transformers import AutoTokenizer, AutoModel
from keybert import KeyBERT
from pathlib import Path
import logging
import torch
import numpy as np
from collections import Counter
import re

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class KeywordExtractor:
    def __init__(self, article_path: str):
        """
        키워드 추출기 초기화

        Args:
            article_path (str): 기사 텍스트 파일 경로
        """
        self.article_path = Path(article_path)
        self.tokenizer = AutoTokenizer.from_pretrained("upskyy/kf-deberta-multitask")
        self.model = AutoModel.from_pretrained("upskyy/kf-deberta-multitask").to(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.kiwi = Kiwi()
        self.keybert_model = KeyBERT("multi-qa-mpnet-base-cos-v1")

    def extract_article_text(self) -> str:
        """
        텍스트 파일에서 제목과 본문을 추출

        Returns:
            str: 제목과 본문을 합친 텍스트
        """
        try:
            with open(self.article_path, 'r', encoding='utf-8') as file:
                lines = file.readlines()
            title = lines[0].strip() if lines else ""
            content = " ".join(line.strip() for line in lines[1:])
            return f"{title} {content}"
        except Exception as e:
            logger.error(f"텍스트 추출 중 오류 발생: {e}")
            return ""

    def extract_nouns(self, text: str) -> List[str]:
        """
        텍스트에서 명사 추출

        Args:
            text (str): 입력 텍스트

        Returns:
            List[str]: 추출된 명사 리스트
        """
        try:
            # Kiwi를 이용한 형태소 분석
            tokens = self.kiwi.analyze(text)[0][0]

            # 명사 추출 (NNG: 일반명사, NNP: 고유명사)
            nouns = [token[0] for token in tokens if token[1].startswith('NN')]
            return nouns

        except Exception as e:
            logger.error(f"명사 추출 중 오류 발생: {e}")
            return []

    def _extract_similarity_keywords(self, text: str, nouns: List[str], top_n: int) -> List[str]:
        """
        KF-DeBERTa를 이용한 유사도 기반 키워드 추출

        Args:
            text (str): 기사 텍스트
            nouns (List[str]): 추출된 명사 리스트
            top_n (int): 추출할 상위 키워드 개수

        Returns:
            List[str]: 추출된 키워드 리스트
        """
        try:
            # 텍스트 인코딩
            text_encoding = self.tokenizer(
                text,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors='pt'
            ).to(self.model.device)

            # 명사 인코딩
            noun_encodings = self.tokenizer(
                nouns,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors='pt'
            ).to(self.model.device)

            # 텍스트 임베딩 생성
            with torch.no_grad():
                text_embedding = self.model(**text_encoding).last_hidden_state.mean(dim=1)
                noun_embeddings = self.model(**noun_encodings).last_hidden_state.mean(dim=1)

            # 코사인 유사도 계산
            similarities = util.pytorch_cos_sim(text_embedding, noun_embeddings)[0]

            # 상위 키워드 선택
            top_indices = similarities.argsort(descending=True)[:top_n]
            return [nouns[idx] for idx in top_indices if len(nouns[idx]) > 1]

        except Exception as e:
            logger.error(f"유사도 기반 키워드 추출 중 오류 발생: {e}")
            return []

    def _extract_tfidf_keywords(self, text: str, top_n: int) -> List[str]:
        """
        TF-IDF를 이용한 키워드 추출

        Args:
            text (str): 기사 텍스트
            top_n (int): 추출할 상위 키워드 개수

        Returns:
            List[str]: 추출된 키워드 리스트
        """
        try:
            # 형태소 분석을 통한 명사 추출
            nouns = self.extract_nouns(text)
            noun_text = ' '.join(nouns)

            # TF-IDF 벡터라이저 초기화 및 적용
            vectorizer = TfidfVectorizer(
                min_df=1,
                max_features=1000,
                token_pattern=r'(?u)\b\w+\b'
            )
            tfidf_matrix = vectorizer.fit_transform([noun_text])

            # 단어별 TF-IDF 점수 계산
            feature_names = vectorizer.get_feature_names_out()
            scores = zip(feature_names, tfidf_matrix.toarray()[0])
            sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)

            # 상위 키워드 선택 (2글자 이상)
            keywords = [word for word, score in sorted_scores if len(word) > 1][:top_n]
            return keywords

        except Exception as e:
            logger.error(f"TF-IDF 키워드 추출 중 오류 발생: {e}")
            return []

    def extract_verbs_from_sentences(self, sentences: List[Dict[str, Union[str, List[str]]]]) -> List[
        Dict[str, Union[str, List[str]]]]:
        """
        각 문장에서 동사 추출

        Args:
            sentences (List[Dict[str, Union[str, List[str]]]]): 키워드 포함 문장 리스트

        Returns:
            List[Dict[str, Union[str, List[str]]]]: 각 문장에 포함된 동사 추가
        """
        results = []
        for item in sentences:
            sentence = item['sentence']
            keywords = item['keywords']

            # Kiwi를 이용해 동사 추출
            tokens = self.kiwi.analyze(sentence)[0][0]
            verbs = [token[0] for token in tokens if token[1].startswith('V')]

            # 결과 저장
            results.append({
                "sentence": sentence,
                "keywords": keywords,
                "verbs": verbs
            })
        return results

    def extract_sentences_with_keywords(self, text: str, keywords: List[str]) -> List[Dict[str, Union[str, List[str]]]]:
        """
        주어진 키워드가 포함된 문장 추출

        Args:
            text (str): 기사 본문 텍스트
            keywords (List[str]): 중복 제거된 키워드 리스트

        Returns:
            List[Dict[str, Union[str, List[str]]]]: 문장과 관련 키워드
        """
        # 문장 분리
        sentences = re.split(r'[.!?]\s*', text)

        # 키워드가 2개 이상 포함된 문장 필터링
        relevant_sentences = []
        for sentence in sentences:
            matched_keywords = [keyword for keyword in keywords if keyword in sentence]
            if len(matched_keywords) >= 2:
                relevant_sentences.append({
                    "sentence": sentence.strip(),
                    "keywords": matched_keywords
                })

        return relevant_sentences

    def extract_keywords(
            self,
            top_n: int = 10,
            methods: Optional[List[str]] = None
    ) -> Dict[str, Union[str, List[str]]]:
        """
        키워드 추출 및 키워드 포함 문장 추출

        Args:
            top_n (int, optional): 추출할 상위 키워드 개수
            methods (Optional[List[str]], optional): 사용할 키워드 추출 방법

        Returns:
            Dict[str, Union[str, List[str]]]: 추출된 키워드와 키워드 포함 문장
        """
        methods = methods or ['kf-deberta', 'keybert', 'tfidf']

        # 기사 텍스트와 명사 리스트 생성
        article_text = self.extract_article_text()
        nouns = self.extract_nouns(article_text)
        results = {}

        # 각 방법별 키워드 추출
        if 'kf-deberta' in methods:
            results['kf-deberta'] = self._extract_similarity_keywords(article_text, nouns, top_n)

        if 'keybert' in methods:
            keybert_keywords = [
                kw[0] for kw in self.keybert_model.extract_keywords(
                    article_text, top_n=top_n, stop_words=None
                )
            ]
            results['keybert'] = [kw for kw in self.extract_nouns(' '.join(keybert_keywords)) if len(kw) > 1]

        if 'tfidf' in methods:
            results['tfidf'] = self._extract_tfidf_keywords(article_text, top_n)

        # 전체 키워드 합치기
        all_keywords_raw = []
        for method_keywords in results.values():
            all_keywords_raw.extend(method_keywords)

        # 메인 키워드: 중복 제거 전 가장 빈도 높은 키워드
        main_keyword = Counter(all_keywords_raw).most_common(1)[0][0] if all_keywords_raw else None
        results['main_keyword'] = main_keyword

        # 중복 제거된 키워드 생성
        all_keywords_unique = list(set(all_keywords_raw))

        # 키워드 포함 문장 추출
        keyword_sentences = self.extract_sentences_with_keywords(article_text, all_keywords_unique)
        keyword_sentences = self.extract_verbs_from_sentences(keyword_sentences)  # 문장별 동사 추가
        results['keyword_sentences'] = keyword_sentences

        return results


def format_keywords_output(keywords_dict: Dict[str, Union[str, List[str]]]) -> str:
    """
    추출된 키워드와 키워드 포함 문장 포맷

    Args:
        keywords_dict (Dict[str, Union[str, List[str]]]): 키워드 결과

    Returns:
        str: 포맷된 출력
    """
    output = ["\n추출된 키워드:"]
    for method, keywords in keywords_dict.items():
        if method == 'main_keyword':
            output.append(f"\n- 메인 키워드: {keywords if keywords else '없음'}")
        elif method == 'keyword_sentences':
            output.append("\n- 키워드 포함 문장:")
            for item in keywords:
                output.append(f"  - 문장: {item['sentence']}")
                output.append(f"    포함된 키워드: {', '.join(item['keywords'])}")
                if 'verbs' in item:  # 동사 정보가 있을 경우 추가
                    output.append(f"    포함된 동사: {', '.join(item['verbs'])}")
        else:
            output.append(f"\n- {method.upper()}: {', '.join(keywords)}")
    return "\n".join(output)


def main():
    article_path = "./summaries/summary_20241211_191802.txt"  # 기사 파일 경로
    extractor = KeywordExtractor(article_path)

    # 키워드와 문장 추출
    keywords = extractor.extract_keywords()

    # 키워드 포함 문장에 동사 추출 추가
    if 'keyword_sentences' in keywords:
        keywords['keyword_sentences'] = extractor.extract_verbs_from_sentences(keywords['keyword_sentences'])

    # 결과 출력
    print(format_keywords_output(keywords))


if __name__ == "__main__":
    main()