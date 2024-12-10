# nlp_processor.py
import os
import json
from typing import List, Dict
from collections import Counter

from src.file_loader import load_article
from src.sentence_splitter import split_sentences
from src.term_extractor import extract_terms
from src.sentence_filter import filter_sentences_by_term_count
from src.relation_extractor import extract_verbs, load_custom_verbs, initialize_jvm


class NLPProcessor:
    def __init__(self):
        self.json_dir = "data/json/"
        self.article_terms_path = os.path.join(self.json_dir, "article_terms.json")
        self.yes_verbs_path = os.path.join(self.json_dir, "yes_verb.json")
        self.not_verbs_path = os.path.join(self.json_dir, "not_verb.json")
        self.verb_mapping_path = os.path.join(self.json_dir, "verb_mapping.json")

        self.article_terms = self._load_json_list(self.article_terms_path, "terms")
        self.yes_verbs = self._load_json_list(self.yes_verbs_path, "yes_verbs")
        self.not_verbs = self._load_json_list(self.not_verbs_path, "not_verbs")
        self.verb_mapping = self._load_verb_mapping(self.verb_mapping_path)

        initialize_jvm()
        load_custom_verbs(self.yes_verbs_path)

    def _load_json_list(self, json_path: str, key: str) -> List[str]:
        try:
            with open(json_path, "r", encoding="utf-8") as file:
                data = json.load(file)
            return data.get(key, [])
        except FileNotFoundError:
            print(f"File not found: {json_path}")
            return []
        except json.JSONDecodeError:
            print(f"Invalid JSON format: {json_path}")
            return []

    def _load_verb_mapping(self, json_path: str) -> str:
        """
        Load a mapping of verbs for normalization.
        """
        try:
            with open(json_path, "r", encoding="utf-8") as file:
                return json.load(file)
        except FileNotFoundError:
            print(f"Mapping file not found: {json_path}")
            return {}
        except json.JSONDecodeError:
            print(f"Invalid JSON format: {json_path}")
            return {}

    def process_text(self, text: str) -> Dict:
        """
        텍스트를 분석하여 하이퍼그래프 데이터 구조를 반환합니다.
        """
        # 문장 분리 및 용어 추출
        sentences = split_sentences(text)

        sentence_with_terms = [
            (sentence, extract_terms(sentence, self.article_terms)) for sentence in sentences
        ]

        filtered_sentences = filter_sentences_by_term_count(
            sentence_with_terms,
            min_count=2,
            max_count=2
        )

        # 하이퍼그래프 데이터 구조 생성
        nodes = []
        edges = []
        all_terms = Counter()

        # 모든 용어를 수집하고 빈도수 계산
        for _, terms in filtered_sentences:
            all_terms.update(terms)

        # 노드 생성 (용어별 중요도 계산)
        for term, freq in all_terms.items():
            importance = round(10 ** -freq, 10)  # 빈도수가 높을수록 중요도가 낮아짐
            nodes.append({
                "id": term,
                "importance": importance
            })

        # 엣지 생성
        for sentence, terms in filtered_sentences:
            verbs = extract_verbs(sentence, self.not_verbs, self.yes_verbs, self.verb_mapping)
            top_verbs = [verb for verb, _ in Counter(verbs).most_common(2)]

            edge_id = f"edge{len(edges) + 1}"
            edges.append({
                "id": edge_id,
                "nodes": terms,
                "description": ", ".join(top_verbs) if top_verbs else "관계 없음",
                "importance": 1.0  # 기본 중요도
            })

        return {
            "nodes": nodes,
            "edges": edges
        }