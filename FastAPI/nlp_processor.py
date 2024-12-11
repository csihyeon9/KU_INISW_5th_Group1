from koalanlp.proc import Tagger
from koalanlp import API
from koalanlp.Util import initialize, finalize
from koalanlp.types import POS
from collections import Counter, defaultdict
import kss
from typing import List, Dict, Tuple, Any
from keybert import KeyBERT
from transformers import AutoTokenizer, AutoModel
from sklearn.feature_extraction.text import TfidfVectorizer
import torch
from sentence_transformers import util
import json

_jvm_initialized = False


class NLPProcessor:
    def __init__(self):
        self.initialize_nlp()
        self.tokenizer = AutoTokenizer.from_pretrained("upskyy/kf-deberta-multitask")
        self.model = AutoModel.from_pretrained("upskyy/kf-deberta-multitask").to(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.keybert_model = KeyBERT("multi-qa-mpnet-base-cos-v1")
        self.tfidf_vectorizer = TfidfVectorizer(
            min_df=1,
            max_features=1000,
            token_pattern=r'(?u)\b\w+\b'
        )
        self.dictionary_file_path = "dictionary.json"
        self.dictionary = self.load_dictionary()
        self.economic_terms_cache = {}

    def initialize_nlp(self):
        global _jvm_initialized
        if not _jvm_initialized:
            try:
                initialize(
                    java_options="-Xmx4g -Dfile.encoding=UTF-8 --add-opens=java.base/java.util=ALL-UNNAMED --add-opens=java.base/java.lang=ALL-UNNAMED --add-opens=java.base/java.lang.reflect=ALL-UNNAMED",
                    DAON="LATEST"
                )
                _jvm_initialized = True
            except Exception as e:
                if "JVM cannot be initialized more than once" not in str(e):
                    raise e
        self.tagger = Tagger(API.DAON)

    def load_dictionary(self):
        try:
            with open(self.dictionary_file_path, 'r', encoding='utf-8') as file:
                return json.load(file)
        except FileNotFoundError:
            return {}

    def is_economic_term(self, term: str) -> bool:
        if term in self.economic_terms_cache:
            return self.economic_terms_cache[term]

        term_cleaned = term.replace(" ", "").lower()

        result = False
        if term_cleaned in self.dictionary:
            result = True
        else:
            for key in self.dictionary.keys():
                key_cleaned = key.split("(")[0].replace(" ", "").lower()
                if key_cleaned == term_cleaned:
                    result = True
                    break
                if "(" in key:
                    inside_parentheses = key.split("(")[1].rstrip(")").lower()
                    if any(item.strip() == term_cleaned for item in inside_parentheses.split(",")):
                        result = True
                        break

        self.economic_terms_cache[term] = result
        return result

    def extract_verbs_and_nouns(self, sentence: Any) -> Tuple[List[str], List[str]]:
        analyzed = self.tagger(sentence)
        verbs = []
        nouns = []
        for sent in analyzed:
            for word in sent:
                if hasattr(word, 'morphemes'):
                    base_form = None
                    endings = []
                    for morpheme in word.morphemes:
                        if morpheme.tag in {"VV", "VX"}:
                            base_form = morpheme.surface
                        elif morpheme.tag.startswith("EP") or morpheme.tag.startswith("EC") or morpheme.tag.startswith(
                                "EF"):
                            endings.append(morpheme.surface)
                        elif morpheme.tag in {"NNG", "NNP", "NNBC"}:
                            nouns.append(morpheme.surface)
                    if base_form:
                        combined_verb = base_form + "".join(endings)
                        verbs.append(combined_verb)
        return list(set(verbs)), list(set(nouns))

    def preprocess_text(self, text: str) -> Dict:
        # 1. TF-IDF 점수 미리 계산
        tfidf_matrix = self.tfidf_vectorizer.fit_transform([text])
        feature_names = self.tfidf_vectorizer.get_feature_names_out()
        tfidf_scores = dict(zip(feature_names, tfidf_matrix.toarray()[0]))

        # 2. KeyBERT 키워드 미리 추출
        keybert_results = dict(self.keybert_model.extract_keywords(text, top_n=50))

        # 3. DeBERTa text embedding 미리 계산
        text_encoding = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors='pt'
        ).to(self.model.device)

        with torch.no_grad():
            text_embedding = self.model(**text_encoding).last_hidden_state.mean(dim=1)

        return {
            'tfidf_scores': tfidf_scores,
            'keybert_results': keybert_results,
            'text_embedding': text_embedding
        }

    def calculate_node_importance(self, preprocessed_data: Dict, node_term: str) -> float:
        importance_scores = []

        # 1. TF-IDF 점수
        tfidf_score = preprocessed_data['tfidf_scores'].get(node_term, 0.0)
        importance_scores.append(tfidf_score)

        # 2. KeyBERT 점수
        keybert_score = preprocessed_data['keybert_results'].get(node_term, 0.0)
        importance_scores.append(keybert_score)

        # 3. DeBERTa 유사도
        try:
            term_encoding = self.tokenizer(
                node_term,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors='pt'
            ).to(self.model.device)

            with torch.no_grad():
                term_embedding = self.model(**term_encoding).last_hidden_state.mean(dim=1)
                similarity = util.pytorch_cos_sim(
                    preprocessed_data['text_embedding'],
                    term_embedding
                )[0][0].item()
            importance_scores.append(similarity)
        except Exception:
            pass

        # 4. 경제 용어 가중치
        if self.is_economic_term(node_term):
            importance_scores = [score * 1.5 for score in importance_scores]

        valid_scores = [score for score in importance_scores if score > 0]
        if not valid_scores:
            return 0.0

        final_importance = sum(valid_scores) / len(valid_scores)
        return round(final_importance, 3)

    def process_text(self, text: str, top_n: int = 5, max_pairs_per_verb: int = 3) -> Dict:
        # 전처리 데이터 미리 계산
        preprocessed_data = self.preprocess_text(text)

        sentences = list(kss.split_sentences(text))
        relationships = []
        nodes_counter = Counter()
        verb_counter = Counter()
        verb_pairs = defaultdict(Counter)

        # 경제 용어 노드 우선 수집
        economic_terms = set()
        for sentence in sentences:
            _, nouns = self.extract_verbs_and_nouns(sentence)
            for noun in nouns:
                if self.is_economic_term(noun):
                    economic_terms.add(noun)

        for sentence in sentences:
            verbs, nouns = self.extract_verbs_and_nouns(sentence)
            if len(nouns) < 2:
                continue

            filtered_verbs = [verb for verb in verbs if len(verb) >= 2]
            if not filtered_verbs:
                continue

            # 경제 용어가 포함된 문장 우선 처리
            has_economic_term = any(term in nouns for term in economic_terms)
            if has_economic_term:
                verb_counter.update(filtered_verbs)
                noun_pairs = [(nouns[i], nouns[j])
                              for i in range(len(nouns))
                              for j in range(i + 1, len(nouns))
                              if nouns[i] in economic_terms or nouns[j] in economic_terms]

                nodes_counter.update(nouns)

                for verb in filtered_verbs:
                    for noun1, noun2 in noun_pairs:
                        if noun1 != noun2:
                            pair = tuple(sorted([noun1, noun2]))
                            verb_pairs[verb][pair] += 2
            else:
                verb_counter.update(filtered_verbs)
                noun_pairs = [(nouns[i], nouns[j])
                              for i in range(len(nouns))
                              for j in range(i + 1, len(nouns))]

                nodes_counter.update(nouns)

                for verb in filtered_verbs:
                    for noun1, noun2 in noun_pairs:
                        if noun1 != noun2:
                            pair = tuple(sorted([noun1, noun2]))
                            verb_pairs[verb][pair] += 1

        top_verbs = dict(verb_counter.most_common(top_n))
        filtered_relationships = []

        for verb in top_verbs:
            pairs = verb_pairs[verb].most_common()
            economic_pairs = [(pair, count) for pair, count in pairs
                              if any(self.is_economic_term(term) for term in pair)]
            normal_pairs = [(pair, count) for pair, count in pairs
                            if not any(self.is_economic_term(term) for term in pair)]

            selected_pairs = (economic_pairs + normal_pairs)[:max_pairs_per_verb]

            for pair, _ in selected_pairs:
                filtered_relationships.append({
                    "verb": verb,
                    "keywords": list(pair)
                })

        nodes = []
        edges = []
        used_nodes = set()

        for rel in filtered_relationships:
            used_nodes.update(rel["keywords"])

        economic_nodes = set(term for term in used_nodes if self.is_economic_term(term))
        other_nodes = used_nodes - economic_nodes

        for term in sorted(economic_nodes):
            importance = self.calculate_node_importance(preprocessed_data, term)
            nodes.append({
                "id": term,
                "importance": importance,
                "is_economic": True
            })

        for term in sorted(other_nodes):
            importance = self.calculate_node_importance(preprocessed_data, term)
            nodes.append({
                "id": term,
                "importance": importance,
                "is_economic": False
            })

        for idx, rel in enumerate(filtered_relationships, 1):
            edges.append({
                "id": f"edge{idx}",
                "nodes": rel["keywords"],
                "description": rel["verb"],
                "importance": 1.0
            })

        return {
            "nodes": nodes,
            "edges": edges
        }

    @staticmethod
    def cleanup():
        global _jvm_initialized
        if _jvm_initialized:
            finalize()
            _jvm_initialized = False