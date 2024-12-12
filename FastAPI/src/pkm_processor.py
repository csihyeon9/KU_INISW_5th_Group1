# pkm_processor.py
from typing import List, Dict
import re
import numpy as np
from collections import defaultdict
from functools import lru_cache


class PKMProcessor:
    """문서 간 관계를 분석하고 통합하는 프로세서"""

    def __init__(self, relation_processor):
        self.relation_processor = relation_processor
        self.processed_docs = {}  # 문서 전처리 결과 저장
        self.pmi_cache = {}  # PMI 결과 캐싱

    def _preprocess_text(self, text: str) -> Dict:
        """텍스트 전처리"""
        words = text.split()
        return {
            'words': words,
            'word_count': len(words)
        }

    def _calculate_pmi(self, kw1: str, kw2: str, window_size: int = 20) -> float:
        """두 키워드 간의 PMI 계산"""
        # 캐시 키 생성
        cache_key = (kw1, kw2)
        if cache_key in self.pmi_cache:
            return self.pmi_cache[cache_key]

        def count_keyword_occurrences(text: str, keyword: str) -> int:
            pattern = r'\b' + re.escape(keyword) + r'\b'
            return len(re.findall(pattern, text))

        def count_co_occurrences(words: List[str], kw1: str, kw2: str, window_size: int) -> int:
            co_count = 0
            for i in range(len(words)):
                window = words[max(0, i - window_size):min(len(words), i + window_size + 1)]
                window_text = ' '.join(window)
                if kw1 in window_text and kw2 in window_text:
                    co_count += 1
            return co_count

        # 전처리된 문서 데이터 사용
        total_kw1_count = 0
        total_kw2_count = 0
        total_co_count = 0
        total_words = 0

        for doc_id, doc_info in self.processed_docs.items():
            text = doc_info['text']
            words = doc_info['words']
            total_kw1_count += count_keyword_occurrences(text, kw1)
            total_kw2_count += count_keyword_occurrences(text, kw2)
            total_co_count += count_co_occurrences(words, kw1, kw2, window_size)
            total_words += doc_info['word_count']

        if total_words == 0:
            result = float('-inf')
        else:
            p_kw1 = (total_kw1_count + 1e-10) / total_words
            p_kw2 = (total_kw2_count + 1e-10) / total_words
            p_joint = (total_co_count + 1e-10) / total_words
            result = float(np.log(p_joint / (p_kw1 * p_kw2)))

        # 결과 캐싱
        self.pmi_cache[cache_key] = result
        return result

    def _find_cross_document_edges(self, docs: List[Dict], pmi_threshold: float = 0.5) -> List[Dict]:
        """문서 간 엣지 찾기"""
        try:
            # 모든 문서의 노드 수집
            doc_nodes = {}  # doc_id -> nodes
            for doc in docs:
                if 'hypergraph_data' in doc and 'nodes' in doc['hypergraph_data']:
                    graph_data = doc['hypergraph_data']
                    nodes = {node['id']: node['importance'] for node in graph_data['nodes']}
                    doc_nodes[str(doc['_id'])] = nodes

            cross_edges = []
            next_edge_id = 1
            processed_pairs = set()

            # 문서 쌍별로 처리
            doc_ids = list(doc_nodes.keys())
            for i, doc1_id in enumerate(doc_ids[:-1]):
                for doc2_id in doc_ids[i + 1:]:
                    # 각 문서의 가장 중요한 노드들 간 PMI 계산
                    nodes1 = sorted(doc_nodes[doc1_id].items(), key=lambda x: x[1], reverse=True)[:3]
                    nodes2 = sorted(doc_nodes[doc2_id].items(), key=lambda x: x[1], reverse=True)[:3]

                    best_connection = None
                    best_pmi = float('-inf')

                    for node1, _ in nodes1:
                        for node2, _ in nodes2:
                            pair_key = tuple(sorted([node1, node2]))
                            if pair_key in processed_pairs:
                                continue
                            processed_pairs.add(pair_key)

                            try:
                                # 수정된 PMI 계산 호출
                                pmi_score = self._calculate_pmi(node1, node2)

                                if pmi_score > best_pmi and pmi_score > pmi_threshold:
                                    # HGNN으로 관계 예측
                                    prediction = self.relation_processor.classify_relations({
                                        "nodes": [],
                                        "edges": [{
                                            "nodes": [node1, node2],
                                            "description": "문서간연결",
                                            "importance": 1.0
                                        }]
                                    })
                                    category_info = prediction['edges'][0]

                                    best_pmi = pmi_score
                                    best_connection = {
                                        "id": f"cross_edge_{next_edge_id}",
                                        "nodes": [node1, node2],
                                        "description": "문서간연결",
                                        "importance": 1.0,
                                        "category": category_info['category'],
                                        "confidence": category_info['confidence'],
                                        "pmi_score": pmi_score,
                                        "source_doc": doc1_id,
                                        "target_doc": doc2_id
                                    }
                            except Exception as e:
                                print(f"Error calculating PMI for {node1} and {node2}: {str(e)}")
                                continue

                    if best_connection:
                        cross_edges.append(best_connection)
                        next_edge_id += 1

            return cross_edges
        except Exception as e:
            print(f"Error in _find_cross_document_edges: {str(e)}")
            return []

    def process_articles(self, articles: List[Dict]) -> List[Dict]:
        """문서들을 처리하고 문서 간 연결 추가"""
        if not articles:
            return []

        # 문서 전처리
        self.processed_docs.clear()  # 이전 캐시 클리어
        for doc in articles:
            text = doc.get('content', '')
            self.processed_docs[str(doc['_id'])] = {
                **self._preprocess_text(text),
                'text': text
            }

        # 문서 간 엣지 찾기
        cross_edges = self._find_cross_document_edges(articles)

        # 각 문서의 hypergraph_data에 cross_edges 추가
        processed_articles = []
        for article in articles:
            article_copy = dict(article)
            if 'hypergraph_data' not in article_copy:
                article_copy['hypergraph_data'] = {'nodes': [], 'edges': []}

            # 이 문서와 관련된 cross_edges 찾기
            doc_id = str(article_copy['_id'])
            relevant_edges = [
                edge for edge in cross_edges
                if edge['source_doc'] == doc_id or edge['target_doc'] == doc_id
            ]

            # 기존 엣지에 cross_edges 추가
            article_copy['hypergraph_data']['edges'].extend(relevant_edges)
            processed_articles.append(article_copy)

        return processed_articles