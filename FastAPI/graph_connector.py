# graph_connector.py
from typing import Dict, List, Tuple, Set
from collections import defaultdict
import re
import numpy as np


class GraphConnector:
    """고립된 서브그래프들을 중앙 노드가 포함된 메인 그래프와 연결"""

    def __init__(self, graph_data: dict, content: str):
        self.graph_data = graph_data
        self.content = content  # 기사 전체 텍스트
        self.nodes = {node['id']: node for node in graph_data['nodes']}
        self.edges = graph_data['edges']

    def _find_central_node(self) -> str:
        """importance가 가장 높은 중앙 노드 찾기"""
        return max(self.graph_data['nodes'], key=lambda x: x['importance'])['id']

    def _find_connected_components(self) -> Tuple[Set[str], List[Set[str]]]:
        """중앙 노드가 포함된 메인 그래프와 고립된 서브그래프들 찾기"""
        connections = defaultdict(set)
        for edge in self.edges:
            node1, node2 = edge['nodes']
            connections[node1].add(node2)
            connections[node2].add(node1)

        central_node = self._find_central_node()
        visited = set()
        components = []

        def dfs(node: str, component: set):
            visited.add(node)
            component.add(node)
            for neighbor in connections[node]:
                if neighbor not in visited:
                    dfs(neighbor, component)

        for node in self.nodes:
            if node not in visited:
                current_component = set()
                dfs(node, current_component)
                components.append(current_component)

        main_graph = next(comp for comp in components if central_node in comp)
        isolated_graphs = [comp for comp in components if central_node not in comp]

        return main_graph, isolated_graphs

    def _calculate_pmi(self, kw1: str, kw2: str, window_size: int = 20) -> float:
        """두 키워드 간의 PMI 계산"""

        def count_keyword_occurrences(text: str, keyword: str) -> int:
            pattern = r'\b' + re.escape(keyword) + r'\b'
            return len(re.findall(pattern, text))

        def count_co_occurrences(text: str, kw1: str, kw2: str, window_size: int) -> int:
            words = text.split()
            co_count = 0
            for i in range(len(words)):
                window = words[max(0, i - window_size):min(len(words), i + window_size + 1)]
                window_text = ' '.join(window)
                if kw1 in window_text and kw2 in window_text:
                    co_count += 1
            return co_count

        # 전체 기사 텍스트 사용
        kw1_count = count_keyword_occurrences(self.content, kw1)
        kw2_count = count_keyword_occurrences(self.content, kw2)
        co_count = count_co_occurrences(self.content, kw1, kw2, window_size)

        total_words = len(self.content.split())

        # PMI 계산 (스무딩 적용)
        p_kw1 = (kw1_count + 1e-10) / total_words
        p_kw2 = (kw2_count + 1e-10) / total_words
        p_joint = (co_count + 1e-10) / total_words

        return float(np.log(p_joint / (p_kw1 * p_kw2)))

    def connect_isolated_graphs(self, relation_processor) -> dict:
        main_graph, isolated_graphs = self._find_connected_components()

        if not isolated_graphs:
            return self.graph_data

        new_edges = []
        next_edge_id = max(int(edge['id'].replace('edge', '')) for edge in self.edges) + 1

        for isolated_graph in isolated_graphs:
            best_pmi = -float('inf')
            best_isolated_node = None
            best_main_node = None

            # PMI 계산으로 최적의 노드 쌍 찾기
            for isolated_node in isolated_graph:
                for main_node in main_graph:
                    pmi_score = self._calculate_pmi(isolated_node, main_node)
                    if pmi_score > best_pmi:
                        best_pmi = pmi_score
                        best_isolated_node = isolated_node
                        best_main_node = main_node

            if best_isolated_node and best_main_node:
                # HGNN으로 관계 카테고리 예측
                prediction = relation_processor.classify_relations({
                    "nodes": [],
                    "edges": [{
                        "nodes": [best_isolated_node, best_main_node],
                        "description": "예측",
                        "importance": 1.0
                    }]
                })
                category_info = prediction['edges'][0]

                new_edge = {
                    "id": f"edge{next_edge_id}",
                    "nodes": [best_isolated_node, best_main_node],
                    "description": "예측",
                    "importance": 1.0,
                    "category": category_info['category'],
                    "confidence": category_info['confidence'],
                    "pmi_score": best_pmi
                }
                new_edges.append(new_edge)
                next_edge_id += 1

        result = self.graph_data.copy()
        result['edges'].extend(new_edges)
        return result