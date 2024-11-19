# typed_visualizer.py
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from typing import Dict, List, Tuple

plt.rcParams['axes.unicode_minus'] = False

class TypedHGNNVisualizer:
    def __init__(self, figsize: Tuple[int, int] = (12, 8)):
        self.figsize = figsize
        plt.style.use('default')
    
    def plot_training_history(
        self,
        losses: Dict[str, List[float]],
        title: str = 'Training History'
    ):
        """학습 과정 시각화"""
        # plt.rc('font', family='Apple Gothic')  # 맥OS를 사용하고 있을 때
        plt.rc('font', family='Malgun Gothic')  # 윈도우를 사용하고 있을 떄
        plt.figure(figsize=self.figsize)
        
        for loss_type, values in losses.items():
            plt.plot(values, label=f'{loss_type} Loss')
        
        plt.title(title)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.show()
    
    def plot_relation_importance(
        self,
        relation_types: List[str],
        attention_weights: np.ndarray,
        title: str = 'Relation Type Importance'
    ):
        """관계 타입별 중요도 시각화"""
        # plt.rc('font', family='Apple Gothic')  # 맥OS를 사용하고 있을 때
        plt.rc('font', family='Malgun Gothic')  # 윈도우를 사용하고 있을 떄
        plt.figure(figsize=(10, 6))
        
        # 중요도에 따라 정렬
        sorted_idx = np.argsort(attention_weights)
        relation_types = [relation_types[i] for i in sorted_idx]
        attention_weights = attention_weights[sorted_idx]
        
        plt.barh(relation_types, attention_weights)
        plt.title(title)
        plt.xlabel('Importance Score')
        plt.tight_layout()
        plt.show()
    
    def plot_typed_similarity_network(
        self,
        keywords: List[str],
        embeddings: np.ndarray,
        relation_types: List[str],
        similarity_threshold: float = 0.5
    ):
        """관계 타입을 고려한 유사도 네트워크 시각화"""
        # 유사도 그래프 생성
        G = nx.Graph()
        
        # 노드 추가
        for i, keyword in enumerate(keywords):
            G.add_node(keyword, embedding=embeddings[i])
        
        # 엣지 추가 (유사도가 높은 경우만)
        for i in range(len(keywords)):
            for j in range(i + 1, len(keywords)):
                similarity = np.dot(embeddings[i], embeddings[j]) / \
                           (np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j]))
                if similarity > similarity_threshold:
                    G.add_edge(keywords[i], keywords[j], weight=similarity)
        
        # 시각화
        plt.figure(figsize=self.figsize)
        
        # 노드 위치 계산
        pos = nx.spring_layout(G)
        
        # 엣지 그리기
        edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
        nx.draw_networkx_edges(
            G, pos,
            alpha=0.5,
            width=[w * 2 for w in edge_weights]
        )
        
        # 노드 그리기
        nx.draw_networkx_nodes(
            G, pos,
            node_size=1000,
            node_color='lightblue'
        )
        
        # 레이블 그리기
        nx.draw_networkx_labels(G, pos)
        
        # plt.rc('font', family='Apple Gothic')  # 맥OS를 사용하고 있을 때
        plt.rc('font', family='Malgun Gothic')  # 윈도우를 사용하고 있을 떄
        plt.title('Keyword Similarity Network')
        plt.axis('off')
        plt.tight_layout()
        plt.show()
    
    def plot_relation_type_distribution(
        self,
        type_wise_metrics: Dict[str, Dict[str, float]],
        title: str = 'Relation Type Distribution and Preservation'
    ):
        """관계 타입별 분포와 보존성 시각화"""
        # plt.rc('font', family='Apple Gothic')  # 맥OS를 사용하고 있을 때
        plt.rc('font', family='Malgun Gothic')  # 윈도우를 사용하고 있을 떄
        plt.figure(figsize=self.figsize)
        
        relation_types = list(type_wise_metrics.keys())
        counts = [metrics['count'] for metrics in type_wise_metrics.values()]
        preservation_scores = [metrics['preservation'] for metrics in type_wise_metrics.values()]
        
        x = np.arange(len(relation_types))
        width = 0.35
        
        fig, ax1 = plt.subplots(figsize=self.figsize)
        ax2 = ax1.twinx()
        
        # 관계 수 막대 그래프
        rects1 = ax1.bar(x - width/2, counts, width, label='Count', color='skyblue')
        ax1.set_ylabel('Count')
        
        # 보존성 점수 막대 그래프
        rects2 = ax2.bar(x + width/2, preservation_scores, width, label='Preservation', color='lightgreen')
        ax2.set_ylabel('Preservation Score')
        
        ax1.set_xticks(x)
        ax1.set_xticklabels(relation_types, rotation=45, ha='right')
        
        ax1.legend(loc='upper left')
        ax2.legend(loc='upper right')
        
        plt.title(title)
        plt.tight_layout()
        plt.show()
    
    def plot_keyword_clusters(
        self,
        keywords: List[str],
        embeddings: np.ndarray,
        n_clusters: int = 5,
        title: str = 'Keyword Clusters'
    ):
        """키워드 클러스터링 시각화"""
        from sklearn.cluster import KMeans
        from sklearn.decomposition import PCA

        # plt.rc('font', family='Apple Gothic')  # 맥OS를 사용하고 있을 때
        plt.rc('font', family='Malgun Gothic')  # 윈도우를 사용하고 있을 떄
        
        # 차원 축소
        pca = PCA(n_components=2)
        embeddings_2d = pca.fit_transform(embeddings)
        
        # 클러스터링
        kmeans = KMeans(n_clusters=n_clusters)
        clusters = kmeans.fit_predict(embeddings)
        
        # 시각화
        plt.figure(figsize=self.figsize)
        
        # 클러스터별로 산점도 그리기
        for i in range(n_clusters):
            mask = clusters == i
            plt.scatter(
                embeddings_2d[mask, 0],
                embeddings_2d[mask, 1],
                label=f'Cluster {i}',
                alpha=0.6
            )
        
        # 키워드 레이블 추가
        for i, keyword in enumerate(keywords):
            plt.annotate(
                keyword,
                (embeddings_2d[i, 0], embeddings_2d[i, 1]),
                xytext=(5, 5),
                textcoords='offset points',
                fontsize=8
            )
        
        plt.title(title)
        plt.legend()
        plt.tight_layout()
        plt.show()