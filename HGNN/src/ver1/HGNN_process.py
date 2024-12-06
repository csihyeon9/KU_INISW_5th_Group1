import json
import numpy as np
import scipy.sparse as sp
from collections import defaultdict
from pathlib import Path
from tqdm import tqdm

class HypergraphProcessor:
    def __init__(self, pmi_path, processed_keywords_path, output_path):
        self.pmi_path = pmi_path
        self.processed_keywords_path = processed_keywords_path
        self.output_path = output_path
        self.node2idx = {}
        self.hyperedges = defaultdict(list)
        self.node_features = None

    def load_data(self):
        # 동사 카테고리 - 키워드, 키워드 PMI 정보 로드
        with open(self.pmi_path, 'r', encoding='utf-8') as f:
            self.pmi_data = json.load(f)

        # Load processed keywords
        with open(self.processed_keywords_path, 'r', encoding='utf-8') as f:
            processed_data = json.load(f)
            self.node2idx = processed_data['keyword2idx']

    def construct_hypergraph(self):
        # Construct hyperedges from PMI data
        for category, relations in tqdm(self.pmi_data.items(), desc="Constructing hyperedges"):
            for pair_str, weight in relations.items():
                pair = eval(pair_str)
                if pair[0] in self.node2idx and pair[1] in self.node2idx:
                    self.hyperedges[category].append((self.node2idx[pair[0]], self.node2idx[pair[1]], weight))

    def create_sparse_matrices(self):
        # Create sparse matrices for hyperedges
        edge_matrices = {}
        for category, edges in self.hyperedges.items():
            rows, cols, data = zip(*edges)
            sparse_matrix = sp.coo_matrix((data, (rows, cols)), shape=(len(self.node2idx), len(self.node2idx)))
            edge_matrices[category] = sparse_matrix
        return edge_matrices

    def generate_features(self, embedding_dim=128):
        # Initialize node features with random embeddings
        self.node_features = np.random.randn(len(self.node2idx), embedding_dim)

    def save_data(self, edge_matrices):
        # Save hypergraph data
        Path(self.output_path).mkdir(parents=True, exist_ok=True)

        # Save node features
        np.save(f"{self.output_path}/node_features.npy", self.node_features)

        # Save hyperedge matrices
        for category, matrix in edge_matrices.items():
            sp.save_npz(f"{self.output_path}/edge_matrix_{category}.npz", matrix)
        print(f"Hypergraph data saved to {self.output_path}")


if __name__ == "__main__":
    pmi_path = "data/pairwise_pmi_values.json"
    processed_keywords_path = "data/processed_keywords.json"
    output_path = "data/hypergraph_data"

    processor = HypergraphProcessor(pmi_path, processed_keywords_path, output_path)
    processor.load_data()
    processor.construct_hypergraph()
    edge_matrices = processor.create_sparse_matrices()
    processor.generate_features()
    processor.save_data(edge_matrices)
