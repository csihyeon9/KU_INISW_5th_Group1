import json
import numpy as np
from gensim.models import Word2Vec
from typing import List, Dict, Set

def load_hyperedges(file_path: str) -> List[Dict]:
    """Load hyperedges from JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def create_hypergraph_embeddings(hyperedges: List[Dict], 
                                  vector_size: int = 100, 
                                  window: int = 5, 
                                  min_count: int = 1) -> Dict[str, np.ndarray]:
    """
    Create embeddings for nodes (keywords) across hyperedges
    """
    # Collect all unique keywords across hyperedges
    all_keywords: List[str] = []
    for edge in hyperedges:
        all_keywords.extend(edge['keywords'])
    
    # Train Word2Vec model directly on keywords
    model = Word2Vec(
        sentences=[all_keywords], 
        vector_size=vector_size, 
        window=window, 
        min_count=min_count, 
        workers=4
    )
    
    # Create embeddings dictionary
    embeddings = {}
    for keyword in set(all_keywords):
        if keyword in model.wv:
            embeddings[keyword] = model.wv[keyword].tolist()
    
    return embeddings

def save_node_embeddings(embeddings: Dict[str, List[float]], output_path: str):
    """Save node embeddings to JSON file."""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(embeddings, f, ensure_ascii=False, indent=2)

def main():
    # File paths
    hyperedges_path = 'hyperedges.json'
    embeddings_output_path = 'node_embeddings.json'
    
    # Load hyperedges
    hyperedges = load_hyperedges(hyperedges_path)
    
    # Create node embeddings
    node_embeddings = create_hypergraph_embeddings(hyperedges)
    
    # Save embeddings
    save_node_embeddings(node_embeddings, embeddings_output_path)
    
    print(f"Node embeddings saved to {embeddings_output_path}")
    print(f"Total nodes embedded: {len(node_embeddings)}")

if __name__ == '__main__':
    main()