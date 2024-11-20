import json
import numpy as np
from gensim.models import Word2Vec
from typing import List, Dict

def load_hyperedges(file_path: str) -> List[Dict]:
    """
    Load hyperedges data from a JSON file.
    
    Args:
        file_path (str): Path to the hyperedges JSON file
    
    Returns:
        List of dictionaries containing hyperedge data
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def preprocess_data(hyperedges: List[Dict]) -> List[List[str]]:
    """
    Preprocess hyperedges data for Word2Vec embedding.
    
    Args:
        hyperedges (List[Dict]): List of hyperedge dictionaries
    
    Returns:
        List of tokenized sentences for Word2Vec training
    """
    sentences = []
    
    for edge in hyperedges:
        # Combine relation type and keywords
        tokens = [edge['relation_type']] + edge['keywords']
        sentences.append(tokens)
    
    return sentences

def train_word2vec(sentences: List[List[str]], 
                   vector_size: int = 100, 
                   window: int = 5, 
                   min_count: int = 1) -> Word2Vec:
    """
    Train Word2Vec model on preprocessed sentences.
    
    Args:
        sentences (List[List[str]]): Preprocessed sentences
        vector_size (int): Dimensionality of word vectors
        window (int): Maximum distance between current and predicted word
        min_count (int): Minimum word count to include in vocabulary
    
    Returns:
        Trained Word2Vec model
    """
    model = Word2Vec(
        sentences=sentences, 
        vector_size=vector_size, 
        window=window, 
        min_count=min_count, 
        workers=4
    )
    return model

def save_embeddings(model: Word2Vec, output_path: str):
    """
    Save word embeddings to a file.
    
    Args:
        model (Word2Vec): Trained Word2Vec model
        output_path (str): Path to save embeddings
    """
    # Create a dictionary of word:embedding
    embeddings = {word: model.wv[word].tolist() for word in model.wv.index_to_key}
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(embeddings, f, ensure_ascii=False, indent=2)

def main():
    # File paths
    hyperedges_path = 'hyperedges.json'
    embeddings_output_path = 'word2vec_embeddings.json'
    
    # Load and preprocess data
    hyperedges = load_hyperedges(hyperedges_path)
    sentences = preprocess_data(hyperedges)
    
    # Train Word2Vec model
    model = train_word2vec(sentences)
    
    # Save embeddings
    save_embeddings(model, embeddings_output_path)
    
    print(f"Embeddings saved to {embeddings_output_path}")
    
    # Optional: Print some example embeddings
    for word in ['기준금리', 'IMF', '당정']:
        if word in model.wv:
            print(f"Embedding for '{word}': {model.wv[word][:5]}...")

if __name__ == '__main__':
    main()