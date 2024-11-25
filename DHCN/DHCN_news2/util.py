import numpy as np
from scipy.sparse import csr_matrix
import json

def preprocess_data(dataset, unique_nodes):
    """
    Preprocess the dataset to create training data and an adjacency matrix.
    
    Args:
        dataset: List of article dictionaries
        unique_nodes: Set of unique keywords
    
    Returns:
        tuple: (train_data, adj_matrix)
    """
    # Create node mappings
    node_to_idx = {node: idx for idx, node in enumerate(unique_nodes)}
    idx_to_node = {idx: node for node, idx in node_to_idx.items()}

    # Save mappings
    with open("node_mapping.json", "w", encoding='utf-8') as f:
        json.dump(idx_to_node, f, ensure_ascii=False, indent=2)
    with open("keyword_to_node.json", "w", encoding='utf-8') as f:
        json.dump(node_to_idx, f, ensure_ascii=False, indent=2)
    print("Mappings saved to node_mapping.json and keyword_to_node.json")

    # Create adjacency matrix data
    rows, cols, values = [], [], []
    for article in dataset:
        keywords = article["keywords"]
        indices = [node_to_idx[keyword] for keyword in keywords if keyword in node_to_idx]
        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                rows.append(indices[i])
                cols.append(indices[j])
                values.append(1)
                # Symmetric edge
                rows.append(indices[j])
                cols.append(indices[i])
                values.append(1)
    
    # Create sparse adjacency matrix
    n_nodes = len(unique_nodes)
    adj_matrix = csr_matrix((values, (rows, cols)), shape=(n_nodes, n_nodes))
    
    # Create training data
    train_data = []
    for article in dataset:
        keyword_indices = [node_to_idx[keyword] for keyword in article["keywords"] 
                         if keyword in node_to_idx]
        if len(keyword_indices) > 1:  # Only include if we have at least 2 keywords
            train_data.append(keyword_indices)
    
    return train_data, adj_matrix

def create_adjacency_matrix(data, num_nodes):
    rows, cols = [], []
    for session in data:
        for i in range(len(session) - 1):
            rows.append(session[i])
            cols.append(session[i + 1])
    adj_matrix = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(num_nodes, num_nodes))
    return adj_matrix

def split_data(data, train_ratio=0.8):
    split_idx = int(len(data) * train_ratio)
    return data[:split_idx], data[split_idx:]

def pad_or_truncate(batch, max_len):
    """
    Pads or truncates sequences in the batch to the same length.
    Args:
        batch (list of lists): The batch of sequences.
        max_len (int): The maximum sequence length.
    Returns:
        padded_batch (list of lists): The batch with equal-length sequences.
    """
    padded_batch = []
    for seq in batch:
        if len(seq) > max_len:
            padded_batch.append(seq[:max_len])  # Truncate
        else:
            padded_batch.append(seq + [0] * (max_len - len(seq)))  # Pad with 0
    return padded_batch