import argparse
import json
from util import preprocess_data, create_adjacency_matrix
from model import DHCN
import torch

def compute_n_node(data):
    """
    Compute the number of unique nodes and return the set of unique nodes.
    """
    unique_nodes = set()
    for article in data:
        for keyword in article["keywords"]:
            unique_nodes.add(keyword)
    return len(unique_nodes), unique_nodes

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="processed_1737.json", 
                       help="Path to the dataset")
    parser.add_argument("--epochs", type=int, default=30, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=100, help="Batch size")
    parser.add_argument("--emb_size", type=int, default=100, help="Embedding size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    args = parser.parse_args()

    # Load dataset
    try:
        with open(args.dataset, "r", encoding='utf-8') as f:
            dataset = json.load(f)
        print(f"Successfully loaded {len(dataset)} articles from {args.dataset}")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    # Compute n_node
    n_node, unique_nodes = compute_n_node(dataset)
    print(f"Number of nodes: {n_node}")

    try:
        # Preprocess data
        train_data, adj_matrix = preprocess_data(dataset, unique_nodes)
        print(f"Generated training data with {len(train_data)} sequences")
        print(f"Adjacency matrix shape: {adj_matrix.shape}")

        # Initialize model
        model = DHCN(num_nodes=n_node, emb_size=args.emb_size)
        
        try:
            state_dict = torch.load("dhcn_model.pth")
            model.load_state_dict(state_dict)
            model.eval()
            print("Model loaded from dhcn_model.pth")
        except Exception as e:
            print(f"Could not load pretrained model: {e}")
            print("Starting with fresh model")

        # Load the mappings
        with open("node_mapping.json", "r", encoding='utf-8') as f:
            idx_to_node = json.load(f)
        with open("keyword_to_node.json", "r", encoding='utf-8') as f:
            node_to_idx = json.load(f)

        # Example input keywords
        input_keywords = ["비트코인", "가상화폐", "트럼프 2기 행정부"]
        
        # Convert keywords to node IDs
        input_sequence = [node_to_idx[keyword] for keyword in input_keywords 
                         if keyword in node_to_idx]
        print(f"Input sequence (node IDs): {input_sequence}")

        # Generate recommendations
        if input_sequence:
            recommended_nodes = model.recommend(input_sequence, adj_matrix, top_k=5)
            recommended_keywords = [idx_to_node[str(node_id)] 
                                 for node_id in recommended_nodes]
            print(f"Recommended keywords for input {input_keywords}: {recommended_keywords}")
        else:
            print("No valid input keywords found in the vocabulary")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()