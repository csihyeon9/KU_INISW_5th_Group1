import argparse
import json
from util import preprocess_data, create_adjacency_matrix
from model import DHCN
import torch

def compute_n_node(data):
    unique_nodes = set()
    for article in data:
        for keyword in article["keywords"]:
            unique_nodes.add(keyword)
    return len(unique_nodes), unique_nodes

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="processed_1737.json", help="Path to the dataset")
    parser.add_argument("--epochs", type=int, default=30, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=100, help="Batch size")
    parser.add_argument("--emb_size", type=int, default=100, help="Embedding size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    args = parser.parse_args()

    # Load dataset
    with open(args.dataset, "r") as f:
        dataset = json.load(f)

    # Compute n_node
    n_node, unique_nodes = compute_n_node(dataset)
    print(f"Number of nodes: {n_node}")

    # Preprocess data
    train_data, adj_matrix = preprocess_data(dataset, unique_nodes)

    # Initialize model
    # Load model weights safely
    model = DHCN(num_nodes=n_node, emb_size=args.emb_size)
    #state_dict = torch.load("dhcn_model.pth", weights_only=True)
    #model.load_state_dict(state_dict)
    #model.eval()
    #print("Model loaded from dhcn_model.pth")

    # Train model
    model.train_model(train_data, adj_matrix, args.epochs, args.batch_size, args.lr)


    # Load the mappings
    with open("node_mapping.json", "r") as f:
        idx_to_node = json.load(f)
    with open("keyword_to_node.json", "r") as f:
        node_to_idx = json.load(f)


    # Generate recommendations

    # Example input keywords
    input_keywords = ["비트코인", "이더리움","가상화폐", "가상자산", "준비자산", "차익 실현", "저항선", "전략비축", "원유", "희토류", "전략적 준비 자산", "국가 준비 자산"]

    # Convert keywords to node IDs
    input_sequence = [node_to_idx[keyword] for keyword in input_keywords if keyword in node_to_idx]
    print(f"Input sequence (node IDs): {input_sequence}")

    # Generate recommendations
    recommended_nodes = model.recommend(input_sequence, adj_matrix, top_k=5)

    # Convert recommended node IDs back to keywords
    recommended_keywords = [idx_to_node[str(node_id)] for node_id in recommended_nodes]
    print(f"Recommended keywords for input {input_keywords}: {recommended_keywords}")




if __name__ == "__main__":
    main()