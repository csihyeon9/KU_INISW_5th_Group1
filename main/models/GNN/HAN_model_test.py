## Han(Heteogeneous) model + Link Prediction task
import json
import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import SAGEConv, to_hetero
from torch_geometric.nn import GATConv
from torch_geometric.data import HeteroData
from torch_geometric.transforms import RandomLinkSplit
from typing import Dict, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer

from torch.optim.lr_scheduler import StepLR
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt


def load_financial_data(json_path: str) -> Tuple[HeteroData, Dict]:
    with open(json_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    # Initialize a HeteroData object
    graph_data = HeteroData()

    # Create id mapping and collect terms
    id_map = {node["id"]: idx for idx, node in enumerate(data["nodes"])}
    terms = [node["word"] for node in data["nodes"]]
    
    # Create term embeddings using TF-IDF
    vectorizer = TfidfVectorizer(max_features=128)
    term_matrix = vectorizer.fit_transform([term.lower() for term in terms])
    x = torch.from_numpy(term_matrix.toarray()).float()
    
    # Add node features
    graph_data['term'].x = x

    # Process edges
    src, dst = [], []
    edge_text = []
    
    for edge in data["edges"]:
        if edge['source'] in id_map and edge['target'] in id_map:
            src.append(id_map[edge['source']])
            dst.append(id_map[edge['target']])
            relation_text = edge.get('relation_description', edge.get('reason', ''))
            edge_text.append(relation_text)

    # Create edge features using TF-IDF on relation texts
    if edge_text:
        edge_vectorizer = TfidfVectorizer(max_features=32)
        edge_features = edge_vectorizer.fit_transform(edge_text)
        edge_attr = torch.from_numpy(edge_features.toarray()).float()
    else:
        edge_attr = torch.ones((len(src), 1), dtype=torch.float)  # 기본 edge attribute

    # Add edges and their features to the graph
    graph_data['term', 'related_to', 'term'].edge_index = torch.tensor([src, dst], dtype=torch.long)
    graph_data['term', 'related_to', 'term'].edge_attr = edge_attr

    # Create reverse mapping for interpretation
    idx_to_term = {idx: term for idx, term in enumerate(terms)}
    
    return graph_data, idx_to_term

def focal_loss(pred, target, alpha=0.25, gamma=2.0):
    prob = torch.sigmoid(pred)
    focal_weight = (1 - prob) ** gamma * target + prob ** gamma * (1 - target)
    loss = -alpha * (target * torch.log(prob) + (1 - target) * torch.log(1 - prob))
    return loss.mean()

class GNNEncoder(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = SAGEConv((in_channels, in_channels), hidden_channels)
        self.conv2 = SAGEConv((hidden_channels, hidden_channels), hidden_channels)
        self.conv3 = SAGEConv((hidden_channels, hidden_channels), out_channels)

        # GAT 레이어 사용 시 
        # self.conv1 = GATConv(in_channels, hidden_channels, heads=2)
        # self.conv2 = GATConv(hidden_channels * 2, hidden_channels, heads=2)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        return self.conv3(x, edge_index)

class EdgeDecoder(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.lin1 = Linear(2 * hidden_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, 1)  # Output scalar value

    def forward(self, z_dict, edge_label_index):
        row, col = edge_label_index
        z = torch.cat([z_dict['term'][row], z_dict['term'][col]], dim=-1)
        z = self.lin1(z).relu()
        return self.lin2(z).squeeze(-1)  # Output shape: [num_edges]

class LinkPredictionModel(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, metadata):
        super().__init__()
        self.encoder = GNNEncoder(in_channels, hidden_channels, hidden_channels)
        self.encoder = to_hetero(self.encoder, metadata, aggr='sum')
        self.decoder = EdgeDecoder(hidden_channels)

    def forward(self, x_dict, edge_index_dict, edge_label_index):
        z_dict = self.encoder(x_dict, edge_index_dict)
        return self.decoder(z_dict, edge_label_index)

def train_test_split(data: HeteroData, val_ratio: float = 0.1, test_ratio: float = 0.1):
    transform = RandomLinkSplit(
        num_val=val_ratio,
        num_test=test_ratio,
        edge_types=[('term', 'related_to', 'term')],
        rev_edge_types=[('term', 'rev_related_to', 'term')],
        is_undirected=True
    )

    train_data, val_data, test_data = transform(data)

     # 확인: positive와 negative sample의 비율
    train_labels = train_data['term', 'related_to', 'term'].edge_label
    num_positive = train_labels.sum().item()
    num_negative = train_labels.size(0) - num_positive
    print(f"Train: Positive edges = {num_positive}, Negative edges = {num_negative}")

    return train_data, val_data, test_data

def train(model, optimizer, train_data):
    model.train()
    optimizer.zero_grad()
    
    edge_label = train_data['term', 'related_to', 'term'].edge_label.float()  # Use generated edge labels
    
    pred = model(
        train_data.x_dict,
        train_data.edge_index_dict,
        train_data['term', 'related_to', 'term'].edge_label_index
    )
    
    # loss = F.binary_cross_entropy_with_logits(pred, edge_label)
    loss = focal_loss(pred, edge_label)
    loss.backward()
    optimizer.step()
    return float(loss)

@torch.no_grad()
def test(model, data):
    model.eval()
    
    edge_label = data['term', 'related_to', 'term'].edge_label.float()
    
    pred = model(
        data.x_dict,
        data.edge_index_dict,
        data['term', 'related_to', 'term'].edge_label_index
    )
    
    return float(F.binary_cross_entropy_with_logits(pred, edge_label))

def predict_relationships(model, data, idx_to_term, source_term_idx, k=5):
    model.eval()
    device = next(model.parameters()).device
    
    # Get all possible target indices
    all_targets = torch.arange(len(idx_to_term), device=device)
    
    # Create edge_label_index for all possible connections
    source_indices = torch.full_like(all_targets, source_term_idx)
    edge_label_index = torch.stack([source_indices, all_targets])
    
    # Get predictions
    with torch.no_grad():
        pred = model(data.x_dict, data.edge_index_dict, edge_label_index)
        scores = torch.sigmoid(pred)  # Convert logits to probabilities
    
    # Get top k predictions
    top_k_scores, top_k_indices = torch.topk(scores, k)
    
    results = []
    for score, idx in zip(top_k_scores, top_k_indices):
        if idx.item() != source_term_idx:  # Exclude self-loops
            target_term = idx_to_term[idx.item()]
            results.append({
                'target_term': target_term,
                'confidence_score': score.item()
            })
    
    return results

def main():
    # Load data
    data, idx_to_term = load_financial_data("backend/app/gnn_15000.json")
    
    # Split data
    train_data, val_data, test_data = train_test_split(data)
    
    # Initialize model
    model = LinkPredictionModel(
        in_channels=data['term'].x.size(1),
        hidden_channels=128,
        metadata=data.metadata()
    )
    
    # Training setup
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Move data to device
    train_data = train_data.to(device)
    val_data = val_data.to(device)
    test_data = test_data.to(device)
    
    # Training loop
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0
    
    
    for epoch in range(1, 301):  # 100 epoch으로 제한
        loss = train(model, optimizer, train_data)
        val_loss = test(model, val_data)
        test_loss = test(model, test_data)
        
        # 학습률 업데이트
        scheduler.step(val_loss)

        # 현재 학습률 확인
        current_lr = scheduler.get_last_lr()[0]

        if epoch % 10 == 0:
            print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Val Loss: {val_loss:.4f}, '
                f'Test Loss: {test_loss:.4f}, LR: {current_lr:.6f}')


    # Example of relationship prediction
    source_term = "금리선물"
    source_idx = list(idx_to_term.keys())[list(idx_to_term.values()).index(source_term)]
    predictions = predict_relationships(model, data.to(device), idx_to_term, source_idx)
    
    print(f"\nPredicted relationships for {source_term}:")
    for pred in predictions:
        print(f"Related term: {pred['target_term']}, Confidence: {pred['confidence_score']:.4f}")

if __name__ == "__main__":
    main()