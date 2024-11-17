from torch_geometric.loader import NeighborLoader
from torch_geometric.utils import to_networkx
import torch
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

# Load the saved data object
data = torch.load('data.pth')

# Create batches with neighbor sampling
train_loader = NeighborLoader(
    data,
    num_neighbors=[5, 10],
    batch_size=16,
    input_nodes=data.train_mask,
)
# data.train_mask는 True로 표시된 노드만 서브그래프 샘플링에 사용됩니다. 즉, 학습 노드만 샘플링하고, 검증/테스트 노드는 제외합니다.


# Print each subgraph
for i, subgraph in enumerate(train_loader):
    print(f'Subgraph {i}: {subgraph}')

# result
# Subgraph 0: Data(x=[424, 500], edge_index=[2, 466], y=[424], train_mask=[424], val_mask=[424], test_mask=[424], n_id=[424], e_id=[466], input_id=[16], batch_size=16)
# Subgraph 1: Data(x=[266, 500], edge_index=[2, 321], y=[266], train_mask=[266], val_mask=[266], test_mask=[266], n_id=[266], e_id=[321], input_id=[16], batch_size=16)
# Subgraph 2: Data(x=[277, 500], edge_index=[2, 322], y=[277], train_mask=[277], val_mask=[277], test_mask=[277], n_id=[277], e_id=[322], input_id=[16], batch_size=16)
# Subgraph 3: Data(x=[189, 500], edge_index=[2, 225], y=[189], train_mask=[189], val_mask=[189], test_mask=[189], n_id=[189], e_id=[225], input_id=[12], batch_size=12)

# Plot each subgraph
fig = plt.figure(figsize=(16,16))
for idx, (subdata, pos) in enumerate(zip(train_loader, [221, 222, 223, 224])):
    G = to_networkx(subdata, to_undirected=True)
    ax = fig.add_subplot(pos)
    ax.set_title(f'Subgraph {idx}', fontsize=24)
    plt.axis('off')
    nx.draw_networkx(G,
                    pos=nx.spring_layout(G, seed=0),
                    with_labels=False,
                    node_color=subdata.y,
                    )
plt.show()

torch.save(train_loader, './data/train_loader.pt')
print("train_loader saved to 'train_loader.pt'")