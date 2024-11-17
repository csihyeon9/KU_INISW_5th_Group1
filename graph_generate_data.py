from torch_geometric.datasets import Planetoid
import torch

dataset = Planetoid(root='.', name="Pubmed")

data = dataset[0]
 
torch.save(data, 'data.pth')
# Print information about the dataset
print(f'Dataset: {dataset}')
print('-------------------')
print(f'Number of graphs: {len(dataset)}')
print(f'Number of nodes: {data.x.shape[0]}')
print(f'Number of features: {dataset.num_features}')
print(f'Number of classes: {dataset.num_classes}')
 
# # Print information about the graph
# print(f'\nGraph:')
# print('------')
# print(f'Training nodes: {sum(data.train_mask).item()}')
# print(f'Evaluation nodes: {sum(data.val_mask).item()}')
# print(f'Test nodes: {sum(data.test_mask).item()}')
# print(f'Edges are directed: {data.is_directed()}')
# print(f'Graph has isolated nodes: {data.has_isolated_nodes()}')
# print(f'Graph has loops: {data.has_self_loops()}')

