import torch
from torch.nn import functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv, SAGEConv

class HeteroGNN(torch.nn.Module):
    def __init__(self, metadata, hidden_channels, out_channels):
        """
        Heterogeneous GNN 모델 초기화
        :param metadata: 노드와 엣지 타입 메타데이터
        :param hidden_channels: 히든 레이어 채널 크기
        :param out_channels: 출력 채널 크기
        """
        super().__init__()
        self.conv1 = HeteroConv(
            {
                ('relation_type', 'relates_to', 'keywords'): SAGEConv((-1, -1), hidden_channels),
                ('keywords', 'linked_to', 'keywords'): SAGEConv((-1, -1), hidden_channels),
                ('relation_type', 'related_to', 'relation_type'): SAGEConv((-1, -1), hidden_channels),
            },
            aggr='sum',
        )
        self.conv2 = HeteroConv(
            {
                ('relation_type', 'relates_to', 'keywords'): SAGEConv((-1, -1), out_channels),
                ('keywords', 'linked_to', 'keywords'): SAGEConv((-1, -1), out_channels),
                ('relation_type', 'related_to', 'relation_type'): SAGEConv((-1, -1), out_channels),
            },
            aggr='sum',
        )

    def forward(self, x_dict, edge_index_dict):
        """
        모델의 순전파 정의
        :param x_dict: 노드 타입별 특성 벡터
        :param edge_index_dict: 엣지 타입별 연결 관계
        """
        x_dict = self.conv1(x_dict, edge_index_dict)
        x_dict = {key: F.relu(x) for key, x in x_dict.items()}
        x_dict = self.conv2(x_dict, edge_index_dict)
        return x_dict


def prepare_data_with_fix(relation_types, keywords, edges, keyword_features, relation_features):
    """
    HeteroData 객체 생성 및 relation_type 갱신 문제 해결.
    :param relation_types: 관계 노드 정보
    :param keywords: 키워드 노드 정보
    :param edges: 엣지 정보 (relation_type ↔ keywords 및 keywords ↔ keywords)
    :param keyword_features: 키워드 노드 특성
    :param relation_features: 관계 노드 특성
    """
    data = HeteroData()
    
    # 노드 추가
    data['relation_type'].x = torch.tensor(relation_features, dtype=torch.float)
    data['keywords'].x = torch.tensor(keyword_features, dtype=torch.float)
    
    # 엣지 추가 (edge_index의 크기를 항상 2로 설정)
    # relation_type ↔ keywords
    relation_to_keywords = edges['relation_to_keywords']
    if len(relation_to_keywords) > 0:
        relation_to_keywords = torch.tensor(relation_to_keywords, dtype=torch.long).t()
        data['relation_type', 'relates_to', 'keywords'].edge_index = relation_to_keywords
    else:
        print("Warning: No edges between relation_type and keywords!")

    # keywords ↔ keywords
    keywords_to_keywords = edges['keywords_to_keywords']
    if len(keywords_to_keywords) > 0:
        keywords_to_keywords = torch.tensor(keywords_to_keywords, dtype=torch.long).t()
        data['keywords', 'linked_to', 'keywords'].edge_index = keywords_to_keywords
    else:
        print("Warning: No edges between keywords!")

    # relation_type 노드 갱신 보장: relation_type ↔ relation_type 연결 추가
    if 'relation_to_relation' in edges:
        relation_to_relation = edges['relation_to_relation']
        if len(relation_to_relation) > 0:
            relation_to_relation = torch.tensor(relation_to_relation, dtype=torch.long).t()
            data['relation_type', 'related_to', 'relation_type'].edge_index = relation_to_relation
        else:
            print("Warning: No edges between relation_type nodes!")
    else:
        # relation_type 노드 갱신을 위해 self-loop 추가
        num_relation_types = len(relation_features)
        self_loops = torch.arange(num_relation_types).repeat(2, 1)
        data['relation_type', 'related_to', 'relation_type'].edge_index = self_loops

    return data


def train_hetero_gnn(data, model, epochs=300, lr=0.01):
    """
    HeteroGNN 모델 학습
    :param data: HeteroData 객체
    :param model: Heterogeneous GNN 모델
    :param epochs: 학습 에포크 수
    :param lr: 학습률
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        # 순전파
        out = model(data.x_dict, data.edge_index_dict)
        
        # 간단한 MSE Loss (임의의 타겟값 생성)
        loss = F.mse_loss(out['keywords'], torch.randn(out['keywords'].shape))
        loss.backward()
        optimizer.step()
        
        if epoch % 10 == 0:
            print(f'Epoch {epoch} Loss: {loss.item():.4f}')

    return model


if __name__ == "__main__":
    # 예제 데이터 정의
    relation_types = ['금융시장', '경제 트렌드']
    keywords = ['비트코인', '인플레이션', '금리', '주식']
    edges = {
        'relation_to_keywords': [[0, 1], [1, 2], [0, 3]],  # 관계 → 키워드
        'keywords_to_keywords': [[0, 1], [1, 2]],           # 키워드 ↔ 키워드
        'relation_to_relation': [[0, 1]]                   # 관계 ↔ 관계
    }
    keyword_features = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]]
    relation_features = [[0.9, 0.1], [0.2, 0.3]]

    # 데이터 준비
    data = prepare_data_with_fix(relation_types, keywords, edges, keyword_features, relation_features)

    # HeteroGNN 모델 생성 및 학습
    model = HeteroGNN(metadata=data.metadata(), hidden_channels=16, out_channels=8)
    model = train_hetero_gnn(data, model, epochs=50)
