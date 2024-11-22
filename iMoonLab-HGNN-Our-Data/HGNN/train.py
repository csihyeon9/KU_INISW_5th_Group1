# train.py
import os
import torch
import torch.nn.functional as F
import pprint as pp
import utils.hypergraph_utils as hgut
from models import HGNN, HGNNRecommender
from config import get_config
from datasets.data_helper import load_json_data, preprocess_json_for_hgnn, create_keyword_embeddings, preprocess_documents, construct_hypergraph_from_json, analyze_data_distribution
from utils.hypergraph_utils import construct_H_with_keywords, generate_G_from_H
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
config = get_config('config/config.yaml')
data_dir = config['econfinnews_raw'] if config['on_dataset'] == 'EconFinNews' \
    else Exception("Data not prepared yet!")

fts, lbls, idx_train, idx_test = preprocess_json_for_hgnn(data_dir)

# Construct hypergraph
H = construct_hypergraph_from_json(fts)
G = hgut.generate_G_from_H(H)

# Convert to PyTorch tensors
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
fts = torch.Tensor(fts).to(device)
lbls = torch.Tensor(lbls).squeeze().long().to(device)
G = torch.Tensor(G).to(device)
idx_train = torch.Tensor(idx_train).long().to(device)
idx_test = torch.Tensor(idx_test).long().to(device)

# Initialize model
n_class = int(lbls.max()) + 1
model = HGNN(in_ch=fts.shape[1],
             n_class=n_class,
             n_hid=32,
             dropout=0.5).to(device)


# def train_model(model, criterion, optimizer, scheduler, num_epochs=25, print_freq=500):
#     since = time.time()

#     best_model_wts = copy.deepcopy(model.state_dict())
#     best_acc = 0.0

#     for epoch in range(num_epochs):
#         if epoch % print_freq == 0:
#             print('-' * 10)
#             print(f'Epoch {epoch}/{num_epochs - 1}')

#         # Each epoch has a training and validation phase
#         for phase in ['train', 'val']:
#             if phase == 'train':
#                 # scheduler.step()
#                 model.train()  # Set model to training mode
#             else:
#                 model.eval()  # Set model to evaluate mode

#             running_loss = 0.0
#             running_corrects = 0

#             idx = idx_train if phase == 'train' else idx_test

#             # Iterate over data.
#             optimizer.zero_grad()
#             with torch.set_grad_enabled(phase == 'train'):
#                 outputs = model(fts, G)
#                 loss = criterion(outputs[idx], lbls[idx])
#                 _, preds = torch.max(outputs, 1)

#                 # backward + optimize only if in training phase
#                 if phase == 'train':
#                     loss.backward()
#                     optimizer.step()
#                     scheduler.step()

#             # statistics
#             running_loss += loss.item() * fts.size(0)
#             running_corrects += torch.sum(preds[idx] == lbls.data[idx])

#             epoch_loss = running_loss / len(idx)
#             epoch_acc = running_corrects.double() / len(idx)

#             if epoch % print_freq == 0:
#                 print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

#             # deep copy the model
#             if phase == 'val' and epoch_acc > best_acc:
#                 best_acc = epoch_acc
#                 best_model_wts = copy.deepcopy(model.state_dict())

#         if epoch % print_freq == 0:
#             print(f'Best val Acc: {best_acc:4f}')
#             print('-' * 20)

#     time_elapsed = time.time() - since
#     print(f'\nTraining complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
#     print(f'Best val Acc: {best_acc:4f}')

#     # load best model weights
#     model.load_state_dict(best_model_wts)
#     return model

class DocumentRecommender:
    def __init__(self, embedding_size=128, hidden_size=256):
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.keyword_model = None
        self.document_embeddings = None
        self.url_to_idx = {}
        self.idx_to_url = {}
        self.documents = None

    def train(self, documents, epochs=100):
        # Store documents
        self.documents = documents

        # Create mappings
        for i, doc in enumerate(documents):
            self.url_to_idx[doc['url']] = i
            self.idx_to_url[i] = doc['url']

        print("Creating keyword embeddings...")
        self.keyword_model = create_keyword_embeddings(documents, self.embedding_size)
        
        print("Preprocessing documents...")
        features = preprocess_documents(documents, self.keyword_model, self.embedding_size)
        
        print("Constructing hypergraph...")
        H = construct_H_with_keywords(documents)
        G = generate_G_from_H(H)
        
        # Convert to PyTorch tensors
        features = torch.FloatTensor(features).to(self.device)
        G = torch.FloatTensor(G).to(self.device)
        
        # Initialize model
        self.model = HGNNRecommender(
            in_dim=self.embedding_size,
            hidden_dim=self.hidden_size,
            embedding_dim=self.embedding_size
        ).to(self.device)
        
        # Training
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        
        print("\nTraining model...")
        self.model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            
            # Get document embeddings
            embeddings = self.model(features, G)
            
            # Compute loss
            sim_matrix = torch.mm(embeddings, embeddings.t())
            mask = (G > 0).float()
            loss = F.mse_loss(sim_matrix, mask)
            
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")
        
        # Save final embeddings
        self.model.eval()
        with torch.no_grad():
            self.document_embeddings = self.model(features, G).cpu().numpy()

    def recommend(self, new_doc, top_k=5):
        """Recommend similar documents for a new document"""
        if self.model is None:
            raise Exception("Model not trained yet!")
        
        print("\nInput document keywords:", new_doc['keywords'])
        
        # Create embedding for new document
        keyword_vectors = [self.keyword_model.wv[word] for word in new_doc['keywords'] 
                         if word in self.keyword_model.wv]
        if not keyword_vectors:
            return []
        
        query_embedding = np.mean(keyword_vectors, axis=0)
        
        # Calculate similarities
        similarities = cosine_similarity([query_embedding], self.document_embeddings)[0]
        
        # Get top-k similar documents
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        recommendations = []
        for idx in top_indices:
            doc_url = self.idx_to_url[idx]
            # Find original document data
            original_doc = next(doc for doc in self.documents if doc['url'] == doc_url)
            recommendations.append({
                'url': doc_url,
                'similarity': float(similarities[idx]),
                'keywords': original_doc['keywords'],
                'relation_type': original_doc.get('relation_type', 'Unknown')
            })
        
        print("\nRecommendations for the new document:")
        for i, rec in enumerate(recommendations, 1):
            print(f"\n{i}. {rec['relation_type']}")
            print(f"URL: {rec['url']}")
            print(f"Similarity: {rec['similarity']:.4f}")
            print(f"Keywords: {', '.join(rec['keywords'])}")
        
        return recommendations

def _main():
    print(f"\nClassification on {config['on_dataset']} dataset!!! class number: {n_class}")
    print('\nConfiguration:')
    pp.pprint(config)
    print()

    # Load data
    documents = load_json_data(data_dir)
    
    # Initialize and train recommender
    recommender = DocumentRecommender()
    recommender.train(documents)
    
    # Example recommendation
    new_hyperedge = {
        "url": "https://www.sedaily.com/NewsView/2DGTELJA35",
        "relation_type": "금융시장",
        "keywords": [
            "비트코인",
            "디지털 자산",
            "가상화폐 정책",
            "준비자산",
            "전략비축",
            "금융 리더십"
        ]
    }
    
    recommendations = recommender.recommend(new_hyperedge)
    
    print("\nRecommendations for the new document:")
    for rec in recommendations:
        print(f"URL: {rec['url']}")
        print(f"Similarity: {rec['similarity']:.4f}")
        print()

# def compute_class_weights(labels):
#     unique_labels = np.unique(labels)
#     class_counts = np.bincount(labels)
#     total_samples = len(labels)
#     weights = torch.FloatTensor([total_samples / (len(unique_labels) * count) for count in class_counts])
#     return weights.to(device)

if __name__ == '__main__':
    analyze_data_distribution(data_dir)
    _main()
