import torch
import torch.nn as nn
import torch.optim as optim
from util import *

class DHCN(torch.nn.Module):
    def __init__(self, num_nodes, emb_size, num_layers=2, dropout=0.2):
        super(DHCN, self).__init__()
        self.embedding = torch.nn.Embedding(num_nodes, emb_size)
        
        # Add multiple layers for better representation
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(emb_size, emb_size),
                nn.ReLU(),
                nn.Dropout(dropout)
            ) for _ in range(num_layers)
        ])
        
        self.attention = nn.MultiheadAttention(
            embed_dim=emb_size,
            num_heads=4,
            dropout=dropout
        )
        
        self.fc = nn.Linear(emb_size, num_nodes)
        
    def forward(self, x, adj_matrix=None):
        x = self.embedding(x)  # [batch_size, seq_length, emb_size]
        
        # Apply attention mechanism
        attn_output, _ = self.attention(x, x, x)
        
        # Process through layers
        for layer in self.layers:
            attn_output = layer(attn_output)
            
        # Aggregate with attention weights
        output = attn_output.mean(dim=1)
        return self.fc(output)
    
    def recommend(self, input_data, adj_matrix, top_k=5, diversity_weight=0.3):
        self.eval()
        input_tensor = torch.tensor(input_data, dtype=torch.long).unsqueeze(0)
        
        with torch.no_grad():
            # Get model predictions
            outputs = self(input_tensor, adj_matrix)
            scores = torch.softmax(outputs, dim=1)
            
            # Apply diversity penalty using adjacency matrix
            if adj_matrix is not None:
                diversity_penalty = torch.tensor(
                    adj_matrix[input_data].mean(axis=0).A[0]
                )
                final_scores = scores - diversity_weight * diversity_penalty
            
            # Get top-k recommendations
            _, top_indices = final_scores.topk(top_k)
        
        return top_indices.squeeze().tolist()
    
    def train_model(self, train_data, adj_matrix, epochs, batch_size, lr, val_data=None):
        optimizer = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=3
        )
        criterion = torch.nn.CrossEntropyLoss()
        
        best_val_loss = float('inf')
        early_stopping_patience = 5
        no_improve_count = 0
        
        for epoch in range(epochs):
            # Training
            self.train()
            train_loss = self._train_epoch(
                train_data, adj_matrix, batch_size, optimizer, criterion
            )
            
            # Validation
            if val_data:
                val_loss = self._validate(val_data, adj_matrix, batch_size, criterion)
                scheduler.step(val_loss)
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'val_loss': val_loss,
                    }, 'best_model.pth')
                    no_improve_count = 0
                else:
                    no_improve_count += 1
                    
                if no_improve_count >= early_stopping_patience:
                    print(f"Early stopping at epoch {epoch}")
                    break

    # def __init__(self, num_nodes, emb_size):
    #     super(DHCN, self).__init__()
    #     self.embedding = torch.nn.Embedding(num_nodes, emb_size)
    #     self.fc = torch.nn.Linear(emb_size, num_nodes)

    # def forward(self, x, adj_matrix=None):
    #     """
    #     Forward pass for the model.
    #     Args:
    #         x: Input tensor of shape [batch_size, seq_length]
    #         adj_matrix: Adjacency matrix (optional, not used in basic implementation)
    #     Returns:
    #         Tensor of shape [batch_size, num_nodes]
    #     """
    #     x = self.embedding(x)  # Shape: [batch_size, seq_length, emb_size]
    #     x = x.mean(dim=1)  # Aggregate embeddings across the sequence, shape: [batch_size, emb_size]
    #     return self.fc(x)  # Shape: [batch_size, num_nodes]
    
    # def recommend(self, input_data, adj_matrix, top_k=5):
    #     """
    #     Generate top-k recommendations for the given input data.
    #     Args:
    #         input_data (list): List of node indices for the input sequence.
    #         adj_matrix (optional): Adjacency matrix if needed.
    #         top_k (int): Number of recommendations to generate.
    #     Returns:
    #         list: List of top-k recommended node indices.
    #     """
    #     self.eval()  # Set model to evaluation mode
    #     input_data = torch.tensor(input_data, dtype=torch.long).unsqueeze(0)  # Add batch dimension
    #     with torch.no_grad():
    #         outputs = self(input_data, adj_matrix)  # Forward pass
    #         _, top_indices = outputs.topk(top_k, dim=1)  # Get top-k predictions
    #     return top_indices.squeeze().tolist()

    # def train_model(self, train_data, adj_matrix, epochs, batch_size, lr, val_data=None):
    #     optimizer = torch.optim.Adam(self.parameters(), lr=lr)
    #     criterion = torch.nn.CrossEntropyLoss()

    #     max_len = 6  # Maximum sequence length for padding/truncation

    #     for epoch in range(epochs):
    #         self.train()  # Set the model to training mode
    #         total_loss = 0
    #         for batch_start in range(0, len(train_data), batch_size):
    #             batch_data = train_data[batch_start:batch_start + batch_size]
    #             batch_data = pad_or_truncate(batch_data, max_len)  # Pad or truncate batch
    #             batch_data = torch.tensor(batch_data, dtype=torch.long)  # Convert to tensor
                
    #             # Split inputs and targets
    #             inputs = batch_data[:, :-1]
    #             targets = batch_data[:, -1]
                
    #             optimizer.zero_grad()
    #             outputs = self(inputs, adj_matrix)
    #             loss = criterion(outputs, targets)
    #             loss.backward()
    #             optimizer.step()

    #             total_loss += loss.item()

    #         print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss:.4f}")

    #         # Validation
    #         if val_data:
    #             self.evaluate(val_data, adj_matrix, batch_size)
        
    #     # Save the model after training
    #     torch.save(self.state_dict(), "dhcn_model.pth")
    #     print("Model saved to dhcn_model.pth")

    # def evaluate(self, val_data, adj_matrix, batch_size):
    #     self.eval()  # Set the model to evaluation mode
    #     total_loss = 0
    #     criterion = torch.nn.CrossEntropyLoss()
    #     max_len = 6

    #     with torch.no_grad():
    #         for batch_start in range(0, len(val_data), batch_size):
    #             batch_data = val_data[batch_start:batch_start + batch_size]
    #             batch_data = pad_or_truncate(batch_data, max_len)
    #             batch_data = torch.tensor(batch_data, dtype=torch.long)
                
    #             inputs = batch_data[:, :-1]
    #             targets = batch_data[:, -1]
    #             outputs = self(inputs, adj_matrix)
    #             loss = criterion(outputs, targets)
    #             total_loss += loss.item()

    #     print(f"Validation Loss: {total_loss:.4f}")

