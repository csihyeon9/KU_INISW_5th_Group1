import torch
import torch.nn as nn
import torch.optim as optim
from util import *

class DHCN(torch.nn.Module):
    def __init__(self, num_nodes, emb_size):
        super(DHCN, self).__init__()
        self.embedding = torch.nn.Embedding(num_nodes, emb_size)
        self.fc = torch.nn.Linear(emb_size, num_nodes)

    def forward(self, x, adj_matrix=None):
        """
        Forward pass for the model.
        Args:
            x: Input tensor of shape [batch_size, seq_length]
            adj_matrix: Adjacency matrix (optional, not used in basic implementation)
        Returns:
            Tensor of shape [batch_size, num_nodes]
        """
        x = self.embedding(x)  # Shape: [batch_size, seq_length, emb_size]
        x = x.mean(dim=1)  # Aggregate embeddings across the sequence, shape: [batch_size, emb_size]
        return self.fc(x)  # Shape: [batch_size, num_nodes]
    
    def recommend(self, input_data, adj_matrix, top_k=5):
        """
        Generate top-k recommendations for the given input data.
        Args:
            input_data (list): List of node indices for the input sequence.
            adj_matrix (optional): Adjacency matrix if needed.
            top_k (int): Number of recommendations to generate.
        Returns:
            list: List of top-k recommended node indices.
        """
        self.eval()  # Set model to evaluation mode
        input_data = torch.tensor(input_data, dtype=torch.long).unsqueeze(0)  # Add batch dimension
        with torch.no_grad():
            outputs = self(input_data, adj_matrix)  # Forward pass
            _, top_indices = outputs.topk(top_k, dim=1)  # Get top-k predictions
        return top_indices.squeeze().tolist()

    def train_model(self, train_data, adj_matrix, epochs, batch_size, lr, val_data=None):
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        criterion = torch.nn.CrossEntropyLoss()

        max_len = 6  # Maximum sequence length for padding/truncation

        for epoch in range(epochs):
            self.train()  # Set the model to training mode
            total_loss = 0
            for batch_start in range(0, len(train_data), batch_size):
                batch_data = train_data[batch_start:batch_start + batch_size]
                batch_data = pad_or_truncate(batch_data, max_len)  # Pad or truncate batch
                batch_data = torch.tensor(batch_data, dtype=torch.long)  # Convert to tensor
                
                # Split inputs and targets
                inputs = batch_data[:, :-1]
                targets = batch_data[:, -1]
                
                optimizer.zero_grad()
                outputs = self(inputs, adj_matrix)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss:.4f}")

            # Validation
            if val_data:
                self.evaluate(val_data, adj_matrix, batch_size)
        
        # Save the model after training
        torch.save(self.state_dict(), "dhcn_model.pth")
        print("Model saved to dhcn_model.pth")

    def evaluate(self, val_data, adj_matrix, batch_size):
        self.eval()  # Set the model to evaluation mode
        total_loss = 0
        criterion = torch.nn.CrossEntropyLoss()
        max_len = 6

        with torch.no_grad():
            for batch_start in range(0, len(val_data), batch_size):
                batch_data = val_data[batch_start:batch_start + batch_size]
                batch_data = pad_or_truncate(batch_data, max_len)
                batch_data = torch.tensor(batch_data, dtype=torch.long)
                
                inputs = batch_data[:, :-1]
                targets = batch_data[:, -1]
                outputs = self(inputs, adj_matrix)
                loss = criterion(outputs, targets)
                total_loss += loss.item()

        print(f"Validation Loss: {total_loss:.4f}")

