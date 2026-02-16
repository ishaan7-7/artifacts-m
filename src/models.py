# File: C:\vehicle_health_factory\src\models.py
import torch
import torch.nn as nn

class LSTMAutoencoder(nn.Module):
    def __init__(self, input_dim, window_size, hidden_dim=64, num_layers=2):
        super().__init__()
        self.window_size = window_size
        self.input_dim = input_dim
        
        # Encoder
        self.encoder = nn.LSTM(
            input_size=input_dim, 
            hidden_size=hidden_dim, 
            num_layers=num_layers, 
            batch_first=True, 
            dropout=0.2
        )
        
        # Decoder
        self.decoder = nn.LSTM(
            input_size=hidden_dim, 
            hidden_size=input_dim, 
            num_layers=num_layers, 
            batch_first=True, 
            dropout=0.2
        )
        
    def forward(self, x):
        # x shape: (Batch, Window, Features)
        
        # Encode
        _, (hidden, _) = self.encoder(x)
        
        # The Context Vector is the last hidden state
        context_vector = hidden[-1] 
        
        # Repeat context for every time step in the window
        repeated_context = context_vector.unsqueeze(1).repeat(1, self.window_size, 1)
        
        # Decode
        reconstructed, _ = self.decoder(repeated_context)
        
        return reconstructed