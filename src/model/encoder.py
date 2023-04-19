import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class PatchEmbedding(nn.Module):
    def __init__(self, in_channels, patch_size, embedding_dim):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embedding_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x
    

class ViTEncoder(nn.Module):
    def __init__(self, in_channels, patch_size, embedding_dim, num_layers, num_heads, mlp_dim, dropout=0.1):
        super().__init__()
        self.embedding = PatchEmbedding(in_channels, patch_size, embedding_dim)
        self.pos_embedding = nn.Parameter(torch.zeros(1, (64 // patch_size) * (64 // patch_size), embedding_dim))
        self.dropout = nn.Dropout(dropout)
        
        encoder_layer = TransformerEncoderLayer(embedding_dim, num_heads, mlp_dim, dropout)
        self.transformer = TransformerEncoder(encoder_layer, num_layers)

    def forward(self, x):
        x = self.embedding(x)
        x += self.pos_embedding
        x = self.dropout(x)
        x = self.transformer(x)
        return x


class CNNEncoder(nn.Module):
    def __init__(self, in_channels, num_features):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(256 * 8 * 8, 1024),
            nn.ReLU(),
            nn.Linear(1024, num_features),
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x