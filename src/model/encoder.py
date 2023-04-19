import torch
import torch.nn as nn


class Encoder3DCNN(nn.Module):
    def __init__(self, in_channels, hidden_dim, num_layers):
        super(Encoder3DCNN, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv3d(in_channels, hidden_dim, kernel_size=(3, 3, 3), padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        )
        for _ in range(num_layers - 1):
            self.layers.add_module("conv3d", nn.Conv3d(hidden_dim, hidden_dim, kernel_size=(3, 3, 3), padding=1))
            self.layers.add_module("relu", nn.ReLU())
            self.layers.add_module("maxpool3d", nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)))

        self.avgpool = nn.AdaptiveAvgPool3d(1)

    def forward(self, x):
        x = self.layers(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x


class EncoderViT(nn.Module):
    def __init__(self, img_size, patch_size, num_frames, channels, num_classes, dim, depth, heads, mlp_dim):
        super().__init__()
        self.num_patches = (img_size // patch_size) * (img_size // patch_size)
        self.patch_size = patch_size
        self.dim = dim
        self.num_frames = num_frames

        self.patch_to_embedding = nn.Linear(patch_size * patch_size * channels, dim)

        self.pos_embedding = nn.Parameter(torch.randn(1, num_frames * self.num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.TransformerEncoderLayer(d_model=dim, nhead=heads, dim_feedforward=mlp_dim))
        

    def forward(self, x):
        batch_size = x.shape[0]
        x = x.view(batch_size, self.num_frames, -1, self.patch_size, self.patch_size)
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = x.view(batch_size, self.num_frames, self.patch_size * self.patch_size, -1)

        x = self.patch_to_embedding(x)
        x = x.permute(1, 0, 2, 3).contiguous()
        x = x.view(self.num_frames, batch_size * self.num_patches, self.dim)

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        x = x + self.pos_embedding
        x = x.permute(1, 0, 2)

        for layer in self.layers:
            x = layer(x)

        x = x[:, 0]
        return x