import torch
import torch.nn as nn
import copy

# input size (b, c, h, w) and output size (b, h, w), where b is the batch size, c is the number of channels, h is the height, and w is the width.
class JEPA(nn.Module):
    def __init__(self, img_size, patch_size, in_channels, embed_dim, num_heads, encoder_x, predictor):
        super(JEPA, self).__init__()

        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        self.patch_dim = self.patch_embed.patch_dim
        self.num_tokens = self.patch_embed.num_tokens
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_tokens, embed_dim))

        self.encoder_x = encoder_x
        self.predictor = predictor

        self.embed_norm = nn.LayerNorm(embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

    # frames_embed: [(b, num_patches, embed_dim) for _ in range(H)]
    # i: the index of the current frame
    def forward(self, frames_embed, i):
        # encode video into patches
        sx = [None for _ in range(self.H)]

        # encode first frame
        # x0: (b, c, h, w)
        x0 = frames_embed[0]
        # sx_0: (b, num_patches, embed_dim)
        sx_0 = self.encoder_x(x0)
        sx[0] = sx_0

        # encode frames i to H
        for j in range(i, self.H):
            # xj: (b, c, h, w)
            xj = frames_embed[j]
            # sx_j: (b, num_patches, embed_dim)
            sx_j = self.encoder_x(self.patch_embed(xj) + self.pos_embed)
            sx[j] = sx_j

        # predict sx_i from sx_0
        # sx_i: (b, num_patches, embed_dim)
        sx_i = sx[i]
        pred_sx_i = self.predictor(sx_0, sx_i)

        return pred_sx_i, sx_i, sx_0, sx

class HJEPA(nn.Module):
    def __init__(self, img_size, patch_size, in_channels, embed_dim, num_heads, encoder_x, encoder_y, predictor, H=11):
        super(HJEPA, self).__init__()

        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        self.patch_dim = self.patch_embed.patch_dim
        self.num_tokens = self.patch_embed.num_tokens
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_tokens, embed_dim))
        
        self.H = H
        self.jepa_layers = nn.ModuleList([JEPA(img_size, patch_size, in_channels, embed_dim, num_heads, encoder_x, encoder_y, predictor) for _ in range(H - 1)])
        self.sx_embeds = []
        self.sy_embeds = []
        self.s_preds = []

    # video: (b, c, h, w * H)
    def forward(self, video):
        
        # encode video into patches
        # video: (b, c, h, w * H)
        # frames[i]: (b, c, h, w)

        frames = torch.split(video, 1, dim=3)
        frames_embed = [None for _ in range(self.H)]
        for i in range(self.H):
            frames_embed[i] = self.patch_embed(frames[i]) + self.pos_embed
        
        for i in range(self.H - 1):
            pred_sx_i, sx_i, sx_0, sx = self.jepa_layers[i](frames_embed, i)

            self.sx_embeds.append(sx_i)
            self.sy_embeds.append(sx_0)
            self.s_preds.append(pred_sx_i)

            frames_embed = sx

        return self.s_preds, self.sx_embeds, self.sy_embeds


        




       
