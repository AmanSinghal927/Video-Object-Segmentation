import torch
import torch.nn as nn
import copy

class LatentMLP(nn.Module):
    def __init__(self, dim, depth, mlp_dim):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.Sequential(
                nn.Linear(dim, mlp_dim),
                nn.BatchNorm1d(mlp_dim),
                nn.LeakyReLU(),
                nn.Linear(mlp_dim, dim),
            ))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class PatchEmbedding(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=64):
        super().__init__()
        if isinstance(img_size, int):
            img_size = img_size, img_size
        if isinstance(patch_size, int):
            patch_size = patch_size, patch_size
        #calculate the number of patches
        self.patch_shape = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.patch_dim = patch_size[0] * patch_size[1] * in_chans
        self.num_tokens = self.patch_shape[0] * self.patch_shape[1]

        #convolutional layer to convert the image into patches
        self.conv = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )
        
    def forward(self, x):
        x = self.conv(x)
        #flatten the patches
        x = x.flatten(2).transpose(1, 2)
        return x

# input size (b, c, h, w) and output size (b, h, w), where b is the batch size, c is the number of channels, h is the height, and w is the width.
class JEPA(nn.Module):
    def __init__(self, img_size, patch_size, in_channels, embed_dim, encoder_x, encoder_y, predictor, skip=0):
        super(JEPA, self).__init__()

        self.skip = skip
        self.encoder_x = encoder_x
        self.encoder_y = encoder_y
        self.predictor = predictor
        self.norm = nn.LayerNorm(embed_dim)
        self.projection = LatentMLP(embed_dim, 2, 256)

    # frames: (b, num_frames=22, c, h, w)
    def forward(self, x, y, enc_xs, enc_ys):
        # print("In JEPA forward")
        # print(frames.shape)

        # TODO: decide architecture for JEPA
        #  1. current architecture: 11 frames together (b, embed_dim) (ViViT) encoded predicting 1 frame (ViViT (could be ViT))
        #  2. 1 frame encoded predicting 1 frame
        #  3. 11 frames encoded individually with ViT (b, 11, embed_dim) predicting 1 frame (b, 1, embed_dim)

        # print("x: ", x.shape)
        # print("y: ", y.shape)

        # encode x and y
        # TODO: decide on strategy for encoding x and y
        #  1. encode each frame of x individually and store in s_embeds
        #     this lets use use the same encoder for x and y, a ViT encoder
        #  2. encode x with ViVit encoder, same with y
        # x_embed: (b, embed_dim)
        # y_embed: (b, embed_dim)
        x_embed = self.encoder_x(x)
        y_embed = self.encoder_y(y)
        # print("x_embed: ", x_embed.shape)
        # print("y_embed: ", y_embed.shape)

        enc_xs.append(x_embed)
        enc_ys.append(y_embed)

        concat_enc_xs = torch.cat(enc_xs, dim=1)
        concat_enc_ys = torch.cat(enc_ys, dim=1)
        print("concat_enc_xs: ", concat_enc_xs.shape)
        print("concat_enc_ys: ", concat_enc_ys.shape)

        # Pass through HSA_x and HSA_y


        x_embed = self.norm(x_embed)
        y_embed = self.norm(y_embed)

        z = self.projection(x_embed)
        # L1 norm of z
        latent_loss = torch.norm(z, p=1, dim=1).mean()
        
        # predictor
        # from first 11 frames, predict the frame that is self.skip frames away
        # first 11 frames: (b, 11, num_tokens, embed_dim)
        # predict the frame that is self.skip frames away
        # pred_y: (b, num_tokens, embed_dim)
        pred_y = self.predictor(x_embed)
        # print('After Predictor: ', pred_y.shape)

        # calculate the residual
        # residual: (b, num_tokens, embed_dim)
        # residual = y - pred_y

        # add the residual to the last 11 frames
        # last_frames: (b, 11, num_tokens, embed_dim)
        # last_frames = last_frames + residual

        # concatenate the first 11 frames and the last 11 frames
        # frames_embed: (b, 22, num_tokens, embed_dim)
        # frames_embed = torch.cat((first_frames, last_frames), dim=1)

        return pred_y, y_embed, latent_loss






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


        




       
