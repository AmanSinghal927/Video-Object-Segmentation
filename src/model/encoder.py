import torch
import torch.nn as nn
from torch import einsum
from einops import rearrange, repeat, reduce
from einops.layers.torch import Rearrange

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = dots.softmax(dim=-1)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out =  self.to_out(out)
        return out


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
    def __init__(self, img_size, patch_size, num_frames, channels, dim, depth, heads, mlp_dim):
        super().__init__()
        self.num_patches = (img_size[0] // patch_size) * (img_size[1] // patch_size)
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
    
# ViViT
# @misc{arnab2021vivit,
#       title={ViViT: A Video Vision Transformer}, 
#       author={Anurag Arnab and Mostafa Dehghani and Georg Heigold and Chen Sun and Mario Lučić and Cordelia Schmid},
#       year={2021},
#       eprint={2103.15691},
#       archivePrefix={arXiv},
#       primaryClass={cs.CV}
# }

# helpers

def exists(val):
    return val is not None

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class ViViT(nn.Module):
    def __init__(
        self,
        *,
        image_size,
        image_patch_size,
        frames,
        frame_patch_size,
        num_classes,
        dim,
        spatial_depth,
        temporal_depth,
        heads,
        mlp_dim,
        pool = 'cls',
        channels = 3,
        dim_head = 64,
        dropout = 0.,
        emb_dropout = 0.
    ):
        super().__init__()
        image_height, image_width = pair(image_size) if type(image_size) == int else image_size
        patch_height, patch_width = pair(image_patch_size) if type(image_patch_size) == int else image_patch_size

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'
        assert frames % frame_patch_size == 0, 'Frames must be divisible by frame patch size'

        num_image_patches = (image_height // patch_height) * (image_width // patch_width)
        num_frame_patches = (frames // frame_patch_size)

        patch_dim = channels * patch_height * patch_width * frame_patch_size

        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.global_average_pool = pool == 'mean'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (f pf) (h p1) (w p2) -> b f (h w) (p1 p2 pf c)', p1 = patch_height, p2 = patch_width, pf = frame_patch_size),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim)
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_frame_patches, num_image_patches, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.spatial_cls_token = nn.Parameter(torch.randn(1, 1, dim)) if not self.global_average_pool else None
        self.temporal_cls_token = nn.Parameter(torch.randn(1, 1, dim)) if not self.global_average_pool else None

        self.spatial_transformer = Transformer(dim, spatial_depth, heads, dim_head, mlp_dim, dropout)
        self.temporal_transformer = Transformer(dim, temporal_depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, video):
        x = self.to_patch_embedding(video)
        b, f, n, _ = x.shape

        x = x + self.pos_embedding

        if exists(self.spatial_cls_token):
            spatial_cls_tokens = repeat(self.spatial_cls_token, '1 1 d -> b f 1 d', b = b, f = f)
            x = torch.cat((spatial_cls_tokens, x), dim = 2)

        x = self.dropout(x)

        x = rearrange(x, 'b f n d -> (b f) n d')

        # attend across space

        x = self.spatial_transformer(x)

        x = rearrange(x, '(b f) n d -> b f n d', b = b)

        # excise out the spatial cls tokens or average pool for temporal attention

        x = x[:, :, 0] if not self.global_average_pool else reduce(x, 'b f n d -> b f d', 'mean')

        # append temporal CLS tokens

        if exists(self.temporal_cls_token):
            temporal_cls_tokens = repeat(self.temporal_cls_token, '1 1 d-> b 1 d', b = b)

            x = torch.cat((temporal_cls_tokens, x), dim = 1)

        # attend across time

        x = self.temporal_transformer(x)

        # excise out temporal cls token or average pool

        x = x[:, 0] if not self.global_average_pool else reduce(x, 'b f d -> b d', 'mean')

        x = self.to_latent(x)
        return self.mlp_head(x)