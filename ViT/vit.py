import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import math


class PatchEmbedding(nn.Module):
    def __init__(self, image_size, patch_size, in_channels, embed_dim):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2

        self.projection = nn.Sequential(
            # Convert image into patches and flatten
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)',
                      p1=patch_size, p2=patch_size),
            nn.Linear(patch_size * patch_size * in_channels, embed_dim)
        )

        # Learnable classification token
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        # Learnable position embeddings
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches + 1, embed_dim))

    def forward(self, x):
        b = x.shape[0]  # batch size
        x = self.projection(x)

        # Repeat cls_token for each item in batch
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
        # Concatenate cls_token with patch embeddings
        x = torch.cat([cls_tokens, x], dim=1)
        # Add position embeddings
        x = x + self.pos_embedding

        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.attn_dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_dropout = nn.Dropout(dropout)

        # Initialize weights with normal distribution
        nn.init.xavier_uniform_(self.qkv.weight)
        self.qkv.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.proj.weight)
        self.proj.bias.data.fill_(0)

    def forward(self, x):
        batch_size, num_tokens, embed_dim = x.shape

        qkv = self.qkv(x)
        qkv = rearrange(qkv, 'b n (h d qkv) -> qkv b h n d',
                        h=self.num_heads, qkv=3)
        q, k, v = qkv

        # Scaled dot-product attention
        scale = self.head_dim ** -0.5
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)

        # Combine heads
        x = torch.matmul(attn, v)
        x = rearrange(x, 'b h n d -> b n (h d)')
        x = self.proj(x)
        x = self.proj_dropout(x)

        return x


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, dropout=0.):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.dropout2(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio=4., dropout=0.1, attn_dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads, attn_dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(
            in_features=embed_dim,
            hidden_features=int(embed_dim * mlp_ratio),
            out_features=embed_dim,
            dropout=dropout
        )

    def forward(self, x):
        # Multi-head attention with residual connection
        x = x + self.attn(self.norm1(x))
        # MLP with residual connection
        x = x + self.mlp(self.norm2(x))
        return x


class VisionTransformer(nn.Module):
    def __init__(
            self,
            image_size=224,
            patch_size=16,
            in_channels=3,
            num_classes=1000,
            embed_dim=768,
            depth=12,
            num_heads=12,
            mlp_ratio=4.,
            dropout=0.1,
            attn_dropout=0.1,
            embed_layer=PatchEmbedding,
    ):
        super().__init__()

        # Patch Embedding
        self.patch_embed = embed_layer(
            image_size=image_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim
        )

        # Transformer Encoder
        self.transformer = nn.Sequential(
            *[TransformerBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                attn_dropout=attn_dropout
            ) for _ in range(depth)]
        )

        # Classification head
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x):
        # Patch embedding
        x = self.patch_embed(x)

        # Transformer encoder
        x = self.transformer(x)

        # Classification head
        x = self.norm(x)
        x = x[:, 0]  # Use [CLS] token only
        x = self.head(x)

        return x


def create_vit(model_size='base', image_size=224, num_classes=1000):
    """
    Create a Vision Transformer model based on specified size.
    Available sizes: 'tiny', 'small', 'base', 'large'
    """
    configs = {
        'tiny': dict(
            patch_size=16,
            embed_dim=192,
            depth=12,
            num_heads=3,
        ),
        'small': dict(
            patch_size=16,
            embed_dim=384,
            depth=12,
            num_heads=6,
        ),
        'base': dict(
            patch_size=16,
            embed_dim=768,
            depth=12,
            num_heads=12,
        ),
        'large': dict(
            patch_size=16,
            embed_dim=1024,
            depth=24,
            num_heads=16,
        ),
    }

    if model_size not in configs:
        raise ValueError(f"Model size {model_size} not supported")

    config = configs[model_size]
    return VisionTransformer(
        image_size=image_size,
        num_classes=num_classes,
        **config
    )