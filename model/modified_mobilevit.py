import torch.nn as nn

# Import from other files in the 'model' directory
from model.transformer_block import TransformerBlock
from model.radial_positional_encoding import RadialPositionEmbedding
from model.utils import fold_tokens_to_grid

class ModifiedMobileViT(nn.Module):
    """
    Custom MobileViT-style block.
    """
    def __init__(self, in_channels=32, embed_dim=192, num_heads=2):
        super().__init__()
        self.local_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.SiLU()
        )
        self.proj_in = nn.Conv2d(in_channels, embed_dim, kernel_size=1)
        self.token_proj = nn.Linear(9, embed_dim)
        # self.pos_encoder = RadialPositionEmbedding(num_rings=16, embed_dim=embed_dim)
        self.pos_encoder = RadialPositionEmbedding(num_rings=8, embed_dim=embed_dim)
        self.transformer = TransformerBlock(embed_dim, num_heads)
        self.proj_out = nn.Conv2d(embed_dim, in_channels, kernel_size=1)
        self.fuse = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.SiLU()
        )

    def forward(self, x, tokens):
        res = x
        x_local = self.local_conv(x)
        x_proj = self.proj_in(x_local)
        tokens_proj = self.token_proj(tokens)
        tokens_encoded = self.pos_encoder(tokens_proj)
        tokens_transformed = self.transformer(tokens_encoded)
        x_global = fold_tokens_to_grid(tokens_transformed, output_size=x_proj.shape[2:])
        x_global = self.proj_out(x_global)
        x = x_global + res
        x = self.fuse(x)
        return x