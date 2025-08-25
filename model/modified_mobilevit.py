import torch
import torch.nn as nn

# Import from other files in the 'model' directory
from model.transformer_block import TransformerBlock
from model.radial_positional_encoding import RadialPositionEmbedding
from model.utils import fold_tokens_to_grid

class ModifiedMobileViT(nn.Module):
    """
    Custom MobileViT-style block with the corrected, intelligent fusion method.
    """
    def __init__(self, in_channels, embed_dim, num_heads=2):
        super().__init__()
        self.local_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.SiLU()
        )
        self.proj_in = nn.Conv2d(in_channels, embed_dim, kernel_size=1)
        self.token_proj = nn.Linear(9, embed_dim)
        self.pos_encoder = RadialPositionEmbedding(num_rings=4, embed_dim=embed_dim)
        self.transformer = TransformerBlock(embed_dim, num_heads)
        self.proj_out = nn.Conv2d(embed_dim, in_channels, kernel_size=1)
        
        # This fuse block correctly accepts 2 * in_channels for concatenation
        self.fuse = nn.Sequential(
            nn.Conv2d(2 * in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.SiLU()
        )

    def forward(self, x, tokens):
        res = x
        
        # This part processes the global information
        x_local = self.local_conv(x)
        x_proj = self.proj_in(x_local)
        tokens_proj = self.token_proj(tokens)
        tokens_encoded = self.pos_encoder(tokens_proj)
        tokens_transformed = self.transformer(tokens_encoded)
        x_global = fold_tokens_to_grid(tokens_transformed, output_size=x_proj.shape[2:])
        x_global = self.proj_out(x_global)
        
        # Concatenate the features instead of adding them.
        x_fused = torch.cat([res, x_global], dim=1)
        
        x = self.fuse(x_fused)
        return x