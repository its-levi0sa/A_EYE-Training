import torch
import torch.nn as nn

class RadialPositionEmbedding(nn.Module):
    def __init__(self, num_rings: int = 4, embed_dim: int = 192):
        super().__init__()
        self.num_rings = num_rings
        self.embed_dim = embed_dim
        self.embedding = nn.Embedding(num_embeddings=num_rings, embedding_dim=embed_dim)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, num_tokens, dim = x.shape
        assert num_tokens == self.num_rings
        assert dim == self.embed_dim
        indices = torch.arange(self.num_rings, device=x.device).unsqueeze(0).expand(B, -1)
        pos_embed = self.embedding(indices)
        return x + pos_embed