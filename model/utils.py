import torch.nn as nn
import torch

def fold_tokens_to_grid(tokens, output_size):
    B, P, D = tokens.shape
    H, W = output_size
    return tokens[:, 0].unsqueeze(-1).unsqueeze(-1).expand(B, D, H, W)

def conv_3x3_bn(inp, oup, stride=1):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.SiLU()
    )

class MV2Block(nn.Module):
    def __init__(self, inp, oup, stride=1, expansion=4):
        super().__init__()
        hidden_dim = int(inp * expansion)
        self.use_res_connect = stride == 1 and inp == oup
        self.conv = nn.Sequential(
            nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU(),
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        )
    def forward(self, x):
        return x + self.conv(x) if self.use_res_connect else self.conv(x)