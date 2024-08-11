import math
import torch
import torch.nn as nn

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim, learnable = False):
        super().__init__()
        self.dim = dim
        self.learnable = learnable
        # if self.learnable:
        self.matrix = nn.Parameter(torch.randn(1, dim // 2))

    def forward(self, x):
        device = x.device
        if self.learnable:
            emb = x[:, None] * self.matrix.weight  * 2 * math.pi
            emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
            return emb
        else:
            half_dim = self.dim // 2
            emb = math.log(10000) / (half_dim - 1)
            emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
            emb = x[:, None] * emb[None, :]
            emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
            return emb
