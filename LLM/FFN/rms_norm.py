import torch.nn as nn
import torch

class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-8,dtype=torch.bfloat16,device=torch.device("cuda")):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(d_model,dtype=dtype,device=device))

    def forward(self, x):
        rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)
        return x / rms * self.scale
        