import torch.nn as nn
import torch
import torch.cuda.nvtx as nvtx


class SwiGLU(nn.Module):
    def __init__(self, d_model, d_ff,dtype=torch.bfloat16,device=torch.device("cuda")):
        super().__init__()
        self.gate = nn.Parameter(torch.empty(d_model,d_ff,dtype=dtype,device=device))
        self.proj = nn.Parameter(torch.empty(d_model, d_ff,dtype=dtype,device=device))
        self.out_proj = nn.Parameter(torch.empty(d_ff,d_model,dtype=dtype,device=device))
        self.activation = nn.SiLU()
        self.init_parameters()

    @nvtx.range("SwiGLU")
    def forward(self, x):
        x_gate = torch.matmul(x, self.gate)
        x_proj = torch.matmul(x, self.proj)
        x_activated = self.activation(x_gate)
        x_ffn = torch.matmul(x_activated * x_proj, self.out_proj)
        return x_ffn

    def init_parameters(self):
        torch.nn.init.trunc_normal_(
            self.gate,
            std=torch.sqrt(
                torch.tensor(1.0 / (self.gate.size(0) + self.gate.size(1)))
            ).item(),
        )
        torch.nn.init.trunc_normal_(
            self.proj,
            std=torch.sqrt(
                torch.tensor(1.0 / (self.proj.size(0) + self.proj.size(1)))
            ).item(),
        )
        torch.nn.init.trunc_normal_(
            self.out_proj,
            std=torch.sqrt(
                torch.tensor(1.0 / (self.out_proj.size(0) + self.out_proj.size(1)))
            ).item(),
        )
