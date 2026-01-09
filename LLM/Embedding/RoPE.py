import torch
import torch.nn as nn


class RoPE(nn.Module):
    def __init__(
        self,
        embed_dim,
        max_position_embeddings=1024,
        rope_theta_base=10000,
        dtype=torch.bfloat16,
        device=torch.device("cuda"),
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_position_embeddings = max_position_embeddings
        inv_freq = 1.0 / (
            rope_theta_base ** (torch.arange(0, embed_dim, 2,dtype=dtype,device=device) / embed_dim)
        )

        self.register_buffer(
            "cos",
            torch.cos(
                torch.arange(0, max_position_embeddings,dtype=dtype,device=device).unsqueeze(1) * inv_freq
            ),
            persistent=False,
        )

        self.register_buffer(
            "sin",
            torch.sin(
                torch.arange(0, max_position_embeddings,dtype=dtype,device=device).unsqueeze(1) * inv_freq
            ),
            persistent=False,
        )

    def forward(self, x, position_ids=None):
        seq_len = x.size(-2)
        if position_ids is None:
            position_ids = torch.arange(0,seq_len)
        cos = self.cos[position_ids].unsqueeze(1)
        sin = self.sin[position_ids].unsqueeze(1)
        cos = cos.repeat_interleave(2, dim=-1)
        sin = sin.repeat_interleave(2, dim=-1)

        x1 = x[..., ::2]
        x2 = x[..., 1::2]
        x_rotated = torch.stack([-x2,x1], dim=-1).reshape_as(x)

        return x * cos + x_rotated * sin
