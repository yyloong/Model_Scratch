import torch
import torch.cuda.nvtx as nvtx
from utils.scaled_dot_attention import scaled_dot_product_attention
from Embedding.RoPE import RoPE
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel


class GQAAttention(torch.nn.Module):
    def __init__(
        self,
        d_model,
        head_dim,
        num_heads,
        kv_heads,
        dropout=0.0,
        max_position_embeddings=1024,
        rope_theta_base=10000,
        dtype=torch.bfloat16,
        device=torch.device("cuda"),
    ):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.head_dim = head_dim
        self.kv_heads = kv_heads

        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        assert num_heads % kv_heads == 0, "num_heads must be divisible by kv_heads"

        self.q_proj = torch.nn.Linear(
            d_model, num_heads * head_dim, dtype=dtype, device=device
        )
        self.kv_proj = torch.nn.Linear(
            d_model, 2 * kv_heads * head_dim, dtype=dtype, device=device
        )
        self.out_proj = torch.nn.Linear(d_model, d_model, dtype=dtype, device=device)
        self.RoPE = RoPE(
            head_dim,
            max_position_embeddings,
            rope_theta_base,
            dtype=dtype,
            device=device,
        )

        self.dropout_p = dropout

    @nvtx.range("GQAAttention")
    def forward(
        self,
        x,
        past_key_values=None,
        attn_mask=None,
        use_cache=False,
        use_standard=True,
    ):
        B, T, C = x.size()

        q = self.q_proj(x)

        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        kv = self.kv_proj(x)
        kv = kv.view(B, T, self.kv_heads, 2 * self.head_dim).transpose(1, 2)
        k, v = kv.chunk(2, dim=-1)
        current_start_pos = 0
        if past_key_values is not None:
            current_start_pos = past_key_values[0].size(-2)

        q = self.RoPE(q, start_pos=current_start_pos)
        k = self.RoPE(k, start_pos=current_start_pos)

        if past_key_values is not None:
            past_k, past_v = past_key_values
            k = torch.cat((past_k, k), dim=2)
            v = torch.cat((past_v, v), dim=2)

        new_cache = (k, v) if use_cache else None
        k = k.repeat_interleave(self.num_heads // self.kv_heads, dim=1)
        v = v.repeat_interleave(self.num_heads // self.kv_heads, dim=1)

        is_causal = attn_mask is None and (past_key_values is None) and (T > 1)
        if not is_causal and attn_mask is not None:
            attn_mask = (
                torch.zeros_like(attn_mask, dtype=q.dtype).masked_fill(
                    ~attn_mask, float("-inf")
                )
                if attn_mask is not None
                else None
            )

        if use_standard:
            with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                attn_output = F.scaled_dot_product_attention(
                    q,
                    k,
                    v,
                    is_causal=is_causal,
                    attn_mask=attn_mask,
                    dropout_p=self.dropout_p,
                )
        else:
            attn_output, _ = scaled_dot_product_attention(
                q, k, v, is_causal=is_causal, mask=attn_mask, scale=None
            )

        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, C)
        output = self.out_proj(attn_output)
        return output, new_cache
