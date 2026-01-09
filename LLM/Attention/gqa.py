import torch
import torch.cuda.nvtx as nvtx
from utils.scaled_dot_attention import scaled_dot_product_attention
from Embedding.RoPE import RoPE
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

try:
    from flash_attn import flash_attn_varlen_func
except ImportError:
    flash_attn_varlen_func = None

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
        self.out_proj = torch.nn.Linear(
            num_heads * head_dim, d_model, dtype=dtype, device=device
        )
        self.RoPE = RoPE(
            head_dim,
            max_position_embeddings,
            rope_theta_base,
            dtype=dtype,
            device=device,
        )

        self.dropout_p = dropout

    def forward(
        self,
        x,
        past_key_values=None,
        position_ids=None,
        attn_mask=None,
        use_cache=False,
        use_standard=True,
        cu_seqlens=None,
        max_seqlen=None,
    ):
        if cu_seqlens is not None:
            total_tokens, _ = x.size()
            B, T = 1, total_tokens 
        else:
            B, T, _ = x.size()

        q = self.q_proj(x)
        kv = self.kv_proj(x)

        if cu_seqlens is not None:
            q = q.view(total_tokens, self.num_heads, self.head_dim)
            kv = kv.view(total_tokens, self.kv_heads, 2 * self.head_dim)
        else:
            q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
            kv = kv.view(B, T, self.kv_heads, 2 * self.head_dim).transpose(1, 2)

        k, v = kv.chunk(2, dim=-1)

        if cu_seqlens is not None:
            q = q.unsqueeze(0).transpose(1, 2) 
            k = k.unsqueeze(0).transpose(1, 2)
            position_ids = position_ids.unsqueeze(0)
            
            q = self.RoPE(q, position_ids=position_ids)
            k = self.RoPE(k, position_ids=position_ids)
            
            q = q.transpose(1, 2).squeeze(0)
            k = k.transpose(1, 2).squeeze(0)
        else:
            q = self.RoPE(q, position_ids=position_ids)
            k = self.RoPE(k, position_ids=position_ids)

        if past_key_values is not None:
            past_k, past_v = past_key_values
            # [T,H,D]格式只在训练阶段开启，暂不支持kv-cache
            if cu_seqlens is None: 
                k = torch.cat((past_k, k), dim=2)
                v = torch.cat((past_v, v), dim=2)

        new_cache = (k, v) if use_cache else None

        rep = self.num_heads // self.kv_heads
        k = k.repeat_interleave(rep, dim=1)
        v = v.repeat_interleave(rep, dim=1)

        if cu_seqlens is not None:
            attn_output = flash_attn_varlen_func(
                q, k, v,
                cu_seqlens_q=cu_seqlens,
                cu_seqlens_k=cu_seqlens,
                max_seqlen_q=max_seqlen,
                max_seqlen_k=max_seqlen,
                dropout_p=self.dropout_p if self.training else 0.0,
                causal=True
            )
            attn_output = attn_output.reshape(total_tokens, self.num_heads * self.head_dim)
        
        else:
            is_causal = attn_mask is None and (past_key_values is None) and (T > 1)
            if not is_causal and attn_mask is not None:
                attn_mask = (
                    torch.zeros_like(attn_mask, dtype=q.dtype).masked_fill(
                        ~attn_mask, float("-inf")
                    )
                    if attn_mask is not None
                    else None
                )

            q = q.contiguous()
            k = k.contiguous()
            v = v.contiguous()

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

            attn_output = (
                attn_output.transpose(1, 2)
                .contiguous()
                .view(B, T, self.num_heads * self.head_dim)
            )

        output = self.out_proj(attn_output)
        return output, new_cache