import torch
from Embedding.RoPE import RoPE
from utils.scaled_dot_attention import scaled_dot_product_attention
import torch.nn.functional as F


class MultiHeadLatentAttention(torch.nn.Module):
    def __init__(
        self,
        d_model,
        head_dim,
        latent_dim,
        rope_dim,
        num_heads,
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
        self.latent_dim = latent_dim
        self.rope_dim = rope_dim
        self.content_dim = head_dim - rope_dim

        assert d_model % (2 * num_heads) == 0, "d_model must be divisible by num_heads"

        self.q_proj = torch.nn.Linear(
            d_model, num_heads * head_dim, dtype=dtype, device=device
        )
        self.kv_down_proj = torch.nn.Linear(
            d_model, 2 * latent_dim, dtype=dtype, device=device
        )
        self.k_rope_proj = torch.nn.Linear(
            d_model, rope_dim, dtype=dtype, device=device
        )
        self.k_up_proj = torch.nn.Linear(
            latent_dim, num_heads * (head_dim - rope_dim), dtype=dtype, device=device
        )
        self.v_up_proj = torch.nn.Linear(
            latent_dim, num_heads * head_dim, dtype=dtype, device=device
        )
        self.out_proj = torch.nn.Linear(d_model, d_model, dtype=dtype, device=device)
        self.RoPE = RoPE(
            self.rope_dim, max_position_embeddings, rope_theta_base=rope_theta_base
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
    ):
        B, T, C = x.size()

        q = self.q_proj(x)

        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        kv = self.kv_down_proj(x)
        k_rope = self.k_rope_proj(x)
        k_rope = k_rope.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
        k_rope = self.RoPE(k_rope, position_ids)
        k_content, v = kv.chunk(2, dim=-1)

        if past_key_values is not None:
            past_k_latent, past_v_latent, past_k_rope = past_key_values
            k_content = torch.cat((past_k_latent, k_content), dim=1)
            k_rope = torch.cat((past_k_rope, k_rope), dim=1)
            v = torch.cat((past_v_latent, v), dim=1)
        new_cache = (k_content, v, k_rope) if use_cache else None

        k_content = (
            self.k_up_proj(k_content)
            .view(B, -1, self.num_heads, self.content_dim)
            .transpose(1, 2)
        )
        k = torch.cat([k_content, k_rope], dim=-1)
        v = self.v_up_proj(v).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        q[..., self.content_dim :] = self.RoPE(
            q[..., self.content_dim :], position_ids
        )

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
