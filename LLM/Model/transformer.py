import torch.nn as nn
import torch
from Embedding.word_embedding import WordEmbedding
from FFN.ffn import SwiGLU
from FFN.rms_norm import RMSNorm
from Attention.gqa import GQAAttention
from Attention.mla import MultiHeadLatentAttention
import torch.cuda.nvtx as nvtx


class transformerblock(nn.Module):
    def __init__(
        self,
        d_model,
        d_ff,
        head_dim,
        num_heads,
        kv_heads,
        dropout=0.0,
        max_position_embeddings=1024,
        attention_type="gqa",
        rms_eps=1e-6,
        rope_dim=None,
        latent_dim=None,
        rope_theta_base=10000,
        dtype=torch.bfloat16,
        device=torch.device("cuda"),
    ):
        super().__init__()
        if attention_type == "gqa":
            self.attn = GQAAttention(
                d_model,
                head_dim,
                num_heads,
                kv_heads,
                dropout,
                max_position_embeddings,
                rope_theta_base,
                dtype,
                device,
            )
        elif attention_type == "mla":
            self.attn = MultiHeadLatentAttention(
                d_model,
                head_dim,
                latent_dim,
                rope_dim,
                num_heads,
                dropout,
                max_position_embeddings,
                rope_theta_base,
                dtype,
                device,
            )
        else:
            raise ValueError(f"Unknown attention type: {attention_type}")
        self.norm1 = RMSNorm(d_model, rms_eps,dtype,device)
        self.ffn = SwiGLU(d_model, d_ff,dtype,device)
        self.norm2 = RMSNorm(d_model, rms_eps,dtype,device)
        self.dropout = nn.Dropout(dropout)
        self.ffn.init_parameters()

    def forward(
        self,
        x,
        past_key_values=None,
        attn_mask=None,
        use_cache=False,
        use_standard=True,
    ):
        residual = x
        x = self.norm1(x)
        attn_output, new_cache = self.attn(
            x, past_key_values, attn_mask, use_cache, use_standard
        )
        x = residual + self.dropout(attn_output)

        residual = x
        x = self.norm2(x)
        ffn_output = self.ffn(x)
        x = residual + self.dropout(ffn_output)

        return x, new_cache


class Transformer(nn.Module):
    def __init__(
        self,
        vocab_size,
        d_model,
        d_ff,
        head_dim,
        num_heads,
        kv_heads,
        num_layers,
        dropout=0.0,
        max_position_embeddings=1024,
        attention_type="gqa",
        rms_eps=1e-6,
        rope_dim=None,
        latent_dim=None,
        rope_theta_base=10000,
        dtype=torch.bfloat16,
        device=torch.device("cuda"),
    ):
        super().__init__()
        self.embedding = WordEmbedding(vocab_size, d_model)
        self.layers = nn.ModuleList(
            [
                transformerblock(
                    d_model=d_model,
                    d_ff=d_ff,
                    head_dim=head_dim,
                    num_heads=num_heads,
                    kv_heads=kv_heads,
                    dropout=dropout,
                    max_position_embeddings=max_position_embeddings,
                    attention_type=attention_type,
                    rope_dim=rope_dim,
                    latent_dim=latent_dim,
                    rms_eps=rms_eps,
                    rope_theta_base=rope_theta_base,
                    dtype=dtype,
                    device=device,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm = RMSNorm(d_model, rms_eps,dtype=dtype,device=device)
        self.d_model = d_model
        self.out_proj = nn.Linear(d_model, vocab_size,dtype=dtype,device=device)

    def forward(
        self,
        input_ids,
        past_key_values=None,
        attn_mask=None,
        use_cache=False,
        use_standard=True,
    ):
        x = self.embedding(input_ids)
        new_past_key_values = []

        for i, layer in enumerate(self.layers):
            past = past_key_values[i] if past_key_values is not None else None
            x, new_cache = layer(x, past, attn_mask, use_cache, use_standard)
            if use_cache:
                new_past_key_values.append(new_cache)

        x = self.norm(x)
        x = self.out_proj(x)

        if use_cache:
            return x, new_past_key_values
        return x
