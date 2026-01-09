import torch.nn as torch
import torch
from dataclasses import dataclass
from Model.transformer import Transformer


@dataclass
class MiniLLMConfig:
    vocab_size: int = 151936
    d_model: int = 1024
    d_ff: int = 3072
    rms_eps: float = 1e-6
    head_dim: int = 128
    num_heads: int = 16
    kv_heads: int = 8
    num_layers: int = 28
    dropout: float = 0.0
    max_position_embeddings: int = 1024
    attention_type: str = "gqa"
    rope_dim: int = None
    latent_dim: int = None
    end_token_id: int = 151645
    rope_theta_base: int = 10000
    dtype: torch.dtype = torch.bfloat16


class MiniLLM(torch.nn.Module):
    def __init__(self, config: MiniLLMConfig):
        super(MiniLLM, self).__init__()
        self.transformer = Transformer(
            vocab_size=config.vocab_size,
            d_model=config.d_model,
            d_ff=config.d_ff,
            head_dim=config.head_dim,
            num_heads=config.num_heads,
            kv_heads=config.kv_heads,
            num_layers=config.num_layers,
            dropout=config.dropout,
            max_position_embeddings=config.max_position_embeddings,
            attention_type=config.attention_type,
            rope_dim=config.rope_dim,
            latent_dim=config.latent_dim,
            rms_eps=config.rms_eps,
            rope_theta_base=config.rope_theta_base,
            dtype=config.dtype,
            device=torch.device("cuda"),
        )
        self.end_token_id = config.end_token_id
        self.loss_fct = torch.nn.CrossEntropyLoss()

    def forward(
        self,
        input_ids,
        label_ids=None,
        past_key_values=None,
        position_ids=None,
        attn_mask=None,
        use_cache=False,
        use_standard=True,
        cu_seqlens=None,
        max_seqlen=None,
    ):
        if label_ids is None:
            self.transformer.eval()
            return (
                self.transformer(
                    input_ids,
                    past_key_values=past_key_values,
                    position_ids=position_ids,
                    attn_mask=attn_mask,
                    use_cache=use_cache,
                    use_standard=use_standard,
                    cu_seqlens=cu_seqlens,
                    max_seqlen=max_seqlen
                ),
                None,
            )

        outputs, _ = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            position_ids=position_ids,
            attn_mask=attn_mask,
            use_cache=use_cache,
            use_standard=use_standard,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen
        )
        # probs = torch.nn.functional.log_softmax(outputs,dim=-1)
        # all_loss = torch.gather(probs,dim=-1,index=label_ids,sparse_grad=True)
        # loss = -torch.mean(all_loss)
        logits = outputs.view(-1, outputs.size(-1))
        labels = label_ids.view(-1)

        loss = self.loss_fct(logits, labels)

        return outputs, _, loss

    def generate(self, input_ids, max_length, temperature=1.0, position_ids=None):
        self.transformer.eval()
        generated = input_ids
        past_key_values = None

        for _ in range(max_length):
            outputs, past_key_values = self.transformer(
                generated,
                past_key_values=past_key_values,
                position_ids=position_ids,
                use_cache=True,
            )
            next_token_logits = outputs[:, -1, :]
            next_token_logits = next_token_logits / temperature
            max_logits = torch.max(next_token_logits, dim=-1, keepdim=True).values
            probs = torch.softmax(next_token_logits - max_logits, dim=-1)
            topk_probs, topk_indices = torch.topk(probs, k=5, dim=-1)
            next_token = torch.multinomial(topk_probs, num_samples=1)
            next_token = torch.gather(topk_indices, -1, next_token)
            generated = torch.cat((generated, next_token), dim=1)
            if next_token.item() == self.end_token_id:
                break

        return generated
