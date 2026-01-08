import torch


def scaled_dot_product_attention(q, k, v, is_causal=True, mask=None, scale=None):
    assert not is_causal or mask is None
    assert is_causal or mask is not None

    d_k = k.size(-1)

    scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(
        torch.tensor(d_k, dtype=torch.float32)
    )

    if mask is None:
        mask = torch.tril(
            torch.ones((q.size(-2), k.size(-2)), device=q.device)
        )
        mask = mask.to(dtype=q.dtype)
        mask = 1.0 - mask
        mask = mask.masked_fill(mask == 1, float("-inf"))
    elif mask.dtype == torch.bool:
        mask = ~mask
        mask = mask.to(dtype=q.dtype).masked_fill(mask, float("-inf"))

    scores = scores + mask
    scores = scores - scores.amax(dim=-1, keepdim=True)

    attn_weights = torch.softmax(scores, dim=-1)

    attn_weights = torch.nan_to_num(attn_weights, nan=0.0)

    output = torch.matmul(attn_weights, v)

    if scale is not None:
        output = output * scale

    return output, attn_weights
