import torch
import torch.nn.functional as F
from utils.scaled_dot_attention import scaled_dot_product_attention
torch.backends.cuda.enable_flash_sdp(True)

def test_scaled_dot_product_attention():

    batch_size = torch.randint(1, 5, (1,)).item()
    seq_len_q = torch.randint(1, 10, (1,)).item()
    seq_len_k = torch.randint(seq_len_q, 10, (1,)).item()
    d_k = torch.randint(1, 20, (1,)).item()

    q = torch.randn(batch_size, seq_len_q, d_k)
    k = torch.randn(batch_size, seq_len_k, d_k)
    v = torch.randn(batch_size, seq_len_k, d_k)

    mask = torch.randint(0, 2, (batch_size, seq_len_q, seq_len_k))
    mask = mask.bool()
    is_causal = torch.randint(0, 2, (1,)).item() == 1
    if is_causal:
        mask = None
    
    output, _ = scaled_dot_product_attention(q, k, v, mask=mask, is_causal=is_causal)

    answer = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=is_causal)

    assert torch.allclose(output, answer, atol=1e-6), "Output does not match the expected result."

if __name__ == "__main__":
    test_time = 10
    for i in range(test_time):
        test_scaled_dot_product_attention()
        print(f"Test {i+1}/{test_time} passed.")
    print("All tests passed!")


    
    

