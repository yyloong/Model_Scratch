import torch
import timeit
import argparse
import numpy as np
import sys
import torch.cuda.nvtx as nvtx

# 假设你的模型定义保存在 transformer.py 中，或者直接将你的类定义粘贴在上方
# 这里我们尝试从当前环境导入，确保你的目录结构正确
try:
    from Model.transformer import Transformer
except ImportError:
    print("Error: Could not import 'Transformer' class. "
          "Make sure this script is in the same directory as your model code "
          "or the 'Transformer' class is defined in this file.")
    sys.exit(1)

def get_model_config(size_name):
    """
    根据作业中的 Table 1 定义模型配置。
    这里提供了一些常见的配置作为示例，请根据实际作业要求修改数值。
    """
    configs = {
        # 示例配置，请根据作业 Table 1 填写具体数值
        "small":  {"d_model": 512,  "num_layers": 4,  "num_heads": 8,  "head_dim": 64, "d_ff": 2048},
        "medium": {"d_model": 1024, "num_layers": 8,  "num_heads": 16, "head_dim": 64, "d_ff": 4096},
        "large":  {"d_model": 2048, "num_layers": 16, "num_heads": 32, "head_dim": 64, "d_ff": 8192},
        "xl":     {"d_model": 4096, "num_layers": 32, "num_heads": 64, "head_dim": 64, "d_ff": 16384},
    }
    
    if size_name not in configs:
        raise ValueError(f"Unknown config size: {size_name}")
    
    return configs[size_name]

def benchmark(model, input_ids, n_steps, warmup_steps, mode="forward"):
    """
    执行基准测试的核心函数
    """
    
    # 1. Warm-up (预热)
    # 预热对于 GPU 及其重要，用于初始化 CUDA context, 分配显存缓存, 以及 cuDNN 的 auto-tuning
    print(f"Running {warmup_steps} warm-up steps ({mode})...")
    model.train() # 启用 Dropout 等
    for _ in range(warmup_steps):
        if mode == "forward":
            with torch.no_grad(): # 纯前向通常不需要梯度，但在训练模式下做前向也可以
                _ = model(input_ids,use_standard=True)
        elif mode == "forward_backward":
            # 清零梯度
            model.zero_grad()
            output = model(input_ids,use_standard=True)
            # 伪造一个 Loss 进行反向传播
            loss = output.mean()
            loss.backward()
        
        # 极为重要：等待 GPU 完成所有任务
        torch.cuda.synchronize()

    # 2. Measurement (正式计时)
    print(f"Running {n_steps} measurement steps ({mode})...")
    times = []
    
    for _ in range(n_steps):
        # 确保计时前 GPU 空闲
        torch.cuda.synchronize()
        start_time = timeit.default_timer()
        
        if mode == "forward":
            with torch.no_grad():
                with nvtx.range("Forward Pass"):
                    _ = model(input_ids,use_standard=True)
        elif mode == "forward_backward":
            model.zero_grad()
            with nvtx.range("Forward + Backward Pass"):
                output = model(input_ids,use_standard=True)
            loss = output.mean()
            loss.backward()
        
        # 确保计时在 GPU 完成后停止
        torch.cuda.synchronize()
        end_time = timeit.default_timer()
        
        times.append(end_time - start_time)

    return times

def main():
    parser = argparse.ArgumentParser(description="Transformer Benchmarking Script")
    parser.add_argument("--model_size", type=str, default="small", choices=["small", "medium", "large", "xl"], help="Model size from Table 1")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--seq_len", type=int, default=128, help="Context length (sequence length)")
    parser.add_argument("--warmup_steps", type=int, default=5, help="Number of warm-up steps")
    parser.add_argument("--n_steps", type=int, default=10, help="Number of measurement steps")
    parser.add_argument("--vocab_size", type=int, default=50257, help="Vocabulary size")
    parser.add_argument("--kv_heads", type=int, default=4, help="KV heads for GQA")
    
    args = parser.parse_args()
    
    # 检查 CUDA 是否可用
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    if device == "cpu":
        print("Warning: Benchmarking on CPU is not representative for this assignment.")
    '''
    from torch.nn.attention import SDPAParams
    from torch.backends.cuda import can_use_flash_attention

    # 准备你的 Q, K, V（必须在 CUDA 上，推荐 fp16 或 bfloat16）
    query = torch.randn(8, 8, 8, 8, dtype=torch.float16, device="cuda")   # [B, H_q, S_q, D]
    key   = torch.randn(8, 8, 8, 8, dtype=torch.float16, device="cuda")   # [B, H_k, S_k, D]
    value = torch.randn(8, 8, 8, 8, dtype=torch.float16, device="cuda")

    # 关键：用 SDPAParams 包装参数
    params = SDPAParams(
        query=query,
        key=key,
        value=value,
        attn_mask=None,          # 如果有 mask 就传进去（布尔或浮点 mask）
        dropout_p=0.0,           # 训练时 >0，推理通常 0.0
        is_causal=False          # True 表示下三角因果掩码
    )

    # 检查是否可以使用 Flash Attention
    result = can_use_flash_attention(params)

    print("是否可以使用 Flash Attention:", result)
    '''

    # 获取配置
    config = get_model_config(args.model_size)
    print(f"Model Configuration: {config}")

    # 初始化模型
    # 注意：这里需要补全你代码中 init 需要的所有参数
    model = Transformer(
        vocab_size=args.vocab_size,
        d_model=config["d_model"],
        d_ff=config["d_ff"],
        head_dim=config["head_dim"],
        num_heads=config["num_heads"],
        kv_heads=args.kv_heads, # 假设 KV heads 固定或通过 args 传入
        num_layers=config["num_layers"],
        max_position_embeddings=2048, # 必须 >= seq_len
        attention_type="gqa"
    ).to(device)

    # 打印参数量
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model Parameters: {num_params / 1e6:.2f} M")

    # 生成随机数据
    input_ids = torch.randint(0, args.vocab_size, (args.batch_size, args.seq_len)).to(device)

    # ==========================
    # Run Forward Pass Benchmark
    # ==========================
    fwd_times = benchmark(model, input_ids, args.n_steps, args.warmup_steps, mode="forward")
    fwd_avg = np.mean(fwd_times)
    fwd_std = np.std(fwd_times)
    print(f"\n[Forward Only] Avg: {fwd_avg*1000:.4f} ms | Std: {fwd_std*1000:.4f} ms")

    # ==========================
    # Run Fwd + Bwd Benchmark
    # ==========================
    # 注意：如果你显存不足，可能需要调小 batch_size
    fb_times = benchmark(model, input_ids, args.n_steps, args.warmup_steps, mode="forward_backward")
    fb_avg = np.mean(fb_times)
    fb_std = np.std(fb_times)
    print(f"\n[Forward + Backward] Avg: {fb_avg*1000:.4f} ms | Std: {fb_std*1000:.4f} ms")
    
    print("-" * 30)
    print(f"Backward pass est. cost: {(fb_avg - fwd_avg)*1000:.4f} ms")
    print(f"Ratio (Fwd+Bwd / Fwd): {fb_avg / fwd_avg:.2f}x")

if __name__ == "__main__":
    main()