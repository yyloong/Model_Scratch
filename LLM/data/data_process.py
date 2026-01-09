import os
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer

# --- 配置参数 ---
model_name = "Qwen/Qwen3-0.6B"
input_file = "/home/u-longyy/data/mini_model/data/pretrain_hq.jsonl"  # 你的数据文件
train_output_path = "/home/u-longyy/data/mini_model/data/train_tokenized_data" # 输出目录
val_output_path = "/home/u-longyy/data/mini_model/data/val_tokenized_data" # 输出目录
block_size = 1024             # 预训练的上下文窗口长度
num_proc = 16                   # 并行进程数（根据 CPU 核心数调整）

def tokenize_and_pack():
    # 1. 初始化 Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    # 2. 加载原始 JSONL 数据
    # streaming=False 会直接处理并生成缓存，适合能放进磁盘的数据
    raw_dataset = load_dataset("json", data_files=input_file, split="train")

    split_dataset = raw_dataset.train_test_split(test_size=0.01, seed=42)
    train_dataset = split_dataset["train"]
    test_dataset = split_dataset["test"]

    # 3. 定义处理函数
    def group_texts(examples):
        # 步骤 A: 将所有文本 Tokenize
        # 不加 Padding, 不截断，保留原始长度
        tokenized_outputs = tokenizer(
            examples["text"], 
            add_special_tokens=False, 
            truncation=False
        )
        
        # 步骤 B: 将当前 Batch 的所有 Token ID 拼成一个长列表
        # 注意：这里我们假设你的文本里已经带了 <|im_end|>，
        # 如果没有，需要手动在这里添加 tokenizer.eos_token_id
        concatenated_ids = []
        for ids in tokenized_outputs["input_ids"]:
            concatenated_ids.extend(ids)
            
        # 步骤 C: 计算可以切分成多少个 block_size
        total_length = len(concatenated_ids)
        # 舍弃最后不足一个 block_size 的余数部分
        total_length = (total_length // block_size) * block_size
        
        # 步骤 D: 切分数据
        result = {
            "input_ids": [
                concatenated_ids[i : i + block_size] 
                for i in range(0, total_length, block_size)
            ]
        }

        return result

    # 4. 执行并行处理
    # batched=True 极大提高处理速度
    train_tokenized_dataset = train_dataset.map(
        group_texts,
        batched=True,
        num_proc=num_proc,
        remove_columns=raw_dataset.column_names, # 移除原始 text 列
        desc=f"Grouping texts into chunks of {block_size}"
    )
    val_tokenized_dataset = test_dataset.map(
        group_texts,
        batched=True,
        num_proc=num_proc,
        remove_columns=raw_dataset.column_names, # 移除原始 text 列
        desc=f"Grouping texts into chunks of {block_size}"
    )

    # 5. 保存到磁盘
    # 这将生成 arrow 格式文件，支持内存映射，加载速度极快
    train_tokenized_dataset.save_to_disk(train_output_path)
    print(f"Tokenization 完成！train 总 Chunk 数: {len(train_tokenized_dataset)}")
    val_tokenized_dataset.save_to_disk(val_output_path)
    print(f"Tokenization 完成！test 总 Chunk 数: {len(val_tokenized_dataset)}")

if __name__ == '__main__':
    tokenize_and_pack()