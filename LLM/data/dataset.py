import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from datasets import load_from_disk

class PretrainDataset(Dataset):
    def __init__(self, dataset_path, rank=0, world_size=1):
        # 1. 加载数据集
        full_dataset = load_from_disk(dataset_path)
        
        # 2. 分布式切片
        if world_size > 1:
            self.dataset = full_dataset.shard(num_shards=world_size, index=rank, contiguous=True)
        else:
            self.dataset = full_dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        return {
            "input_ids": torch.tensor(item["input_ids"], dtype=torch.long)
        }

class VarlenCollator:
    def __init__(self, eot_token_id, ignore_index=-100):
        # 修改变量名为 eot_token_id 以符合语义
        self.eot_token_id = eot_token_id
        self.ignore_index = ignore_index

    def __call__(self, batch):
        # 1. 提取并堆叠 input_ids
        input_ids_list = [item["input_ids"] for item in batch]
        input_ids_batch = torch.stack(input_ids_list) 
        
        batch_size, seq_len = input_ids_batch.shape
        total_tokens = batch_size * seq_len

        # 2. 摊平 Input IDs
        flatten_input_ids = input_ids_batch.view(-1)
        
        # 3. 计算 cu_seqlens (核心修改部分)
        # 基础边界：物理上的 Batch 截断点 [0, 1024, 2048, ..., total_tokens]
        seq_boundaries = torch.arange(0, total_tokens + 1, seq_len, dtype=torch.int32, device=flatten_input_ids.device)
        
        # --- 修改开始 ---
        # 寻找 <|end_of_text|> 的位置
        eot_indices = (flatten_input_ids == self.eot_token_id).nonzero(as_tuple=True)[0]
        
        # 关键逻辑：Position 重启发生在 end_of_text 之后
        # 所以切分点应该是 eot_index + 1
        doc_end_split_indices = eot_indices + 1
        doc_end_split_indices = doc_end_split_indices.to(dtype=torch.int32)
        
        # 合并边界并排序去重
        # 注意：如果最后一个 token 是 end_of_text，doc_end_split_indices 会包含 total_tokens
        # 此时 seq_boundaries 也包含 total_tokens，torch.unique 会自动处理重复，不会报错
        cu_seqlens = torch.cat([seq_boundaries, doc_end_split_indices])
        cu_seqlens = torch.unique(cu_seqlens, sorted=True)
        # --- 修改结束 ---
        
        # 4. 生成 Position IDs
        token_indices = torch.arange(total_tokens, device=flatten_input_ids.device)
        # 使用 bucketize 找到每个 token 属于哪个 segment
        segment_ids = torch.bucketize(token_indices, cu_seqlens, right=True) - 1
        segment_starts = cu_seqlens[segment_ids]
        position_ids = token_indices - segment_starts

        # 5. 生成 Labels (Shifted)
        labels = flatten_input_ids.clone()
        labels[:-1] = flatten_input_ids[1:]
        
        # 将每个 segment 的末尾 token 的 label 设为 ignore_index
        # 原理：
        # 1. 物理边界截断处：无法预测下一个物理块的开头（无关联），需要 mask。
        # 2. End_of_text 处：这是文档结尾，虽然 labelsShift 后它的 label 是下一篇文档的开头，
        #    但我们不希望模型去学习跨文档的预测，所以需要 mask。
        # cu_seqlens[1:] - 1 正好涵盖了这两种情况。
        seq_end_indices = cu_seqlens[1:] - 1
        
        # 防止越界（虽然通常 seq_end_indices 最大也就是 total_tokens-1，但在 shift 操作中需要小心）
        # 这里 labels 长度也是 total_tokens，所以直接索引是安全的
        labels[seq_end_indices.long()] = self.ignore_index

        # 6. 计算 Max SeqLen
        max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()

        return {
            "input_ids": flatten_input_ids,
            "labels": labels,
            "position_ids": position_ids.long(),
            "cu_seqlens": cu_seqlens.int(),
            "max_seqlen": max_seqlen
        }

def get_dataloader(dataset_path, batch_size, eot_token_id, rank=0, world_size=1):
    # 修正：实例化 Dataset 时传入 rank 和 world_size，否则分布式切片不生效
    dataset = PretrainDataset(dataset_path, rank=rank, world_size=world_size)
    
    sampler = None
    if world_size > 1:
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(sampler is None), 
        sampler=sampler,
        # 传入 eot_token_id
        collate_fn=VarlenCollator(eot_token_id),
        drop_last=True
    )
    return dataloader