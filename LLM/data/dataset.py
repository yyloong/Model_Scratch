import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from datasets import load_from_disk

class PretrainDataset(Dataset):
    def __init__(self, dataset_path, rank=0, world_size=1):
        # 1. 加载数据集
        full_dataset = load_from_disk(dataset_path)
        
        # 2. 关键修改：分布式切片
        # num_shards 等于总 GPU 数，index 为当前 GPU 的编号
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
    def __init__(self, im_start_id, ignore_index=-100):
        self.im_start_id = im_start_id
        self.ignore_index = ignore_index

    def __call__(self, batch):
        # 1. 提取并堆叠 input_ids
        # 这里假设你的每条数据 input_ids 长度都是固定的 1024
        input_ids_list = [item["input_ids"] for item in batch]
        input_ids_batch = torch.stack(input_ids_list) 
        
        batch_size, seq_len = input_ids_batch.shape
        total_tokens = batch_size * seq_len

        # 2. 摊平 Input IDs
        flatten_input_ids = input_ids_batch.view(-1)
        
        # 3. 计算 cu_seqlens (用于 Flash Attention / Varlen Attn)
        # 基础边界：[0, 1024, 2048, ...]
        seq_boundaries = torch.arange(0, total_tokens + 1, seq_len, dtype=torch.int32, device=flatten_input_ids.device)
        
        # 寻找文档起始符 <|im_start|> 的位置
        doc_start_indices = (flatten_input_ids == self.im_start_id).nonzero(as_tuple=True)[0]
        doc_start_indices = doc_start_indices.to(dtype=torch.int32)
        
        # 合并边界并排序去重
        cu_seqlens = torch.cat([seq_boundaries, doc_start_indices])
        cu_seqlens = torch.unique(cu_seqlens, sorted=True)
        
        # 4. 生成 Position IDs
        token_indices = torch.arange(total_tokens, device=flatten_input_ids.device)
        segment_ids = torch.bucketize(token_indices, cu_seqlens, right=True) - 1
        segment_starts = cu_seqlens[segment_ids]
        position_ids = token_indices - segment_starts

        # 5. 生成 Labels (Shifted)
        labels = flatten_input_ids.clone()
        labels[:-1] = flatten_input_ids[1:]
        
        # 将每个 segment 的末尾 token 的 label 设为 ignore_index
        seq_end_indices = cu_seqlens[1:] - 1
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

# 修改：增加 rank 和 world_size 参数
def get_dataloader(dataset_path, batch_size, im_start_token_id, rank=0, world_size=1):
    dataset = PretrainDataset(dataset_path) # 移除内部的 shard 逻辑
    
    sampler = None
    if world_size > 1:
        # 使用 DistributedSampler 确保每个 Epoch 数据分配不同
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(sampler is None), 
        sampler=sampler,
        collate_fn=VarlenCollator(im_start_token_id),
        drop_last=True
    )
    return dataloader