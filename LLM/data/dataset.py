import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from datasets import load_from_disk

class PretrainDataset(Dataset):
    def __init__(self, dataset_path):
        self.dataset =  load_from_disk(dataset_path)

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
        """
        将 batch 内的样本摊平，并生成 cu_seqlens, position_ids 和 shift 后的 labels
        """
        # 1. 提取并堆叠 input_ids
        input_ids_list = [item["input_ids"] for item in batch]
        input_ids_batch = torch.stack(input_ids_list) 
        
        batch_size, seq_len = input_ids_batch.shape
        total_tokens = batch_size * seq_len

        # 2. 摊平 Input IDs
        flatten_input_ids = input_ids_batch.view(-1)
        
        # 3. 计算 cu_seqlens
        seq_boundaries = torch.arange(0, total_tokens + 1, seq_len, dtype=torch.int32, device=flatten_input_ids.device)
        
        doc_start_indices = (flatten_input_ids == self.im_start_id).nonzero(as_tuple=True)[0]
        doc_start_indices = doc_start_indices.to(dtype=torch.int32)
        
        cu_seqlens = torch.cat([seq_boundaries, doc_start_indices])
        cu_seqlens = torch.unique(cu_seqlens, sorted=True)
        
        # 4. 生成 Position IDs
        token_indices = torch.arange(total_tokens, device=flatten_input_ids.device)
        segment_ids = torch.bucketize(token_indices, cu_seqlens, right=True) - 1
        segment_starts = cu_seqlens[segment_ids]
        position_ids = token_indices - segment_starts

        # 5. 生成 Labels (核心修改逻辑)
        # -------------------------------------------------------------------------
        # 逻辑：Label[i] 应该是 Input[i+1]。
        # 这是一个 "左移" 操作（数据左移，让 i 对齐 i+1）。
        labels = flatten_input_ids.clone()
        
        # 5.1 整体移位：将 input_ids 向后读一位作为 label
        # labels[:-1] = input_ids[1:]
        # 注意：这会导致 labels 最后一个元素是脏数据，稍后会被 mask 掉
        labels[:-1] = flatten_input_ids[1:]
        
        # 5.2 Mask 掉每个片段的最后一个 Token
        # 既然我们用 cu_seqlens 切断了注意力，那么片段 A 的最后一个 token 
        # 就不应该预测片段 B 的第一个 token。
        # cu_seqlens 里的值不仅是片段的 Start，也是上一个片段的 End (exclusive)。
        # 所以 cu_seqlens[1:] - 1 也就是所有“切断点”的前一个位置（即段末）。
        
        # 获取每个 segment 结束位置的索引
        # cu_seqlens: [start_0, start_1, ..., total_tokens]
        # end_indices: [start_1 - 1, start_2 - 1, ..., total_tokens - 1]
        seq_end_indices = cu_seqlens[1:] - 1
        
        # 将所有段末的 label 设为 ignore_index
        # 这同时也自然处理了 total_tokens - 1 (整个 batch 的最后一个 token)
        labels[seq_end_indices.long()] = self.ignore_index
        # -------------------------------------------------------------------------

        # 6. 计算 Max SeqLen
        max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()

        return {
            "input_ids": flatten_input_ids,
            "labels": labels,                         # 已处理好的 labels
            "position_ids": position_ids.long(),
            "cu_seqlens": cu_seqlens.int(),
            "max_seqlen": max_seqlen
        }

def get_dataloader(dataset_path, batch_size, im_start_token_id, shuffle=True):
    # 这里为了演示 Mock 数据，实际使用请换回 PretrainDataset(dataset_path)
    dataset = PretrainDataset(dataset_path) 
    
    collator = VarlenCollator(im_start_token_id)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        # num_workers=0, # 调试时建议设为 0
        collate_fn=collator
    )
    return dataloader

# --- 验证逻辑 ---
if __name__ == "__main__":
    IMSTART_ID = 9999
    print(f"{'='*20} Checking Labels Logic {'='*20}")
    
    # 这里的 dataset_path 没用到，因为用了 MockDataset
    loader = get_dataloader("dummy_path", batch_size=2, im_start_token_id=IMSTART_ID, shuffle=False)

    for batch in loader:
        input_ids = batch['input_ids']
        labels = batch['labels']
        cu_seqlens = batch['cu_seqlens']
        
        print(f"Batch Flatten Shape: {input_ids.shape}")
        print(f"cu_seqlens: {cu_seqlens.tolist()}")
        
        print("\n--- Token vs Label Check ---")
        print(f"{'Idx':<4} | {'Input':<6} | {'Label':<6} | {'Note'}")
        
        inputs_list = input_ids.tolist()
        labels_list = labels.tolist()
        
        # 找出切割点以便高亮显示
        cut_points = set((cu_seqlens[1:] - 1).tolist())
        
        for i in range(len(inputs_list)):
            inp = inputs_list[i]
            lbl = labels_list[i]
            
            note = ""
            if i in cut_points:
                note = "<--- Segment End (Label should be -100)"
            elif inp == IMSTART_ID:
                note = "<--- Doc Start"
                
            # 验证逻辑：正常情况下 Label 应该是 Input 的下一个
            # 也就是 labels[i] == inputs[i+1]
            if i not in cut_points and i + 1 < len(inputs_list):
                 if lbl != inputs_list[i+1]:
                     note += " [ERROR: Shift incorrect]"
            
            print(f"{i:<4} | {inp:<6} | {lbl:<6} | {note}")
        
        break