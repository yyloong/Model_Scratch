import torch
import torch.distributed as dist
import logging
import wandb
import os
import argparse
import deepspeed
import json
import time  
import math

from Model.mini_llm import MiniLLM, MiniLLMConfig
from data.dataset import get_dataloader
from dataclasses import dataclass, asdict
from transformers import AutoTokenizer

os.environ["WANDB_START_METHOD"] = "thread"
os.environ["WANDB_CONSOLE"] = "off" 

def init_loger(log_file):
    class AnsiColor:
        BLACK = "\x1b[30m"; RED = "\x1b[31m"; GREEN = "\x1b[32m"; YELLOW = "\x1b[33m"
        BLUE = "\x1b[34m"; MAGENTA = "\x1b[35m"; CYAN = "\x1b[36m"; WHITE = "\x1b[37m"
        RESET = "\x1b[0m"
    class CustomFormatter(logging.Formatter):
        FORMATS = {
            logging.DEBUG: f"{AnsiColor.CYAN}%(levelname)s{AnsiColor.RESET}: %(message)s",
            logging.INFO: f"{AnsiColor.GREEN}%(levelname)s{AnsiColor.RESET}: %(message)s",
            logging.WARNING: f"{AnsiColor.YELLOW}%(levelname)s{AnsiColor.RESET}: %(message)s",
            logging.ERROR: f"{AnsiColor.RED}%(levelname)s{AnsiColor.RESET}: %(message)s",
            logging.CRITICAL: f"{AnsiColor.RED}%(levelname)s{AnsiColor.RESET}: %(message)s",
        }
        def format(self, record):
            log_fmt = self.FORMATS.get(record.levelno)
            formatter = logging.Formatter(log_fmt)
            return formatter.format(record)
    logger = logging.getLogger(__name__); logger.setLevel(logging.DEBUG)
    if logger.handlers: logger.handlers.clear()
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    console_handler = logging.StreamHandler(); console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(CustomFormatter())
    logger.addHandler(file_handler); logger.addHandler(console_handler)
    return logger

def init_wandb(project_name, run_name, run_id, config_dict=None):
    return wandb.init(
        project=project_name, 
        name=run_name, 
        id=run_id, 
        resume="allow",
        config=config_dict
    )

def train(args, config, ds_config_dict):
    model_config = MiniLLMConfig()
    model = MiniLLM(model_config)

    model_engine, optimizer, _, scheduler = deepspeed.initialize(
        args=args,
        model=model,
        model_parameters=model.parameters()
    )

    model_engine = torch.compile(model_engine)

    is_master = model_engine.local_rank == 0
    device = model_engine.device

    full_config = asdict(config)
    full_config.update(ds_config_dict)

    if is_master:
        logger = init_loger(config.log_file)
        wandb_id = config.wandb_run_id if config.wandb_run_id else config.run_name
        init_wandb(config.project_name, config.run_name, wandb_id, full_config)
    else:
        logger = None
    
    gradient_accumulation_steps = model_engine.gradient_accumulation_steps()
    micro_batchsize = ds_config_dict['train_micro_batch_size_per_gpu']
    global_batch_size = model_engine.train_batch_size()
    world_size = dist.get_world_size() if dist.is_initialized() else 1

    if is_master:
        logger.info(f"DeepSpeed Config: {ds_config_dict}")
        logger.info(f"Gradient Accumulation Steps: {gradient_accumulation_steps}")
        logger.info(f"Micro Batch Size (per GPU): {micro_batchsize}")
        logger.info(f"Global Target Train Batch Size: {global_batch_size}")

    # --- 断点续连逻辑 ---
    # load_checkpoint 需要所有进程都调用
    if config.resume_train:
        load_path, client_sd = model_engine.load_checkpoint(config.save_path)
        if is_master and load_path is None:
            logger.warning("No checkpoint found, starting from scratch.")
            now_step, start_epoch = 0, 0
        else:
            # 确保 client_sd 在所有进程都有值（通常 load_checkpoint 会处理同步，但为了安全建议所有进程都读取）
            now_step = client_sd.get('step', 0) if client_sd else 0
            start_epoch = client_sd.get('epoch', 0) if client_sd else 0
            if is_master and now_step > 0:
                logger.info(f"Successfully resumed from step {now_step} at epoch {start_epoch}")
    else:
        now_step, start_epoch = 0, 0
    
    skip_step = 0
    start_step = now_step
    
    train_dataloader = get_dataloader(config.train_data_path, micro_batchsize, config.im_start_id, world_size=world_size,rank=model_engine.local_rank)
    val_dataloader = get_dataloader(config.val_data_path, config.val_batchsize, config.im_start_id,world_size = world_size,rank=model_engine.local_rank)

    model_engine.train()
    
    last_log_time = time.time()
    
    for epoch in range(start_epoch, config.total_epochs):

        if is_master:
            logger.info(f"epoch:{epoch} start")

        if hasattr(train_dataloader.sampler, 'set_epoch'):
            train_dataloader.sampler.set_epoch(epoch)

        for batch in train_dataloader:
            if skip_step < start_step:
                skip_step += 1
                if is_master:
                    logger.info(f"Skip step {skip_step} for continue train")
                continue
            
            batch_input_ids = batch['input_ids'].to(device)
            batch_labels = batch['labels'].to(device)
            batch_position_ids = batch['position_ids'].to(device)
            
            batch_cu_seqlens, batch_max_seqlen, batch_attn_mask = None, None, None
            if config.use_varlen_attn:
                if 'cu_seqlens' in batch:
                    batch_cu_seqlens = batch['cu_seqlens'].to(device)
                    batch_max_seqlen = batch['max_seqlen']
            else:
                if 'attention_mask' in batch: batch_attn_mask = batch['attention_mask'].to(device)
                batch_input_ids, batch_labels, batch_position_ids = batch_input_ids[..., :-1], batch_labels[..., 1:], batch_position_ids[..., :-1]
                if batch_attn_mask is not None: batch_attn_mask = batch_attn_mask[..., :-1, :-1]

            _, _, loss = model_engine(
                batch_input_ids, label_ids=batch_labels, position_ids=batch_position_ids,
                attn_mask=batch_attn_mask, cu_seqlens=batch_cu_seqlens, max_seqlen=batch_max_seqlen
            )

            model_engine.backward(loss)
            model_engine.step()
            
            if model_engine.is_gradient_accumulation_boundary():
                # --- 日志记录逻辑 (仅 Master) ---
                if is_master:
                    if now_step % config.train_log_every_step == 0:
                        current_time = time.time()
                        elapsed_time = current_time - last_log_time
                        
                        gpu_mem_allocated = torch.cuda.max_memory_allocated(device) / 1024 / 1024 / 1024 
                        gpu_mem_reserved = torch.cuda.memory_reserved(device) / 1024 / 1024 / 1024 
                        
                        current_lr = model_engine.get_lr()[0]
                        grad_norm = model_engine.get_global_grad_norm() if hasattr(model_engine, 'get_global_grad_norm') else 0.0

                        logger.info(
                            f"Step:{now_step} | Loss:{loss.item():.4f} | LR:{current_lr:.6f} | "
                            f"Time:{elapsed_time:.2f}s | Mem:{gpu_mem_allocated:.2f}GB"
                        )
                        
                        log_dict = {
                            "train/loss": loss.item(),
                            "train/learning_rate": current_lr,
                            "train/grad_norm": grad_norm,
                            "perf/step_time_sec": elapsed_time,
                            "perf/samples_per_sec": global_batch_size / elapsed_time * config.train_log_every_step, 
                            "sys/gpu_memory_allocated_gb": gpu_mem_allocated,
                            "sys/gpu_memory_reserved_gb": gpu_mem_reserved,
                            "epoch": epoch + (now_step / len(train_dataloader) * gradient_accumulation_steps) 
                        }
                        
                        wandb.log(log_dict, step=now_step)
                        last_log_time = current_time
                        torch.cuda.reset_peak_memory_stats(device)
                
                # --- 保存逻辑 (所有进程必须同时参与) ---
                # 修改点：移除了 if is_master 的缩进，确保所有进程都执行 save_checkpoint
                if now_step > 0 and now_step % config.save_every_step == 0:
                    client_sd = {'step': now_step, 'epoch': epoch}
                    model_engine.save_checkpoint(config.save_path, tag=f'step_{now_step}', client_state=client_sd)
                    if is_master:
                        logger.info(f"Saved checkpoint at step {now_step}")

                now_step += 1
                
                # --- 验证逻辑 ---
                if now_step > 0 and now_step % config.val_every_steps == 0:
                    model_engine.eval()
                    val_loss, count = 0.0, 0
                    if is_master: logger.info("Starting validation...")
                    
                    with torch.no_grad():
                        for val_batch in val_dataloader:
                            val_input_ids, val_labels, val_pos_ids = val_batch['input_ids'].to(device), val_batch['labels'].to(device), val_batch['position_ids'].to(device)
                            val_cu, val_max, val_mask = None, None, None
                            if config.use_varlen_attn:
                                if 'cu_seqlens' in val_batch: val_cu, val_max = val_batch['cu_seqlens'].to(device), val_batch['max_seqlen']
                            else:
                                if 'attention_mask' in val_batch: val_mask = val_batch['attention_mask'].to(device)
                                val_input_ids, val_labels, val_pos_ids = val_input_ids[..., :-1], val_labels[..., 1:], val_pos_ids[..., :-1]
                                if val_mask is not None: val_mask = val_mask[..., :-1, :-1]
                            _, _, val_batch_loss = model_engine(
                                val_input_ids, label_ids=val_labels, position_ids=val_pos_ids, attn_mask=val_mask, cu_seqlens=val_cu, max_seqlen=val_max
                            )
                            count += 1
                            val_loss += val_batch_loss.item()
                    
                    # 验证集的Loss聚合（可选）：通常只在 Master 打印即可，
                    # 但如果想获得精确的全局 Loss，应该使用 dist.all_reduce
                    stats = torch.tensor(val_loss, device=device)
                    dist.reduce(stats,dst=0,op=dist.ReduceOp.SUM)
                    
                    if is_master:
                        val_loss = stats.item() / world_size
                        avg_val_loss = val_loss / count
                        logger.info(f"step:{now_step}, val_loss: {avg_val_loss:.4f}")
                        wandb.log({"val/loss": avg_val_loss}, step=now_step)
                    
                    model_engine.train()

@dataclass
class Train_config:
    # --- 基础配置 ---
    project_name: str = "MiniLLM-Train-DeepSpeed"
    run_name: str = "deepspeed-run-0005-compile"
    wandb_run_id: str = "deepspeed-run-0005-compile" 
    
    log_file: str = "train_deepspeed.log"
    resume_train: bool = False
    
    # --- 关键开关 ---
    use_varlen_attn: bool = True 

    # --- 路径配置 ---
    train_data_path: str = "/home/u-longyy/data/mini_model/data/train_tokenized_data"
    val_data_path: str = "/home/u-longyy/data/mini_model/data/val_tokenized_data"
    save_path: str = "/home/u-longyy/data/mini_model/ds_checkpoints" 

    # --- Batch 和 Epoch ---
    val_batchsize: int = 16
    total_epochs: int = 5

    # --- 数据 ---
    im_start_id: int = 151644

    # --- 步数控制 ---
    val_every_steps: int = 500
    train_log_every_step: int = 10
    save_every_step: int = 1000

def get_args():
    parser = argparse.ArgumentParser(description="DeepSpeed MiniLLM Training")
    parser.add_argument('--local_rank', type=int, default=-1,
                        help='local rank passed from distributed launcher')
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()
    config = Train_config()

    with open(args.deepspeed_config, 'r') as f:
        ds_config_dict = json.load(f)

    train(
        args=args,
        config=config,
        ds_config_dict=ds_config_dict
    )