import torch
import logging
import wandb
import os
import argparse
import deepspeed
import json  

from Model.mini_llm import MiniLLM, MiniLLMConfig
from data.dataset import get_dataloader
from dataclasses import dataclass
from transformers import AutoTokenizer

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

def init_wandb(project_name, run_name, notes=None, tags=None):
    return wandb.init(project=project_name, name=run_name, notes=notes, tags=tags)

def train(args, config, ds_config_dict):
    model_config = MiniLLMConfig()
    model = MiniLLM(model_config)
    model_engine, optimizer, _, scheduler = deepspeed.initialize(
        args=args,
        model=model,
        model_parameters=model.parameters()
    )

    is_master = model_engine.local_rank == 0
    device = model_engine.device

    if is_master:
        logger = init_loger(config.log_file)
        init_wandb(config.project_name, config.run_name)
    else:
        logger = None
    
    gradient_accumulation_steps = model_engine.gradient_accumulation_steps()
    if is_master:
        logger.info(f"DeepSpeed Config: {ds_config_dict}")
        logger.info(f"Gradient Accumulation Steps: {gradient_accumulation_steps}")
        logger.info(f"Micro Batch Size (per GPU): {model_engine.train_micro_batch_size_per_gpu()}")
        logger.info(f"Global Target Train Batch Size: {model_engine.train_batch_size()}")

    if config.resume_train:
        _, client_sd = model_engine.load_checkpoint(config.save_path)
        now_step = client_sd.get('step', 0) if client_sd else 0
        start_epoch = client_sd.get('epoch', 0) if client_sd else 0
        if is_master and now_step > 0:
            logger.info(f"Successfully resumed from step {now_step} at epoch {start_epoch}")
    else:
        now_step, start_epoch = 0, 0
    
    # [关键] 从 ds_config_dict 获取 micro_batch_size 以初始化 dataloader
    micro_batchsize = ds_config_dict['train_micro_batch_size_per_gpu']
    train_dataloader = get_dataloader(config.train_data_path, micro_batchsize, config.im_start_id)
    val_dataloader = get_dataloader(config.val_data_path, config.val_batchsize, config.im_start_id)

    model_engine.train()
    
    for epoch in range(start_epoch, config.total_epochs):
        if is_master:
            logger.info(f"epoch:{epoch} start")

        for batch in train_dataloader:
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
                if is_master:
                    if now_step % config.train_log_every_step == 0:
                        logger.info(f"train loss:{loss.item():.4f}, step:{now_step}, lr:{model_engine.get_lr()[0]:.6f}")
                        wandb.log({"train_loss": loss.item(), "step": now_step, "learning_rate": model_engine.get_lr()[0]})
                    
                    if now_step > 0 and now_step % config.save_every_step == 0:
                        client_sd = {'step': now_step, 'epoch': epoch}
                        model_engine.save_checkpoint(config.save_path, tag=f'step_{now_step}', client_state=client_sd)
                        logger.info(f"Saved checkpoint at step {now_step}")

                now_step += 1
                
                if now_step > 0 and now_step % config.val_every_steps == 0:
                    model_engine.eval()
                    val_loss, count = 0.0, 0
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
                    
                    if is_master:
                        avg_val_loss = val_loss / count
                        logger.info(f"step:{now_step}, val_loss: {avg_val_loss:.4f}")
                        wandb.log({"val_loss": avg_val_loss}, step=now_step)
                    
                    model_engine.train()

@dataclass
class Train_config:
    # --- 基础配置 ---
    project_name: str = "MiniLLM-Train-DeepSpeed"
    run_name: str = "deepspeed-run-001"
    log_file: str = "train_deepspeed.log"
    resume_train: bool = True
    
    # --- 关键开关 ---
    use_varlen_attn: bool = True 

    # --- 路径配置 ---
    train_data_path: str = "/home/u-longyy/data/mini_model/data/train_tokenized_data"
    val_data_path: str = "/home/u-longyy/data/mini_model/data/val_tokenized_data"
    save_path: str = "/home/u-longyy/data/mini_model/ds_checkpoints" # 检查点目录

    # --- 训练超参 (现在由 ds_config.json 控制) ---
    # num_warmup_steps 和 num_training_steps 仍在 JSON 中
    
    # --- Batch 和 Epoch (Batch size 由 ds_config.json 控制) ---
    val_batchsize: int = 16
    total_epochs: int = 10

    # --- 数据 ---
    im_start_id: int = 151644

    # --- 步数控制 ---
    val_every_steps: int = 500
    train_log_every_step: int = 10
    save_every_step: int = 1000

def get_args():
    parser = argparse.ArgumentParser(description="DeepSpeed MiniLLM Training")
    # [关键] DeepSpeed 会自动添加 --local_rank, --deepspeed, --deepspeed_config 等参数
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()
    config = Train_config()

    # [关键] 加载 DeepSpeed 配置文件以获取 dataloader 所需的参数
    with open(args.deepspeed_config, 'r') as f:
        ds_config_dict = json.load(f)

    # 启动训练，传入解析后的参数和配置
    train(
        args=args,
        config=config,
        ds_config_dict=ds_config_dict
    )