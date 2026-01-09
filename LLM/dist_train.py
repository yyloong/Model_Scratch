import torch
import logging
import torch.nn as nn
from Model.mini_llm import MiniLLM, MiniLLMConfig
from Model.transformer import Transformer
import wandb
from data.dataset import get_dataloader
from dataclasses import dataclass
from torch.amp import autocast
import torch.distributed as dist
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from transformers import get_cosine_schedule_with_warmup
from torch.optim import AdamW
from transformers import AutoTokenizer
import os


def distributed_init():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])  # 之前修复: 获取 world_size
        torch.cuda.set_device(local_rank)
        device = torch.device(local_rank)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        world_size = 1  # 之前修复
        local_rank = 0  # 之前修复

    capability = torch.cuda.get_device_capability()
    if capability[0] >= 7:
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        if local_rank == 0:
            print("Uses tensor cores")
    return local_rank, device, world_size  # 之前修复


def train_init(config):
    model = MiniLLM(MiniLLMConfig)
    optimizer = AdamW(
        params=model.parameters(),
        lr=config.lr,
        betas=config.betas,
        eps=config.eps,
        weight_decay=config.weight_decay,
        fused=True,
    )
    '''
    scheduler = CosineAnnealingWarmRestarts(
        optimizer=optimizer,
        T_0=config.T_0,
        T_mult=config.T_mult,
        eta_min=config.eta_min,
        last_epoch=config.last_epoch,
    )
    '''
    scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=config.num_warmup_steps,
        num_training_steps=config.num_training_steps,
    )

    return model, optimizer, scheduler


def save_checkpoint(model, epoch, step, optimizer, scheduler, path, max_checkpoint):
    if not os.path.exists(path):
        os.makedirs(path)

    checkpoints = [f for f in os.listdir(path) if f.endswith('.pth')]

    checkpoints.sort(key=lambda x: os.path.getmtime(os.path.join(path, x)))

    while len(checkpoints) >= max_checkpoint:
        oldest_checkpoint = checkpoints.pop(0)
        os.remove(os.path.join(path, oldest_checkpoint))
        print(f"Removed old checkpoint: {oldest_checkpoint}")

    checkpoint = {
        'epoch': epoch,
        'step': step,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
    }

    save_path = os.path.join(path, f'epoch_{epoch}_step_{step}_checkpoint.pth')
    torch.save(checkpoint, save_path)
    print(f"Saved checkpoint: {save_path}")


def load_checkpoint(path):
    checkpoint = torch.load(path, map_location='cpu')
    epoch = checkpoint['epoch']
    step = checkpoint['step']
    model = checkpoint['model']
    optimizer = checkpoint['optimizer']
    scheduler = checkpoint['scheduler']
    return epoch, step, model, optimizer, scheduler


def init_loger(log_file):
    class AnsiColor:
        BLACK = "\x1b[30m"
        RED = "\x1b[31m"
        GREEN = "\x1b[32m"
        YELLOW = "\x1b[33m"
        BLUE = "\x1b[34m"
        MAGENTA = "\x1b[35m"
        CYAN = "\x1b[36m"
        WHITE = "\x1b[37m"
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

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    if logger.handlers:  # 之前修复: 防止重复添加 handler
        logger.handlers.clear()

    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(
        logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    )  # 之前修复

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(CustomFormatter())  # 之前修复

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)  # 之前修复

    return logger


def init_wandb(project_name, run_name, notes=None, tags=None):
    run = wandb.init(
        project=project_name,
        name=run_name,
        notes=notes,
        tags=tags,
    )
    return run


def all_reduce_grads(model, world_size):
    grad_list = [param.grad for param in model.parameters() if param.grad is not None]
    if not grad_list: return 
    grad_tensor = torch.cat([g.view(-1) for g in grad_list])
    dist.all_reduce(grad_tensor, op=dist.ReduceOp.SUM)
    grad_tensor.div_(world_size)

    offset = 0
    for param in model.parameters():
        if param.grad is not None:
            numel = param.grad.numel()
            param.grad.copy_(grad_tensor[offset : offset + numel].view_as(param.grad))
            offset += numel

def train(config, local_rank, device, model, optimizer, scheduler):
    is_master = local_rank == 0

    if is_master:
        init_wandb(config.project_name, config.run_name)
        logger = init_loger(config.log_file)
    else:
        logger = None

    assert config.train_batchsize % config.micro_batchsize == 0, \
        f"train_batchsize ({config.train_batchsize}) must be divisible by micro_batchsize ({config.micro_batchsize})"
    
    gradient_accumulation_steps = config.train_batchsize // config.micro_batchsize
    if is_master:
        logger.info(f"Gradient Accumulation Steps: {gradient_accumulation_steps}")
        logger.info(f"Micro Batch Size: {config.micro_batchsize}")
        logger.info(f"Target Train Batch Size (per GPU): {config.train_batchsize}")

    start_epoch = 0
    now_step = 0 


    if config.resume_train:
        load_path = config.resume_path
        if load_path is None and os.path.exists(config.save_path):
            checkpoints = [
                f for f in os.listdir(config.save_path) if f.endswith('.pth')
            ]
            if checkpoints:
                checkpoints.sort(
                    key=lambda x: os.path.getmtime(os.path.join(config.save_path, x))
                )
                load_path = os.path.join(config.save_path, checkpoints[-1])

        if load_path and os.path.exists(load_path):
            if is_master:
                logger.info(f"Resuming training from {load_path}")

            r_epoch, r_step, r_model_state, r_optim_state, r_scheduler_state = (
                load_checkpoint(load_path)
            )

            model.load_state_dict(r_model_state)
            optimizer.load_state_dict(r_optim_state)
            scheduler.load_state_dict(r_scheduler_state)

            start_epoch = r_epoch
            now_step = r_step

            if is_master:
                logger.info(
                    f"Successfully resumed. Start Epoch: {start_epoch}, Step: {now_step}"
                )
        else:
            if is_master:
                logger.warning(
                    "Resume requested but no checkpoint found. Starting from scratch."
                )

    train_dataloader = get_dataloader(
        config.train_data_path, config.micro_batchsize, config.im_start_id
    )
    val_dataloader = get_dataloader(
        config.val_data_path, config.val_batchsize, config.im_start_id
    )

    model.train()
    optimizer.zero_grad() 
    
    current_accumulation_step = 0
    train_loss_accumulator = 0.0 

    for epoch in range(start_epoch, config.total_epochs):
        if is_master:
            logger.info(f"epoch:{epoch} start")

        for batch in train_dataloader:
            
            batch_input_ids = batch['input_ids'].to(device)
            batch_labels = batch['labels'].to(device)
            batch_position_ids = batch['position_ids'].to(device)
            
            batch_cu_seqlens = None
            batch_max_seqlen = None
            batch_attn_mask = None

            if config.use_varlen_attn:
                if 'cu_seqlens' in batch:
                    batch_cu_seqlens = batch['cu_seqlens'].to(device)
                    batch_max_seqlen = batch['max_seqlen']
            else:
                if 'attention_mask' in batch:
                    batch_attn_mask = batch['attention_mask'].to(device)
                
                batch_input_ids = batch_input_ids[..., :-1]
                batch_labels = batch_labels[..., 1:]
                batch_position_ids = batch_position_ids[..., :-1]
                if batch_attn_mask is not None:
                    batch_attn_mask = batch_attn_mask[..., :-1, :-1]

            with autocast(device_type="cuda", dtype=config.mix_dtype):
                _, _, loss = model(
                    batch_input_ids,
                    label_ids=batch_labels,
                    position_ids=batch_position_ids,
                    attn_mask=batch_attn_mask,
                    cu_seqlens=batch_cu_seqlens,
                    max_seqlen=batch_max_seqlen
                )

            loss = loss / gradient_accumulation_steps
            train_loss_accumulator += loss.item()
            loss.backward()

            current_accumulation_step += 1

            if current_accumulation_step % gradient_accumulation_steps == 0:
                all_reduce_grads(model, config.world_size)
                
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad() 
                
                train_loss_val = train_loss_accumulator 
                
                train_loss_tensor = torch.tensor(train_loss_val, device=device)
                dist.all_reduce(train_loss_tensor, op=dist.ReduceOp.AVG) 
                train_loss_val = train_loss_tensor.item()

                if is_master:
                    if now_step % config.train_log_every_step == 0:
                        logger.info(f"train loss:{train_loss_val:.4f}, step:{now_step}")
                        wandb.log({"train_loss": train_loss_val, "step": now_step})
                    
                    if now_step > 0 and now_step % config.save_every_step == 0:
                        save_checkpoint(model, epoch, now_step, optimizer, scheduler, config.save_path, config.max_checkpoint)

                train_loss_accumulator = 0.0
                now_step += 1 
                
                if now_step > 0 and now_step % config.val_every_steps == 0:
                    model.eval()
                    val_loss = 0.0
                    count = 0
                    with torch.no_grad():
                        for val_batch in val_dataloader:
                            val_input_ids = val_batch['input_ids'].to(device)
                            val_labels = val_batch['labels'].to(device)
                            val_pos_ids = val_batch['position_ids'].to(device)
                            
                            val_cu = None
                            val_max = None
                            val_mask = None
                            
                            if config.use_varlen_attn:
                                if 'cu_seqlens' in val_batch:
                                    val_cu = val_batch['cu_seqlens'].to(device)
                                    val_max = val_batch['max_seqlen']
                            else:
                                if 'attention_mask' in val_batch:
                                    val_mask = val_batch['attention_mask'].to(device)
                                val_input_ids = val_input_ids[..., :-1]
                                val_labels = val_labels[..., 1:]
                                val_pos_ids = val_pos_ids[..., :-1]
                                if val_mask is not None: val_mask = val_mask[..., :-1, :-1]

                            _, _, val_batch_loss = model(
                                val_input_ids, label_ids=val_labels, position_ids=val_pos_ids,
                                attn_mask=val_mask, cu_seqlens=val_cu, max_seqlen=val_max
                            )
                            count += 1
                            val_loss += val_batch_loss.item()
                    
                    val_loss_tensor = torch.tensor(val_loss / count, device=device)
                    dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.AVG)
                    val_loss = val_loss_tensor.item()
                    
                    if is_master:
                        logger.info(f"step:{now_step}, val_loss: {val_loss:.4f}")
                        wandb.log({"val_loss": val_loss}, step=now_step)
                    
                    model.train()


@dataclass
class Train_config:
    # --- 基础配置 ---
    project_name: str = "MiniLLM-Train"
    run_name: str = "run-001"
    log_file: str = "train.log"
    resume_train: bool = True
    resume_path: str = None

    # --- 关键开关 ---
    use_varlen_attn: bool = True 

    # --- 路径配置 ---
    train_data_path: str = "/home/u-longyy/data/mini_model/data/train_tokenized_data"
    val_data_path: str = "/home/u-longyy/data/mini_model/data/val_tokenized_data"
    save_path: str = "/home/u-longyy/data/mini_model/checkpoints"

    # --- 训练超参 ---
    lr: float = 1e-4
    betas: tuple = (0.9, 0.95)
    eps: float = 1e-8
    weight_decay: float = 0.1
    num_warmup_steps = 250
    num_training_steps = 25000

    # --- Batch 和 Epoch ---
    # [修改] 定义更加清晰
    # train_batchsize: 这里指单卡在进行一次 Optimizer Step 前需要看到的总样本数 (Local Gradient Accumulation Target)
    train_batchsize: int = 16
    # micro_batchsize: DataLoader 每次 spit 出来的实际 batch size (显存限制瓶颈)
    micro_batchsize: int = 16
    
    val_batchsize: int = 16
    total_epochs: int = 10

    # --- 分布式与数据 ---
    world_size: int = 4
    im_start_id: int = 151644

    # --- 步数控制 ---
    val_every_steps: int = 500
    train_log_every_step: int = 10
    save_every_step: int = 1000
    max_checkpoint: int = 5

    # dtype
    model_dtype = torch.bfloat16
    mix_dtype = torch.bfloat16


if __name__ == "__main__":
    local_rank, device, world_size = distributed_init()

    config = Train_config()
    config.world_size = world_size

    model, optimizer, scheduler = train_init(config)
    
    model = torch.compile(model)
    model.to(device=device, dtype=config.model_dtype)

    train(
        config,
        local_rank=local_rank,
        device=device,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
    )