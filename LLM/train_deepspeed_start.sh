export CUDA_VISIBLE_DEVICES=4,5,6,7
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=INFO                      
export NCCL_TIMEOUT=3600000                 
export TORCH_DISTRIBUTED_DEBUG=DETAIL
deepspeed deepspeed_train.py --deepspeed_config ds_config.json
