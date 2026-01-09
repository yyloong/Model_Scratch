export CUDA_VISIBLE_DEVICES=4,5,6,7
CUDA_LAUNCH_BLOCKING=1 torchrun --nproc_per_node=4 dist_train.py