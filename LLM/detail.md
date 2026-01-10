#### 数据处理
- 给原始文本分段,加上<im_start>,<im_end>等特殊符号
- 按照上下文长度使用tokenizer 进行tokenize并进行存储
- 重新加载数据的时候需要计算position_ids,因为不同段可能会拼接在一起需要从每段开始的地方重置position

#### 训练踩坑
- 数据集应该不同进程使用不同子集,不同epoch需要sample
- 优化器状态使用fp32
- deepspeed应该不同进程都保存权重，不然会死锁
- 生成标签仔细考虑维度，不然loss降不下来

#### 推理踩坑
- 记得开torch.no_grad()
- kv-cache每次只需要传入一个token+kv_cache
- 需要计算position_ids