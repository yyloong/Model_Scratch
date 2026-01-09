#### 数据处理
- 给原始文本分段,加上<im_start>,<im_end>等特殊符号
- 按照上下文长度使用tokenizer 进行tokenize并进行存储
- 重新加载数据的时候需要计算position_ids,因为不同段可能会拼接在一起需要从每段开始的地方重置position