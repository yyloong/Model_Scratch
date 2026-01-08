import torch
import torch.nn as nn


class WordEmbedding(torch.nn.Module):
    def __init__(self, vocab_size, embed_dim, dtype=torch.bfloat16, device=torch.device("cuda")):
        super().__init__()
        self.embedding = nn.Parameter(
            torch.empty(vocab_size, embed_dim,dtype=dtype,device=device)
        )
        self.init_parameters()

    def forward(self, input_ids):
        return self.embedding[input_ids]

    def init_parameters(self):
        torch.nn.init.trunc_normal_(self.embedding)
