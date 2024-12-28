
import torch
import torch.nn.functional as F

from torch import nn, optim

import pytorch_lightning as pl


class MaskedMultihead_attention(pl.LightningModule):
    def __init__(self,n_heads=8, d_model=512, context_len=6):
        super().__init__()

        self.context_len = context_len
        self.n_heads = n_heads
        self.d_model = d_model
        self.dk = d_model // n_heads

        self.Wq = nn.Linear(d_model, d_model)
        self.Wk = nn.Linear(d_model, d_model)
        self.Wv = nn.Linear(d_model, d_model)
        self.Wo = nn.Linear(d_model, d_model)

        self.register_buffer(
            "causal_mask",
            torch.triu(torch.ones(context_len, context_len), diagonal=1).bool()
        )

    def forward(self, q,k,v):
        batch_size = q.size(0)

        q = self.Wq(q)
        k = self.Wk(k)
        v = self.Wv(v)

        # Reshaping into (Batchsize, n_heads, context len, d_k)
        q = q.view(batch_size, self.n_heads, self.context_len, self.d_k)
        k = k.view(batch_size, self.n_heads, self.context_len, self.d_k)
        v = v.view(batch_size, self.n_heads, self.context_len, self.d_k)

        # Attention between all the heads
        kt = k.transpose(-1, -2)  # (Batchsize, n_heads,d_k,context len)
        scores = torch.matmul(q, kt)  # (Batchsize, n_heads,context len,context len)
        scores = scores / torch.sqrt(torch.tensor(self.dk).float())  # (Batchsize, n_heads,context len,context len)

        # Causal mask
        scores = scores.masked_fill(self.causal_mask[:self.context_len, :self.context_len], float('-inf'))

        scores = F.softmax(scores, dim=-1)  # (Batchsize, n_heads,context len,context len)

        # (Batchsize, n_heads,context len,context len) ** (Batchsize, n_heads, context len, d_k)
        output = torch.matmul(scores, v)  # (Batchsize, n_heads,context len,d_k)
        output = output.contiguous().view(batch_size, self.context_len, self.d_model)
        output = self.Wo(output)

        return output  # (Batch size, context length, d_model)

