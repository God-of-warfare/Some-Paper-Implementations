
import torch

import math

import pytorch_lightning as pl


class PositionalEncoding(pl.LightningModule):
    def __init__(self,d_model=512, context_len=6):
        super().__init__()
       # self.x = x  # (Batch size, context length, d_model)
        self.d_model = d_model
        self.context_len = context_len
        pe = torch.zeros(context_len, d_model)

        position = torch.arange(0, context_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, context_len, d_model)

        # Register buffer (won't be updated during backprop)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe


