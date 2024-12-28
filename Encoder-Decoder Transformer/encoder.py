from multihead_attention import Multihead_attention
from positional_embedding import PositionalEncoding

from torch import nn, optim

import pytorch_lightning as pl


class Feedforward(pl.LightningModule):
    def __init__(self, d_model = 512, dff = 2048, dropout = 0.1):
        super().__init__()
        self.d_model = d_model
        self.dff = dff
        self.fc1 = nn.Linear(d_model,dff)
        self.fc2 = nn.Linear(dff,d_model)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()


    def forward(self,x):

        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x  # (Batch size, context_len, d_model)


class Encoder(pl.LightningModule):
    def __init__(self, n_heads, d_model=512, context_len=6):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.context_len = context_len
       # self.x = x # (Batch size, context_len, 1)
        self.projector = nn.Linear(1,d_model)
        self.poe = PositionalEncoding(d_model,context_len)
        self.self_attention = Multihead_attention(n_heads,d_model=d_model,context_len=context_len)

        self.feedforward = Feedforward()
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)


    def forward(self,x):

        x = self.projector(x)  # (Batch size, context_len, d_model)
        x = self.poe(x)  # (Batch size, context_len, d_model) with positional embedding

        res_connection = x      # storing x for residual skip connection

        x = self.self_attention(x, x, x)  # (Batch size, context length, d_model) after attention

        x = x + res_connection   # (Batch size, context length, d_model) after residual skip connection
        x = self.layer_norm1(x)    # (Batch size, context length, d_model) after layer norm

        res_connection = x  # storing x for residual skip connection

        x = self.feedforward(x)  # (Batch size, context length, d_model) after two layers of feedforward

        x = x + res_connection  # (Batch size, context length, d_model) after residual skip connection
        x = self.layer_norm2(x)    # (Batch size, context length, d_model) after layer norm

        return x   # (Batch size, context length, d_model)












