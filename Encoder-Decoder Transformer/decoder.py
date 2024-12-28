from masked_multihead_attention import MaskedMultihead_attention
from multihead_attention import Multihead_attention
from positional_embedding import PositionalEncoding
from typing import Any
import torch
import math
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.utils.data import random_split
import pytorch_lightning as pl
import torchmetrics
from torchmetrics import Metric
import multiprocessing
from encoder import Encoder

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

class Decoder(pl.LightningModule):
    def __init__(self, n_heads, d_model=512, context_len=6, vocab_size=10000):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.context_len = context_len
        # self.x = x # (Batch size, context_len, 1)
        self.projector = nn.Linear(1, d_model)
        self.poe = PositionalEncoding(d_model, context_len)
        self.cross_attention = Multihead_attention(n_heads, d_model=d_model, context_len=context_len)
        self.masked_attention = MaskedMultihead_attention()


        self.feedforward = Feedforward()
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.layer_norm3 = nn.LayerNorm(d_model)

        self.output_linear = nn.Linear(d_model, vocab_size)
        self.softmax = nn.Softmax(dim=-1)



    def forward(self,x, encoder_output):
        x = self.projector(x)  # (Batch size, context_len, d_model)
        x = self.poe(x)  # (Batch size, context_len, d_model) with positional embedding

        # Self-attention block
        res_connection = x  # storing x for residual skip connection
        x = self.masked_attention(x,x,x)  # (Batch size, context length, d_model) after attention
        x = x + res_connection  # (Batch size, context length, d_model) after residual skip connection
        x = self.layer_norm1(x)  # (Batch size, context length, d_model) after layer norm

        # Cross-attention block
        res_connection = x  # storing x for residual skip connection
        x = self.cross_attention(q=x, k=encoder_output, v=encoder_output)  # (Batch size, context length, d_model) after cross attention
        x = x + res_connection  # (Batch size, context length, d_model) after residual skip connection
        x = self.layer_norm2(x)  # (Batch size, context length, d_model) after layer norm

        # Feedforward block
        res_connection = x  # storing x for residual skip connection
        x = self.feedforward(x)  # (Batch size, context length, d_model) after two layers of feedforward
        x = x + res_connection  # (Batch size, context length, d_model) after residual skip connection
        x = self.layer_norm3(x)  # (Batch size, context length, d_model) after layer norm

        # Output layers
        x = self.output_linear(x)  # (Batch size, context_len, vocab_size)
        x = self.softmax(x)  # (Batch size, context_len, vocab_size) as probabilities

        return x   # (Batch size, context_len, vocab_size)








