import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import numpy as np

class VisionConfig():
    def __init__(self, hidden_size=768,
                 num_hidden_layers=12,
                 num_attention_heads=12,
                 intermediate_size=3072,
                 num_channels=3,
                 image_size=224,
                 patch_size=16,
                 dropout=0.0,
                 attention_dropout=0.0,
                 layer_norm_eps=1e-6,
                 num_image_tokens: int = None,
                 **kwargs):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.num_channels = num_channels
        self.image_size = image_size
        self.patch_size = patch_size
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.num_image_tokens = num_image_tokens
        self.layer_norm_eps = layer_norm_eps


class VisionTransformer(nn.Module):
    # Receives a tensor of shape [batch_size, num_channels, image_size, image_size]
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size

        self.embeddings = VisionEmbedding(config)
        self.encoder = Encoder(config)
        self.post_layernorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)

    def forward(self, pixel_values: torch.Tensor):
        hidden_states = self.embeddings(pixel_values) # [batch_size, num_patches, embedding_dim]
        last_hidden_state = self.encoder(hidden_states)  # [batch_size, num_patches, embedding_dim]
        last_hidden_state = self.post_layernorm(last_hidden_state)
        return last_hidden_state


class EncoderLayer(nn.Module):
    # Receives a tensor of shape [batch_size, num_patches, embedding_dim]
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.self_attn = Attention(config)
        self.layer_norm1 = nn.LayerNorm(self.embedding_dim, eps=config.layer_norm_eps)
        self.mlp = MLP(config)
        self.layer_norm2 = nn.LayerNorm(self.embedding_dim, eps=config.layer_norm_eps)

    def forward(self, hidden_states: torch.Tensor):
        # hidden_states has shape [batch_size, num_patches, embedding_dim]
        residual = hidden_states
        hidden_states = self.layer_norm1(hidden_states)
        hidden_states, _ = self.self_attn(hidden_states)

        hidden_states = residual + hidden_states

        residual = hidden_states

        hidden_states = self.layer_norm2(hidden_states)

        hidden_states = self.mlp(hidden_states)

        hidden_states = hidden_states + residual

        return hidden_states


class Encoder(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([EncoderLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states: torch.Tensor):
        for layer in self.layer:
            hidden_states = layer(hidden_states)

        return hidden_states


class MLP(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.config = config
        self.fc1 = nn.Linear(self.config.hidden_size, self.config.intermediate_size)
        self.fc2 = nn.Linear(self.config.intermediate_size, self.config.hidden_size)

    def forward(self, hidden_states: torch.Tensor):
        hidden_states = self.fc1(hidden_states)
        hidden_states = F.gelu(hidden_states, approximate="tanh")
        hidden_states = self.fc2(hidden_states)
        return hidden_states


class Attention(nn.Module):
    # Receives a tensor of shape [batch_size, num_patches, embedding_dim]
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // config.num_attention_heads

        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.dropout = nn.Dropout(config.attention_dropout)
        self.scale = self.head_dim ** -0.5
        self.training = True

    def forward(self, hidden_states: torch.Tensor):
        # hidden_states has shape [batch_size, seq_len, embedding_dim]
        batch_size, seq_len, _ = hidden_states.shape

        query_states = self.q_proj(hidden_states)  # [batch_size, seq_len, embed_dim]
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.reshape(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1,
                                                                                                        3)  # [batch_size, num_heads, seq_len, head_dim]
        key_states = key_states.reshape(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1,
                                                                                                    3)  # [batch_size, num_heads, seq_len, head_dim]
        value_states = value_states.reshape(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1,
                                                                                                        3)  # [batch_size, num_heads, seq_len, head_dim]

        attn_weights = torch.matmul(query_states, key_states.transpose(-1,
                                                                       -2)) * self.scale  # [batch_size, num_heads, seq_len, seq_len]
        attn_weights = F.softmax(attn_weights, dim=-1).to(
            query_states.dtype)  # [batch_size, num_heads, seq_len, seq_len]
        attn_weights = self.dropout(attn_weights, p=self.config.attention_dropout, training=self.training)

        attn_output = torch.matmul(attn_weights, value_states)  # [batch_size, num_heads, seq_len, head_dim]
        attn_output = attn_output.transpose(1, 2).contiguous()  # [batch_size, seq_len, num_heads, embedding_dim]
        attn_output = attn_output.reshape(batch_size, seq_len, self.embed_dim)
        attn_output = self.out_proj(attn_output)
        return attn_output, attn_weights


class VisionEmbedding(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.config = config
        self.embedding_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size
        self.num_patches = (self.image_size // self.patch_size) ** 2

        self.patch_embedding = nn.Conv2d(
            in_channels=self.config.num_channels,
            out_channels=self.embedding_dim,
            kernel_size=(self.patch_size, self.patch_size),
            stride=(self.patch_size, self.patch_size),
            padding="valid",
        )

        self.embedding = nn.Embedding(self.num_patches, self.embedding_dim)

        self.register_buffer("position_ids", torch.arange(self.num_patches).expand((1, -1)))

    def forward(self, pixel_values: torch.Tensor):
        # pixel_values has shape [batch_size, num_channels, image_size, image_size]

        patch_embeddings = self.patch_embedding(
            pixel_values)  # [batch_size, embedding_dim, num_patches_H, num_patches_W]
        patch_embeddings = patch_embeddings.flatten(2)  # [batch_size, embedding_dim, num_patches_H*num_patches_W]
        patch_embeddings = patch_embeddings.transpose(1, 2)  # [batch_size, num_patches, embedding_dim]

        embeddings = patch_embeddings + self.embedding(self.position_ids)

        return embeddings  # [batch_size, num_patches, embedding_dim]


class VisionModel(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.config = config
        self.vision_model = VisionTransformer(config)

    def forward(self, pixel_values) -> Tuple:
        # [Batch_Size, Channels, Height, Width] -> [Batch_Size, Num_Patches, Embed_Dim]
        return self.vision_model(pixel_values=pixel_values)
