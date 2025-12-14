"""
DiT (Diffusion Transformer)

Standard discrete diffusion transformer with global sigma conditioning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange
from dataclasses import dataclass


@dataclass
class DiTConfig:
    hidden_size: int = 512
    cond_dim: int = 128
    length: int = 1024
    vocab: int = 32768
    n_blocks: int = 12
    n_heads: int = 8
    dropout: float = 0.1

#################################################################################
#                                  Rotary                                       #
#################################################################################

class Rotary(torch.nn.Module):
    def __init__(self, dim, base=10_000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None

    def forward(self, seq_len, device):
        if seq_len != self.seq_len_cached:
            self.seq_len_cached = seq_len
            t = torch.arange(seq_len, device=device).type_as(self.inv_freq)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq.clone())
            emb = torch.cat((freqs, freqs), dim=-1).to(device)
            # dims are: batch, seq_len, head, dim - broadcasts with (B, S, H, D)
            self.cos_cached = emb.cos()[None, :, None, :]
            self.sin_cached = emb.sin()[None, :, None, :]

        return self.cos_cached, self.sin_cached


def rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat(
        (-x2, x1), dim=-1
    )

def apply_rotary_pos_emb(qkv, cos, sin):
    return (qkv * cos) + (rotate_half(qkv) * sin)

#################################################################################
#                                  Utils                                        #
#################################################################################

def modulate(x, shift, scale):
    return x * (1 + scale) + shift

#################################################################################
#                                  Layers                                       #
#################################################################################

class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones([dim]))
        self.dim = dim
    def forward(self, x):
        with torch.cuda.amp.autocast(enabled=False):
            x = F.layer_norm(x.float(), [self.dim])
        return x * self.weight[None,None,:]

#################################################################################
#                          Timestep Embedder                                    #
#################################################################################

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256, silu=True):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size


    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb

class DDiTBlock(nn.Module):
    def __init__(self, dim, n_heads, cond_dim, mlp_ratio=4, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.dim = dim
        self.dropout = dropout

        self.norm1 = LayerNorm(dim)
        self.attn_qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.attn_out = nn.Linear(dim, dim, bias=False)

        self.norm2 = LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_ratio * dim, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_ratio * dim, dim, bias=True)
        )

        self.adaLN_modulation = nn.Linear(cond_dim, 6 * dim, bias=True)
        nn.init.zeros_(self.adaLN_modulation.weight)
        nn.init.zeros_(self.adaLN_modulation.bias)

    def forward(self, x, rotary_cos_sin, c):
        B, S, D = x.shape

        # adaLN modulation
        mod = self.adaLN_modulation(c)[:, None, :]
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mod.chunk(6, dim=-1)

        # Attention
        x_norm = modulate(self.norm1(x), shift_msa, scale_msa)
        qkv = self.attn_qkv(x_norm)
        qkv = rearrange(qkv, "b s (three h d) -> three b s h d", three=3, h=self.n_heads)
        q, k, v = qkv.unbind(dim=0)

        # Apply rotary
        cos, sin = rotary_cos_sin
        q = apply_rotary_pos_emb(q, cos, sin)
        k = apply_rotary_pos_emb(k, cos, sin)

        # Attention
        x_attn = F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False
        ).transpose(1, 2)

        x_attn = rearrange(x_attn, "b s h d -> b s (h d)")
        x = x + gate_msa * F.dropout(self.attn_out(x_attn), p=self.dropout, training=self.training)

        # MLP
        x_norm = modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = x + gate_mlp * F.dropout(self.mlp(x_norm), p=self.dropout, training=self.training)

        return x


class EmbeddingLayer(nn.Module):
    def __init__(self, dim, vocab_dim):
        super().__init__()
        self.embedding = nn.Parameter(torch.empty((vocab_dim, dim)))
        nn.init.kaiming_uniform_(self.embedding, a=math.sqrt(5))

    def forward(self, x):
        return self.embedding[x]


class DDiTFinalLayer(nn.Module):
    def __init__(self, hidden_size, out_channels, cond_dim):
        super().__init__()
        self.norm_final = LayerNorm(hidden_size)
        self.linear = nn.Linear(hidden_size, out_channels, bias=True)
        self.adaLN_modulation = nn.Linear(cond_dim, 2 * hidden_size, bias=True)

        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)
        nn.init.zeros_(self.adaLN_modulation.weight)
        nn.init.zeros_(self.adaLN_modulation.bias)

    def forward(self, x, c):
        mod = self.adaLN_modulation(c)[:, None, :]
        shift, scale = mod.chunk(2, dim=-1)
        x = modulate(self.norm_final(x), shift, scale)
        return self.linear(x)


#################################################################################
#                               Core Module                                     #
#################################################################################

class DiT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.vocab_embed = EmbeddingLayer(config.hidden_size, config.vocab)
        self.sigma_map = TimestepEmbedder(config.cond_dim)
        self.rotary_emb = Rotary(config.hidden_size // config.n_heads)
        self.blocks = nn.ModuleList([
            DDiTBlock(config.hidden_size, config.n_heads, config.cond_dim, dropout=config.dropout)
            for _ in range(config.n_blocks)
        ])
        self.output_layer = DDiTFinalLayer(config.hidden_size, config.vocab, config.cond_dim)

    def forward(self, indices, sigma):
        """
        Args:
            indices: (B, S) token indices
            sigma: (B,) noise level per sequence

        Returns:
            logits: (B, S, V)
        """
        B, S = indices.shape

        x = self.vocab_embed(indices)
        c = F.silu(self.sigma_map(sigma))
        rotary_cos_sin = self.rotary_emb(S, indices.device)

        for blk in self.blocks:
            x = blk(x, rotary_cos_sin, c)

        logits = self.output_layer(x, c)

        # Set log score for input token to 0 (discrete diffusion convention)
        logits = logits.scatter(-1, indices[..., None], torch.zeros_like(logits[..., :1]))
        return logits
