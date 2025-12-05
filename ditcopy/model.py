import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
import math

# -----------------------------
# Configuration Dataclass
# -----------------------------
@dataclass
class DiTConfig:
    seq_len: int = 1024
    vocab_size: int = 50304
    n_layers: int = 4
    n_heads: int = 4
    n_embd: int = 64
    c_dim: int = 32
    dropout: float = 0.1


# -----------------------------
# Utility Modules
# -----------------------------
class LayerNorm(nn.Module):
    def __init__(self, dim, eps: float = 1e-6):
        super().__init__()
        self.ln = nn.LayerNorm(dim, eps=eps)

    def forward(self, x):
        return self.ln(x)


def modulate(x, shift, scale):
    return x * (1 + scale) + shift


# -----------------------------
# Rotary Positional Embeddings
# -----------------------------
class Rotary(nn.Module):
    def __init__(self, dim, max_seq_len, base=10000):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base

        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len):
        t = torch.arange(seq_len, device=self.inv_freq.device).type_as(self.inv_freq)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)

    def forward(self, x):
        seq_len = x.shape[1]
        if seq_len > self.max_seq_len:
            self._build_cache(seq_len)
        return self.cos_cached[:, :, :seq_len, :], self.sin_cached[:, :, :seq_len, :]


def apply_rotary(x, rotary_cos_sin):
    cos, sin = rotary_cos_sin
    *leading_dims, d = x.shape
    x_reshaped = x.view(*leading_dims, d // 2, 2)
    x1, x2 = x_reshaped.unbind(dim=-1)
    x_rot = torch.stack((-x2, x1), dim=-1).view(*leading_dims, d)
    return x * cos + x_rot * sin


# -----------------------------
# Token & Timestep Embeddings
# -----------------------------
class EmbeddingLayer(nn.Module):
    def __init__(self, vocab_dim, dim):
        super().__init__()
        self.embedding = nn.Parameter(torch.empty((vocab_dim, dim)))
        nn.init.kaiming_uniform_(self.embedding, a=math.sqrt(5))

    def forward(self, x):
        return self.embedding[x]


class TimestepEmbedder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.freq_dim = config.c_dim
        self.mlp = nn.Sequential(
            nn.Linear(self.freq_dim, config.n_embd),
            nn.SiLU(),
            nn.Linear(config.n_embd, config.n_embd)
        )

    @staticmethod
    def timestep_embedding(t, dim):
        half = dim // 2
        freqs = torch.exp(-math.log(10000.0) * torch.arange(half, dtype=torch.float32, device=t.device) / (half - 1))
        args = t[:, None].float() * freqs[None]
        return torch.cat([args.cos(), args.sin()], dim=-1)

    def forward(self, t):
        t_emb = self.timestep_embedding(t, self.freq_dim)
        return self.mlp(t_emb)


# -----------------------------
# Multi-Head Attention
# -----------------------------
class MHA(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_heads = config.n_heads
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_heads

        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)

    def forward(self, x, rotary_cos_sin, kv_cache=None, prefix_len=0):
        B, S, _ = x.shape

        # Full attention (no KV cache)
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)

        # Reshape for multi-head
        q = q.view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, S, self.n_heads, self.head_dim).transpose(1, 2)

        q = apply_rotary(q, rotary_cos_sin)
        k = apply_rotary(k, rotary_cos_sin)

        y = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        y = y.transpose(1, 2).contiguous().view(B, S, self.n_embd)
        return self.c_proj(y), (k, v)


# -----------------------------
# Transformer Block
# -----------------------------
class DiTBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.norm1 = LayerNorm(config.n_embd)
        self.norm2 = LayerNorm(config.n_embd)
        self.attn = MHA(config)

        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(approximate='tanh'),
            nn.Linear(4 * config.n_embd, config.n_embd)
        )

        self.adaLN = nn.Linear(config.n_embd, 6 * config.n_embd)
        nn.init.zeros_(self.adaLN.weight)
        nn.init.zeros_(self.adaLN.bias)

        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x, c, rotary_cos_sin, kv_cache=None, prefix_len=0):
        shift1, scale1, gate1, shift2, scale2, gate2 = self.adaLN(c)[:, None].chunk(6, dim=2)

        x_skip = x
        h = modulate(self.norm1(x), shift1, scale1)
        attn_out, new_kv_cache = self.attn(h, rotary_cos_sin, kv_cache, prefix_len)
        x = x_skip + self.dropout(attn_out) * gate1

        h = modulate(self.norm2(x), shift2, scale2)
        mlp_out = self.mlp(h)
        x = x + self.dropout(mlp_out) * gate2

        return x, new_kv_cache


# -----------------------------
# Final Output Layer
# -----------------------------
class DiTFinalLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.final_norm = LayerNorm(config.n_embd)
        self.linear = nn.Linear(config.n_embd, config.vocab_size)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

        self.adaLN = nn.Linear(config.n_embd, 2 * config.n_embd)
        nn.init.zeros_(self.adaLN.weight)
        nn.init.zeros_(self.adaLN.bias)

    def forward(self, x, c):
        shift, scale = self.adaLN(c)[:, None].chunk(2, dim=2)
        x = modulate(self.final_norm(x), shift, scale)
        return self.linear(x)


# -----------------------------
# Main DiT Model
# -----------------------------
class DLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.vocab_embed = EmbeddingLayer(config.vocab_size, config.n_embd)
        self.c_emb = TimestepEmbedder(config)
        self.rotary_emb = Rotary(config.n_embd // config.n_heads, config.seq_len)

        self.blocks = nn.ModuleList([DiTBlock(config) for _ in range(config.n_layers)])
        self.proj_out = DiTFinalLayer(config)

    def _forward_impl(self, x, c, rotary_cos_sin, tok_id, kv_cache=None, prefix_len=0):
        new_kv_cache = [] if kv_cache is not None else None
        for i, block in enumerate(self.blocks):
            layer_kv_cache = kv_cache[i] if kv_cache is not None else None
            x, layer_new_kv = block(x, c, rotary_cos_sin, layer_kv_cache, prefix_len)
            if new_kv_cache is not None:
                new_kv_cache.append(layer_new_kv)

        x = self.proj_out(x, c)
        x = x.scatter(-1, tok_id.unsqueeze(-1), torch.zeros_like(x[..., :1]))
        return x, new_kv_cache

    def forward(self, tok_id, sigma, kv_cache=None, prefix_len=0):
        x = self.vocab_embed(tok_id)
        c = F.silu(self.c_emb(sigma))
        rotary_cos_sin = self.rotary_emb(x)
        logits, new_kv_cache = self._forward_impl(x, c, rotary_cos_sin, tok_id, kv_cache, prefix_len)

        if prefix_len > 0 and kv_cache is not None:
            logits = logits[:, prefix_len:]

        return logits, new_kv_cache
