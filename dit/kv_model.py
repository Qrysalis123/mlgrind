"""
DiT (Diffusion Transformer) with Block-wise KV Caching

API:
    # Standard forward (no caching)
    logits = model(x, sigma)

    # Build cache from clean prefix
    kv_cache = model.build_cache(prefix_tokens)

    # Denoise using cache (fast - only computes on x_noisy)
    logits = model(x_noisy, sigma, kv_cache=kv_cache)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange
from dataclasses import dataclass
from typing import Optional, List, Tuple


@dataclass
class DiTConfig:
    hidden_size: int = 512
    cond_dim: int = 128
    length: int = 1024
    vocab: int = 32768
    n_blocks: int = 12
    n_heads: int = 8
    dropout: float = 0.1



# KV cache: list of (K, V) per layer, each (B, H, S, D)
KVCache = List[Tuple[torch.Tensor, torch.Tensor]]


#################################################################################
#                                  Rotary                                       #
#################################################################################

class Rotary(nn.Module):
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
            self.cos_cached = emb.cos()[None, :, None, :]
            self.sin_cached = emb.sin()[None, :, None, :]
        return self.cos_cached, self.sin_cached

    def get_range(self, start: int, length: int, device):
        """Get rotary embeddings for positions [start, start + length)"""
        total = start + length
        if self.seq_len_cached is None or total > self.seq_len_cached:
            self.forward(total, device)
        return self.cos_cached[:, start:start+length], self.sin_cached[:, start:start+length]


def rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(qkv, cos, sin):
    return (qkv * cos) + (rotate_half(qkv) * sin)


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
        return x * self.weight[None, None, :]


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
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
        return self.mlp(t_freq)


class EmbeddingLayer(nn.Module):
    def __init__(self, dim, vocab_dim):
        super().__init__()
        self.embedding = nn.Parameter(torch.empty((vocab_dim, dim)))
        nn.init.kaiming_uniform_(self.embedding, a=math.sqrt(5))

    def forward(self, x):
        return self.embedding[x]


#################################################################################
#                             DDiT Block                                        #
#################################################################################

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps
        self.dim = dim

    def forward(self, x):
        with torch.amp.autocast('cuda', enabled=False):
            x_fp32 = x.float()
            rms = torch.sqrt(torch.mean(x_fp32 ** 2, dim=-1, keepdim=True) + self.eps)
            return ((x_fp32 / rms) * self.weight).type_as(x)


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

        # QK-Norm for attention stability
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)

        self.norm2 = LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_ratio * dim, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_ratio * dim, dim, bias=True)
        )

        self.adaLN_modulation = nn.Linear(cond_dim, 6 * dim, bias=True)
        nn.init.zeros_(self.adaLN_modulation.weight)
        nn.init.zeros_(self.adaLN_modulation.bias)

    def forward(
        self,
        x: torch.Tensor,
        rotary_cos_sin: Tuple[torch.Tensor, torch.Tensor],
        c: torch.Tensor,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: (B, S, D) hidden states
            rotary_cos_sin: (cos, sin) for positions of x
            c: (B, cond_dim) sigma conditioning
            kv_cache: optional (K, V) from prefix, shape (B, H, S_prefix, D_head)
        """
        B, S, D = x.shape

        # adaLN modulation
        mod = self.adaLN_modulation(c)[:, None, :]
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mod.chunk(6, dim=-1)

        # Norm + modulate
        x_norm = modulate(self.norm1(x), shift_msa, scale_msa)

        # QKV for current tokens
        qkv = self.attn_qkv(x_norm)
        qkv = rearrange(qkv, "b s (three h d) -> three b s h d", three=3, h=self.n_heads)
        q, k, v = qkv.unbind(dim=0)

        # QK-Norm before rotary
        q = self.q_norm(q)
        k = self.k_norm(k)

        # Apply rotary
        cos, sin = rotary_cos_sin
        q = apply_rotary_pos_emb(q, cos, sin)
        k = apply_rotary_pos_emb(k, cos, sin)

        # Transpose for attention: (B, S, H, D) -> (B, H, S, D)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Concatenate with cache if available
        if kv_cache is not None:
            k_cache, v_cache = kv_cache
            k = torch.cat([k_cache, k], dim=2)
            v = torch.cat([v_cache, v], dim=2)

        # Attention
        x_attn = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False
        )
        x_attn = x_attn.transpose(1, 2)
        x_attn = rearrange(x_attn, "b s h d -> b s (h d)")

        x = x + gate_msa * F.dropout(self.attn_out(x_attn), p=self.dropout, training=self.training)

        # MLP
        x_norm = modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = x + gate_mlp * F.dropout(self.mlp(x_norm), p=self.dropout, training=self.training)

        return x

    def forward_split(
        self,
        x: torch.Tensor,
        rotary_cos_sin: Tuple[torch.Tensor, torch.Tensor],
        c: torch.Tensor,
        split_idx: int,
    ) -> torch.Tensor:
        """
        Forward with adaLN applied only to positions >= split_idx.
        Used for training where we need gradients through both clean and noisy.
        """
        B, S, D = x.shape

        mod = self.adaLN_modulation(c)[:, None, :]
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mod.chunk(6, dim=-1)

        # Norm with split modulation
        x_clean_norm = self.norm1(x[:, :split_idx])
        x_noisy_norm = modulate(self.norm1(x[:, split_idx:]), shift_msa, scale_msa)
        x_norm = torch.cat([x_clean_norm, x_noisy_norm], dim=1)

        # QKV
        qkv = self.attn_qkv(x_norm)
        qkv = rearrange(qkv, "b s (three h d) -> three b s h d", three=3, h=self.n_heads)
        q, k, v = qkv.unbind(dim=0)

        # QK-Norm
        q = self.q_norm(q)
        k = self.k_norm(k)

        cos, sin = rotary_cos_sin
        q = apply_rotary_pos_emb(q, cos, sin)
        k = apply_rotary_pos_emb(k, cos, sin)

        x_attn = F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False
        ).transpose(1, 2)
        x_attn = rearrange(x_attn, "b s h d -> b s (h d)")
        attn_out = F.dropout(self.attn_out(x_attn), p=self.dropout, training=self.training)

        # Residual with split gating
        x_clean = x[:, :split_idx] + attn_out[:, :split_idx]
        x_noisy = x[:, split_idx:] + gate_msa * attn_out[:, split_idx:]
        x = torch.cat([x_clean, x_noisy], dim=1)

        # MLP with split modulation
        x_clean_norm = self.norm2(x[:, :split_idx])
        x_noisy_norm = modulate(self.norm2(x[:, split_idx:]), shift_mlp, scale_mlp)
        x_norm = torch.cat([x_clean_norm, x_noisy_norm], dim=1)

        mlp_out = F.dropout(self.mlp(x_norm), p=self.dropout, training=self.training)
        x_clean = x[:, :split_idx] + mlp_out[:, :split_idx]
        x_noisy = x[:, split_idx:] + gate_mlp * mlp_out[:, split_idx:]
        x = torch.cat([x_clean, x_noisy], dim=1)

        return x

    def get_kv(
        self,
        x: torch.Tensor,
        rotary_cos_sin: Tuple[torch.Tensor, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute K, V for clean prefix (no adaLN - sigma=0).

        Returns:
            h_out: (B, S, D) output hidden states
            k: (B, H, S, D_head)
            v: (B, H, S, D_head)
        """
        # No modulation for clean tokens (sigma=0 means scale=0, shift=0)
        x_norm = self.norm1(x)
        qkv = self.attn_qkv(x_norm)
        qkv = rearrange(qkv, "b s (three h d) -> three b s h d", three=3, h=self.n_heads)
        q, k, v = qkv.unbind(dim=0)

        # QK-Norm
        q = self.q_norm(q)
        k = self.k_norm(k)

        cos, sin = rotary_cos_sin
        q = apply_rotary_pos_emb(q, cos, sin)
        k = apply_rotary_pos_emb(k, cos, sin)

        # Self-attention for clean tokens
        q_t = q.transpose(1, 2)
        k_t = k.transpose(1, 2)
        v_t = v.transpose(1, 2)

        x_attn = F.scaled_dot_product_attention(q_t, k_t, v_t, is_causal=False)
        x_attn = x_attn.transpose(1, 2)
        x_attn = rearrange(x_attn, "b s h d -> b s (h d)")

        # Residual (no gating for clean)
        x = x + F.dropout(self.attn_out(x_attn), p=self.dropout, training=self.training)

        # MLP (no modulation)
        x_norm = self.norm2(x)
        x = x + F.dropout(self.mlp(x_norm), p=self.dropout, training=self.training)

        return x, k_t, v_t


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

    def forward(
        self,
        x: torch.Tensor,
        sigma: torch.Tensor,
        kv_cache: Optional[KVCache] = None,
        x_clean: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: (B, S) token indices to denoise
            sigma: (B,) noise level
            kv_cache: optional KV cache from build_cache()
            x_clean: optional (B, S_clean) clean prefix for training mode

        Returns:
            logits: (B, S, V) or (B, S_noisy, V) if x_clean provided
        """
        # Training mode with clean prefix
        if x_clean is not None:
            return self._forward_train(x, sigma, x_clean)

        B, S = x.shape
        device = x.device

        h = self.vocab_embed(x)
        c = F.silu(self.sigma_map(sigma))

        # Get rotary positions (offset by cache length if using cache)
        prefix_len = kv_cache[0][0].shape[2] if kv_cache else 0
        rotary_cos_sin = self.rotary_emb.get_range(prefix_len, S, device)

        for i, blk in enumerate(self.blocks):
            cache_i = kv_cache[i] if kv_cache else None
            h = blk(h, rotary_cos_sin, c, kv_cache=cache_i)

        logits = self.output_layer(h, c)
        logits = logits.scatter(-1, x[..., None], torch.zeros_like(logits[..., :1]))
        return logits

    def forward_train(
        self,
        x_noisy: torch.Tensor,
        sigma: torch.Tensor,
        x_clean: torch.Tensor,
    ) -> torch.Tensor:
        """Wrapper for backward compatibility. Use forward(x, sigma, x_clean=x_clean) instead."""
        return self._forward_train(x_noisy, sigma, x_clean)

    def _forward_train(
        self,
        x_noisy: torch.Tensor,
        sigma: torch.Tensor,
        x_clean: torch.Tensor,
    ) -> torch.Tensor:
        """
        Training forward with clean prefix + noisy block.

        Concatenates [x_clean, x_noisy], runs full attention,
        but only applies adaLN to noisy portion. Gradients flow through both.

        Args:
            x_noisy: (B, S_noisy) noisy tokens to denoise
            sigma: (B,) noise level
            x_clean: (B, S_clean) clean prefix tokens

        Returns:
            logits: (B, S_noisy, V) predictions for noisy block only
        """
        S_clean = x_clean.shape[1]
        S_noisy = x_noisy.shape[1]

        # Embed and concat
        h_clean = self.vocab_embed(x_clean)
        h_noisy = self.vocab_embed(x_noisy)
        h = torch.cat([h_clean, h_noisy], dim=1)

        c = F.silu(self.sigma_map(sigma))
        rotary_cos_sin = self.rotary_emb.get_range(0, S_clean + S_noisy, h.device)

        # Process with split adaLN
        for blk in self.blocks:
            h = blk.forward_split(h, rotary_cos_sin, c, split_idx=S_clean)

        # Output only for noisy portion
        h_noisy = h[:, S_clean:]
        logits = self.output_layer(h_noisy, c)
        logits = logits.scatter(-1, x_noisy[..., None], torch.zeros_like(logits[..., :1]))
        return logits

    def build_cache(
        self,
        tokens: torch.Tensor,
        kv_cache: Optional[KVCache] = None,
    ) -> KVCache:
        """
        Build or extend KV cache from clean tokens.

        Args:
            tokens: (B, S) clean token indices to add to cache
            kv_cache: optional existing cache to extend

        Returns:
            kv_cache: list of (K, V) per layer
        """
        S = tokens.shape[1]
        offset = kv_cache[0][0].shape[2] if kv_cache else 0

        h = self.vocab_embed(tokens)
        rotary_cos_sin = self.rotary_emb.get_range(offset, S, h.device)

        new_cache = []
        for i, blk in enumerate(self.blocks):
            h, k, v = blk.get_kv(h, rotary_cos_sin)

            # Append to existing cache if provided
            if kv_cache is not None:
                k = torch.cat([kv_cache[i][0], k], dim=2)
                v = torch.cat([kv_cache[i][1], v], dim=2)

            new_cache.append((k, v))

        return new_cache
