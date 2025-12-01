import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
import math

# Check if torch.compile is available (PyTorch 2.0+)
TORCH_COMPILE_AVAILABLE = hasattr(torch, 'compile')


@dataclass
class DiTConfig():
    vocab_size: int = 50257
    seq_len: int = 1024
    n_embd: int = 1
    n_heads: int = 1
    n_layers: int = 1
    c_dim: int = 1 #128



class DLM(nn.Module):
    def __init__(self, config, compile_model=False):
        super().__init__()
        self.config = config
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.c_emb = TimestepEmbedder(config)
        self.rotary_emb = Rotary(config.n_embd // config.n_heads, config.seq_len)

        self.blocks = nn.ModuleList([
            DiTBlock(config) for _ in range(config.n_layers)
        ])

        self.proj_out = DiTFinalLayer(config)

        # Optional: compile model for 2-3x speedup (PyTorch 2.0+)
        if compile_model and TORCH_COMPILE_AVAILABLE:
            # Don't compile when using kv cache to avoid recompilation overhead
            self._forward_impl = torch.compile(self._forward_impl)

    def _forward_impl(self, x, c, rotary_cos_sin, tok_id, kv_cache=None, prefix_len=0):
        """
        Compiled forward implementation for speed

        Args:
            x: (B, S, n_embd) token embeddings
            c: (B, n_embd) conditioning embeddings
            rotary_cos_sin: tuple of (cos, sin) rotary embeddings
            tok_id: (B, S) token indices
            kv_cache: optional list of (k, v) tuples for each layer
            prefix_len: length of cached prefix (0 means no caching)

        Returns:
            logits: (B, S, vocab_size) output logits
            new_kv_cache: updated kv cache if caching enabled
        """
        new_kv_cache = [] if kv_cache is not None else None

        for i, block in enumerate(self.blocks):
            layer_kv_cache = kv_cache[i] if kv_cache is not None else None
            x, layer_new_kv = block(x, c, rotary_cos_sin, layer_kv_cache, prefix_len)

            if new_kv_cache is not None:
                new_kv_cache.append(layer_new_kv)

        x = self.proj_out(x, c)  # (B, S, vocab_size)

        # Mask out current token (zero diagonal for Markov transition matrix)
        # This ensures the model doesn't predict staying at the same token
        x = x.scatter(-1, tok_id.unsqueeze(-1), 0.0)

        return x, new_kv_cache

    def forward(self, tok_id, sigma, kv_cache=None, prefix_len=0):
        """
        Args:
            tok_id: (B, S) token indices
            sigma: (B, 1) noise level
            kv_cache: optional list of (k, v) tuples from previous forward pass
            prefix_len: number of tokens at start that are cached

        Returns:
            logits: (B, S, vocab_size) if prefix_len == 0
                    (B, S - prefix_len, vocab_size) if prefix_len > 0 (only new tokens)
            new_kv_cache: updated kv cache for next forward pass
        """
        # tok_id: (B, S), sigma: (B, 1)
        x = self.tok_emb(tok_id)  # (B, S, n_embd)
        c = self.c_emb(sigma)  # (B, n_embd)
        rotary_cos_sin = self.rotary_emb(x)  # ((1, 1, S, head_dim), (1, 1, S, head_dim))

        logits, new_kv_cache = self._forward_impl(x, c, rotary_cos_sin, tok_id, kv_cache, prefix_len)

        # Only return logits for new tokens if using cache
        if prefix_len > 0 and kv_cache is not None:
            logits = logits[:, prefix_len:]

        return logits, new_kv_cache


class DiTBlock(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.adaLN = nn.Sequential(
            nn.SiLU(),
            nn.Linear(config.n_embd, 6*config.n_embd, bias=True)
        )
        self.adaLN[-1].weight.data.zero_()
        self.adaLN[-1].bias.data.zero_()

        self.norm1 = RMSNorm(config.n_embd)
        self.attn = MHA(config)
        self.norm2 = RMSNorm(config.n_embd)

        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd, bias=True),
            nn.SiLU(),
            nn.Linear(4 * config.n_embd, config.n_embd, bias=True)
        )

    def forward(self, x, c, rotary_cos_sin, kv_cache=None, prefix_len=0):
        """
        Args:
            x: (B, S, n_embd) input embeddings
            c: (B, 1, n_embd) or (B, n_embd) conditioning
            rotary_cos_sin: rotary embeddings
            kv_cache: optional (k, v) tuple from previous pass
            prefix_len: number of cached prefix tokens

        Returns:
            x: (B, S, n_embd) output
            new_kv_cache: (k, v) tuple for next pass
        """
        c = c.squeeze(1)
        # Faster reshape instead of rearrange
        ada_params = self.adaLN(c).reshape(c.shape[0], 6, -1)
        shift1, scale1, gate1, shift2, scale2, gate2 = ada_params.unbind(1)

        attn_out, new_kv_cache = self.attn(
            modulate(self.norm1(x), shift1, scale1),
            rotary_cos_sin,
            kv_cache=kv_cache,
            prefix_len=prefix_len
        )
        x = x + gate1.unsqueeze(1) * attn_out
        x = x + gate2.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift2, scale2))
        return x, new_kv_cache


class DiTFinalLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.adaLN = nn.Sequential(
            nn.SiLU(),
            nn.Linear(config.n_embd, 2 * config.n_embd, bias=True)
        )
        self.adaLN[-1].weight.data.zero_()
        self.adaLN[-1].bias.data.zero_()

        self.final_norm = RMSNorm(config.n_embd)
        self.proj = nn.Linear(config.n_embd, config.vocab_size)
        self.proj.weight.data.zero_()
        self.proj.bias.data.zero_()

    def forward(self, x, c):
        c = c.squeeze(1)
        # Faster reshape
        shift, scale = self.adaLN(c).reshape(c.shape[0], 2, -1).unbind(1)
        x = modulate(self.final_norm(x), shift, scale)
        x = self.proj(x)
        return x


class MHA(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=True)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=True)
        self.n_heads = config.n_heads
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_heads

    def forward(self, x, rotary_cos_sin, kv_cache=None, prefix_len=0):
        """
        Args:
            x: (B, S, n_embd) input
            rotary_cos_sin: rotary embeddings
            kv_cache: optional (k_cache, v_cache) tuple
                     k_cache, v_cache: (B, n_heads, prefix_len, head_dim)
            prefix_len: number of tokens cached

        Returns:
            y: (B, S, n_embd) attention output
            new_kv_cache: (k, v) tuple for caching
        """
        B, S, _ = x.size()

        if kv_cache is not None and prefix_len > 0:
            # Only compute QKV for new tokens
            x_new = x[:, prefix_len:]
            qkv = self.c_attn(x_new)
            q, k_new, v_new = qkv.split(self.n_embd, dim=2)

            # Reshape new tokens
            S_new = S - prefix_len
            q = q.view(B, S_new, self.n_heads, self.head_dim).transpose(1, 2)
            k_new = k_new.view(B, S_new, self.n_heads, self.head_dim).transpose(1, 2)
            v_new = v_new.view(B, S_new, self.n_heads, self.head_dim).transpose(1, 2)

            # Apply rotary to new tokens only (slice rotary embeddings)
            cos, sin = rotary_cos_sin
            cos_new = cos[:, :, prefix_len:, :]
            sin_new = sin[:, :, prefix_len:, :]
            q = apply_rotary(q, (cos_new, sin_new))
            k_new = apply_rotary(k_new, (cos_new, sin_new))

            # Concatenate with cached k, v
            k_cache, v_cache = kv_cache
            k = torch.cat([k_cache, k_new], dim=2)  # (B, n_heads, S, head_dim)
            v = torch.cat([v_cache, v_new], dim=2)

            # Attention over full sequence, but only for query positions
            y = F.scaled_dot_product_attention(q, k, v, is_causal=False)
            y = y.transpose(1, 2).contiguous().view(B, S_new, self.n_embd)

        else:
            # Normal forward pass without caching
            qkv = self.c_attn(x)
            q, k, v = qkv.split(self.n_embd, dim=2)

            # Reshape: (b, s, n_embd) -> (b, s, h, d) -> (b, h, s, d)
            q = q.view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
            k = k.view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
            v = v.view(B, S, self.n_heads, self.head_dim).transpose(1, 2)

            q = apply_rotary(q, rotary_cos_sin)
            k = apply_rotary(k, rotary_cos_sin)

            # Use Flash Attention if available, otherwise falls back to standard attention
            y = F.scaled_dot_product_attention(q, k, v, is_causal=False)
            # Reshape back: (b, h, s, d) -> (b, s, h, d) -> (b, s, n_embd)
            y = y.transpose(1, 2).contiguous().view(B, S, self.n_embd)

        y = self.c_proj(y)

        # Return updated cache (full k, v)
        new_kv_cache = (k, v)
        return y, new_kv_cache

#############################################
#                   Layers                  #
#############################################

def modulate(x, shift, scale):
    # Reshape: (b, d) -> (b, 1, d)
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        # RMS in FP32 for stability (works on CPU and GPU)
        rms = (x.to(torch.float32).pow(2).mean(dim=-1, keepdim=True) + self.eps).sqrt()
        return (x / rms) * self.scale


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
        freqs = torch.exp(-math.log(10000.0) * torch.arange(half, dtype=torch.float32, device=t.device) / (half-1))
        args = t[:, None].float() * freqs[None] # (B, half)
        embedding = torch.cat([args.cos(), args.sin()], dim=-1) # (N, dim)
        return embedding  # (N, c_dim)

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.freq_dim)
        t_emb = self.mlp(t_freq)
        return t_emb # (N, n_embd)


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
        """ precompute cos and sin """
        t = torch.arange(seq_len, device=self.inv_freq.device).type_as(self.inv_freq)
        freqs = torch.einsum('i,j -> ij', t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)

    def forward(self, x):
        """
        Args:
            x: input tensor with shape (batch, seq_len, n_embd)
        Returns:
            tuple of (cos, sin) tensors with shape (1, 1, seq_len, head_dim)
        """
        seq_len = x.shape[1]

        # Rebuild cache if sequence is longer than cached
        if seq_len > self.max_seq_len:
            self._build_cache(seq_len)

        # Return cos and sin up to current sequence length
        return (
            self.cos_cached[:, :, :seq_len, :],
            self.sin_cached[:, :, :seq_len, :]
        )


def apply_rotary(x, rotary_cos_sin):
    cos, sin = rotary_cos_sin
    # Reshape: (..., d) -> (..., d/2, 2)
    *leading_dims, d = x.shape
    x_reshaped = x.view(*leading_dims, d // 2, 2)
    x1, x2 = x_reshaped.unbind(dim=-1)
    x_rotated = torch.stack((-x2, x1), dim=-1)
    # Reshape back: (..., d/2, 2) -> (..., d)
    x_rotated = x_rotated.view(*leading_dims, d)

    return x * cos + x_rotated * sin
