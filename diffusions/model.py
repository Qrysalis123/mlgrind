import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat, reduce


@dataclass
class DLMConfig:
    seq_len: int = 1024  # sequence length
    vocab_size: int = 50257  # GPT-2 tokenizer
    n_layer: int = 6  # number of transformer blocks
    n_head: int = 8  # number of attention heads
    n_embd: int = 512  # hidden_size (main model dimension)
    cond_dim: int = 128  # conditioning dimension (SEDD standard)


def sigma_schedule(t, sigma_min=0.001, sigma_max=1.0):
    """Noise schedule: t in [0,1] -> sigma"""
    return sigma_min * (sigma_max / sigma_min) ** t


def forward_noise(x_0, sigma, vocab_size):
    """Add Q_uniform noise to tokens"""
    B, L = x_0.shape
    device = x_0.device
    move_prob = 1 - torch.exp(-sigma)  # (B,)

    # For each token, decide if it moves
    move_mask = torch.rand(B, L, device=device) < repeat(move_prob, 'b -> b l', l=L)

    # Sample random replacement tokens
    random_tokens = torch.randint(0, vocab_size, (B, L), device=device)

    # Apply noise
    x_t = torch.where(move_mask, random_tokens, x_0)
    return x_t


class Rotary(nn.Module):
    """RoPE: Rotary Position Embeddings"""
    def __init__(self, dim, base=10000):
        super().__init__()
        # Precompute inverse frequencies: theta_i = 10000^(-2i/d)
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None

    def forward(self, x, seq_dim=1):
        """
        Precompute and cache cos/sin for the sequence length.

        Args:
            x: Input tensor (to determine device and seq_len)
            seq_dim: Which dimension is the sequence length

        Returns:
            cos_cached: (seq_len, 1, 1, dim) for broadcasting
            sin_cached: (seq_len, 1, 1, dim) for broadcasting
        """
        seq_len = x.shape[seq_dim]

        if seq_len != self.seq_len_cached:
            self.seq_len_cached = seq_len
            # Position indices: [0, 1, 2, ..., seq_len-1]
            t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
            # Outer product: position x frequency -> (seq_len, dim/2)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            # Duplicate frequencies for each half: (seq_len, dim)
            emb = torch.cat([freqs, freqs], dim=-1)

            # Cache cos/sin with shape for broadcasting: (seq_len, 1, 1, dim)
            self.cos_cached = emb.cos()[None, :, None, :]
            self.sin_cached = emb.sin()[None, :, None, :]

        return self.cos_cached, self.sin_cached


def apply_rotary_pos_emb(x, cos, sin):
    """
    Apply rotary position embeddings to input tensor.
    Standard RoPE: rotate pairs of features.

    Args:
        x: (B, L, H, D) tensor (query or key)
        cos: (1, L, 1, D) precomputed cosines
        sin: (1, L, 1, D) precomputed sines

    Returns:
        (B, L, H, D) tensor with RoPE applied
    """
    # Split into two halves: first half and second half
    d = x.shape[-1]
    x1, x2 = x[..., :d//2], x[..., d//2:]

    # Get corresponding cos/sin (also split in half)
    cos = cos[..., :d//2]
    sin = sin[..., :d//2]

    # Apply 2D rotation:
    # [x1']   [cos  -sin] [x1]
    # [x2'] = [sin   cos] [x2]
    rotated_x1 = x1 * cos - x2 * sin
    rotated_x2 = x1 * sin + x2 * cos

    # Concatenate back
    return torch.cat([rotated_x1, rotated_x2], dim=-1)


class TimestepEmbedder(nn.Module):
    """Embeds scalar timesteps into vector representations (SEDD style)."""
    def __init__(self, cond_dim, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, cond_dim, bias=True),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.

        Args:
            t: (B,) timesteps
            dim: embedding dimension
            max_period: maximum period (default 10000)
        """
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = rearrange(t, 'b -> b 1').float() * rearrange(freqs, 'd -> 1 d')
        embedding = rearrange([torch.cos(args), torch.sin(args)], 'two b d -> b (two d)', two=2)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        return self.mlp(t_freq)


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(reduce(x.pow(2), '... d -> ... 1', 'mean') + self.eps)

    def forward(self, x):
        return self.weight * self._norm(x.float()).type_as(x)


def modulate(x, shift, scale):
    """Apply affine modulation: x * (1 + scale) + shift"""
    return x * (1 + scale) + shift


class DDiTFinalLayer(nn.Module):
    def __init__(self, hidden_size, vocab_size, cond_dim):
        super().__init__()
        self.norm_final = RMSNorm(hidden_size)
        self.linear = nn.Linear(hidden_size, vocab_size, bias=True)

        # Zero init
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

        self.adaLN_modulation = nn.Linear(cond_dim, 2 * hidden_size, bias=True)
        nn.init.zeros_(self.adaLN_modulation.weight)
        nn.init.zeros_(self.adaLN_modulation.bias)

    def forward(self, x, c):
        modulation = rearrange(self.adaLN_modulation(c), 'b (two d) -> b 1 two d', two=2)
        shift, scale = modulation[:, :, 0], modulation[:, :, 1]
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class DDiTBlock(nn.Module):
    def __init__(self, dim, n_heads, cond_dim, rotary, mlp_ratio=4, dropout=0.0):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads

        self.norm1 = RMSNorm(dim)
        self.attn_qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.attn_out = nn.Linear(dim, dim, bias=False)

        # RoPE (shared across all blocks)
        self.rotary = rotary

        self.norm2 = RMSNorm(dim)
        # SwiGLU MLP (LLaMA-style)
        hidden_dim = int(mlp_ratio * dim)
        # For SwiGLU, we need hidden_dim * 2 for gate and up projections
        self.mlp_gate = nn.Linear(dim, hidden_dim, bias=False)
        self.mlp_up = nn.Linear(dim, hidden_dim, bias=False)
        self.mlp_down = nn.Linear(hidden_dim, dim, bias=False)

        # AdaLN modulation: 6 params (shift, scale, gate) for attn and mlp
        self.adaLN_modulation = nn.Linear(cond_dim, 6 * dim, bias=True)
        nn.init.zeros_(self.adaLN_modulation.weight)
        nn.init.zeros_(self.adaLN_modulation.bias)

    def forward(self, x, c):
        """
        Args:
            x: (B, L, D) input tokens
            c: (B, cond_dim) conditioning
        """
        B, L, D = x.shape

        # Get modulation parameters
        modulation = rearrange(self.adaLN_modulation(c), 'b (six d) -> b 1 six d', six=6)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
            modulation[:, :, 0], modulation[:, :, 1], modulation[:, :, 2], \
            modulation[:, :, 3], modulation[:, :, 4], modulation[:, :, 5]

        # Attention with AdaLN
        x_skip = x
        x_norm = self.norm1(x)
        x_mod = modulate(x_norm, shift_msa, scale_msa)

        # Multi-head attention with RoPE
        qkv = self.attn_qkv(x_mod)
        qkv = rearrange(qkv, 'b l (three h d) -> b l three h d', three=3, h=self.n_heads)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]

        # Apply RoPE separately to Q and K (not V!)
        device_type = 'cuda' if q.is_cuda else 'cpu'
        with torch.amp.autocast(device_type=device_type, enabled=False):
            cos, sin = self.rotary(q, seq_dim=1)
            q = apply_rotary_pos_emb(q, cos.to(q.dtype), sin.to(q.dtype))
            k = apply_rotary_pos_emb(k, cos.to(k.dtype), sin.to(k.dtype))

        # Rearrange to (B, H, L, D)
        q = rearrange(q, 'b l h d -> b h l d')
        k = rearrange(k, 'b l h d -> b h l d')
        v = rearrange(v, 'b l h d -> b h l d')

        # Scaled dot-product attention (uses FlashAttention when available)
        attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=False)

        attn_out = rearrange(attn_out, 'b h l d -> b l (h d)')
        attn_out = self.attn_out(attn_out)

        # Residual with gate
        x = x_skip + gate_msa * attn_out

        # MLP with AdaLN (SwiGLU)
        x_skip = x
        x_norm = self.norm2(x)
        x_mod = modulate(x_norm, shift_mlp, scale_mlp)

        # SwiGLU: (gate_proj * silu) ⊙ up_proj
        gate = F.silu(self.mlp_gate(x_mod))
        up = self.mlp_up(x_mod)
        mlp_out = self.mlp_down(gate * up)

        # Residual with gate
        x = x_skip + gate_mlp * mlp_out

        return x


class DiffusionLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.vocab_embd = nn.Embedding(config.vocab_size, config.n_embd)
        self.sigma_map = TimestepEmbedder(config.cond_dim)

        # Shared rotary embeddings across all blocks
        head_dim = config.n_embd // config.n_head
        self.rotary = Rotary(head_dim)

        self.blocks = nn.ModuleList([
            DDiTBlock(config.n_embd, config.n_head, config.cond_dim, self.rotary)
            for _ in range(config.n_layer)
        ])

        self.norm = RMSNorm(config.n_embd)
        self.output = DDiTFinalLayer(config.n_embd, config.vocab_size, config.cond_dim)

    def forward(self, tok_id, sigma):
        """
        Args:
            tok_id: (B, L) token IDs
            sigma: (B,) noise levels

        Returns:
            logits: (B, L, vocab_size)
        """
        # Embed tokens
        x = self.vocab_embd(tok_id)  # (B, L, D)

        # Embed timestep with DOUBLE SiLU (SEDD does this!)
        c = F.silu(self.sigma_map(sigma))  # (B, cond_dim)

        # Apply transformer blocks
        for block in self.blocks:
            x = block(x, c)

        # Final norm and output projection
        x = self.norm(x)
        logits = self.output(x, c)  # (B, L, vocab_size)

        # mask out the noised tokens in the logits.
        # this way learns to denoise instead of x_t-1 being the exact same tokens
        # since forward noising replaces the tokens
        # so curerntly nosied tokens should not be x_t-1 tokens
        # forces x_t-1 tokens != x_t tokens. moves to clean state
        logits = torch.scatter(logits, -1, rearrange(tok_id, 'b l -> b l 1'), torch.zeros_like(logits[..., :1]))

        return logits


def score_entropy_loss(logits, x_t, x_0, sigma, vocab_size):
    """
    Score entropy loss for discrete diffusion (SEDD style).

    logits: (B, L, V) predicted scores
    x_t: (B, L) noised tokens
    x_0: (B, L) clean tokens
    sigma: (B,) noise levels
    """
    B, L, V = logits.shape

    # Convert logits to probabilities (scores)
    score = F.softmax(logits, dim=-1)  # (B, L, V)

    # Compute e^sigma - 1 with numerical stability
    esigm1 = torch.where(
        sigma < 0.5,
        torch.expm1(sigma),
        torch.exp(sigma) - 1
    )  # (B,)

    # Ratio for uniform graph
    ratio = 1 - vocab_size / (repeat(esigm1, 'b -> b l', l=L) + vocab_size)  # (B, L)

    # Mask for positions where x_t == x_0 (no perturbation)
    no_perturb = (x_t == x_0)  # (B, L)

    # Negative term
    score_mean = reduce(score, 'b l v -> b l', 'mean')  # (B, L) - mean over vocab
    score_at_xt = rearrange(torch.gather(score, -1, rearrange(x_t, 'b l -> b l 1')), 'b l 1 -> b l')
    score_at_x0 = rearrange(torch.gather(score, -1, rearrange(x_0, 'b l -> b l 1')), 'b l 1 -> b l')

    neg_term = score_mean - score_at_xt / vocab_size
    neg_term = torch.where(
        no_perturb,
        ratio * neg_term,
        score_at_x0 / repeat(esigm1, 'b -> b l', l=L) + neg_term
    )

    # Constant term
    const = torch.where(
        no_perturb,
        (vocab_size - 1) / vocab_size * ratio * (ratio.log() - 1),
        ((-ratio.log() - 1) / ratio - (vocab_size - 2)) / vocab_size
    )

    # Positive term
    pos_term = reduce(score, 'b l v -> b l', 'mean') - rearrange(torch.gather(score, -1, rearrange(x_t, 'b l -> b l 1')), 'b l 1 -> b l') / vocab_size

    # Combine all terms
    loss = pos_term - neg_term + const

    # Mean over batch and sequence
    return reduce(loss, 'b l -> ', 'mean')
