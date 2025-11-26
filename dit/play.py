import math
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat, reduce

"""
Main components:

--------------DiT Block----------------
carries condition info by time embd -> cond dim -> 6x hidden scale shift gate


-----------Score entropy loss----------
cross entropy
score function





"""

@dataclass
class DiTLMConfig:
    seq_len: int = 1024
    vocab_size: int = 50257
    n_blocks: int = 6
    n_heads: int = 8
    n_embd: int = 512
    cond_dim: int = 128 # cond dim for noise/time


class DiTLM(nn.Module):
    """
    input:
        tok_id: (B, L) token ids
        sigma: (B,) noise/timestep level

    return:
        logits: (B, L, vocab_size)

    """
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.vocab_embd = nn.Embedding(config.vocab_size, config.n_embd) # token embedding
        self.sigma_map = TimestepEmbedder(config.cond_dim) # sinusoidal timestep embedding
        self.rotary = Rotary(config.n_embd // config.n_heads) # Rope positional embedding

        self.blocks = nn.ModuleList([
            DiTBlock(config.n_embd, config.n_heads, config.cond_dim, self.rotary)
            for _ in range(config.n_blocks)])

        self.norm = RMSNorm(config.n_embd)
        self.output = DiTFinalLayer(config.n_embd, config.vocab_size, config.cond_dim)

    def forward(self, tok_id, sigma):
        x = self.vocab_embd(tok_id)
        c = F.silu(self.sigma_map(sigma))

        for block in self.blocks:
            x = block(x, c)

        x = self.norm(x)
        logits = self.output(x, c)

        # logits at current noised tokens set to 0, mask it out
        # forces x_t-1 tokens != x_t tokens. so its actually denoising
        logits = torch.scatter(logits, -1, rearrange(tok_id, 'b l -> b l 1'), torch.zeros_like(logits[..., :1]))

        return logits


class TimestepEmbedder(nn.Module):
    def __init__(self, cond_dim, freq_dim=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(freq_dim, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim)
        )
        self.freq_dim = freq_dim

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
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
        t_freq = self.timestep_embedding(t, self.cond_dim)
        return self.mlp(t_freq)


class Rotary(nn.Module):
    def __init__(self, dim, base=10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None

    def forward(self, x, seq_dim=1):
        seq_len = x.shape[seq_dim]
        if seq_len != self.seq_len_cached:
            self.seq_len_cached = seq_len
            t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            emb = torch.cat([freqs, freqs], dim=-1)
            self.cos_cached = emb.cos()[None, :, None, :]
            self.sin_cached = emb.sin()[None, :, None, :]
        return self.cos_cached, self.sin_cached

def apply_rotary_pos_emb(x, cos, sin):
    d = x.shape[-1]
    x1, x2 = x[..., :d//2], x[..., d//2:]
    cos = cos[..., :d//2]
    sin = sin[..., :d//2]
    rotated_x1 = x1 * cos - x2 * sin
    rotated_x2 = x1 * sin + x2 * cos
    return torch.cat([rotated_x1, rotated_x2], dim=-1)


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(reduce(x.pow(2), '... d -> ... 1', 'mean') + self.eps)

    def forward(self, x):
        return self.weight * self._norm(x.float()).type_as(x)





class DDiTBlock(nn.Module):
    def __init__()


















def score_entropy_loss(logits, x_t, x_0, sigma, vocab_size):
    """
    score entropy loss (SEDD)

    logits: (B,L,V)     predicted scores
    x_t:    (B,L)       noised tokens
    x_0:    (B,L)       target tokens
    sigma:  (B,)        noise level

    return:
        mean loss

    """

    B,L,V = logits.shape

    score = F.softmax(logits, dim=-1) # (B, L, V)

    esigm1 = torch.where(
        sigma < 0.5,
        torch.expm1(sigma),
        torch.exp(sigma) - 1
    ) # (B,)

    ratio = 1 - vocab_size / (repeat(esigm1, 'b -> b l', l=L) + vocab_size) # (B, L)

    # mask where x_t == x_0 (no perturbation)
    no_perturb = (x_t == x_0) # (B,L)

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
