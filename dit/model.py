import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange
from dataclasses import dataclass

#################################################################################
#                                  Config                                       #
#################################################################################

@dataclass
class DLMConfig:
    hidden_size: int = 512
    cond_dim: int = 128
    length: int = 1024
    vocab: int = 50304
    n_blocks: int = 8
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

    def forward(self, x, seq_dim=1):
        seq_len = x.shape[seq_dim]
        if seq_len != self.seq_len_cached:
            self.seq_len_cached = seq_len
            t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
            # dims are: batch, seq_len, qkv, head, dim
            self.cos_cached = emb.cos()[None, :, None, None, :].repeat(1,1,3,1,1)
            self.sin_cached = emb.sin()[None, :, None, None, :].repeat(1,1,3,1,1)
            # This makes the transformation on v an identity.
            self.cos_cached[:,:,2,:,:].fill_(1.)
            self.sin_cached[:,:,2,:,:].fill_(0.)

        return self.cos_cached, self.sin_cached

def rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

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
        self.weight = nn.Parameter(torch.ones(dim))
        self.dim = dim

    def forward(self, x):
        return F.layer_norm(x.float(), [self.dim]).to(x.dtype) * self.weight[None, None, :]

#################################################################################
#                          Timestep Embedder                                    #
#################################################################################

class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True)
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(half, dtype=torch.float32, device=t.device) / half)
        args = t[:, None].float() * freqs[None]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return emb

    def forward(self, t):
        return self.mlp(self.timestep_embedding(t, self.frequency_embedding_size))

#################################################################################
#                                 Core Model                                    #
#################################################################################

class DDiTBlock(nn.Module):
    def __init__(self, dim, n_heads, cond_dim, mlp_ratio=4, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads
        self.dropout = dropout

        self.norm1 = LayerNorm(dim)
        self.attn_qkv = nn.Linear(dim, 3*dim, bias=False)
        self.attn_out = nn.Linear(dim, dim, bias=False)

        self.norm2 = LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_ratio*dim, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_ratio*dim, dim, bias=True)
        )

        self.adaLN_modulation = nn.Linear(cond_dim, 6*dim, bias=True)
        nn.init.zeros_(self.adaLN_modulation.weight)
        nn.init.zeros_(self.adaLN_modulation.bias)

    def forward(self, x, rotary_cos_sin, c, kv_cache=None, prefix_len=0):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c)[:, None].chunk(6, dim=2)

        # Self-attention with modulation
        x_norm = modulate(self.norm1(x), shift_msa, scale_msa)
        qkv = rearrange(self.attn_qkv(x_norm), "b s (three h d) -> b s three h d", three=3, h=self.n_heads)
        qkv = apply_rotary_pos_emb(qkv, *rotary_cos_sin)

        # Flash Attention 2 with optional KV cache
        q, k, v = qkv.unbind(dim=2)  # (B, S, H, D) each

        # Handle KV cache for inference
        if kv_cache is not None and prefix_len > 0:
            k_cached, v_cached = kv_cache
            k = torch.cat([k_cached, k], dim=1)
            v = torch.cat([v_cached, v], dim=1)

        new_kv = (k, v) if kv_cache is not None else None

        # Flash Attention 2 (PyTorch automatically selects Flash Attention on H100)
        # With BF16/FP16 on H100, this uses Flash Attention 2 for optimal performance
        x_attn = F.scaled_dot_product_attention(
            q.transpose(1, 2),  # (B, H, S, D)
            k.transpose(1, 2),
            v.transpose(1, 2),
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False
        ).transpose(1, 2)  # (B, S, H, D)

        x_attn = rearrange(x_attn, "b s h d -> b s (h d)")
        x = x + gate_msa * F.dropout(self.attn_out(x_attn), p=self.dropout, training=self.training)

        # MLP with modulation
        x_norm = modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = x + gate_mlp * F.dropout(self.mlp(x_norm), p=self.dropout, training=self.training)

        return x, new_kv

class EmbeddingLayer(nn.Module):
    def __init__(self, dim, vocab_dim):
        super().__init__()
        self.embedding = nn.Parameter(torch.empty((vocab_dim, dim)))
        nn.init.kaiming_uniform_(self.embedding, a=math.sqrt(5))

    def forward(self, x):
        return self.embedding[x]

class DDitFinalLayer(nn.Module):
    def __init__(self, hidden_size, out_channels, cond_dim):
        super().__init__()
        self.norm_final = LayerNorm(hidden_size)
        self.linear = nn.Linear(hidden_size, out_channels, bias=True)
        self.adaLN_modulation = nn.Linear(cond_dim, 2*hidden_size, bias=True)

        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)
        nn.init.zeros_(self.adaLN_modulation.weight)
        nn.init.zeros_(self.adaLN_modulation.bias)

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c)[:, None].chunk(2, dim=2)
        x = modulate(self.norm_final(x), shift, scale)
        return self.linear(x)

class SEDD(nn.Module):
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
        self.output_layer = DDitFinalLayer(config.hidden_size, config.vocab, config.cond_dim)

    def forward(self, indices, sigma, kv_cache=None, prefix_len=0):
        """
        Forward pass with optional KV caching for inference.

        Args:
            indices: (B, S) token indices
            sigma: (B,) noise levels
            kv_cache: optional list of (k, v) tuples per layer from previous step (inference only)
            prefix_len: number of prefix tokens to use from cache (inference only)

        Returns:
            logits: (B, S, vocab) output logits
            new_kv_cache: list of (k, v) tuples per layer for caching (None during training)
        """
        x = self.vocab_embed(indices)
        c = F.silu(self.sigma_map(sigma))
        rotary_cos_sin = self.rotary_emb(x)

        # Build new cache only if kv_cache is provided (inference mode)
        new_kv_cache = [] if kv_cache is not None else None

        for i, blk in enumerate(self.blocks):
            layer_kv_cache = kv_cache[i] if kv_cache is not None else None
            x, layer_new_kv = blk(x, rotary_cos_sin, c, layer_kv_cache, prefix_len)
            if new_kv_cache is not None:
                new_kv_cache.append(layer_new_kv)

        x = self.output_layer(x, c)

        # Zero out logits corresponding to input indices
        x = x.scatter(-1, indices[..., None], torch.zeros_like(x[..., :1]))

        # If using cache, only return logits for new tokens
        if prefix_len > 0 and kv_cache is not None:
            x = x[:, prefix_len:]

        return x, new_kv_cache

#################################################################################
#                                  Noise                                        #
#################################################################################

class GeometricNoise():
    def __init__(self, sigma_min=1e-3, sigma_max=20):
        self.sigmas = 1.0 * torch.tensor([sigma_min, sigma_max])

    def get_noise(self, t):
        rate_noise = self.sigmas[0] ** (1 - t) * self.sigmas[1] ** t * (self.sigmas[1].log() - self.sigmas[0].log())
        total_noise = self.sigmas[0] ** (1 - t) * self.sigmas[1] ** t
        return total_noise, rate_noise




class LogLinearNoise:
    def __init__(self, eps=1e-3):
        self.eps = eps

    def get_noise(self, t):
        """Returns (total_noise, rate_noise)"""
        rate_noise = (1 - self.eps) / (1 - (1 - self.eps) * t)
        total_noise = -torch.log1p(-(1 - self.eps) * t)
        return total_noise, rate_noise

#################################################################################
#                                  Sampling                                     #
#################################################################################

def sample_categorical(categorical_probs, method="hard"):
    if method == "hard":
        gumbel_norm = 1e-10 - (torch.rand_like(categorical_probs) + 1e-10).log()
        return (categorical_probs / gumbel_norm).argmax(dim=-1)
    else:
        raise ValueError(f"Method {method} for sampling categorical variables is not valid.")

#################################################################################
#                                  Graph                                        #
#################################################################################

class UniformGraph:
    """Uniform graph for discrete diffusion."""

    def __init__(self, dim):
        self.dim = dim

    def rate(self, i):
        """Forward rate matrix column for state i."""
        edge = torch.ones(*i.shape, self.dim, device=i.device) / self.dim
        edge.scatter_(-1, i[..., None], -(self.dim - 1) / self.dim)
        return edge

    def reverse_rate(self, i, score):
        """Reverse rate for sampling."""
        normalized_rate = self.rate(i) * score
        normalized_rate.scatter_(-1, i[..., None], torch.zeros_like(normalized_rate[..., :1]))
        normalized_rate.scatter_(-1, i[..., None], -normalized_rate.sum(dim=-1, keepdim=True))
        return normalized_rate

    def sample_rate(self, i, rate):
        """Sample from rate matrix."""
        return sample_categorical(F.one_hot(i, num_classes=self.dim).to(rate) + rate)

    def sample_transition(self, i, sigma):
        """Sample next state according to transition probabilities."""
        move_chance = 1 - (-sigma).exp()
        move_indices = torch.rand(*i.shape, device=i.device) < move_chance
        return torch.where(move_indices, torch.randint_like(i, self.dim), i)

    def sample_limit(self, *batch_dims):
        """Sample from the limiting uniform distribution."""
        return torch.randint(0, self.dim, batch_dims)

    def score_entropy(self, score, sigma, x, x0):
        """
        Compute score entropy for training.
        Optimized version with fused operations.
        """
        esigm1 = torch.where(sigma < 0.5, torch.expm1(sigma), torch.exp(sigma) - 1)
        ratio = 1 - self.dim / (esigm1 + self.dim)

        # Compute negative term
        score_mean = score.mean(dim=-1)
        score_at_x = torch.gather(score, -1, x[..., None]).squeeze(-1)
        neg_term = score_mean - score_at_x / self.dim

        score_at_x0 = torch.gather(score, -1, x0[..., None]).squeeze(-1)
        neg_term = torch.where(
            x == x0,
            ratio * neg_term,
            score_at_x0 / esigm1 + neg_term
        )

        # Compute constant term
        const = torch.where(
            x == x0,
            (self.dim - 1) / self.dim * ratio * (ratio.log() - 1),
            ((-ratio.log() - 1) / ratio - (self.dim - 2)) / self.dim
        )

        # Compute positive term
        sexp = score.exp()
        pos_term = sexp.mean(dim=-1) - torch.gather(sexp, -1, x[..., None]).squeeze(-1) / self.dim

        return pos_term - neg_term + const





###########################
#    resource accnting    #
###########################
# FLOPs calculation for transformer
# Per token: 6 * N (forward) + 12 * N (backward with gradients)
# N = number of parameters
# For attention models: ~6ND where D is sequence length (approximately)
def print_flops(config, batch_size, seq_len, num_params, n_iters, ngpus):
    tokens_per_iter = batch_size * seq_len
    flops_per_token_fwd = 6 * num_params
    flops_per_token_bwd = 12 * num_params
    flops_per_token_total = flops_per_token_fwd + flops_per_token_bwd

    flops_per_iter = tokens_per_iter * flops_per_token_total
    total_flops = flops_per_iter * n_iters

    print("="*60)
    print("MODEL STATISTICS")
    print("="*60)
    print(f"Model params: {num_params/1e6:.2f}M ({num_params:,})")
    print(f"Hidden size: {config.hidden_size}")
    print(f"Num blocks: {config.n_blocks}")
    print(f"Num heads: {config.n_heads}")
    print(f"Sequence length: {config.length}")
    print(f"Vocab size: {config.vocab}")
    print()

    print("="*60)
    print("FLOPS ACCOUNTING")
    print("="*60)
    print(f"Tokens per iteration: {tokens_per_iter:,}")
    print(f"FLOPs per token (fwd): {flops_per_token_fwd/1e9:.2f} GFLOPs")
    print(f"FLOPs per token (bwd): {flops_per_token_bwd/1e9:.2f} GFLOPs")
    print(f"FLOPs per token (total): {flops_per_token_total/1e9:.2f} GFLOPs")
    print(f"FLOPs per iteration: {flops_per_iter/1e12:.2f} TFLOPs")
    print(f"Total FLOPs ({n_iters:,} iters): {total_flops/1e15:.2f} PFLOPs")
    print()

    # Estimate training time for both A100 and H100
    a100_tflops = 312  # TFLOP/s for A100 in bf16
    h100_tflops = 989  # TFLOP/s for H100 in bf16
    h100_fp8_tflops = 1979  # TFLOP/s for H100 in FP8 (with Transformer Engine)

    # A100 estimates
    a100_time_per_iter = (flops_per_iter / 1e12) / (a100_tflops * ngpus)  # seconds
    a100_total_time = a100_time_per_iter * n_iters / 3600  # hours

    # H100 estimates (bf16)
    h100_time_per_iter = (flops_per_iter / 1e12) / (h100_tflops * ngpus)  # seconds
    h100_total_time = h100_time_per_iter * n_iters / 3600  # hours

    # H100 estimates (fp8)
    h100_fp8_time_per_iter = (flops_per_iter / 1e12) / (h100_fp8_tflops * ngpus)  # seconds
    h100_fp8_total_time = h100_fp8_time_per_iter * n_iters / 3600  # hours

    print("="*60)
    print("ESTIMATED TRAINING TIME")
    print("="*60)
    print(f"A100 (bf16) - {a100_tflops} TFLOP/s per GPU:")
    print(f"  Time per iter: {a100_time_per_iter:.3f}s")
    print(f"  Total time: {a100_total_time:.2f} hours ({a100_total_time/24:.2f} days)")
    print()
    print(f"H100 (bf16) - {h100_tflops} TFLOP/s per GPU:")
    print(f"  Time per iter: {h100_time_per_iter:.3f}s")
    print(f"  Total time: {h100_total_time:.2f} hours ({h100_total_time/24:.2f} days)")
    print(f"  Speedup vs A100: {a100_total_time/h100_total_time:.2f}x")
    print()
    print(f"H100 (fp8) - {h100_fp8_tflops} TFLOP/s per GPU:")
    print(f"  Time per iter: {h100_fp8_time_per_iter:.3f}s")
    print(f"  Total time: {h100_fp8_total_time:.2f} hours ({h100_fp8_total_time/24:.2f} days)")
    print(f"  Speedup vs A100: {a100_total_time/h100_fp8_total_time:.2f}x")
    print()
    print(f"Total compute: {ngpus} GPUs × {n_iters:,} iterations")
    print()
