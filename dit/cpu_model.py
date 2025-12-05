import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange
from dataclasses import dataclass
from typing import Optional
from torch import Tensor

#################################################################################
#                                  Config                                       #
#################################################################################

@dataclass
class DLMConfig:
    hidden_size: int = 768
    cond_dim: int = 256
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
            t = torch.arange(x.shape[seq_dim], device=x.device).type_as(self.inv_freq)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq.clone())
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
    return torch.cat(
        (-x2, x1), dim=-1
    )

def apply_rotary_pos_emb(qkv, cos, sin):
    # qkv shape: (batch, seq_len, 3, n_heads, head_dim)
    # cos/sin shape: (1, seq_len, 3, 1, head_dim)
    # Apply rotary embeddings using the rotate_half helper
    return (qkv * cos) + (rotate_half(qkv) * sin)

#################################################################################
#                                  Dropout Scale                                #
#################################################################################

def bias_dropout_add_scale(x, bias: Optional[Tensor], scale: Tensor, residual: Optional[Tensor], prob: float, training: bool) -> Tensor:
    out = scale * F.dropout(x + bias if bias is not None else x, p=prob, training=training)
    if residual is not None:
        out = residual + out
    return out

def bias_dropout_add_scale_train(x, bias, scale, residual, prob):
    return bias_dropout_add_scale(x, bias, scale, residual, prob, True)

def bias_dropout_add_scale_inference(x, bias, scale, residual, prob):
    return bias_dropout_add_scale(x, bias, scale, residual, prob, False)

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
        x_norm = F.layer_norm(x.float(), [self.dim])
        return x_norm.to(x.dtype) * self.weight[None, None, :]

def residual_linear(x, W, x_skip, residual_scale):
    dim_out, dim_in = W.shape
    return torch.addmm(x_skip.view(-1, dim_out), x.view(-1, dim_in), W.T, alpha=residual_scale).view(*x.shape[:-1], dim_out)

#################################################################################
#                          Timestep Embedder                                    #
#################################################################################

class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size)
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
#                         Attention fallback (CPU SDPA)                          #
#################################################################################

def attention_fallback_sdpa(qkv, n_heads, dropout, training, kv_cache=None, prefix_len=0):
    """
    Attention with optional KV caching.
    
    Args:
        qkv: (B, S, 3, H, D) query, key, value
        n_heads: number of attention heads
        dropout: dropout probability
        training: training mode flag
        kv_cache: optional (k, v) tuple from previous step
        prefix_len: number of prefix tokens to use from cache
        
    Returns:
        out: (B, S, H, D) attention output
        new_kv: (k, v) tuple for caching
    """
    q, k, v = qkv.unbind(dim=2)
    b, s, h, d = q.shape
    
    # If we have a cache and prefix_len > 0, concatenate cached K/V
    if kv_cache is not None and prefix_len > 0:
        k_cached, v_cached = kv_cache
        # k_cached, v_cached: (B, prefix_len, H, D)
        k = torch.cat([k_cached, k], dim=1)
        v = torch.cat([v_cached, v], dim=1)
    
    # Store full K/V for next iteration
    new_kv = (k, v)
    
    # Reshape for SDPA
    q_sdpa = q.reshape(b*h, s, d)
    k_sdpa = k.reshape(b*h, k.shape[1], d)
    v_sdpa = v.reshape(b*h, v.shape[1], d)
    
    out = F.scaled_dot_product_attention(q_sdpa, k_sdpa, v_sdpa, dropout_p=dropout if training else 0.0, is_causal=False)
    out = out.reshape(b, h, s, d).permute(0, 2, 1, 3)
    
    return out, new_kv

#################################################################################
#                                 Core Model                                    #
#################################################################################

class DDiTBlock(nn.Module):
    def __init__(self, dim, n_heads, cond_dim, mlp_ratio=4, dropout=0.1):
        super().__init__()
        self.n_heads, self.dropout = n_heads, dropout
        self.norm1 = LayerNorm(dim)
        self.attn_qkv = nn.Linear(dim, 3*dim, bias=False)
        self.attn_out = nn.Linear(dim, dim, bias=False)
        self.norm2 = LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_ratio*dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_ratio*dim, dim)
        )
        self.adaLN_modulation = nn.Linear(cond_dim, 6*dim)
        nn.init.zeros_(self.adaLN_modulation.weight)
        nn.init.zeros_(self.adaLN_modulation.bias)

    def _get_bias_dropout_scale(self):
        return bias_dropout_add_scale_train if self.training else bias_dropout_add_scale_inference

    def forward(self, x, rotary_cos_sin, c, kv_cache=None, prefix_len=0, seqlens=None):
        B, S = x.shape[:2]
        bias_dropout_scale_fn = self._get_bias_dropout_scale()
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c)[:, None].chunk(6, dim=2)
        x_skip = x
        x = modulate(self.norm1(x), shift_msa, scale_msa)
        qkv = rearrange(self.attn_qkv(x), "b s (three h d) -> b s three h d", three=3, h=self.n_heads)
        qkv = apply_rotary_pos_emb(qkv, *rotary_cos_sin)
        x, new_kv = attention_fallback_sdpa(qkv, self.n_heads, self.dropout, self.training, kv_cache, prefix_len)
        x = rearrange(x, "b s h d -> b s (h d)")
        x = bias_dropout_scale_fn(self.attn_out(x), None, gate_msa, x_skip, self.dropout)
        mlp_out = self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        x = bias_dropout_scale_fn(mlp_out, None, gate_mlp, x, self.dropout)
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
        self.linear = nn.Linear(hidden_size, out_channels)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)
        self.adaLN_modulation = nn.Linear(cond_dim, 2*hidden_size)
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
        Forward pass with optional KV caching.
        
        Args:
            indices: (B, S) token indices
            sigma: (B, 1) noise levels
            kv_cache: optional list of (k, v) tuples per layer from previous step
            prefix_len: number of prefix tokens to use from cache
            
        Returns:
            logits: (B, S_out, vocab) output logits (S_out = S if no cache, else S - prefix_len)
            new_kv_cache: list of (k, v) tuples per layer for caching
        """
        x = self.vocab_embed(indices)
        c = F.silu(self.sigma_map(sigma))
        rotary_cos_sin = self.rotary_emb(x)
        
        # Build new cache
        new_kv_cache = [] if kv_cache is not None else None
        
        for i, blk in enumerate(self.blocks):
            layer_kv_cache = kv_cache[i] if kv_cache is not None else None
            x, layer_new_kv = blk(x, rotary_cos_sin, c, layer_kv_cache, prefix_len)
            if new_kv_cache is not None:
                new_kv_cache.append(layer_new_kv)
        
        x = self.output_layer(x, c)
        
        # zero out logits corresponding to input indices
        x = torch.scatter(x, -1, indices[..., None], torch.zeros_like(x[..., :1]))
        
        # If using cache, only return logits for new tokens
        if prefix_len > 0 and kv_cache is not None:
            x = x[:, prefix_len:]
        
        return x, new_kv_cache



#################################################################################
#                                  Noise                                        #
#################################################################################
"""
Usage:
    # create GeometricNoise instance
    noise_schedule = GeometricNoise(sigma_min=1e-3, sigma_max=1.0)

    # batch of times t (e.g., 32 samples)
    t = torch.rand(4)  # uniform in [0,1]
    t = torch.tensor(1)

    # compute total noise and its rate
    sigma, dsigma = noise_schedule.forward(t)

    print("t:", t)
    print("sigma(t):", sigma)       # noise at each t
    print("dsigma(t)/dt:", dsigma)  # derivative of noise
"""

class GeometricNoise():
    def __init__(self, sigma_min=1e-3, sigma_max=1):
        self.sigmas = 1.0 * torch.tensor([sigma_min, sigma_max])

    def get_noise(self, t):
        rate_noise = self.sigmas[0] ** (1 - t) * self.sigmas[1] ** t * (self.sigmas[1].log() - self.sigmas[0].log())
        total_noise = self.sigmas[0] ** (1 - t) * self.sigmas[1] ** t
        return total_noise, rate_noise

class LogLinearNoise():
    """
    Log Linear noise schedule built so that 1 - 1/e^(n(t)) interpolates between 0 and ~1
    when t goes from 0 to 1. Used for absorbing

    Total noise is -log(1 - (1 - eps) * t), so the sigma will be (1 - eps) * t
    """
    def __init__(self, eps=1e-3):
        self.eps = eps

    def get_noise(self, t):
        rate_noise = (1 - self.eps) / (1 - (1 - self.eps) * t)
        total_noise =  -torch.log1p(-(1 - self.eps) * t)
        return total_noise, rate_noise


#################################################################################
#                                  Cat Sample                                   #
#################################################################################
def gumbel_softmax(categorical_probs, hard=False, eps=1e-9):
    logits = categorical_probs.clamp(min=1e-9).log()
    return F.gumbel_softmax(logits, hard=hard)


def sample_categorical(categorical_probs, method="hard"):
    if method == "hard":
        gumbel_norm = 1e-10 - (torch.rand_like(categorical_probs) + 1e-10).log()
        return (categorical_probs / gumbel_norm).argmax(dim=-1)
    else:
        raise ValueError(f"Method {method} for sampling categorical variables is not valid.")





#################################################################################
#                                  Transition Matrix                            #
#################################################################################


class UniformGraph:
    """
    Simple uniform graph: every state can transition to every other state equally.
    """
    def __init__(self, dim):
        self.dim = dim

    def rate(self, i):
        """
        Compute the forward rate matrix column for state i.
        """
        edge = torch.ones(*i.shape, self.dim, device=i.device) / self.dim
        edge = edge.scatter(-1, i[..., None], - (self.dim - 1) / self.dim)
        return edge

    def transp_rate(self, i):
        """
        Transpose of rate (same for uniform).
        """
        return self.rate(i)

    def reverse_rate(self, i, score):
        """
        Construct the reverse rate for a uniform graph.
        Reverse rate = score * transp_rate, normalized to sum to zero along last dim.
        """
        normalized_rate = self.transp_rate(i) * score
        # subtract sum to make row sum = 0
        normalized_rate.scatter_(-1, i[..., None], torch.zeros_like(normalized_rate[..., :1]))
        normalized_rate.scatter_(-1, i[..., None], -normalized_rate.sum(dim=-1, keepdim=True))
        return normalized_rate

    def sample_rate(self, i, rate):
        return sample_categorical(F.one_hot(i, num_classes=self.dim).to(rate) + rate)


    def transition(self, i, sigma):
        """
        Exponential of rate matrix: e^{sigma Q}.
        """
        trans = torch.ones(*i.shape, self.dim, device=i.device) * (1 - (-sigma[..., None]).exp()) / self.dim
        trans = trans.scatter(-1, i[..., None], torch.zeros_like(trans))
        trans = trans.scatter(-1, i[..., None], 1 - trans.sum(dim=-1, keepdim=True))
        return trans

    def sample_transition(self, i, sigma):
        """
        Sample next state according to transition probabilities.
        """
        move_chance = 1 - (-sigma).exp()
        move_indices = torch.rand(*i.shape, device=i.device) < move_chance
        return torch.where(move_indices, torch.randint_like(i, self.dim), i)

    def staggered_score(self, score, dsigma):
        """
        Compute score for staggered timestep.
        """
        dim = score.shape[-1]
        epow = (-dsigma).exp()[..., None]
        return ((epow - 1) / (dim * epow)) * score.sum(dim=-1, keepdim=True) + score / epow

    def sample_limit(self, *batch_dims):
        """
        Sample from the limiting uniform distribution.
        """
        return torch.randint(0, self.dim, batch_dims, device='cpu')

    def score_entropy(self, score, sigma, x, x0):
        """
        Compute score entropy for training.
        """
        esigm1 = torch.where(
            sigma < 0.5,
            torch.expm1(sigma),
            torch.exp(sigma) - 1
        )
        ratio = 1 - self.dim / (esigm1 + self.dim)

        neg_term = score.mean(dim=-1) - torch.gather(score, -1, x[..., None]).squeeze(-1) / self.dim
        neg_term = torch.where(
            x == x0,
            ratio * neg_term,
            torch.gather(score, -1, x0[..., None]).squeeze(-1) / esigm1 + neg_term
        )

        const = torch.where(
            x == x0,
            (self.dim - 1) / self.dim * ratio * (ratio.log() - 1),
            ((-ratio.log() - 1) / ratio - (self.dim - 2)) / self.dim
        )

        sexp = score.exp()
        pos_term = sexp.mean(dim=-1) - torch.gather(sexp, -1, x[..., None]).squeeze(-1) / self.dim
        return pos_term - neg_term + const
