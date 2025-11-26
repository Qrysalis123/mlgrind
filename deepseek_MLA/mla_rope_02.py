
"""

1. low rank KV compression
2. **decouple RoPE to avoid recompute during inference


############# DECOUPLING ROPE #############3
    1.
        w_base: to be compressed
        w_rope: to apply rope

    2.
        q_base <- compress(w_base)
        q_top <- rotate(w_rope)
        Q = concat( q_base, q_top)

        k_base <- compress(w_base)
        k_top <- rotate(w_rope)
        K = concat( k_base, k_top)

    3.
        cache



    Decoupled RoPE:
        “We keep a small set of dimensions completely
        outside the low-rank compression"

    compressing KV loses the positioning:
        need to uncompress -> rotate again


    # Decoupled RoPE
        another comrpession for its RoPE

    x_base = W_base(X)
    x_rope = W_rope(X)

    x_rope = apply_rope(x_rope, positions)

    concat x_base + x_rope back to dim

    but x_rope is not comrpessed

    when new tokens: apply rope to new x_rope.

    RoPE subspace: 128
    Base comrpesseion: 512-1024 per head



######33### LOW RANK KV COMRPESSION ############3
    KV projected to latent space
    c_kv = W_d,kv @ h_t

    (b t h) @ (b h kv) -> (b t kv) cache this

    project K,V back out
    K = (b t kv) -> (b t k)
    V = (b t kv) -> (b t v)
"""

from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange, repeat


@dataclass
class Config:
    # Core MODEL
    vocab_size = 32000 # vocab size
    d_model = 5120 # token embd
    n_layers = 2 # transformer blocks Nx
    n_heads = 8 # number of heads in MLA

    # MLA
    d_kv_comp = 128  # latent dim for KV
    d_rope = 16 # dim for rope

    # MOE
    n_experts = 32 # number of experts
    n_shared = 2 # number of always active experts
    top_k = 2 # experts activated per token

    # Training configs
    seq_len = 256
    batch_size = 1
    ffn_dim = 384
    device_groups = 4


class DeepSeekV2(nn.Module):
    """
    emd -> rmsnorm -> mla (compress, rope) -> rmsnorm -> moe -> rmsnorm -> linear output

    MLA:
        input:
            -> W_q
            -> W_dkv:
                -> compress -> W_uk = K
                -> comrpess -> W_uv = V
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embd = nn.Embedding(config.seq_len, config.d_model)
        self.norm = RMSNorm(config.d_model)

        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])

        self.lm_head = nn.Linear(config.d_model, config.vocab_size)

    def forward(self, token_ids):
        x = self.embd(token_ids)
        for block in self.blocks:
            x, _ = block(x)

        return self.lm_head(self.norm(x))



class TransformerBlock(nn.Module):
    """
    norm1 -> MLA -> norm 2 -> moe
    """
    def __init__(self, config):
        super().__init__()
        self.norm1 = RMSNorm(config.d_model)
        self.attn = MLA()
        self.norm2 = RMSNorm(config.d_model)
        self.moe = DeepSeekMoE()

    def forward(self, x, past_kv=None):
        pass



""" MLA """
class MLA(nn.Module):
    def __init__(self, config):
        self.config = config
        self.d_head = config.d_model // config.n_heads
        self.split_dim = self.d_head - config.d_rope

        # Projections
        self.W_dkv = nn.Linear(config.d_model, config.d_kv_comp)
        self.W_dq = nn.Linear(config.d_model, config.d_kv_comp)

        # Base Query dim
        self.W_uq = nn.Linear(config.d_kv_comp, config.n_heads * self.split_dim)
        # Query Rope dim
        self.W_qr = nn.Linear(config.d_kv_comp, config.n_heads * config.d_rope)
        self.W_kr = nn.Linear(config.d_model, config.n_heads * config.d_rope)

        self.rotary = RotaryEmbedding(config.d_rope)

        self.output = nn.Linear(config.n_heads * self.d_head, config.d_model)


    def forward(self, h, past_kv=None):
        batch_size, seq_len, _ = h.shape

        # KV Compression
        c_kv = self.W_dkv(h) # (b, s, embd) -> (b s d_kv)
        # divide into heads
        k = rearrange(c_kv, 'b s (h d) -> b h s d', h=self.config.n_heads)
        v = rearrange(c_kv, 'b s (h d) -> b h s d', h=self.config.n_heads)
        # Split K into rot part
        k_rot = rearrange(self.W_kr(k), 'b s (h r) -> b h s r', h=self.config.n_heads)

        # Query Compression
        c_q = self.W_dq(h) # (b s embd) -> (b s d_kv_comp)
        # Split into Base & Rot and divide into heads
        q_base = rearrange(self.W_uq(c_q), 'b s (h d) -> b h s d', h=self.config.n_heads)
        q_rot = rearrange(self.W_qr(c_q), 'b s (h r) -> b h s r', h=self.config.n_heads)

        # Apply rope to Q & K
        rotary_emb = self.rotary(seq_len)
        cos = repeat(rotary_emb.cos, 'l d -> b h l d', b=batch_size, h=self.config.n_heads)
        sin = repeat(rotary_emb.sin, 'l d -> b h l d', b=batch_size, h=self.config.n_heads)

        q_rot = apply_rotary(q_rot, cos, sin)
        k_rot = apply_rotary(k_rot, cos, sin)

        # Concatenate latent + rope
        q = torch.cat([q_base, q_rot], dim=-1)
        v = torch.cat([k, k_rot], dim=-1)

        # Attention score
        scores = torch.einsum("b h q d, b h k d -> b h q k", q, k) / math.sqrt(self.d_head)
        attn = scores.softmax(dim=-1)
        out = torch.einsum("b h q k, b h v d -> b h q d", attn, v)

        # project out
        out = rearrange(out, 'b h q d -> b q (h d)')
        out = self.output(out)
        return out, (c_kv, k_rot)
