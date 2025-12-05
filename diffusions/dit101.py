
"""
DiT paper
Scalable Diffusion Models with Transformers
https://arxiv.org/pdf/2212.09748

https://apxml.com/courses/advanced-diffusion-architectures


---------------traditional layer norm (no conditioning)-----------------
x_norm = (x - mean) / std
x_out = x_norm * gamma + beta
    problem:
        no timestep condition



---------------AdaLN: incorporates `timestamp` condition--------------

Adaptive Layer Normalization (AdaLN)
    AdaLN:
        # makes shift & scale depend on timestep
        # given a c condition (timestep/noise embedding)
        shift, scale, gate = MLP(c).chunk(3)

        then adaptive norm
        x_norm = LayerNorm(x)
        x_mod = x_norm * (1 + scale) + shift

        MLP

        # adaptive residual
        x = x_skip + gate * out


    Shift: moves features in embd space based on timestep
    # different noise levels may need different features centered differently

    Scale: amplify or dampen features based on timestep
    # high noise, model needs stronger feature responses, low noise gentle refinements

    Gate: control how much of the features to add to residual
    # high noise: model needs big changes, low noise: preserve most input




---------------------DiT Block------------------------
* needs additional conditional info: noise timestep. embd separately
* so introduces adaLN-Zero block

* paper did variations of architecture to see which works:
    1. in context conditioning: appending t & c  and treat them same as image tokens
    2. treat t & c separate of len 2. then cross attention.
    3. adaLN (adaptive Layer Norm): learns the mean, std instead of ~(0,1)
    4. adaLN-Zero block: adaLN with additional `scale` alpha to residual connections
        * this helps guide how much info from original is passed through to next layer
        * on the side is:
            * conditioning embd -> MLP -> carries conditional embd info to the other layers
            * like y1, B1, a1, y2, B2, a2
            (scale, shift) scale, scale, shift, scale





-------------------Scalable Diffusion Models with Transformers---------------
* transformer for spatial self attention
* adaptive norm layers (adaLN) to inject conditional info and channel counts

-------------adaLN-Zero-------------
condition embd -> MLP -> 6 * condition embd (2x scale, shift, scale)
MLP outputs 6 params
    scale, shift: applied to input tokens before MHA
    scale: applied to output of MHA
    scale, shift: applied to output of MHA
    scale: applied to output of MLP


---------------DiT: forward pass-------------------
Patchify (tokenizing):
    (3,256,256) -> z (4,32,32) -> patchify by p (T,d) -> sinusoidal positioning

DiT block:
    tokens -> transformer

    conditional embd:
        * noise tinmesteps t
        * class labels c
        * text,.. etc


    1. in-context conditioning t & c. just add 2 extra tokens ViT then remove
    2. concat t & c embd. add heads
    3. adaptive layer norm (adaLN): learn scale & shift params from t & c
    4. adaLN-Zero: zero-initializing the final batch norm scale factor
        * scale & shift
        * scaling params immediately prior to any residual conenctions

Decoder:
    * decode tokens into output noise pred
EMA of DiT weights with decay of 0.9999


------ timestep embd ----
256 dim
2 layer MLP with dim == transformers hidden size
SiLU activations

adaLN -> timestep -> SiLU linear -> 4x or 6x transformer hidden size
GELU in core transformer


c_out: shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp
init with zero to maintain training stability

"""





"""
------------https://github.com/xinli2008/diffusion_from_scratch/blob/main/models/dit.py-------------------

            DiT Block:

       scale <----------------|
         ff                   |
    scale, shift <------------|
    layer norm                |
         |                    |
       scale <----------------|
        MHA                   |
    scale, shift <------------|
     layer norm              MLP
         |                    |
    input tokens       conditioning




"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from dataclasses import dataclass
import numpy as np


@dataclass
class DiTConfig():
    seq_len: int
    n_embd: int

    n_heads: int
    n_blocks: int
    n_layers: int

    c_dim: int

class DiT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_emb = TimestepEmbedder(config.c_dim)

        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.rotary_emb = Rotary(config.hidden_size // config.n_heads)

        self.blocks = nn.ModuleList([
            DiTBlock(config.n_embd, config.n_heads, config.c_dim) for _ in range(config.n_blocks)
        ])

        self.proj_out = FinalLayer(config.n_embd, config.c_dim, config.vocab_size)

    def forward(self, tok_id, sigma):
        x = self.tok_emb(tok_id) # (batch, seq, emb)
        c = F.silu(self.c_emb(sigma)) # (batch, c_dim)
        rotary_cos_sin = self.rotary_emb(x)

        for block in self.blocks:
            x = block(x, c, rotary_cos_sin)

        x = self.proj_out(x, c)

        return x # (batch, seq, vocab)


def modulate(x, scale, shift):
    return x + (1 + rearrange(scale, 'b d -> b 1 d')) + rearrange(shift, 'b d -> b 1 d')

class DiTBlock(nn.Module):
    def __init__(self, n_embd, n_heads, c_dim):
        super().__init__()

        self.adaLN = nn.Sequential(
            nn.SiLU(),
            nn.Linear(c_dim, n_embd * 6, bias=False)
        )

        self.norm1 = nn.LayerNorm(n_embd)
        self.attn = MHA(n_embd, n_heads)
        self.norm2 = nn.LayerNomr(n_embd)
        self.mlp = nn.Sequential

        self.mlp = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd, bias=False),
            nn.SiLU(),
            nn.Linear(4 * n_embd, n_embd, bias=False)
        )

    def forward(self, x, c):
        shift1, scale1, gate1, shift2, scale2, gate2 = rearrange(self.adaLN(c), 'b (six d) -> six b d', six=6)
        x = x + rearrange(gate1, 'b d -> b 1 d') * self.attn(modulate(self.norm1(x), scale1, shift1))
        x = x + rearrange(gate2, 'b d -> b 1 d') * self.mlp(modulate(self.norm2(x), scale2, shift2))
        return x


class FinalLayer(nn.Module):
    def __init__(self, n_embd, c_dim, vocab_size):
        super().__init__()
        self.final_norm = nn.LayerNorm(n_embd)
        self.proj = nn.Linear(n_embd, vocab_size)

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(c_dim, n_embd * 2, bias=False)
        )

    def forward(self, x, c):
        shift, scale = rearrange(self.adaLN_modulation(c), 'b (two d) -> two b d', two=2)
        x = self.final_norm(x)
        x = modulate(x, scale, shift)
        x = self.proj(x)
        return x




"""
KV cachign

init:
    new_kv_cache = []

1.
    - splits x_constant x_new

2. qkv on new block & attn on all
    - computes qkv on x_new
    - new_kv_cache <- concats qkv
    - x, layer_new_kv <- attn(kv_cache, prefix_len)

3. append new kv to cache
    - new_kv_cache.append(layer_new_kv)

4. compute proj(x) and return y, kv_cache
    - compute projection
    - return new_kv_cache
    - return proj(x), kv_cache


"""
