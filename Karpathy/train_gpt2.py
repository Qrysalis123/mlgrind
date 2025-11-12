
'''

124M
12 layers
768 dim


softmax
linear

-------BLOCK---------
+ skip
mlp
ln 2
skip

+ skip
mha
ln 1
skip
-------BLOCK----------
+ pos
emb


'''


from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
from einops import rearrange
import math

@dataclass
class GPT(nn.Module):
    block_size: int = 256 # max seq len
    vocab_size: int = 65 # vocab size 50257 (50000 bpe + 256 bytes + 1 <eos>)
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        '''
        hf schema:
            wte: (vocab, emb) token embd
            wpe: (seq, emb) pos embd
            h: num heads
            ln_f: layer norm
            lm_head: (embd, vocab_size) the output clasifier

        '''
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))

        self.lm_head = nn.Embedding(config.n_embd, config.vocab_size, bias=False)

    def forward(self, x):
        pass


class Block(nn.Module):
    """
    + skip
    mlp
    ln 2
    skip

    + skip
    mha
    ln 1
    skip
    """

    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd),
        self.attn = CausalSelfAttention(config) # placeholder for now
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config) # placeholder for now

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        '''
        ff part

        c_fc(embd, 4 * embd) -> gelu -> c_proj(4*embd, embd)

        '''
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)

    def forward(self, x):
        return self.c_proj(self.gelu(self.c_fc(x)))


class CausalSelfAttention(nn.Module):
    '''
    MHA

    x = (b, seq, n_embd)

    init:
    d_k = n_embd / n_head
    c_attn = (n_embd, 3 * n_embd)
    c_proj = (n_embd, n_embd)

    forward:
        - qkv = c_attn(x)
        - (b, seq, n_embd) -> (b, seq, 3 * n_embd)

        - q,k,v = qkv.split(n_embd, dim=2)
        - (b, seq, 3 * n_embd) -> (b, seq, n_embd)

        - split q,k,v into nheads
        - (b, s, d) -> (b, nh, s, dk)

        - attn q @ k.T
        - (b, nh, s, dk) @ (b, nh, s, dk) -> (b, nh, s, s) / sqrt(dk)

        - mask -> softmax -> @ v
        - (b, nh, s, s) @ (b, nh, s, dk) -> (b, nh, s, dk)

        - reassemble to nh * dk
        - (b, nh, s, dk) -> (b, s, nh * dk)

        - proj (b, s, n_embd) -> (b, s, n_embd)
        - c_proj(y)
    '''

    def __init__(self, config):
        super().__init__()
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)

        self.nh = config.n_head
        self.n_embd = config.n_embd
        self.dk = config.n_embd // config.n_head

        # Causal mask
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
            .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        b, s, d = x.size()

        qkv = self.c_attn(x) # (b, s, embd) -> (b, s, 3 * embd)
        q,k,v = qkv.split(self.n_embd, dim=2)

        q = rearrange(q, 'b s (h dk) -> b h s dk', h=self.nh, dk=self.dk)
        k = rearrange(k, 'b s (h dk) -> b h s dk', h=self.nh, dk=self.dk)
        v = rearrange(v, 'b s (h dk) -> b h s dk', h=self.nh, dk=self.dk)

        # replace this with flash attention
        att = torch.einsum('b h s dk, b h t dk -> b h s t', q, k) / math.sqrt(self.dk)
        att = att.masked_fill(self.bias[:, :, :s, :s] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        y = torch.einsum('b h s t, b h t dk -> b h s dk', att, v)
        # replace this with flash attention


        y = rearrange(y, 'b h s dk -> b s (h dk)')
        y = self.c_proj(y)

        return y
