
"""
------------GPU---------------------
p3:
    gpu training, flash attetnion, quantization,

p4:
    hyperparams, scaling laws,

p5:
    distributed data parallel
    validation, evaluation



"""

from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
from einops import rearrange, repeat
import math
import tiktoken


@dataclass
class GPTConfig:
    block_size: int = 1024 # max seq len
    vocab_size: int = 50257 # vocab size 50257 (50000 bpe + 256 bytes + 1 <eos>)
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # 1st weight sharing: point wte -> lm_head.weight. since they same size
        # this is 1024*50257 / 124M = ~30% of the model
        self.transformer.wte.weight = self.lm_head.weight

        # init params. iterates all the sub modules in this moduel
        # and applies the weights functions
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std += (2 * self.config.n_layer) ** -0.5

            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        # idx (B, T)
        B, T = idx.size()
        assert T <= self.config.block_size, "tooooooooooo big"

        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        pos_emb = self.transformer.wpe(pos) # (T) -> (B, T, n_embd)
        tok_emb = self.transformer.wte(idx) # (B, T) -> (B, T, n_embd)
        x = tok_emb + pos_emb

        for block in self.transformer.h:
            x = block(x) # (B, T, n_embd)

        x = self.transformer.ln_f(x)
        logits = self.lm_head(x) # (B,T,n_embd) -> (B,T,vocab_size)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                rearrange(logits, 'b t c -> (b t) c'),
                rearrange(targets, 'b t -> (b t)')
            )
        return logits, loss


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        return self.c_proj(self.gelu(self.c_fc(x)))


class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        # scale by 1/sqrt(N) to avoid stddev growing
        self.c_proj.NANOGPT_SCALE_INIT = 1

        self.n_head = config.n_head
        self.n_embd = config.n_embd

        assert config.n_embd % config.n_head == 0
        self.dk = config.n_embd // config.n_head # // for int instead of float

        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
            .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        b, s, d = x.size()

        qkv = self.c_attn(x) # (b, s, embd) -> (b, s, 3 * embd)
        q,k,v = qkv.split(self.n_embd, dim=2)

        q = rearrange(q, 'b s (h dk) -> b h s dk', h=self.n_head, dk=self.dk)
        k = rearrange(k, 'b s (h dk) -> b h s dk', h=self.n_head, dk=self.dk)
        v = rearrange(v, 'b s (h dk) -> b h s dk', h=self.n_head, dk=self.dk)

        # ===replace this with flash attention===
        att = torch.einsum('bhsd,bhtd->bhst', q, k) / math.sqrt(self.dk)
        att = att.masked_fill(self.bias[:, :, :s, :s] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        y = torch.einsum('bhst,bhtd->bhsd', att, v)
        # ===replace this with flash attention===

        y = rearrange(y, 'b h s dk -> b s (h dk)')
        y = self.c_proj(y)

        return y # (b, s, embd)




# ----------------------------------------------------------------
class DataLoaderLite:
    def __init__(self, B, T):
        self.B = B
        self.T = T

        with open('input.txt', 'r') as f:
            text = f.read()
        enc = tiktoken.get_encoding('gpt2')
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens)
        print(f"loaded: {len(self.tokens)} tokens")
        print(f"1 epoch = {len(self.tokens) // self.B} batches")

        self.current_position = 0

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position+B*T+1]
        x = rearrange(buf[:-1], '(B T) -> B T', B=B)
        y = rearrange(buf[1:], '(B T) -> B T', B=B)

        self.current_position += B*T

        # if next batch is out of bounds reset/rerun from start
        if self.current_position + (B*T+1) >= len(self.tokens):
            self.current_position = 0
        return x,y

# ----------------------------------------------------------------

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"using device: {device}")


train_loader = DataLoaderLite(B=4, T=32)

model = GPT(GPTConfig())
model.to(device)

# optimize
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
for i in range(50):
    x, y = train_loader.next_batch()
    x, y = x.to(device), y.to(device)
    optimizer.zero_grad()
    logits, loss = model(x, y)
    loss.backward()
    optimizer.step()
    print(f"step {i}: loss={loss.item():.4f}")


import sys; sys.exit(0)



# count params
total_params = sum(p.numel() for p in model.parameters())
print(f"Total params: {total_params:,}")



# Print the initial prompt
tokens = enc.encode('helllo im an ai model')
tokens = torch.tensor(tokens, dtype=torch.long) # (T)
tokens = repeat(tokens, 'T -> B T', B=B)
x = tokens.to(device)

print(enc.decode(x[0].tolist()), end='', flush=True)

while x.size(1) < T:
    with torch.no_grad():
        logits, loss = model(x) # (B, T, vocab_size)
        logits = logits[:, -1, :] # last vocab size (B, vocab_size)
        probs = F.softmax(logits, dim=-1)
        topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)  # (B, 50)
        ix = torch.multinomial(topk_probs, 1) # (B, 1)
        xcol = torch.gather(topk_indices, 1, ix) # (B, 1)
        x = torch.cat((x, xcol), dim=1) # append to sequence

        # Print only the new token
        new_token = xcol[0].item()
        decoded_token = enc.decode([new_token])
        print(decoded_token, end='', flush=True)

print()  # New line at the end
