
"""
---------CPU---------------
p2:
    add the loss function
    optimizer adamW
    build a data loader

    ##### WEIGHT SHARING THE wte & lm_head ####
    ** Using the Output Embedding to Improve Language Models **

    "for the input embedding, we would like the network to react similarly
        to synonyms, while in the output embedding, we would like the scores of words
        that are interchangeable to be similar"

    so sharing weights would make the output embedding react like the input embedding
    where synonyms are reacted similarly
    similar embedding output words should have similar weights

    achieved by contributing wte at lm_head, and head:


    ### Model init ###
        * wte stddev = 0.02
        * wpe stddev = 0.01

    #### SCALE WEIGHTS OF RESIDUAL LAYERS #####
    " We scale the weights of residual layers at initialization by a
    factor of 1/sqrt(N) where N is the number of residual layers"

        accounts for the accumulation on the residual path with model depth
        example:
            for i in range(n):
                x += randn(768)
            this variance in the residual conns grows instead of ~N(0,1)

            with 1/sqrt(N)
            for i in range(N):
                x += n**-0.5 * randn(768)
            this keeps the stddev at ~1

        implementation:
            self.c_proj.NANOGPT_SCALE_INIT = 1
            then in _init_weights():
                if hasattr(module, 'NANOGPT_SCALE_INIT'):
                    scale by 2 * 1/sqrt(N)
                    '2x' because theres 2 residuals




------------GPU---------------------
p3:
    gpu training, flash attetnion, quantization,
    hyperparams, scaling laws,

p4:
    grad accum

p5:
    distributed data parallel

p6:
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

enc = tiktoken.get_encoding('gpt2')

# create log
import os
log_dir = "log"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "shakespeare_loss.txt")
with open(log_file, 'w') as f:
    # train loss, val loss, hellaswag accuracy
    pass

# optimize
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
for i in range(5000):
    x, y = train_loader.next_batch()
    x, y = x.to(device), y.to(device)
    optimizer.zero_grad()
    logits, loss = model(x, y)
    loss.backward()
    optimizer.step()
    print(f"step {i}: loss={loss.item():.4f}")
    with open(log_file, 'a') as f:
        f.write(f"{i} shakespeare {loss:.4f}\n")

    # Save model weights periodically
    if (i + 1) % 50 == 0:
        os.makedirs("checkpoints", exist_ok=True)
        torch.save(model.state_dict(), f"checkpoints/step_{i}.pt")
        torch.save(model.state_dict(), "checkpoints/latest.pt")
        print(f"Saved checkpoint at step {i}")

    # Inference every 100 steps
    if (i + 1) % 100 == 0:
        model.eval()
        print(f"\n{'='*60}")
        print(f"Step {i} - Generating text...")
        print('='*60)

        prompt = "hi i am an ai model"
        tokens = enc.encode(prompt)
        tokens = torch.tensor(tokens, dtype=torch.long, device=device)
        tokens = tokens.unsqueeze(0)

        with torch.no_grad():
            for _ in range(50):
                tokens_cond = tokens if tokens.size(1) <= model.config.block_size else tokens[:, -model.config.block_size:]
                logits, _ = model(tokens_cond)
                logits = logits[:, -1, :]
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                tokens = torch.cat([tokens, next_token], dim=1)

        generated_text = enc.decode(tokens[0].tolist())
        print(generated_text)
        print()

        model.train()


# ----------------------------------------------------------------
# Inference
# ----------------------------------------------------------------

print("\n" + "="*60)
print("Generating text...")
print("="*60)

model.eval()

# Get the tokenizer

# Start with a prompt
prompt = "hi i am an ai model"
tokens = enc.encode(prompt)
tokens = torch.tensor(tokens, dtype=torch.long, device=device)
tokens = tokens.unsqueeze(0)  # (1, T)

# Generate
max_new_tokens = 100
with torch.no_grad():
    for _ in range(max_new_tokens):
        # Crop to block_size if needed
        tokens_cond = tokens if tokens.size(1) <= model.config.block_size else tokens[:, -model.config.block_size:]

        # Forward
        logits, _ = model(tokens_cond)

        # Get logits at the last position
        logits = logits[:, -1, :]  # (B, vocab_size)

        # Sample from the distribution
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)  # (B, 1)

        # Append to sequence
        tokens = torch.cat([tokens, next_token], dim=1)

# Decode and print
generated_text = enc.decode(tokens[0].tolist())
print(generated_text)

# Save to log file
with open(log_file, 'a') as f:
    f.write(f"\n{'='*60}\n")
    f.write(f"Generated text after training:\n")
    f.write(f"{'='*60}\n")
    f.write(f"{generated_text}\n")

import sys; sys.exit(0)
