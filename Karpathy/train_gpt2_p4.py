
"""
p4:
    gradient accumulation (bc the hyperparams are correlated to the batch size)

    batch size 0.5M: (is the total tokesn a batch. not batch number)
        0.5M,
            0.5e6 / 1024 seq len = ~488 batch size (B=488, T=1024)

    total_batch_size = 524288 # 2**19 (~0.5M) number of tokens per batch
    B, T = 16, 1024

    assert total_batch_size % (B * T) == 0
    grad steps = total_batch_size // (B * T) accum gradient at each batch step

    so accum gradients until learnt a token size of 0.5M before updating


    for micro_step in range(grad_accum_steps):
        loss
        += accum loss
        loss / N normalizer






p5:
    distributed data parallel

p6:
    validation, evaluation



"""

from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
from einops import rearrange
import math
import tiktoken
import time
import inspect


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
        self.transformer.wte.weight = self.lm_head.weight
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

    def configure_optimizers(self, weight_decay, learning_rate, device):
        param_dict = {pn: p for pn,p in self.named_parameters()}
        param_dict = {pn: p for pn,p in param_dict.items() if p.requires_grad}

        decay_params = [p for n,p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n,p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed param tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed param tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")

        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and 'cuda' in device
        print(f'using fused AdamW: {use_fused}')
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8)
        return optimizer





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

        # # ===replace this with flash attention===
        # att = torch.einsum('bhsd,bhtd->bhst', q, k) / math.sqrt(self.dk)
        # att = att.masked_fill(self.bias[:, :, :s, :s] == 0, float('-inf'))
        # att = F.softmax(att, dim=-1)
        # y = torch.einsum('bhst,bhtd->bhsd', att, v)
        # # ===replace this with flash attention===
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)

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


total_batch_size = 524288 # 2**19, using GPT-2 small of 0.5M tokens per batch
B, T = 8, 1024
assert total_batch_size % (B*T) ==0
grad_accum_steps = total_batch_size // (B*T)
print('total desired tokens per batch:', total_batch_size)
print('grad_accum_steps:', grad_accum_steps)



train_loader = DataLoaderLite(B, T)

torch.set_float32_matmul_precision('high')

model = GPT(GPTConfig(vocab_size=50304))
model.to(device)
model = torch.compile(model)


max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_steps = 10
max_steps = 50
def get_lr(it):
    # linear increase for warmup
    if it < warmup_steps:
        return max_lr * (it+1) / warmup_steps

    # stable 10% of max lr
    if it > max_steps:
        return min_lr

    # cosine decay
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)




optimizer = model.configure_optimizers(weight_decay=0.1, learning_rate=6e-4, device=device)

for step in range(max_steps):
    t0 = time.time()
    optimizer.zero_grad()
    loss_accum = 0.0
    for micro_step in range(grad_accum_steps):
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            logits, loss = model(x, y)
        loss = loss / grad_accum_steps
        loss_accum += loss.detach() # detach from tensor
        loss.backward()

    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    optimizer.step()
    torch.cuda.synchronize()
    t1 = time.time()
    dt = (t1 - t0)
    tokens_processed = train_loader.B * train_loader.T * grad_accum_steps
    tokens_per_sec = tokens_processed / dt
    print(f"step {step} | loss: {loss_accum.item():.6f} | norm: {norm:.4f} | lr: {lr:.4f} | dt: {dt*1000:.2f}ms | tok/sec: {tokens_per_sec:.2f}")
