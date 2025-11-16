
"""

p5: 8x A100
    distributed data parallel

    launch 8 processes -> each assigned a gpu -> distribute data -> average gradient

    from torch.distributed import init_process_group, destroy_process_group

    # torchrun command sets the env variable RANK, WORLD_SIZE
        WORLD_SIZE: total num processes (8 in this eaxmple)
        DDP RANK: gpu0 gpu1 gpu2... 8
        master_process = ddp_rank == 0 # arbitrary gpu do the logging, checkpointing, etc


INITIALIZE DDP:
    # check for ddp
    ddp = int(os.environ.get('RANK', -1)) != -1
    if ddp:
        ddp_rank =
        ddp_local_rank
        ddp_world_size =
        master_process = ddp_rank ==0
    else:
        ddp_rank = 0
        device = cuda:{ddp_local_rank}

    so can do (B * T * ddp_world_size) tokens at once batch size
        * need to change grad_accum_steps

    logging on master_process only


    python train_gpt2.py -> torchrun --standalone --nproc_per_node=8 train.py

SETUP DATA DISTRIBUTION:
    all 8 processes have diferent ddp rank reading same code
    so split data by ddp_rank

    add a stride to the data loader

    wrap model in DDP

    in training loop:
        if ddp:
            model.require_backward_grad_sync = (micro_step == grad_accum_steos - 1)
            # average grad on last stpe

    # loss acum tensor exists on all the ranks.
    # call all_reduce() to create the average and desposits the average on al lthe ranks
    # so al lthe ranks now contains averaged loss_acum
    # so now master_process will print the synced up averaged loss_acum
    if ddp:
        dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)


# Change:
    model.configure_optimizers()...
    to
    raw_model.configure_optimizers()

RUN:
    `torchrun --standalone --nproc_per_node=8 train.py`

    NOT `python train.py`


total desired tokens per batch: 524288
grad_accum_steps: 4
ddp 0 | loaded: 338025 tokens
ddp 7 | loaded: 338025 tokens
ddp 4 | loaded: 338025 tokens
ddp 6 | loaded: 338025 tokens
ddp 5 | loaded: 338025 tokens
ddp 1 | loaded: 338025 tokens
ddp 3 | loaded: 338025 tokens
ddp 2 | loaded: 338025 tokens
num decayed param tensors: 50, with 124,354,560 parameters
num non-decayed param tensors: 98, with 121,344 parameters
using fused AdamW: True
step 0 | loss: 10.986195 | norm: 9.3117 | lr: 0.0001 | dt: 11472.60ms | tok/sec: 45699.15
step 1 | loss: 10.021605 | norm: 5.2635 | lr: 0.0001 | dt: 377.09ms | tok/sec: 1390352.52
step 2 | loss: 9.451401 | norm: 3.0508 | lr: 0.0002 | dt: 369.25ms | tok/sec: 1419888.07
step 3 | loss: 9.157027 | norm: 4.1729 | lr: 0.0002 | dt: 370.00ms | tok/sec: 1417007.82
step 4 | loss: 8.855445 | norm: 2.3548 | lr: 0.0003 | dt: 368.84ms | tok/sec: 1421459.39
step 5 | loss: 8.594171 | norm: 2.8933 | lr: 0.0004 | dt: 368.80ms | tok/sec: 1421613.77
step 6 | loss: 8.283277 | norm: 2.4613 | lr: 0.0004 | dt: 371.85ms | tok/sec: 1409959.28
step 7 | loss: 7.941079 | norm: 2.0192 | lr: 0.0005 | dt: 368.53ms | tok/sec: 1422643.84
step 8 | loss: 7.557868 | norm: 1.5707 | lr: 0.0005 | dt: 370.55ms | tok/sec: 1414891.69
step 9 | loss: 7.178166 | norm: 1.9787 | lr: 0.0006 | dt: 371.24ms | tok/sec: 1412264.71
step 10 | loss: 6.802391 | norm: 1.2884 | lr: 0.0006 | dt: 369.05ms | tok/sec: 1420623.73
step 11 | loss: 6.516446 | norm: 2.0517 | lr: 0.0006 | dt: 368.08ms | tok/sec: 1424400.23
step 12 | loss: 6.283520 | norm: 0.8673 | lr: 0.0006 | dt: 370.63ms | tok/sec: 1414604.99
step 13 | loss: 6.123775 | norm: 1.2755 | lr: 0.0006 | dt: 369.61ms | tok/sec: 1418482.17
step 14 | loss: 6.013155 | norm: 1.4687 | lr: 0.0006 | dt: 369.87ms | tok/sec: 1417496.49
step 15 | loss: 5.918060 | norm: 0.6570 | lr: 0.0006 | dt: 371.05ms | tok/sec: 1412977.96
step 16 | loss: 5.874942 | norm: 0.7043 | lr: 0.0006 | dt: 371.01ms | tok/sec: 1413120.52
step 17 | loss: 5.836116 | norm: 1.4112 | lr: 0.0006 | dt: 370.37ms | tok/sec: 1415571.15
step 18 | loss: 5.788655 | norm: 0.6009 | lr: 0.0005 | dt: 369.30ms | tok/sec: 1419673.57
step 19 | loss: 5.750583 | norm: 1.1282 | lr: 0.0005 | dt: 372.08ms | tok/sec: 1409082.01
step 20 | loss: 5.713661 | norm: 0.8668 | lr: 0.0005 | dt: 370.56ms | tok/sec: 1414843.45
step 21 | loss: 5.673497 | norm: 0.4830 | lr: 0.0005 | dt: 371.50ms | tok/sec: 1411259.57
step 22 | loss: 5.628794 | norm: 0.4424 | lr: 0.0005 | dt: 371.93ms | tok/sec: 1409636.62
step 23 | loss: 5.649997 | norm: 1.0285 | lr: 0.0005 | dt: 370.45ms | tok/sec: 1415261.40
step 24 | loss: 5.586600 | norm: 0.3825 | lr: 0.0005 | dt: 369.24ms | tok/sec: 1419914.66
step 25 | loss: 5.554578 | norm: 0.6659 | lr: 0.0004 | dt: 369.58ms | tok/sec: 1418589.23
step 26 | loss: 5.516109 | norm: 0.3487 | lr: 0.0004 | dt: 373.68ms | tok/sec: 1403053.93
step 27 | loss: 5.484050 | norm: 0.5476 | lr: 0.0004 | dt: 370.67ms | tok/sec: 1414426.65
step 28 | loss: 5.456867 | norm: 0.5547 | lr: 0.0004 | dt: 372.72ms | tok/sec: 1406651.98
step 29 | loss: 5.428117 | norm: 0.3514 | lr: 0.0004 | dt: 372.73ms | tok/sec: 1406615.09
step 30 | loss: 5.393528 | norm: 0.4710 | lr: 0.0003 | dt: 372.08ms | tok/sec: 1409088.33
step 31 | loss: 5.364033 | norm: 0.3523 | lr: 0.0003 | dt: 370.95ms | tok/sec: 1413365.75
step 32 | loss: 5.337593 | norm: 0.3888 | lr: 0.0003 | dt: 370.21ms | tok/sec: 1416180.13
step 33 | loss: 5.310970 | norm: 0.3238 | lr: 0.0003 | dt: 371.18ms | tok/sec: 1412503.29
step 34 | loss: 5.284133 | norm: 0.3680 | lr: 0.0002 | dt: 371.16ms | tok/sec: 1412579.50
step 35 | loss: 5.259714 | norm: 0.3427 | lr: 0.0002 | dt: 374.43ms | tok/sec: 1400247.73
step 36 | loss: 5.235972 | norm: 0.2874 | lr: 0.0002 | dt: 372.84ms | tok/sec: 1406219.31
step 37 | loss: 5.212223 | norm: 0.3169 | lr: 0.0002 | dt: 371.35ms | tok/sec: 1411826.77
step 38 | loss: 5.191065 | norm: 0.2762 | lr: 0.0002 | dt: 371.54ms | tok/sec: 1411127.35
step 39 | loss: 5.171147 | norm: 0.3451 | lr: 0.0002 | dt: 373.89ms | tok/sec: 1402251.39
step 40 | loss: 5.151126 | norm: 0.2765 | lr: 0.0001 | dt: 373.79ms | tok/sec: 1402628.84
step 41 | loss: 5.132643 | norm: 0.2243 | lr: 0.0001 | dt: 373.45ms | tok/sec: 1403895.92
step 42 | loss: 5.116573 | norm: 0.2526 | lr: 0.0001 | dt: 372.57ms | tok/sec: 1407228.08
step 43 | loss: 5.101176 | norm: 0.2265 | lr: 0.0001 | dt: 372.49ms | tok/sec: 1407513.61
step 44 | loss: 5.086498 | norm: 0.2672 | lr: 0.0001 | dt: 373.75ms | tok/sec: 1402788.10
step 45 | loss: 5.073082 | norm: 0.2701 | lr: 0.0001 | dt: 373.79ms | tok/sec: 1402619.89
step 46 | loss: 5.060590 | norm: 0.2006 | lr: 0.0001 | dt: 372.10ms | tok/sec: 1408986.31
step 47 | loss: 5.048855 | norm: 0.2076 | lr: 0.0001 | dt: 372.95ms | tok/sec: 1405784.21
step 48 | loss: 5.037683 | norm: 0.2304 | lr: 0.0001 | dt: 372.64ms | tok/sec: 1406948.97
step 49 | loss: 5.026746 | norm: 0.2144 | lr: 0.0001 | dt: 371.77ms | tok/sec: 1410266.72
ubuntu@155-248-162-60:~/Karpathy$



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
import os


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
        if master_process:
            print(f"num decayed param tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
            print(f"num non-decayed param tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")

        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and 'cuda' in device
        if master_process:
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

        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        y = rearrange(y, 'b h s dk -> b s (h dk)')
        y = self.c_proj(y)

        return y # (b, s, embd)




# ----------------------------------------------------------------
class DataLoaderLite:
    def __init__(self, B, T, process_rank, num_processes):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes

        with open('input.txt', 'r') as f:
            text = f.read()
        enc = tiktoken.get_encoding('gpt2')
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens)
        print(f"ddp {self.process_rank} | loaded: {len(self.tokens)} tokens")

        self.current_position = self.B * self.T * self.process_rank

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position+B*T+1]
        x = rearrange(buf[:-1], '(B T) -> B T', B=B)
        y = rearrange(buf[1:], '(B T) -> B T', B=B)

        self.current_position += B*T*self.num_processes

        # if next batch is out of bounds reset/rerun from start
        if self.current_position + (B*T*self.num_processes+1) >= len(self.tokens):
            self.current_position = self.B * self.T * self.process_rank
        return x,y
# ----------------------------------------------------------------
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

ddp = int(os.environ.get('RANK', -1)) != -1
if ddp:
    assert torch.cuda.is_available()
    init_process_group(backend='nccl')
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0
else:
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"using device: {device}")


device_type = "cuda" if device.startswith("cuda") else "cpu"
torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

total_batch_size = 524288 # 2**19, using GPT-2 small of 0.5M tokens per batch
B, T = 16, 1024
assert total_batch_size % (B*T*ddp_world_size) ==0
grad_accum_steps = total_batch_size // (B*T*ddp_world_size)
if master_process:
    print('total desired tokens per batch:', total_batch_size)
    print('grad_accum_steps:', grad_accum_steps)

train_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size)

torch.set_float32_matmul_precision('high')


# create model
model = GPT(GPTConfig(vocab_size=50304))
model.to(device)
model = torch.compile(model)
# wrap model into ddp
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])
    # once backward pass is over. it calls allreduce which averages gradients
raw_model = model.module if ddp else model


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


optimizer = raw_model.configure_optimizers(weight_decay=0.1, learning_rate=6e-4, device=device)

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
        if ddp:
            # sync on last step
            model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)
        loss.backward()
    if ddp:
        dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)

    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    optimizer.step()
    torch.cuda.synchronize()
    t1 = time.time()
    dt = (t1 - t0)
    tokens_processed = train_loader.B * train_loader.T * grad_accum_steps * ddp_world_size
    tokens_per_sec = tokens_processed / dt
    if master_process:
        print(f"step {step} | loss: {loss_accum.item():.6f} | norm: {norm:.4f} | lr: {lr:.4f} | dt: {dt*1000:.2f}ms | tok/sec: {tokens_per_sec:.2f}")

if ddp:
    destroy_process_group()

import sys;sys.exit(0)
