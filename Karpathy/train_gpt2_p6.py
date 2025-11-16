
"""
p6:
    added edu_fineweb10B

    change DataLoaderLite to work with shards

    added hellaswag for eval + logging


def load_tokens(filename):
    npt = np.load(filename)
    ptt = torch.tensor(npt, dtype=torch.long)
    return ptt

class DataLoaderLite:
    def __init__(self, B, T, process_rank, num_processes, split):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        assert split in {'train', 'val'}

        # get the shard filenames
        data_root = 'edu_fineweb10B'
        shards = os.listdir(data_root)
        shards = [s for s in shards if split in s]
        shards = sorted(shards)
        shards = [os.path.join(data_root, s) for s in shards]
        self.shards = shards

        assert len(self.shards) > 0, f"no shards for split {split}"
        if master_process:
            print(f"found {len(shards)} shards for split {split}")


        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.curernt_shard])
        self.current_position = self.B * self.T * self.process_rank

    def next_batch(self):
        # shift position
        # if next batch out of bounds -> load next shard

        B, T = self.B, self.T
        buf = self.tokens[self.current_position: self.current_position+B*T+1]
        x = rearrange(buf[:-1], '(B T) -> B T', B=B)
        y = rearrange(buf[1:], '(B T) -> B T', B=B)
        self.current_position += B * T * self.num_processes


        if self.current_position + (B * T * self.num_processes + 1) >= len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_position = self.B * self.T * self.process_rank
        return x, y


- 10B tokens in total

GPT-2 trained on 0.5M tokens per batch size. next power of 2 is 2**19
- 2**19 = 524,288 Tokens per step

Datset of 10B tokens:
    1 epoch:
        max_steps = 10e8 / 2**19 = 19,073 steps

GPT-2 warmup over first 375M tokens:
        warmup_steps = 375e6 / 2**19 = 715

batch size = 64
    64 * 1024 tokens * 8 = 524,288 tokens = 1 step
    so wont do grad acucm



# Add validation loader
# in DataLoaderLite:
        def reset():
            to start at 0 for validation

# 8x A100
torchrun --standalone --nproc_per_node=8 train_gpt2_p6.py


"""

from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
from einops import rearrange, repeat
import math
import tiktoken
import time
import inspect
import os
import numpy as np
from hellaswag import render_example, iterate_examples


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

def load_tokens(filename):
    npt = np.load(filename)
    ptt = torch.tensor(npt, dtype=torch.long)
    return ptt

class DataLoaderLite:
    def __init__(self, B, T, process_rank, num_processes, split):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        assert split in {'train', 'val'}

        # get the shard filenames
        data_root = 'edu_fineweb10B'
        shards = os.listdir(data_root)
        shards = [s for s in shards if split in s]
        shards = sorted(shards)
        shards = [os.path.join(data_root, s) for s in shards]
        self.shards = shards

        assert len(self.shards) > 0, f"no shards for split {split}"
        if master_process:
            print(f"found {len(shards)} shards for split {split}")


        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = self.B * self.T * self.process_rank

    def reset(self):
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = self.B * self.T * self.process_rank

    def next_batch(self):
        # shift position
        # if next batch out of bounds -> load next shard

        B, T = self.B, self.T
        buf = self.tokens[self.current_position: self.current_position+B*T+1]
        x = rearrange(buf[:-1], '(B T) -> B T', B=B)
        y = rearrange(buf[1:], '(B T) -> B T', B=B)
        self.current_position += B * T * self.num_processes


        if self.current_position + (B * T * self.num_processes + 1) >= len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_position = self.B * self.T * self.process_rank
        return x, y


# ----------------------------------------------------------------
#
def get_most_likely_row(tokens, mask, logits):
    # evaluate the autoregressive loss at all positions
    shift_logits = (logits[..., :-1, :]).contiguous()
    shift_tokens = (tokens[..., 1:]).contiguous()
    flat_shift_logits = shift_logits.view(-1, shift_logits.size(-1))
    flat_shift_tokens = shift_tokens.view(-1)
    shift_losses = F.cross_entropy(flat_shift_logits, flat_shift_tokens, reduction='none')
    shift_losses = shift_losses.view(tokens.size(0), -1)
    # now get the average loss just for the completion region (where mask == 1), in each row
    shift_mask = (mask[..., 1:]).contiguous() # we must shift mask, so we start at the last prompt token
    masked_shift_losses = shift_losses * shift_mask
    # sum and divide by the number of 1s in the mask
    sum_loss = masked_shift_losses.sum(dim=1)
    avg_loss = sum_loss / shift_mask.sum(dim=1)
    # now we have a loss for each of the 4 completions
    # the one with the lowest loss should be the most likely
    pred_norm = avg_loss.argmin().item()
    return pred_norm

# ----------------------------------------------------------------------------

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

enc = tiktoken.get_encoding("gpt2")

total_batch_size = 524288 # 2**19, using GPT-2 small of 0.5M tokens per batch
B, T = 16, 1024
assert total_batch_size % (B*T*ddp_world_size) ==0
grad_accum_steps = total_batch_size // (B*T*ddp_world_size)
if master_process:
    print('total desired tokens per batch:', total_batch_size)
    print('grad_accum_steps:', grad_accum_steps)

train_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split='train')
val_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split='val')

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
warmup_steps = 715
max_steps = 19073
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

# create log
log_dir = "log"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "log.txt")
with open(log_file, 'w') as f:
    # train loss, val loss, hellaswag accuracy
    pass



for step in range(max_steps):
    t0 = time.time()
    last_step = (step == max_steps - 1)

    # validation every 100th
    if step % 250 == 0 or last_step:
        model.eval()
        val_loader.reset()
        with torch.no_grad():
            val_loss_accum = 0.0
            val_loss_steps = 20
            for _ in range(val_loss_steps):
                x, y = val_loader.next_batch()
                x, y = x.to(device), y.to(device)
                with torch.autocast(device_type=device, dtype=torch.bfloat16):
                    logits, loss = model(x, y)
                loss = loss / val_loss_steps
                val_loss_accum += loss.detach()
            if ddp:
                dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)
            if master_process:
                print(f"validation loss: {val_loss_accum.item():.4f}")
                with open(log_file, 'a') as f:
                    f.write(f"{step} val {val_loss_accum.item():.4f}\n")

    # evaluate hellaswag
    if (step % 250 == 0 or last_step):
        num_correct_norm = 0
        num_total = 0
        for i, example in enumerate(iterate_examples('val')):
            if i % ddp_world_size != ddp_rank:
                continue
            _, tokens, mask, label = render_example(example)
            tokens = tokens.to(device)
            mask = mask.to(device)
            with torch.no_grad():
                with torch.autocast(device_type=device, dtype=torch.bfloat16):
                    logits, loss = model(tokens)
                pred_norm = get_most_likely_row(tokens, mask, logits)
            num_total += 1
            num_correct_norm += int(pred_norm == label)

        # reduce stats across all processes
        if ddp:
            num_total = torch.tensor(num_total, dtype=torch.long, device=device)
            num_correct_norm = torch.tensor(num_correct_norm, dtype=torch.long, device=device)
            dist.all_reduce(num_total, op=dist.ReduceOp.SUM)
            dist.all_reduce(num_correct_norm, op=dist.ReduceOp.SUM)
            num_total = num_total.item()
            num_correct_norm = num_correct_norm.item()
        acc_norm = num_correct_norm / num_total
        if master_process:
            print(f"Hellaswag accuracy: {num_correct_norm}/{num_total}={acc_norm:.4f}")
            with open(log_file, 'a') as f:
                f.write(f"{step} hella {acc_norm:.4f}\n")


    # generate text to see results
    if ((step > 0 and step % 250 == 0) or last_step):
        model.eval()
        num_return_sequences = 1
        max_length = 32
        tokens = enc.encode("hello i am an ai model, and ")
        tokens = torch.tensor(tokens, dtype=torch.long)
        tokens = repeat(tokens, 't -> b t', b=num_return_sequences)
        xgen = tokens.to(device)
        sample_rng = torch.Generator(device=device) #separate rng so doenst affect global for training
        model(xgen).manual_seed(42 + ddp_rank)
        while xgen.size(1) < max_length:
            with torch.no_grad():
                with torch.autocast(device_type=device, dtype=torch.bfloat16):
                    logits, loss = model(xgen)
                logits = logits[:, -1,:]
                probs = F.softmax(logits, dim=-1)
                topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
                ix = torch.multinomial(topk_probs, num_samples=1, generator=sample_rng)
                xcol = torch.gather(topk_indices, -1, ix)
                xgen = torch.cat([xgen, xcol], dim=1)
        for i in range(num_return_sequences):
            tokens = xgen[i, :max_length].tolist()
            decoded = enc.decode(tokens)
            print(f"rank {ddp_rank} sample {i}: {decoded}")

    # train
    model.train()
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
        with open(log_file, 'a') as f:
            f.write(f"{step} train {loss_accum.item():.6f}\n")

if ddp:
    destroy_process_group()
