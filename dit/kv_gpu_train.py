import os
import time
import math
import torch
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from kv_model import DiT, DiTConfig
from kv_noise import LogLinearNoise, UniformGraph
from finewebedu import get_dataloader
from transformers import PreTrainedTokenizerFast

#################
#    config     #
#################

init_from = "resume"  # "scratch" or "resume"

out_dir = "gpu_run01"
batch_size = 32  # per GPU
seq_len = 1024
data_dir = ".cache/fineweb-edu-10BT"



# Block-wise training config (independent sampling)
min_prefix = 2
max_prefix = 1024
min_block = 2
max_block = 1024

# sampling
sampling_eps = 1e-5  # increased from 1e-5 to avoid numerical instability at very small t
noise_eps = 1e-3

# optimizer
lr = 3e-4
min_lr = 1e-5
beta1 = 0.9
beta2 = 0.999
weight_decay = 0.1
warmup_iters = 2000
max_iters = 100000
grad_clip = 1.0

# logging/checkpointing
eval_iters = 100
save_iters = 100


# bf16 mixed precision + compile
use_amp = True
amp_dtype = torch.bfloat16
use_compile = False

#################
#    DDP setup  #
#################
ddp = int(os.environ.get("RANK", -1)) != -1
if ddp:
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = dist.get_world_size()
    device = f"cuda:{local_rank}"
    torch.cuda.set_device(device)
    master_process = rank == 0
else:
    rank = 0
    local_rank = 0
    world_size = 1
    device = "cuda"
    master_process = True

if master_process:
    print(f"DDP: {ddp}, world_size: {world_size}")

#####################
#    data           #
#####################
if master_process:
    print("Loading tokenizer...")
tokenizer = PreTrainedTokenizerFast(tokenizer_file="tokenizer/bpe-32k-uncased-bytelevel.json")


if master_process:
    print("Loading data...")
train_loader = get_dataloader(data_dir, batch_size=batch_size, distributed=ddp, num_workers=4)

def infinite_loader(loader, distributed=False):
    epoch = 0
    while True:
        if distributed and hasattr(loader, 'sampler'):
            loader.sampler.set_epoch(epoch)
        for batch in loader:
            yield batch["input_ids"]
        epoch += 1

train_iter = infinite_loader(train_loader, distributed=ddp)

#####################
#    dirs           #
#####################
if master_process:
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "logs"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "samples"), exist_ok=True)

#####################
#    model          #
#####################
config = DiTConfig(
    hidden_size=512,
    cond_dim=128,
    length=128,
    vocab=32768,
    n_blocks=12,
    n_heads=8,
    dropout=0.1
)

if master_process:
    print(f"\nConfig: {config}")

model = DiT(config).to(device)
graph = UniformGraph(config.vocab)
noise = LogLinearNoise(eps=noise_eps)

num_params = sum(p.numel() for p in model.parameters())
if master_process:
    print(f"Model params: {num_params/1e6:.2f}M")
    print(f"Using AMP: {use_amp}, dtype: {amp_dtype}")
    print(f"Using compile: {use_compile}\n")

# Compile before DDP wrap
if use_compile:
    model = torch.compile(model)

# Wrap with DDP
if ddp:
    model = DDP(model, device_ids=[local_rank])
raw_model = model.module if ddp else model

#####################
#    optimizer      #
#####################
optimizer = optim.AdamW(model.parameters(), lr=lr, betas=(beta1, beta2), weight_decay=weight_decay)

# GradScaler not needed for bf16, but we keep it optional for fp16
scaler = torch.amp.GradScaler('cuda', enabled=(use_amp and amp_dtype == torch.float16))

def get_lr(step):
    if step < warmup_iters:
        return lr * step / warmup_iters
    if step > max_iters:
        return min_lr
    decay_ratio = (step - warmup_iters) / (max_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (lr - min_lr)

#####################
#    resume         #
#####################
start_step = 0
latest_ckpt = os.path.join(out_dir, "checkpoints", "latest.pt")
if init_from == "resume" and os.path.exists(latest_ckpt):
    if master_process:
        print(f"Resuming from {latest_ckpt}...")
    ckpt = torch.load(latest_ckpt, map_location=device, weights_only=False)
    raw_model.load_state_dict(ckpt['model'])
    optimizer.load_state_dict(ckpt['optimizer'])
    start_step = ckpt['step'] + 1
    if master_process:
        print(f"Resumed at step {start_step}")
elif init_from == "scratch":
    if master_process:
        print("Starting from scratch")

#####################
#    training       #
#####################
log_file = os.path.join(out_dir, "logs", "train.txt")
sample_file = os.path.join(out_dir, "samples", "samples.txt")

if master_process:
    print(f"Logging to: {log_file}\n")
    print("=" * 60)
    print("TRAINING - DiT (block-wise KV cache) - GPU bf16")
    print(f"Batch size: {batch_size} x {world_size} = {batch_size * world_size}, Seq len: {seq_len}")
    print("=" * 60)

t0 = time.time()
step = start_step

while True:
    x0 = next(train_iter).to(device)
    B, S = x0.shape

    # LR scheduling
    current_lr = get_lr(step)
    for g in optimizer.param_groups:
        g['lr'] = current_lr

    # Sample timestep t ~ [eps, 1]
    t = (1 - sampling_eps) * torch.rand(B, device=device) + sampling_eps
    sigma, dsigma = noise.get_noise(t)

    # Block-wise training: independent log-uniform sampling of prefix and block sizes
    log_prefix = torch.empty(1).uniform_(math.log(min_prefix), math.log(max_prefix + 1))
    prefix_len = int(log_prefix.exp().item())
    log_block = torch.empty(1).uniform_(math.log(min_block), math.log(max_block + 1))
    block_len = int(log_block.exp().item())

    # Clamp to fit within sequence
    prefix_len = min(prefix_len, S - min_block)
    block_len = min(block_len, S - prefix_len)

    x_clean = x0[:, :prefix_len]
    x0_noisy_block = x0[:, prefix_len : prefix_len + block_len]

    # Perturb only the noisy block
    x_noisy = graph.sample_transition(x0_noisy_block, sigma[:, None])

    # Forward pass with autocast
    with torch.amp.autocast('cuda', dtype=amp_dtype, enabled=use_amp):
        log_score = model(x_noisy, sigma, x_clean=x_clean)

    # Loss computation (stays in float32 due to score_entropy casting)
    loss_per_token = graph.score_entropy(log_score, sigma[:, None], x_noisy, x0_noisy_block)
    weighted_loss = (dsigma[:, None] * loss_per_token).flatten()
    weighted_loss = weighted_loss[~torch.isnan(weighted_loss)]
    loss = weighted_loss.mean()

    # Backward
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad()

    #----------------------
    # Log
    #----------------------
    t1 = time.time()
    dt = t1 - t0
    t0 = t1

    # All-reduce loss for accurate logging
    if ddp:
        loss_reduced = loss.detach().clone()
        dist.all_reduce(loss_reduced, op=dist.ReduceOp.AVG)
    else:
        loss_reduced = loss

    if master_process:
        msg = f"step {step:06d} | loss {loss_reduced.item():.4f} | norm {norm:.4f} | lr {current_lr:.6f} | prefix {prefix_len} block {block_len} | {dt*1000:.0f}ms/step"
        print(msg)
        with open(log_file, "a") as f:
            f.write(msg + "\n")

    #----------------------
    # Sample & save checkpoint
    #----------------------
    if step % eval_iters == 0 and step > 0 and master_process:
        print("=" * 60)
        print(f"SAMPLING at step {step}")
        print("=" * 60)

        raw_model.eval()
        with torch.no_grad():
            with open(sample_file, "a") as f:
                f.write(f"\n=== Step {step} ===\n\n")

            for b in [4, 8, 16, 32, 64, 128, 512, 1024]:
                num_steps = b
                # Block-wise sampling
                prefix = "the capital of france"
                prefix_tok = torch.tensor([tokenizer.encode(prefix)], device=device)

                with torch.amp.autocast('cuda', dtype=amp_dtype, enabled=use_amp):
                    kv_cache = raw_model.build_cache(prefix_tok)

                # initialize noisy block
                x_block = graph.sample_limit(1, b).to(device)
                timesteps = torch.linspace(1, sampling_eps, num_steps + 1, device=device)
                dt_sample = (1 - sampling_eps) / num_steps

                for i in range(num_steps):
                    t_sample = timesteps[i]
                    sigma_sample, dsigma_sample = noise.get_noise(t_sample.expand(1))

                    with torch.amp.autocast('cuda', dtype=amp_dtype, enabled=use_amp):
                        log_score = raw_model(x_block, sigma_sample, kv_cache=kv_cache)
                    score = log_score.float().exp()
                    rev_rate = dt_sample * dsigma_sample[:, None, None] * graph.reverse_rate(x_block, score)
                    x_block = graph.sample_rate(x_block, rev_rate)
                    x_block = x_block.clamp(0, config.vocab - 1)

                text_response = torch.cat([prefix_tok, x_block], dim=1)
                text_response = tokenizer.decode(text_response[0].tolist())
                kv_cache = None
                print(f"block & step: {b} \n{text_response}")
                print()

                with open(sample_file, "a") as f:
                    f.write(f"[{b} tokens, block={b}]\n{text_response}\n\n")

        # Save checkpoint
        torch.save({
            'model': raw_model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'step': step,
            'config': config,
        }, latest_ckpt)
        print(f"Saved: {latest_ckpt}")

        raw_model.train()
        print("=" * 60)

    step += 1
