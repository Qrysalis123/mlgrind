import os
import glob
import time

import tiktoken
import torch
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from model import SEDD, DLMConfig, GeometricNoise, UniformGraph, print_flops
from data import get_dataloaders


#################
#   finetuning  #
#################
# ddp
ngpus = 8
accum = 1
backend = 'nccl'

# system
device = "cuda"
dtype = torch.bfloat16
compile = True

# finetuning
batch_size = 256  # smaller batch for finetuning
batch_size_per_gpu = batch_size // (ngpus * accum)

n_iters = 50000  # fewer iters for finetuning
log_freq = 10
ckpt_freq = 1000
seq_len = 1024
out_dir = "/mnt/persistent/dit_finetune"

# pretrained checkpoint to load
pretrained_ckpt = "/mnt/persistent/dit/checkpoints/ckpt_0600000.pt"

# data - instruction dataset
train_dataset = "HuggingFaceTB/smoltalk"
train_split = "train"
train_subset = None  # no subset for this dataset
cache_dir = None
num_proc = 8

# geometric noise
sigma_min = 1e-3
sigma_max = 20

# sampling
sampling_steps = 32
sampling_eps = 1e-5
sample_batch = 2
sample_seq_len = 128

# adamW optimizer - lower LR for finetuning
weight_decay = 0.01  # slight weight decay for finetuning
lr = 3e-5  # 10x lower than pretraining
beta1 = 0.9
beta2 = 0.999
eps = 1e-8
warmup_iters = int(0.1 * n_iters)  # shorter warmup
grad_clip = 1.0


###########################################
#                setup ddp                #
###########################################
ddp = int(os.environ.get("RANK", -1)) != -1

if ddp:
    dist.init_process_group(backend=backend)
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    device = f"cuda:{local_rank}"
    is_master = rank == 0
    seed = 42 + rank
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
else:
    rank = 0
    local_rank = 0
    world_size = 1
    is_master = True
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

enc = tiktoken.get_encoding("gpt2")


###########################################
#                dir for logs             #
###########################################
if is_master:
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "samples"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "logs"), exist_ok=True)

if ddp:
    dist.barrier()


#####################
#   data loading    #
#####################
train_loader, val_loader = get_dataloaders(
    train_dataset_name=train_dataset,
    train_split=train_split,
    train_subset=train_subset,
    valid_dataset_name=None,
    valid_split=None,
    batch_size=batch_size_per_gpu,
    seq_len=seq_len,
    cache_dir=cache_dir,
    num_proc=num_proc,
    distributed=ddp,
    num_workers=4
)

num_train_tokens = len(train_loader.dataset) * seq_len
tokens_per_iter = batch_size * seq_len
iters_per_epoch = num_train_tokens / tokens_per_iter
num_epochs = n_iters / iters_per_epoch

if is_master:
    print("=" * 60)
    print("FINETUNING DATA STATISTICS")
    print("=" * 60)
    print(f"Train dataset: {len(train_loader.dataset):,} sequences")
    print(f"Train tokens: {num_train_tokens:,}")
    print()
    print(f"Batch size per GPU: {batch_size_per_gpu}")
    print(f"Gradient accumulation steps: {accum}")
    print(f"Effective batch size: {batch_size_per_gpu * world_size * accum}")
    print(f"Tokens per iteration: {tokens_per_iter:,}")
    print(f"Iterations per epoch: {iters_per_epoch:,.1f}")
    print(f"Total iterations: {n_iters:,}")
    print(f"Total epochs: {num_epochs:.2f}")
    print()


#####################
#    build model    #
#####################
config = DLMConfig(
    hidden_size=512,
    cond_dim=128,
    length=1024,
    vocab=50304,
    n_blocks=8,
    n_heads=8,
    dropout=0.05  # lower dropout for finetuning
)

if is_master:
    print(config)

score_fn = SEDD(config).to(device=device, dtype=dtype)
graph = UniformGraph(config.vocab)
noise = GeometricNoise(sigma_min, sigma_max)

num_params = sum(p.numel() for p in score_fn.parameters())

if is_master:
    print_flops(config, batch_size, seq_len, num_params, n_iters, ngpus)


###########################################
#       load pretrained weights           #
###########################################
if is_master:
    print(f"Loading pretrained weights from {pretrained_ckpt}...")

checkpoint = torch.load(pretrained_ckpt, map_location=device, weights_only=False)
score_fn.load_state_dict(checkpoint['model'])

if is_master:
    print(f"Loaded pretrained model (trained for {checkpoint['step']} steps, loss {checkpoint['loss']:.4f})")
    print()

# compile after loading weights
if compile:
    if is_master:
        print("Compiling model with torch.compile...")
    score_fn = torch.compile(score_fn)

# wrap with DDP
if ddp:
    score_fn = DDP(score_fn, device_ids=[local_rank])

raw_model = score_fn.module if ddp else score_fn


###########################################
#            utils for saving             #
###########################################
def save_checkpoint(model, optimizer, scaler, step, loss, config, path):
    checkpoint = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scaler': scaler.state_dict(),
        'step': step,
        'loss': loss,
        'config': config,
        'pretrained_from': pretrained_ckpt,
    }
    tmp_path = path + ".tmp"
    torch.save(checkpoint, tmp_path)
    os.rename(tmp_path, path)


def load_checkpoint(path, model, optimizer, scaler):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    scaler.load_state_dict(checkpoint['scaler'])
    return checkpoint['step'], checkpoint['loss']


def find_latest_checkpoint(ckpt_dir):
    ckpts = glob.glob(os.path.join(ckpt_dir, "ckpt_*.pt"))
    if not ckpts:
        return None
    ckpts.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))
    return ckpts[-1]


###########################################
#         fresh optimizer for finetune    #
###########################################
scaler = torch.amp.GradScaler('cuda')
optimizer = optim.AdamW(
    raw_model.parameters(),
    lr=lr,
    betas=(beta1, beta2),
    eps=eps,
    weight_decay=weight_decay
)

if is_master:
    print(f"Optimizer: AdamW (lr={lr}, betas=({beta1}, {beta2}), wd={weight_decay})")
    print()


###########################################
#              lr schedule                #
###########################################
def get_lr(step):
    if step < warmup_iters:
        return lr * step / warmup_iters
    return lr


###########################################
#              compute loss               #
###########################################
def compute_loss(model, x0, noise, graph, sampling_eps):
    B = x0.shape[0]
    t = (1 - sampling_eps) * torch.rand(B, device=x0.device) + sampling_eps
    sigma, dsigma = noise.get_noise(t)
    xt = graph.sample_transition(x0, sigma[:, None])

    with torch.amp.autocast('cuda', dtype=dtype):
        log_score, _ = model(xt, sigma)

    loss = graph.score_entropy(log_score, sigma[:, None], xt, x0)
    loss = (dsigma[:, None] * loss).sum(dim=-1).mean()
    return loss


###########################################
#              mfu estimate               #
###########################################
def estimate_mfu(num_params, batch_size, seq_len, dt, ngpus):
    flops_per_token = 18 * num_params
    tokens_per_iter = batch_size * seq_len
    flops_per_iter = flops_per_token * tokens_per_iter
    peak_flops = 989e12 * ngpus
    achieved_flops = flops_per_iter / dt
    return achieved_flops / peak_flops


###########################################
#          infinite dataloader            #
###########################################
def infinite_dataloader(loader, sampler):
    epoch = 0
    while True:
        if sampler is not None:
            sampler.set_epoch(epoch)
        for batch in loader:
            yield batch
        epoch += 1


###############################################
#              finetuning loop                #
###############################################
step = 0

train_sampler = train_loader.sampler if ddp else None
train_iter = infinite_dataloader(train_loader, train_sampler)

if is_master:
    print("=" * 60)
    print("STARTING FINETUNING")
    print("=" * 60)

score_fn.train()
t0 = time.time()
log_loss = 0.0
log_steps = 0

while step < n_iters:
    current_lr = get_lr(step)
    for g in optimizer.param_groups:
        g['lr'] = current_lr

    optimizer.zero_grad(set_to_none=True)
    accum_loss = 0.0

    for micro_step in range(accum):
        batch = next(train_iter)
        x0 = batch["input_ids"].to(device)

        loss = compute_loss(score_fn, x0, noise, graph, sampling_eps)
        loss = loss / accum
        scaler.scale(loss).backward()
        accum_loss += loss.item() * accum

    scaler.unscale_(optimizer)
    grad_norm = torch.nn.utils.clip_grad_norm_(raw_model.parameters(), grad_clip)

    if torch.isnan(grad_norm) or torch.isinf(grad_norm):
        if is_master:
            print(f"step {step}: NaN grad_norm, skipping")
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        continue

    scaler.step(optimizer)
    scaler.update()

    step += 1
    log_loss += accum_loss
    log_steps += 1

    # logging
    if step % log_freq == 0 and is_master:
        t1 = time.time()
        dt = t1 - t0
        avg_loss = log_loss / log_steps
        tokens_per_sec = tokens_per_iter * log_steps / dt
        mfu = estimate_mfu(num_params, batch_size, seq_len, dt / log_steps, world_size)

        msg = f"step {step:06d} | loss {avg_loss:.4f} | lr {current_lr:.2e} | grad {grad_norm:.2f} | tok/s {tokens_per_sec:,.0f} | mfu {mfu*100:.1f}%"
        print(msg)
        with open(os.path.join(out_dir, "logs", "finetune_log.txt"), "a") as f:
            f.write(msg + "\n")

        t0 = time.time()
        log_loss = 0.0
        log_steps = 0

    # checkpointing
    if step % ckpt_freq == 0 and is_master:
        ckpt_path = os.path.join(out_dir, "checkpoints", f"ckpt_{step:07d}.pt")
        save_checkpoint(raw_model, optimizer, scaler, step, accum_loss, config, ckpt_path)
        print(f"Saved checkpoint to {ckpt_path}")

    # sampling
    if step % ckpt_freq == 0 and is_master:
        score_fn.eval()

        x = graph.sample_limit(sample_batch, sample_seq_len).to(device)
        timesteps = torch.linspace(1, sampling_eps, sampling_steps + 1, device=device)
        dt_sample = (1 - sampling_eps) / sampling_steps

        print("=" * 60)
        print(f"SAMPLING at step {step}")
        print("=" * 60)

        for i in range(sampling_steps):
            t = timesteps[i] * torch.ones(x.shape[0], device=device)
            sigma, dsigma = noise.get_noise(t)

            with torch.no_grad(), torch.amp.autocast('cuda', dtype=dtype):
                log_score_out, _ = raw_model(x, sigma)

            score = log_score_out.exp()
            rev_rate = dt_sample * dsigma[:, None, None] * graph.reverse_rate(x, score)
            x = graph.sample_rate(x, rev_rate)

            tokens = x[0].tolist()
            tokens = [min(t, enc.n_vocab - 1) for t in tokens]
            text = enc.decode(tokens)
            print(f"denoise step {i+1}/{sampling_steps} | sigma {sigma[0]:.4f}")
            print(text[:200])
            print("-" * 40)

        sample_path = os.path.join(out_dir, "samples", f"samples_{step:07d}.txt")
        with open(sample_path, "w") as f:
            for j in range(x.shape[0]):
                tokens = x[j].tolist()
                tokens = [min(t, enc.n_vocab - 1) for t in tokens]
                text = enc.decode(tokens)
                f.write(f"=== Sample {j+1} ===\n")
                f.write(text)
                f.write("\n\n")
        print(f"Saved samples to {sample_path}")
        print("=" * 60)

        score_fn.train()
        t0 = time.time()

# final save
if is_master:
    ckpt_path = os.path.join(out_dir, "checkpoints", f"ckpt_{step:07d}.pt")
    save_checkpoint(raw_model, optimizer, scaler, step, log_loss / max(log_steps, 1), config, ckpt_path)
    print(f"Finetuning complete. Final checkpoint: {ckpt_path}")

if ddp:
    dist.destroy_process_group()

# shutdown instance when done
if is_master:
    import subprocess
    print("Finetuning complete. Shutting down instance...")
    subprocess.run(["sudo", "shutdown", "-h", "now"])
