import os
import time
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from kv_model import DiT, DiTConfig
from kv_noise import LogLinearNoise, UniformGraph
from transformers import PreTrainedTokenizerFast

out_dir = "cpu_testing"
device = "cpu"

#################
#    config     #
#################
"""
memory per batch for 1024 seq length

hidden size: 512
n_blocks: 12
seq_len: 1024
attn weights: 1024**2 * n_heads
MLP: 1024 * 2048 * 4

total 1.7GB on training step
"""


batch_size = 16
seq_len = 32

# sampling
sampling_eps = 1e-5
noise_eps = 1e-3

# optimizer
lr = 3e-4
beta1 = 0.9
beta2 = 0.999
warmup_iters = 100
grad_clip = 1.0

# logging/checkpointing
eval_iters = 100
save_iters = 100


init_from = "scratch"  # "scratch" or "resume"
tokens_cache = "tokenizer/fineweb_1mb_tokens.pt"
text_data = "tokenizer/fineweb_1mb.txt"


#####################
#    data           #
#####################
print("Loading tokenizer...")
tokenizer = PreTrainedTokenizerFast(tokenizer_file="tokenizer/bpe-32k-uncased-bytelevel.json")

if os.path.exists(tokens_cache):
    print(f"Loading cached tokens from {tokens_cache}...")
    tokens = torch.load(tokens_cache, weights_only=True)
    print(f"Loaded {tokens.numel():,} tokens")
else:
    print(f"Loading {text_data}...")
    with open(f"{text_data}", "r", encoding="utf-8") as f:
        text = f.read()

    print(f"Text length: {len(text):,} chars")
    print("Tokenizing (this may take a moment)...")
    tokens = tokenizer.encode(text, add_special_tokens=False)
    print(f"Total tokens: {len(tokens):,}")

    tokens = torch.tensor(tokens, dtype=torch.long)
    print(f"Saving tokens to {tokens_cache}...")
    torch.save(tokens, tokens_cache)

tokens = tokens[:len(tokens) // seq_len * seq_len].view(-1, seq_len)
print(f"Sequences: {tokens.shape[0]:,}, Seq len: {seq_len}")

train_loader = DataLoader(tokens, batch_size=batch_size, shuffle=True)

def infinite_loader(loader):
    while True:
        for batch in loader:
            yield batch

train_iter = infinite_loader(train_loader)

#####################
#    dirs           #
#####################
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
    length=1024,
    vocab=32768,
    n_blocks=12,
    n_heads=8,
    dropout=0.1
)

print(f"\nConfig: {config}")
model = DiT(config).to(device)
graph = UniformGraph(config.vocab)
noise = LogLinearNoise(eps=noise_eps)

num_params = sum(p.numel() for p in model.parameters())
print(f"Model params: {num_params/1e6:.2f}M\n")

#####################
#    optimizer      #
#####################
optimizer = optim.AdamW(model.parameters(), lr=lr, betas=(beta1, beta2))

def get_lr(step):
    if step < warmup_iters:
        return lr * step / warmup_iters
    return lr

#####################
#    resume         #
#####################
start_step = 0
latest_ckpt = os.path.join(out_dir, "checkpoints", "latest.pt")
if init_from == "resume" and os.path.exists(latest_ckpt):
    print(f"Resuming from {latest_ckpt}...")
    ckpt = torch.load(latest_ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model'])
    optimizer.load_state_dict(ckpt['optimizer'])
    start_step = ckpt['step'] + 1
    print(f"Resumed at step {start_step}")
elif init_from == "scratch":
    print("Starting from scratch")

#####################
#    training       #
#####################
log_file = os.path.join(out_dir, "logs", "train.txt")
sample_file = os.path.join(out_dir, "samples", "samples.txt")

print(f"Logging to: {log_file}\n")
print("=" * 60)
print("TRAINING - DiT (global sigma, bidirectional)")
print(f"Batch size: {batch_size}, Seq len: {seq_len}")
print("=" * 60)

t0 = time.time()
running_loss = 0.0
step = start_step

while True:  # infinite training
    x0 = next(train_iter)
    B, S = x0.shape

    # LR warmup
    current_lr = get_lr(step)
    for g in optimizer.param_groups:
        g['lr'] = current_lr

    # Sample timestep t ~ [eps, 1]
    t = (1 - sampling_eps) * torch.rand(B, device=device) + sampling_eps
    sigma, dsigma = noise.get_noise(t)  # (B,)

    # Perturb: x0 -> xt
    xt = graph.sample_transition(x0, sigma[:, None])

    # Forward pass - global sigma (same for all positions)
    log_score = model(xt, sigma)

    # SEDD loss
    loss_per_token = graph.score_entropy(log_score, sigma[:, None], xt, x0)  # (B, S)
    loss = (dsigma[:, None] * loss_per_token).mean()

    # Backward
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()
    optimizer.zero_grad()

    #----------------------
    # Log
    #----------------------
    t1 = time.time()
    dt = t1 - t0
    t0 = t1

    msg = f"step {step:06d} | loss {loss.item():.4f} | norm {norm:.4f} | lr {current_lr:.6f} | {dt*1000:.0f}ms/step"
    print(msg)
    with open(log_file, "a") as f:
        f.write(msg + "\n")

    #----------------------
    # Sample & save checkpoint
    #----------------------
    if step % eval_iters == 0 and step > 0:
        print("=" * 60)
        print(f"SAMPLING at step {step}")
        print("=" * 60)

        model.eval()
        with torch.no_grad():
            with open(sample_file, "a") as f:
                f.write(f"\n=== Step {step} ===\n\n")

            # Standard sampling (shorter for CPU)
            for sample_len in [8, 16, 32]:
                num_steps = sample_len

                x = graph.sample_limit(1, sample_len).to(device)
                timesteps = torch.linspace(1, sampling_eps, num_steps + 1, device=device)
                dt_sample = (1 - sampling_eps) / num_steps

                for i in range(num_steps):
                    t_sample = timesteps[i]
                    sigma_sample, dsigma_sample = noise.get_noise(t_sample.expand(1))

                    log_score = model(x, sigma_sample)
                    score = log_score.exp()

                    rev_rate = dt_sample * dsigma_sample[:, None, None] * graph.reverse_rate(x, score)
                    x = graph.sample_rate(x, rev_rate)

                x = x.clamp(0, config.vocab - 1)
                text_out = tokenizer.decode(x[0].tolist())
                print(f"[{sample_len} tokens] {text_out}")
                print()

                with open(sample_file, "a") as f:
                    f.write(f"[{sample_len} tokens]\n{text_out}\n\n")

        # Save checkpoint
        torch.save({
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'step': step,
            'config': config,
        }, latest_ckpt)
        print(f"Saved: {latest_ckpt}")

        model.train()
        print("=" * 60)


    step += 1
