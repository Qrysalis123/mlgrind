import os
import time
import math
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from kv_model import DiT, DiTConfig
from kv_noise import LogLinearNoise, UniformGraph
from transformers import PreTrainedTokenizerFast

out_dir = "poem_overfit"
device = "cpu"

#################
#    config     #
#################

batch_size = 16
seq_len = 64

# Block-wise training config
# split_idx ~ Uniform[0, max_prefix] - includes 0 for "no prefix" case
max_prefix_ratio = 0.9  # Maximum prefix length as ratio of seq_len
min_prefix = 4
max_prefix = 32

# sampling
sampling_eps = 1e-5
noise_eps = 1e-3

# optimizer
lr = 1e-3
min_lr = 1e-4  # for cosine decay
beta1 = 0.9
beta2 = 0.999
weight_decay = 0.0  # no weight decay for overfitting
warmup_iters = 100
max_iters = 10000  # for cosine decay
grad_clip = 1.0

# logging/checkpointing
eval_iters = 50
save_iters = 50


init_from = "resume"  # "scratch" or "resume"
text_data = "shakespeare.txt"


#####################
#    data           #
#####################
print("Loading tokenizer...")
tokenizer = PreTrainedTokenizerFast(tokenizer_file="tokenizer/bpe-32k-uncased-bytelevel.json")

print(f"Loading {text_data}...")
with open(text_data, "r", encoding="utf-8") as f:
    text = f.read()

print(f"Text length: {len(text):,} chars")
print("Tokenizing...")
tokens = tokenizer.encode(text, add_special_tokens=False)
print(f"Total tokens: {len(tokens):,}")

tokens = torch.tensor(tokens, dtype=torch.long)

# For small text, repeat to get more sequences
num_seqs = len(tokens) // seq_len
if num_seqs < batch_size * 4:
    # Repeat tokens to have enough data
    repeats = (batch_size * 8 * seq_len) // len(tokens) + 1
    tokens = tokens.repeat(repeats)
    print(f"Repeated {repeats}x for overfitting")

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
    length=128,
    vocab=32768,
    n_blocks=16,
    n_heads=8,
    dropout=0.1  # no dropout for overfitting
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
optimizer = optim.AdamW(model.parameters(), lr=lr, betas=(beta1, beta2), weight_decay=weight_decay)

def get_lr(step):
    # Linear warmup
    if step < warmup_iters:
        return lr * step / warmup_iters
    # Cosine decay to min_lr
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
print("TRAINING - DiT (block-wise KV cache)")
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

    # Block-wise training: random split into clean prefix + noisy block
    # split_idx=0 means entire sequence is noisy (standard diffusion)
    split_idx = torch.randint(min_prefix, max_prefix + 1, (1,)).item()

    x_clean = x0[:, :split_idx]  # Clean prefix (ground truth)
    x0_noisy_block = x0[:, split_idx:]  # Ground truth for noisy block

    # Perturb only the noisy block
    x_noisy = graph.sample_transition(x0_noisy_block, sigma[:, None])

    # Forward pass
    log_score = model(x_noisy, sigma, x_clean=x_clean)

    # SEDD loss (only on noisy block)
    loss_per_token = graph.score_entropy(log_score, sigma[:, None], x_noisy, x0_noisy_block)
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

    msg = f"step {step:06d} | loss {loss.item():.4f} | norm {norm:.4f} | lr {current_lr:.6f} | split {split_idx} | {dt*1000:.0f}ms/step"
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


            for b in [8, 16, 32, 64]:
                # Block-wise sampling - use poem prefix
                prefix = "romeo"
                prefix_tok = torch.tensor([tokenizer.encode(prefix)])

                kv_cache = model.build_cache(prefix_tok)
                num_steps = b  # denoising steps (independent of block size)

                # initialize noisy block
                x_block = graph.sample_limit(1, b).to(device)
                timesteps = torch.linspace(1, sampling_eps, num_steps + 1, device=device)
                dt_sample = (1 - sampling_eps) / num_steps

                for i in range(num_steps):
                    t_sample = timesteps[i]
                    sigma_sample, dsigma_sample = noise.get_noise(t_sample.expand(1))

                    log_score = model(x_block, sigma_sample, kv_cache=kv_cache)
                    score = log_score.exp()
                    rev_rate = dt_sample * dsigma_sample[:, None, None] * graph.reverse_rate(x_block, score)
                    x_block = graph.sample_rate(x_block, rev_rate)
                    x_block = x_block.clamp(0, config.vocab-1)

                text_response = torch.cat([prefix_tok, x_block], dim=1)
                text_response = tokenizer.decode(text_response[0].tolist())
                kv_cache = None
                print(f"block & step: {b} \n{text_response}")
                print()

                with open(sample_file, "a") as f:
                    f.write(f"[{b} tokens, block={b}]\n{text_response}\n\n")

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
