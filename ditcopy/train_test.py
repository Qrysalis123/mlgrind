import tiktoken
import torch
from model import DLM, DiTConfig
import noise
import time

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")


# -----------------------
# MINIMAL CONFIG - Start small!
# -----------------------
T = 32  # sequence length
VOCAB_SIZE = 50304  # Keep full GPT-2 vocab

config = DiTConfig(
    seq_len=T,
    vocab_size=VOCAB_SIZE,
    n_layers=4,
    n_heads=4,
    n_embd=256,
    c_dim=64,
    dropout=0.0  # No dropout for overfitting
)

# -----------------------
# HYPERPARAMETERS
# -----------------------
learning_rate = 1e-4
batch_size = 1  # SINGLE BATCH - overfit on this one sequence
max_iters = 5000
log_interval = 1 # do not change this. log everystep
eval_interval = 50 # do not change this. eval every 50th

# -----------------------
# DATA - SINGLE FIXED BATCH
# -----------------------
enc = tiktoken.get_encoding('gpt2')
with open('shakespeare.txt', 'r') as f:
    text = f.read()[:1000]
tokens = enc.encode(text)
tokens = torch.tensor(tokens)
print(f"Loaded {len(tokens)} tokens")

# SINGLE FIXED BATCH - overfit on this
fixed_batch = tokens[:T].unsqueeze(0).to(device)  # (1, T)
print(f"Fixed batch shape: {fixed_batch.shape}")
print(f"Target text to overfit on:")
print(f"  {enc.decode(fixed_batch[0].tolist())}")
print()

# -----------------------
# MODEL
# -----------------------
model = DLM(config).to(device)
num_params = sum(p.numel() for p in model.parameters())
print(f'Model params: {num_params / 1e6:.3f}M')

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.0)

# Learning rate scheduler - warmup for first 5% of steps
warmup_steps = int(0.05 * max_iters)

def get_lr(step):
    if step < warmup_steps:
        return learning_rate * (step + 1) / warmup_steps
    return learning_rate

# -----------------------
# TRAINING LOOP
# -----------------------
model.train()
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

print(f"\nTraining for {max_iters} steps on fixed batch...")
print("=" * 80)

for step in range(max_iters):
    t0 = time.time()

    # Update learning rate
    current_lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = current_lr

    # Sample random timestep
    t = torch.rand(batch_size, device=device) * 0.999 + 0.001
    sigma, dsigma = noise.geometric_noise_schedule(t, sigma_min=1e-3, sigma_max=6.907)

    # Add noise
    x_t = noise.sample_transition(fixed_batch, sigma.unsqueeze(-1), VOCAB_SIZE)

    # Forward pass
    optimizer.zero_grad()
    log_score, _ = model(x_t, sigma)

    # Compute loss
    loss_per_token = noise.score_entropy(log_score, sigma, x_t, fixed_batch, VOCAB_SIZE)
    loss = (dsigma.unsqueeze(-1) * loss_per_token).mean()

    # Backward pass
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    # Logging
    if step % log_interval == 0:
        t1 = time.time()
        score_mean = log_score.mean().item()
        score_std = log_score.std().item()
        print(f"step {step:4d} | loss: {loss.item():.4f} | lr: {current_lr:.4f} | norm: {norm.item():.2f} | dt: {(t1-t0)*1000:.0f}ms")

    # Sampling
    if step % eval_interval == 0 or step == max_iters - 1:
        model.eval()
        print("\n" + "="*80)
        print("SAMPLING")
        print("="*80)

        with torch.no_grad():
            # Start from noise
            x_t = torch.randint(0, VOCAB_SIZE, (1, T), device=device)

            # Reverse diffusion
            steps = 32
            for i in range(steps):
                t_val = 1.0 - (i / steps)
                t_tensor = torch.tensor([t_val], device=device)
                curr_sigma, _ = noise.geometric_noise_schedule(t_tensor)

                t_next = max(1.0 - ((i + 1) / steps), 0.001)
                t_next_tensor = torch.tensor([t_next], device=device)
                next_sigma, _ = noise.geometric_noise_schedule(t_next_tensor)
                dsigma = curr_sigma - next_sigma

                # Get model predictions
                log_score, _ = model(x_t, curr_sigma)
                score = log_score.exp()

                # Apply staggered score and transition
                stag_score = noise.staggered_score(score, dsigma, VOCAB_SIZE)
                trans = noise.transp_transition(x_t, dsigma, VOCAB_SIZE)
                probs = stag_score * trans

                # Sample
                x_t = noise.sample_categorical(probs)

            sample_text = enc.decode(x_t[0].tolist())
            target_text = enc.decode(fixed_batch[0].tolist())
            matches = (x_t[0] == fixed_batch[0]).sum().item()
            print(f"\nSample:\n'{sample_text}'")
            print("="*80 + "\n")
            print(f"\nTarget:\n'{target_text}'")
            print(f"\nMatch:  {matches}/{T} tokens ({100*matches/T:.1f}%)")
            print("="*80 + "\n")

        model.train()

print("\nTraining complete!")
