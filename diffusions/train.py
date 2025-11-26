"""
Training script for Discrete Diffusion LM (Q_uniform)

Features:
- DDP (Distributed Data Parallel) multi-GPU support
- Mixed precision training (bfloat16)
- Learning rate warmup + cosine decay
- Gradient clipping
- Periodic sampling and checkpointing

Usage:
    python train.py                              # Single GPU/CPU
    torchrun --nproc_per_node=8 train.py        # Multi-GPU
"""

import os
import time
import math
import torch
import torch.nn.functional as F
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import tiktoken

from model import DLMConfig, DiffusionLM, sigma_schedule, forward_noise, score_entropy_loss


# ============================================================================
# DDP Setup
# ============================================================================

ddp = int(os.environ.get('RANK', -1)) != -1
if ddp:
    assert torch.cuda.is_available(), "DDP requires CUDA"
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

device_type = "cuda" if device.startswith("cuda") else "cpu"

if master_process:
    print(f"Device: {device}")


# ============================================================================
# Data Loader
# ============================================================================

class DataLoaderLite:
    def __init__(self, B, T, process_rank, num_processes, data_path):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes

        # Load and tokenize text
        with open(data_path, 'r', encoding='utf-8') as f:
            text = f.read()

        enc = tiktoken.get_encoding("gpt2")
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens, dtype=torch.long)

        if master_process:
            print(f"Loaded {len(self.tokens):,} tokens")

        # Starting position for this process (DDP sharding)
        self.current_position = self.B * self.T * self.process_rank

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position + B*T]

        # Wrap around if not enough tokens
        if len(buf) < B*T:
            needed = B*T - len(buf)
            buf = torch.cat([buf, self.tokens[:needed]])
            self.current_position = needed

        x = buf.view(B, T)
        self.current_position += B * T * self.num_processes

        # Wrap around
        if self.current_position >= len(self.tokens):
            self.current_position = self.B * self.T * self.process_rank

        return x


# ============================================================================
# Quick Sampling for Evaluation
# ============================================================================

@torch.no_grad()
def quick_sample(model, prompt_text, tokenizer, config, device, max_tokens=256, block_size=64, num_steps=20):
    """Generate sample text during training (fast, low quality)"""
    model.eval()

    prompt_tokens = tokenizer.encode(prompt_text)
    clean_tokens = torch.tensor(prompt_tokens, dtype=torch.long, device=device)

    blocks_generated = 0
    max_blocks = max_tokens // block_size

    while blocks_generated < max_blocks and len(clean_tokens) + block_size <= config.seq_len:
        # Initialize noisy block
        noisy_block = torch.randint(0, config.vocab_size, (block_size,), device=device)

        # Denoise block
        for step in range(num_steps):
            t = 1.0 - step / num_steps
            sigma = sigma_schedule(torch.tensor([t], device=device))

            # Concatenate clean context with noisy block
            if len(clean_tokens) > 0:
                full_seq = torch.cat([clean_tokens, noisy_block])
                logits = model(full_seq.unsqueeze(0), sigma)
                # Take only logits for the noisy block positions
                logits = logits[:, -block_size:, :]
            else:
                logits = model(noisy_block.unsqueeze(0), sigma)

            probs = F.softmax(logits[0], dim=-1)
            noisy_block = torch.multinomial(probs, 1).squeeze(-1)

        # Add clean block to context
        clean_tokens = torch.cat([clean_tokens, noisy_block])
        blocks_generated += 1

        # Check for EOS
        if tokenizer.eot_token in noisy_block.tolist():
            break

    model.train()
    return tokenizer.decode(clean_tokens.tolist())


# ============================================================================
# Training
# ============================================================================

def train():
    # Hyperparameters
    batch_size = 4
    seq_len = 1024
    max_steps = 10000
    learning_rate = 6e-4
    warmup_steps = 100
    weight_decay = 0.1
    grad_clip = 1.0

    # Paths
    data_path = 'input.txt'
    save_dir = '.'

    # Seed
    torch.manual_seed(1337)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(1337)

    # Model
    config = DLMConfig(seq_len=seq_len)
    model = DiffusionLM(config).to(device)

    if master_process:
        num_params = sum(p.numel() for p in model.parameters())
        print(f"Model parameters: {num_params/1e6:.2f}M")

    # DDP wrap
    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank])
    raw_model = model.module if ddp else model

    # Data
    train_loader = DataLoaderLite(
        B=batch_size,
        T=seq_len,
        process_rank=ddp_rank,
        num_processes=ddp_world_size,
        data_path=data_path
    )

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        betas=(0.9, 0.95),
        weight_decay=weight_decay,
        fused=(device_type == 'cuda')
    )

    # LR schedule
    def get_lr(step):
        if step < warmup_steps:
            return learning_rate * (step + 1) / warmup_steps
        if step > max_steps:
            return learning_rate * 0.1
        decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return learning_rate * 0.1 + coeff * (learning_rate - learning_rate * 0.1)

    # Tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")

    # Training loop
    if master_process:
        print(f"\nTraining for {max_steps} steps...\n")
        log_file = open(os.path.join(save_dir, 'train_log.txt'), 'w')

    step_times = []

    for step in range(max_steps):
        t0 = time.time()

        # Update LR
        lr = get_lr(step)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # Get batch
        x_0 = train_loader.next_batch().to(device)

        # Sample timesteps and add noise
        t = torch.rand(batch_size, device=device)
        sigma = sigma_schedule(t)
        x_t = forward_noise(x_0, sigma, config.vocab_size)

        # Forward pass
        if device_type == 'cuda':
            with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                logits = model(x_t, sigma)
                loss = score_entropy_loss(logits, x_t, x_0, sigma, config.vocab_size)
        else:
            logits = model(x_t, sigma)
            loss = score_entropy_loss(logits, x_t, x_0, sigma, config.vocab_size)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        # Timing
        if device_type == 'cuda':
            torch.cuda.synchronize()
        dt = time.time() - t0
        tokens_per_sec = batch_size * seq_len * ddp_world_size / dt

        # Track step times for ETA (skip first step)
        if step > 0:
            step_times.append(dt)
            if len(step_times) > 10:
                step_times.pop(0)

        # Calculate ETA
        if step > 0 and master_process:
            avg_time = sum(step_times) / len(step_times)
            remaining = max_steps - (step + 1)
            eta_sec = remaining * avg_time
            eta_min = eta_sec / 60
            eta_hr = eta_min / 60
            
            if eta_hr >= 1:
                eta_str = f"{eta_hr:.1f}h"
            elif eta_min >= 1:
                eta_str = f"{eta_min:.0f}m"
            else:
                eta_str = f"{eta_sec:.0f}s"
        else:
            eta_str = "..."

        # Logging
        if master_process:
            log_msg = f"Step {step+1:5d}/{max_steps} | Loss: {loss.item():.4f} | LR: {lr:.6f} | {dt*1000:.1f}ms | {tokens_per_sec:.0f} tok/s | ETA: {eta_str}"
            print(log_msg)
            log_file.write(log_msg + '\n')
            log_file.flush()

        # Sample every 100 steps
        if master_process and (step + 1) % 100 == 0:
            print("\n" + "="*80)
            sample = quick_sample(
                model=raw_model,
                prompt_text="Once upon a time",
                tokenizer=tokenizer,
                config=config,
                device=device,
                max_tokens=256,
                block_size=64,
                num_steps=20
            )
            print(f"Sample: {sample[:200]}...")
            log_file.write(f"\nSample at step {step+1}:\n{sample}\n\n")
            log_file.flush()
            print("="*80 + "\n")

        # Save checkpoint every 100 steps
        if master_process and (step + 1) % 100 == 0:
            checkpoint = {
                'step': step + 1,
                'model': raw_model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'config': config,
            }
            save_path = os.path.join(save_dir, f'model_step{step+1}.pt')
            torch.save(checkpoint, save_path)
            print(f"Saved: {save_path}\n")

    # Final save
    if master_process:
        torch.save({
            'step': max_steps,
            'model': raw_model.state_dict(),
            'config': config,
        }, os.path.join(save_dir, 'model.pt'))
        log_file.close()
        print("\nTraining complete!")

    if ddp:
        destroy_process_group()


if __name__ == "__main__":
    train()
