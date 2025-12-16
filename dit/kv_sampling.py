"""
Block-wise Diffusion Sampling with KV Caching

Generates text block-by-block:
1. Build KV cache from clean prefix
2. Denoise noisy block using cached K,V (fast)
3. Append block to prefix, rebuild cache, repeat
"""

import sys
import time
import torch
from kv_model import DiT, DiTConfig
from kv_noise import LogLinearNoise, UniformGraph
from transformers import PreTrainedTokenizerFast

device = "cuda" if torch.cuda.is_available() else "cpu"

# Model config (matches kv_gpu_train.py)
config = DiTConfig(
    hidden_size=512,
    cond_dim=128,
    length=128,
    vocab=32768,
    n_blocks=12,
    n_heads=8,
    dropout=0.0
)

# Sampling config
block_size = 128
num_steps = 128
sampling_eps = 1e-3
noise_eps = 1e-3
max_seq_len = 512
batch_size = 1

# AMP settings
use_amp = True
amp_dtype = torch.bfloat16

# Display
BLUE = '\033[38;5;153m'
RESET = '\033[0m'
GRAY = '\033[90m'

#################
#    setup      #
#################
print(f"Device: {device}")

tokenizer = PreTrainedTokenizerFast(tokenizer_file="tokenizer/bpe-32k-uncased-bytelevel.json")
model = DiT(config).to(device)
graph = UniformGraph(config.vocab)
noise = LogLinearNoise(eps=noise_eps)

# Load checkpoint
ckpt_path = "/home/jackson/kitchen/mlgrind/dit/gpu_run01/checkpoints/latest.pt"
torch.serialization.add_safe_globals([DiTConfig])
ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
model.load_state_dict(ckpt["model"])
print(f"Loaded checkpoint: {ckpt_path}")
loss = ckpt.get('loss', 'N/A')
loss_str = f"{loss:.4f}" if isinstance(loss, (int, float)) else loss
print(f"  Step: {ckpt.get('step', 'N/A')}, Loss: {loss_str}")

num_params = sum(p.numel() for p in model.parameters())
print(f"Model params: {num_params/1e6:.1f}M")
print(f"Block size: {block_size}, Steps: {num_steps}")
print()

model.eval()

#################
#    prompt     #
#################
prompt = "hello i am an ai model"
prompt_tokens = torch.tensor([tokenizer.encode(prompt, add_special_tokens=False)], device=device)
prompt_len = prompt_tokens.shape[1]
print(f"Prompt: '{prompt}' ({prompt_len} tokens)")
print()

print("=" * 60)
print("BLOCK-WISE GENERATION WITH KV CACHING")
print("=" * 60)

total_time = 0
block_idx = 0

with torch.no_grad():
    # Build initial cache from prompt
    with torch.amp.autocast('cuda', dtype=amp_dtype, enabled=use_amp):
        kv_cache = model.build_cache(prompt_tokens)
    all_tokens = prompt_tokens.clone()

    while all_tokens.shape[1] < max_seq_len:
        curr_block_size = min(block_size, max_seq_len - all_tokens.shape[1])
        if curr_block_size <= 0:
            break

        # Initialize noisy block
        x_block = graph.sample_limit(batch_size, curr_block_size).to(device)

        # Denoise using cached prefix
        timesteps = torch.linspace(1, sampling_eps, num_steps + 1, device=device)
        dt = (1 - sampling_eps) / num_steps

        block_start = time.time()
        for step in range(num_steps):
            t = timesteps[step]
            sigma, dsigma = noise.get_noise(t.expand(batch_size))

            with torch.amp.autocast('cuda', dtype=amp_dtype, enabled=use_amp):
                log_score = model(x_block, sigma, kv_cache=kv_cache)

            score = log_score.float().exp()
            rev_rate = dt * dsigma[:, None, None] * graph.reverse_rate(x_block, score)
            x_block = graph.sample_rate(x_block, rev_rate)
            x_block = x_block.clamp(0, config.vocab - 1)

            # Display
            sys.stdout.write('\033[2J\033[H')
            clean_text = tokenizer.decode(all_tokens[0].tolist())
            noisy_text = tokenizer.decode(x_block[0].tolist())
            sys.stdout.write(f"{clean_text}{BLUE}{noisy_text}{RESET}\n")
            sys.stdout.write(f"\n{GRAY}Block {block_idx} | Step {step+1}/{num_steps} | ")
            sys.stdout.write(f"Tokens {all_tokens.shape[1] + curr_block_size}/{max_seq_len}{RESET}\n")
            sys.stdout.flush()

        block_time = time.time() - block_start
        total_time += block_time

        # Extend cache with the new block (incremental)
        with torch.amp.autocast('cuda', dtype=amp_dtype, enabled=use_amp):
            kv_cache = model.build_cache(x_block, kv_cache=kv_cache)
        all_tokens = torch.cat([all_tokens, x_block], dim=1)
        block_idx += 1

# Final
sys.stdout.write('\033[2J\033[H')
print(tokenizer.decode(all_tokens[0].tolist()))
print()
print("=" * 60)
print(f"Generated {all_tokens.shape[1] - prompt_len} tokens in {block_idx} blocks")
print(f"Total time: {total_time:.2f}s ({total_time/block_idx:.2f}s/block)")
print("=" * 60)
