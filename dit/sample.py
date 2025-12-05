import torch
import torch.nn as nn
from multi_gpu_model import SEDD, DLMConfig, UniformGraph, LogLinearNoise
import tiktoken
import sys
import argparse

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Parse command line arguments
parser = argparse.ArgumentParser(description='Sample from discrete diffusion model with KV caching')
parser.add_argument('--prompt', type=str, default="hello i am an ai model i can", help='Prompt text to start generation')
args = parser.parse_args()

# Configuration
config = DLMConfig(
    hidden_size=512,
    cond_dim=128,
    length=1024,
    vocab=50304,
    n_blocks=8,
    n_heads=8,
    dropout=0.1
)

batch_size = 1
sampling_eps = 1e-5
max_seq_len = config.length

# Initialize model
model = SEDD(config).to(device)

# Initialize random weights for testing (since we don't have trained weights yet)
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean=0.0, std=0.02)
        if m.bias is not None:
            nn.init.normal_(m.bias, mean=0.0, std=0.02)
    elif hasattr(m, 'embedding') and isinstance(m.embedding, nn.Parameter):
        nn.init.normal_(m.embedding, mean=0.0, std=0.02)
    elif hasattr(m, 'weight') and isinstance(m.weight, nn.Parameter) and m.weight.dim() == 1:
        # LayerNorm weight - keep at 1.0 for stability
        nn.init.ones_(m.weight)

model.apply(init_weights)
model.eval()  # Set to eval mode for inference

# Initialize graph and noise schedule
graph = UniformGraph(config.vocab)
noise = LogLinearNoise(eps=1e-3)

# ------------------------------
# Main Sampling with KV Cache
# ------------------------------

# Setup tokenizer and prefix
eos_token_id = 9999999
enc = tiktoken.get_encoding('gpt2')

prefix = args.prompt

# Prepare input conditioning
prompt_tokens = torch.tensor(enc.encode(prefix), device=device)
# Repeat for batch: (T,) -> (B, T)
prompt_tokens = prompt_tokens.unsqueeze(0).repeat(batch_size, 1)
prompt_len = prompt_tokens.size(1)
tokens_remaining = max_seq_len - prompt_len

# ------------------------------
# Adaptive step size & block size
# https://arxiv.org/pdf/2311.14768
# ------------------------------
steps = 2 ** torch.randint(2, 3, (1,)).item()
block_size = 2 ** torch.randint(3, 9, (1,)).item()
block_size = min(block_size, tokens_remaining)

# Initialize: prompt + noise block
noise_block = torch.randint(0, config.vocab, (batch_size, block_size), device=device)
x = torch.cat([prompt_tokens, noise_block], dim=1)

BLUE = '\033[38;5;153m'  # Faint blue for denoising

# Cache prompt text (only decode once)
cached_prompt_text = enc.decode(prompt_tokens[0].tolist())
cached_prompt_len = len(cached_prompt_text)

with torch.no_grad():
    while tokens_remaining > 0:
        # Timesteps
        timesteps = torch.linspace(1, sampling_eps, steps + 1, device=device)
        dt = (1 - sampling_eps) / steps

        # Denoise the current block
        prefix_len = x.size(1) - block_size

        # Build prefix cache: cache all clean tokens (everything except the noisy block)
        if prefix_len > 0:
            prefix_tokens = x[:, :prefix_len]
            # Dummy sigma for prefix caching (doesn't matter, just need KV)
            t_dummy = torch.ones(batch_size, 1, device=device) * 0.5
            sigma_dummy, _ = noise.get_noise(t_dummy)
            _, prefix_kv_cache = model(prefix_tokens, sigma_dummy.reshape(-1), kv_cache=None, prefix_len=0)
        else:
            prefix_kv_cache = None

        for step in range(steps):
            t = timesteps[step] * torch.ones(x.shape[0], 1, device=device)  # (B, 1)
            sigma, dsigma = noise.get_noise(t)  # (B, 1), (B, 1)

            # Always process only the noisy block, using cached prefix
            x_block = x[:, -block_size:]

            if prefix_kv_cache is not None:
                # Use cached prefix KV
                log_score, _ = model(x_block, sigma.reshape(-1), kv_cache=prefix_kv_cache, prefix_len=prefix_len)
            else:
                # No prefix, just denoise the block
                log_score, _ = model(x_block, sigma.reshape(-1), kv_cache=None, prefix_len=0)

            score = log_score.exp()

            # Reverse diffusion step on the block
            rev_rate = dt * dsigma[..., None] * graph.reverse_rate(x_block, score)
            x_block_new = graph.sample_rate(x_block, rev_rate)

            # Update x with new block
            if prefix_len > 0:
                x = torch.cat([x[:, :prefix_len], x_block_new], dim=1)
            else:
                x = x_block_new

            """ ------ display effect ------"""
            # Decode with rainbow colors for denoising!
            sys.stdout.write('\033[2J\033[H')  # Clear screen + home

            # Decode and color new tokens (clip to tokenizer vocab range)
            clipped_tokens = torch.clamp(x[0], 0, enc.n_vocab - 1)
            full_text = enc.decode(clipped_tokens.tolist())
            new_tokens = full_text[cached_prompt_len:]
            colored_new = f"{BLUE}{new_tokens}\033[0m"

            tokens_used = x.size(1)
            tokens_remaining_display = max_seq_len - tokens_used

            # Mode indicator based on steps vs block size
            mode = "Thinking..." if steps > block_size else "Yapping..."
            sys.stdout.write(f"{cached_prompt_text}{colored_new}\n")
            sys.stdout.write(f"\n\033[90m{mode}")
            sys.stdout.write(f"\n\033[90mStep {step+1}/{steps} | Block {block_size} | Tokens {tokens_used}/{max_seq_len} | Cached {prefix_len} | Denoising {block_size}\033[0m\n")
            sys.stdout.flush()

        sys.stdout.write('\033[2J\033[H')  # Clear screen + home
        sys.stdout.write(f"{full_text}\n")
        sys.stdout.write(f"\n\033[90m{mode}")
        sys.stdout.write(f"\n\033[90mStep {step+1}/{steps} | Block {block_size} | Tokens {tokens_used}/{max_seq_len} | Cached {prefix_len} | Denoising {block_size}\033[0m\n")
        sys.stdout.flush()

        # Check for EOS
        if (x[:, -block_size:] == eos_token_id).any():
            break

        # Next iteration - the denoised block becomes part of the prefix
        prompt_tokens = x.clone()

        # Update cached prompt text (clip to tokenizer vocab range)
        clipped_tokens = torch.clamp(prompt_tokens[0], 0, enc.n_vocab - 1)
        cached_prompt_text = enc.decode(clipped_tokens.tolist())
        cached_prompt_len = len(cached_prompt_text)

        # Adaptive select steps, block size
        steps = 2 ** torch.randint(3, 7, (1,)).item()
        block_size = 2 ** torch.randint(3, 9, (1,)).item()

        # Clip block_size to tokens remaining
        tokens_remaining = max_seq_len - prompt_tokens.size(1)
        block_size = min(block_size, tokens_remaining)

        # Generate new x with noise block appended
        noise_block = torch.randint(0, config.vocab, (batch_size, block_size), device=device)
        x = torch.cat([prompt_tokens, noise_block], dim=1)

print()  # Final newline when done
