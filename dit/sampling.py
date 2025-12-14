"""
Block-wise Diffusion Sampling with Pre-Rotary QKV Caching

Generates text block-by-block:
1. Denoise a noisy block using cached context
2. After block done, append its QKV to cache
3. Repeat for next block
"""

import sys
import time
import random
import torch
from kv_model import DiT, DiTConfig
from kv_noise import LogLinearNoise, UniformGraph
from transformers import PreTrainedTokenizerFast

device = "cuda" if torch.cuda.is_available() else "cpu"

# Model config - should match training
config = DiTConfig(
    hidden_size=512,
    cond_dim=128,
    length=1024,
    vocab=32768,
    n_blocks=12,
    n_heads=8,
    dropout=0.1
)

# Sampling config
block_size = 64
num_steps = 64
sampling_eps = 1e-5
noise_eps = 1e-3
max_seq_len = 1024
batch_size = 1

# Display colors
BLUE = '\033[38;5;153m'
RESET = '\033[0m'
GRAY = '\033[90m'


def main():
    print(f"Device: {device}")

    # Load model
    tokenizer = PreTrainedTokenizerFast(tokenizer_file="tokenizer/bpe-32k-uncased.json")
    model = DiT(config).to(device)
    graph = UniformGraph(config.vocab)
    noise = LogLinearNoise(eps=noise_eps)

    # Try to load checkpoint
    ckpt_path = "outputs_dit/checkpoints/latest.pt"
    try:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model'])
        print(f"Loaded checkpoint: {ckpt_path} (step {ckpt['step']})")
    except Exception as e:
        print(f"No checkpoint found at {ckpt_path}, using random weights")
        print(f"  Error: {e}")

    model.eval()
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {num_params/1e6:.1f}M")
    print(f"Block size: {block_size}, Steps: {num_steps}")
    print()

    # === PROMPT ===
    prompt = "the meaning of life is"
    prompt_tokens = torch.tensor([tokenizer.encode(prompt, add_special_tokens=False)], device=device)
    prompt_len = prompt_tokens.shape[1]

    # Cache prompt text for display
    cached_prompt_text = tokenizer.decode(prompt_tokens[0].tolist())

    print(f"Prompt: '{prompt}' ({prompt_len} tokens)")
    print()

    # Clear cache and initialize with prompt
    model.clear_cache()

    # Process prompt through model to build initial cache
    # Use sigma=0 since prompt tokens are "clean" (not noised)
    with torch.no_grad():
        sigma_zero = torch.zeros(batch_size, device=device)
        _ = model.forward_with_cache(prompt_tokens, sigma_zero)
        model.append_to_cache()

    all_tokens = prompt_tokens.clone()

    print("=" * 60)
    print("BLOCK-WISE GENERATION WITH QKV CACHING")
    print("=" * 60)

    total_time = 0
    block_idx = 0

    with torch.no_grad():
        while all_tokens.shape[1] < max_seq_len:
            # Clip block size to remaining tokens
            tokens_remaining = max_seq_len - all_tokens.shape[1]
            current_block_size = min(block_size, tokens_remaining)

            if current_block_size <= 0:
                break

            # Initialize noisy block
            x_block = graph.sample_limit(batch_size, current_block_size).to(device)

            # Timesteps for denoising
            timesteps = torch.linspace(1, sampling_eps, num_steps + 1, device=device)
            dt = (1 - sampling_eps) / num_steps

            block_start = time.time()

            for step in range(num_steps):
                t = timesteps[step]
                sigma, dsigma = noise.get_noise(t.expand(batch_size))

                # Forward with cache (Q on current block, K/V on cache + current)
                log_score = model.forward_with_cache(x_block, sigma)

                # Sample
                score = log_score.exp()
                rev_rate = dt * dsigma[:, None, None] * graph.reverse_rate(x_block, score)
                x_block = graph.sample_rate(x_block, rev_rate)
                x_block = x_block.clamp(0, config.vocab - 1)

                # === Display effect ===
                sys.stdout.write('\033[2J\033[H')  # Clear screen + home

                # Decode clean tokens (cached) and noisy block separately
                clean_text = tokenizer.decode(all_tokens[0].tolist())
                noisy_text = tokenizer.decode(x_block[0].tolist())

                # Display: clean (white) + denoising (blue)
                sys.stdout.write(f"{clean_text}{BLUE}{noisy_text}{RESET}\n")
                total_tokens = all_tokens.shape[1] + x_block.shape[1]
                sys.stdout.write(f"\n{GRAY}Block {block_idx} | Step {step+1}/{num_steps} | ")
                sys.stdout.write(f"Tokens {total_tokens}/{max_seq_len} | ")
                sys.stdout.write(f"Cache {model.get_cache_len()}{RESET}\n")
                sys.stdout.flush()

            block_time = time.time() - block_start
            total_time += block_time

            # Block done - recompute KV with sigma=0 and cache it
            # This is necessary because during denoising we used sigma > 0,
            # but for caching we need K,V computed with sigma=0 (clean tokens)
            sigma_zero = torch.zeros(batch_size, device=device)
            _ = model.forward_with_cache(x_block, sigma_zero)
            model.append_to_cache()

            # Append to all tokens
            all_tokens = torch.cat([all_tokens, x_block], dim=1)

            block_idx += 1

    # Final display
    sys.stdout.write('\033[2J\033[H')
    final_text = tokenizer.decode(all_tokens[0].tolist())
    print(final_text)
    print()
    print("=" * 60)
    print(f"Generated {all_tokens.shape[1] - prompt_len} tokens in {block_idx} blocks")
    print(f"Total time: {total_time:.2f}s ({total_time/block_idx:.2f}s/block)")
    print(f"Final cache length: {model.get_cache_len()}")
    print("=" * 60)


def benchmark():
    """Compare cached vs non-cached generation."""
    print(f"\nDevice: {device}")
    print("=" * 60)
    print("BENCHMARK: CACHED vs NON-CACHED")
    print("=" * 60)

    tokenizer = PreTrainedTokenizerFast(tokenizer_file="tokenizer/bpe-32k-uncased.json")
    model = DiT(config).to(device)
    graph = UniformGraph(config.vocab)
    noise = LogLinearNoise(eps=noise_eps)
    model.eval()

    # Try to load checkpoint
    ckpt_path = "outputs_dit/checkpoints/latest.pt"
    try:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model'])
        print(f"Loaded checkpoint: {ckpt_path}")
    except:
        print("Using random weights")

    test_len = 64
    test_steps = 32
    prefix_len = 64

    print(f"\nTest: {prefix_len} prefix + {test_len} new tokens, {test_steps} steps")

    with torch.no_grad():
        # === NON-CACHED: full forward every step ===
        x_full = graph.sample_limit(batch_size, prefix_len + test_len).to(device)
        timesteps = torch.linspace(1, sampling_eps, test_steps + 1, device=device)
        dt = (1 - sampling_eps) / test_steps

        # Warmup
        for _ in range(2):
            sigma, _ = noise.get_noise(torch.tensor([0.5], device=device))
            _ = model(x_full, sigma)

        if device == "cuda":
            torch.cuda.synchronize()

        t0 = time.time()
        for step in range(test_steps):
            t = timesteps[step]
            sigma, dsigma = noise.get_noise(t.expand(batch_size))
            log_score = model(x_full, sigma)  # Full forward
            score = log_score.exp()
            rev_rate = dt * dsigma[:, None, None] * graph.reverse_rate(x_full, score)
            x_full = graph.sample_rate(x_full, rev_rate)

        if device == "cuda":
            torch.cuda.synchronize()
        non_cached_time = time.time() - t0

        # === CACHED: only denoise new block ===
        model.clear_cache()

        # Cache prefix with sigma=0 (clean tokens)
        x_prefix = graph.sample_limit(batch_size, prefix_len).to(device)
        sigma_zero = torch.zeros(batch_size, device=device)
        _ = model.forward_with_cache(x_prefix, sigma_zero)
        model.append_to_cache()

        # Denoise new block with cache
        x_block = graph.sample_limit(batch_size, test_len).to(device)
        timesteps = torch.linspace(1, sampling_eps, test_steps + 1, device=device)

        # Warmup
        for _ in range(2):
            sigma, _ = noise.get_noise(torch.tensor([0.5], device=device))
            _ = model.forward_with_cache(x_block, sigma)

        if device == "cuda":
            torch.cuda.synchronize()

        t0 = time.time()
        for step in range(test_steps):
            t = timesteps[step]
            sigma, dsigma = noise.get_noise(t.expand(batch_size))
            log_score = model.forward_with_cache(x_block, sigma)
            score = log_score.exp()
            rev_rate = dt * dsigma[:, None, None] * graph.reverse_rate(x_block, score)
            x_block = graph.sample_rate(x_block, rev_rate)

        if device == "cuda":
            torch.cuda.synchronize()
        cached_time = time.time() - t0

    print(f"\nNon-cached ({prefix_len + test_len} tokens): {non_cached_time*1000:.1f}ms ({non_cached_time/test_steps*1000:.2f}ms/step)")
    print(f"Cached ({prefix_len} cached + {test_len} new): {cached_time*1000:.1f}ms ({cached_time/test_steps*1000:.2f}ms/step)")
    print(f"Speedup: {non_cached_time/cached_time:.2f}x")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", action="store_true", help="Run benchmark comparison")
    args = parser.parse_args()

    if args.benchmark:
        benchmark()
    else:
        main()
