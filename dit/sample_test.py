import torch
from model import DLM, DiTConfig
from transformers import GPT2TokenizerFast
import sys

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Configuration
config = DiTConfig()
batch_size = 1
eps = 1e-5
max_seq_len = config.seq_len


# Initialize model with torch.compile for faster inference
model = DLM(config, compile_model=False).to(device)  # Disable compile with kv cache

# For testing: wrap model to output random scores (replace with trained model)
class RandomScoreWrapper:
    """Wrapper for testing - outputs random scores. Replace with trained model."""
    def __init__(self, model):
        self.model = model

    def __call__(self, x, sigma, kv_cache=None, prefix_len=0):
        B, S = x.shape
        if kv_cache is not None and prefix_len > 0:
            # Only return logits for new tokens
            return torch.randn(B, S - prefix_len, config.vocab_size, device=device), None
        return torch.randn(B, S, config.vocab_size, device=device), None

score_fn = RandomScoreWrapper(model)

# ------------------------------
# Sampling Utilities
# ------------------------------

def geometric_noise_schedule(t, sigma_min=1e-3, sigma_max=1.0):
    """Geometric noise schedule: σ(t) = sigma_min^(1−t) * sigma_max^t"""
    return sigma_min ** (1 - t) * sigma_max ** t


def staggered_score(score, dsigma):
    """
    Compute staggered score for reverse-step scaling.
    Args:
        score: (B, S, V) model output scores
        dsigma: (B, 1) noise difference
    Returns:
        (B, S, V) staggered scores
    """
    V = score.shape[-1]
    epow = (-dsigma).exp()[..., None]  # (B, 1, 1)
    return ((epow - 1) / (V * epow)) * score.sum(dim=-1, keepdim=True) + score / epow


def transp_transition(x_t, sigma, vocab_size):
    """
    Compute transpose transition probabilities (uniform graph).

    Args:
        x_t: (B, S) current token indices
        sigma: (B, 1) noise level
        vocab_size: vocabulary size
    Returns:
        (B, S, V) transition probabilities
    """
    B, S = x_t.shape

    # Initialize with off-diagonal probability: (1 - exp(-sigma)) / V
    trans = torch.ones(B, S, vocab_size, device=x_t.device, dtype=torch.float32)
    trans = trans * (1 - (-sigma[..., None]).exp()) / vocab_size

    # Zero out diagonal
    trans.scatter_(-1, x_t.unsqueeze(-1), 0.0)

    # Set diagonal to remaining probability (ensures rows sum to 1)
    trans.scatter_(-1, x_t.unsqueeze(-1), 1 - trans.sum(dim=-1, keepdim=True))

    return trans


def sample_categorical(probs):
    """
    Sample from categorical distribution using Gumbel-max trick.

    Args:
        probs: (B, S, V) probability distribution
    Returns:
        (B, S) sampled token indices
    """
    gumbel_noise = 1e-10 - (torch.rand_like(probs) + 1e-10).log()
    return (probs / gumbel_noise).argmax(dim=-1)

# ------------------------------
# Main Sampling with KV Cache
# ------------------------------

# Setup tokenizer and prefix
eos_token_id = 9999999
tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
prefix = "helllo i am an ai model i can "

# Prepare input conditioning
prompt_tokens = torch.tensor(tokenizer(prefix).input_ids, device=device)
# Repeat for batch: (T,) -> (B, T)
prompt_tokens = prompt_tokens.unsqueeze(0).repeat(batch_size, 1)
prompt_len = prompt_tokens.size(1)
tokens_remaining = max_seq_len - prompt_len

# ------------------------------
# Adaptive step size & block size
# https://arxiv.org/pdf/2311.14768
# ------------------------------
steps = 2 ** torch.randint(2, 7, (1,)).item()
block_size = 2 ** torch.randint(3, 7, (1,)).item()
block_size = min(block_size, tokens_remaining)

# Initialize: prompt + noise block
noise_block = torch.randint(0, config.vocab_size, (batch_size, block_size), device=device)
x_t = torch.cat([prompt_tokens, noise_block], dim=1)

CREAM = '\033[38;5;229m'  # Faint gold cream for denoising

# Cache prompt text (only decode once)
cached_prompt_text = tokenizer.decode(prompt_tokens[0])
cached_prompt_len = len(cached_prompt_text)

with torch.no_grad():
    iteration = 0
    while tokens_remaining > 0:
        iteration += 1

        # Timesteps
        timesteps = torch.linspace(1, eps, steps + 1, device=device)
        dt = (1 - eps) / steps

        # KV cache for this denoising sequence
        # We'll build it on first step, then reuse for subsequent steps
        kv_cache = None
        prefix_len = x_t.size(1) - block_size

        # Sampling loop
        for step in range(steps):
            t = timesteps[step] * torch.ones(batch_size, 1, device=device)

            # Compute noise levels
            curr_sigma = geometric_noise_schedule(t)
            next_sigma = geometric_noise_schedule(t - dt)
            dsigma = curr_sigma - next_sigma

            # Get model score with KV caching
            # On first step, cache is None so we compute full attention
            # On subsequent steps, we only compute attention for new block
            score, kv_cache = score_fn(x_t, t, kv_cache=kv_cache, prefix_len=prefix_len if step > 0 else 0)

            # If we used cache, score is only for the block
            # Otherwise it's for the full sequence
            if step > 0 and kv_cache is not None:
                score_block = score  # Already just the block
            else:
                score_block = score[:, -block_size:]

            # Compute staggered score
            stag_score = staggered_score(score_block, dsigma)

            # Compute transition probabilities
            x_block = x_t[:, -block_size:]
            trans = transp_transition(x_block, curr_sigma, config.vocab_size)

            # Reverse-step probabilities
            probs = stag_score * trans

            # Sample next tokens
            x_block_new = sample_categorical(probs)

            # concat back to prompt block
            x_t = torch.cat([x_t[:, :-block_size], x_block_new], dim=1)

            # NOTE: We keep kv_cache! The prefix K/V are still valid.
            # On the next step, we'll recompute K/V only for the changed block tokens
            # and concatenate with the cached prefix K/V

            """ ------ display effect ------"""
            # Decode with rainbow colors for denoising!
            sys.stdout.write('\033[2J\033[H')  # Clear screen + home

            # Use cached prompt text, only decode new tokens
            full_text = tokenizer.decode(x_t[0])
            new_tokens = full_text[cached_prompt_len:]

            # Show tokens remaining
            tokens_used = x_t.size(1)
            tokens_remaining = max_seq_len - tokens_used

            # Color denoising tokens cream, normal when complete
            if tokens_remaining > 0:
                colored_new = f"{CREAM}{new_tokens}\033[0m"
            else:
                colored_new = new_tokens

            # Compute cache memory usage
            cache_memory_bytes = (2 * config.n_layers * config.n_heads * prefix_len *
                                 (config.n_embd // config.n_heads) * 4 * batch_size)

            # Format memory appropriately
            if cache_memory_bytes < 1024:
                cache_mem_str = f"{cache_memory_bytes}B"
            elif cache_memory_bytes < 1024**2:
                cache_mem_str = f"{cache_memory_bytes/1024:.1f}KB"
            else:
                cache_mem_str = f"{cache_memory_bytes/(1024**2):.1f}MB"

            # Mode indicator based on steps vs block size
            mode = "Thinking..." if steps / block_size >= 4 else "Retarding..."


            sys.stdout.write(f"{cached_prompt_text}{colored_new}\n")
            sys.stdout.write(f"\n\033[90m{mode}")
            sys.stdout.write(f"\n\033[90mStep {step+1}/{steps} | Block {block_size} | Tokens {tokens_used}/{max_seq_len} | Cache {cache_mem_str}\033[0m\n")
            sys.stdout.flush()




        # Check for EOS
        if (x_t[:, -block_size:] == eos_token_id).any():
            break

        # Next iteration
        prompt_tokens = x_t.clone()

        # Update cached prompt text
        cached_prompt_text = tokenizer.decode(prompt_tokens[0])
        cached_prompt_len = len(cached_prompt_text)

        # Adaptive select steps, block size
        steps = 2 ** torch.randint(1, 7, (1,)).item()
        block_size = 2 ** torch.randint(1, 7, (1,)).item()

        # Clip block_size to tokens remaining
        tokens_remaining = max_seq_len - prompt_tokens.size(1)
        block_size = min(block_size, tokens_remaining)

        # generate new x_t
        noise_block = torch.randint(0, config.vocab_size, (batch_size, block_size), device=device)
        x_t = torch.cat([prompt_tokens, noise_block], dim=1)

        print()  # Final newline when done



"""
KV Caching Strategy Explained:
==============================

1. WHAT WE CACHE:
   - For each transformer layer, we cache K and V matrices for the prefix tokens
   - These don't change during the denoising steps for a given block
   - Cache shape: List of (k, v) tuples, one per layer
     - k: (batch, n_heads, prefix_len, head_dim)
     - v: (batch, n_heads, prefix_len, head_dim)

2. WHEN WE USE IT:
   - First denoising step: Compute full attention, store K/V for prefix
   - Subsequent steps: Only compute Q/K/V for the block being denoised
   - Concatenate cached prefix K/V with new block K/V for attention

3. WHY IT WORKS:
   - Attention is: softmax(Q @ K.T) @ V
   - If prefix tokens don't change, their K and V don't change
   - We only need new Q for full sequence, new K/V for block
   - Saves ~(prefix_len / total_len) of attention computation

4. IMPLEMENTATION DETAILS:
   - Modified MHA.forward() to accept kv_cache and prefix_len
   - On cache hit: only compute QKV for x[:, prefix_len:]
   - Concatenate cached K/V with new K/V before attention
   - Return updated cache for next iteration

5. SPEEDUP ANALYSIS:
   With P = prefix_len, B = block_size, L = num_layers, S = num_steps:

   Without cache: L * S * (P + B)^2 attention ops
   With cache:    L * (P + B)^2 + L * (S-1) * B * (P + B) attention ops

   For P=512, B=64, L=12, S=16:
     Without: 12 * 16 * 576^2 = 63.8M ops
     With:    12 * 576^2 + 12 * 15 * 64 * 576 = 10.6M ops
     Speedup: 6x

6. LIMITATIONS:
   - Memory: Need to store K/V for all layers
   - Currently invalidates cache after each denoising step
     (could be optimized to update incrementally)
   - torch.compile may not optimize as well with dynamic caching
"""
