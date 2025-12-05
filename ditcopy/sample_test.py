import torch
import torch.nn as nn
from model import DLM, DiTConfig
import tiktoken
import sys

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Configuration
config = DiTConfig()
batch_size = 1
eps = 1e-5
max_seq_len = config.seq_len


# Initialize model with torch.compile for faster inference
model = DLM(config).to(device)  # Disable compile with kv cache

# Initialize random weights for testing (since we don't have trained weights yet)
# Override the zero initialization in the model architecture
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean=0.0, std=0.02)
        if m.bias is not None:
            nn.init.normal_(m.bias, mean=0.0, std=0.02)
    elif isinstance(m, nn.Embedding):
        nn.init.normal_(m.weight, mean=0.0, std=0.02)
    elif hasattr(m, 'scale') and isinstance(m.scale, nn.Parameter):
        # RMSNorm scale - keep at 1.0 for stability
        nn.init.ones_(m.scale)

model.apply(init_weights)

# Use model directly (random weights) to test KV caching
score_fn = model

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
enc = tiktoken.get_encoding('gpt2')

prefix = "helllo i am an ai model i can "

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
block_size = 2 ** torch.randint(3, 7, (1,)).item()
block_size = min(block_size, tokens_remaining)

# Initialize: prompt + noise block
noise_block = torch.randint(0, config.vocab_size, (batch_size, block_size), device=device)
x_t = torch.cat([prompt_tokens, noise_block], dim=1)

BLUE = '\033[38;5;153m'  # Faint blue for denoising

# Cache prompt text (only decode once)
cached_prompt_text = enc.decode(prompt_tokens[0].tolist())
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
            t = timesteps[step] * torch.ones(batch_size, device=device) # (B,)

            # Compute noise levels
            curr_sigma = geometric_noise_schedule(t) # (B,)
            next_sigma = geometric_noise_schedule(t - dt) # (B,)
            dsigma = curr_sigma - next_sigma

            # Get model score with KV caching
            # On first step, cache is None so we compute full attention
            # On subsequent steps, we only compute attention for new block
            score, kv_cache = score_fn(x_t, curr_sigma, kv_cache=kv_cache, prefix_len=prefix_len if step > 0 else 0)

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

            # Decode and color new tokens (clip to tokenizer vocab range)
            clipped_tokens = torch.clamp(x_t[0], 0, enc.n_vocab - 1)
            full_text = enc.decode(clipped_tokens.tolist())
            new_tokens = full_text[cached_prompt_len:]
            colored_new = f"{BLUE}{new_tokens}\033[0m"

            tokens_used = x_t.size(1)
            tokens_remaining = max_seq_len - tokens_used

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
            mode = "Thinking..." if steps > block_size else "Yapping..."


            sys.stdout.write(f"{cached_prompt_text}{colored_new}\n")
            sys.stdout.write(f"\n\033[90m{mode}")
            sys.stdout.write(f"\n\033[90mStep {step+1}/{steps} | Block {block_size} | Tokens {tokens_used}/{max_seq_len} | Cache {cache_mem_str}\033[0m\n")
            sys.stdout.flush()




        # Check for EOS
        if (x_t[:, -block_size:] == eos_token_id).any():
            break

        # Next iteration
        prompt_tokens = x_t.clone()

        # Update cached prompt text (clip to tokenizer vocab range)
        clipped_tokens = torch.clamp(prompt_tokens[0], 0, enc.n_vocab - 1)
        cached_prompt_text = enc.decode(clipped_tokens.tolist())
        cached_prompt_len = len(cached_prompt_text)

        # Adaptive select steps, block size
        steps = 2 ** torch.randint(3, 7, (1,)).item()
        block_size = 2 ** torch.randint(3, 7, (1,)).item()

        # Clip block_size to tokens remaining
        tokens_remaining = max_seq_len - prompt_tokens.size(1)
        block_size = min(block_size, tokens_remaining)

        # generate new x_t
        noise_block = torch.randint(0, config.vocab_size, (batch_size, block_size), device=device)
        x_t = torch.cat([prompt_tokens, noise_block], dim=1)

        print()  # Final newline when done
