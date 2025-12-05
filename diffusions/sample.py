"""
Sampling script for Discrete Diffusion LM

Generates text using block-based denoising with KV caching for efficiency.

Usage:
    python sample.py --prompt "Once upon a time" --steps 50 --tokens 512
"""

import torch
import torch.nn.functional as F
import tiktoken
import argparse

from model import DLMConfig, DiffusionLM, sigma_schedule


@torch.no_grad()
def generate(
    model,
    prompt_text,
    tokenizer,
    config,
    device,
    block_size=128,
    num_steps=50,
    max_tokens=512,
    show_progress=True
):
    """
    Generate text using block-based denoising with KV caching.

    Args:
        model: Trained DiffusionLM
        prompt_text: Text prompt to condition on
        tokenizer: GPT-2 tokenizer
        config: Model config
        device: Device (cuda/cpu)
        block_size: Tokens per block (default 128)
        num_steps: Denoising steps per block (default 50)
        max_tokens: Maximum total tokens to generate
        show_progress: Show denoising progress

    Returns:
        Generated text string
    """
    model.eval()

    # Encode prompt
    prompt_tokens = tokenizer.encode(prompt_text)
    clean_tokens = torch.tensor(prompt_tokens, dtype=torch.long, device=device)

    if show_progress:
        print(f"Prompt: {prompt_text}\n")
        print("="*80)

    # Generate blocks
    block_num = 0
    while len(clean_tokens) + block_size <= min(max_tokens, config.seq_len):
        block_num += 1

        # Initialize random noisy block
        noisy_block = torch.randint(0, config.vocab_size, (block_size,), device=device)

        # Build KV cache from clean context
        if len(clean_tokens) > 0:
            kv_cache = model.build_kv_cache(
                clean_tokens.unsqueeze(0),
                torch.tensor([0.001], device=device)  # Dummy sigma for caching
            )
        else:
            kv_cache = None

        # Denoise block
        for step in range(num_steps):
            t = 1.0 - step / num_steps
            sigma = sigma_schedule(torch.tensor([t], device=device))

            # Forward pass (only new block, context cached!)
            if kv_cache is not None:
                logits = model(noisy_block.unsqueeze(0), sigma, kv_cache=kv_cache)
            else:
                logits = model(noisy_block.unsqueeze(0), sigma)

            # Sample next state
            probs = F.softmax(logits[0], dim=-1)
            noisy_block = torch.multinomial(probs, 1).squeeze(-1)

            # Show progress
            if show_progress and (step % 10 == 0 or step == num_steps - 1):
                current = torch.cat([clean_tokens, noisy_block]).tolist()
                current_text = tokenizer.decode(current)
                print(f"\r[Block {block_num}, Step {step+1}/{num_steps}] {current_text[:150]}...", end='', flush=True)

        if show_progress:
            print()  # Newline after block completes

        # Add clean block to context
        clean_tokens = torch.cat([clean_tokens, noisy_block])

        # Check for EOS
        if tokenizer.eot_token in noisy_block.tolist():
            if show_progress:
                print("\n[EOS detected]")
            break

    if show_progress:
        print("="*80)

    final_text = tokenizer.decode(clean_tokens.tolist())
    return final_text


def main():
    parser = argparse.ArgumentParser(description="Generate text with Discrete Diffusion LM")
    parser.add_argument("--model", type=str, default="model.pt", help="Path to checkpoint")
    parser.add_argument("--prompt", type=str, default="Once upon a time", help="Text prompt")
    parser.add_argument("--block", type=int, default=128, help="Block size")
    parser.add_argument("--steps", type=int, default=50, help="Denoising steps per block")
    parser.add_argument("--tokens", type=int, default=512, help="Max tokens to generate")
    parser.add_argument("--quiet", action="store_true", help="Don't show progress")
    args = parser.parse_args()

    # Setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}\n")

    # Load model
    config = DLMConfig()
    model = DiffusionLM(config).to(device)

    checkpoint = torch.load(args.model, map_location=device, weights_only=False)
    if 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    else:
        model.load_state_dict(checkpoint)

    model.eval()

    # Load tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")

    # Generate
    text = generate(
        model=model,
        prompt_text=args.prompt,
        tokenizer=tokenizer,
        config=config,
        device=device,
        block_size=args.block,
        num_steps=args.steps,
        max_tokens=args.tokens,
        show_progress=not args.quiet
    )

    print(f"\nGenerated Text:\n{text}\n")


if __name__ == "__main__":
    main()
