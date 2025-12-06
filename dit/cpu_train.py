import time
import torch
import torch.optim as optim
import tiktoken

from model import SEDD, DLMConfig, GeometricNoise, UniformGraph


device = "cuda" if torch.cuda.is_available() else "cpu"


training_data = "i like to think of a cybernetic meadow"\
    " where mammals and computers live together"\
    " in mutually programming harmony"\
    ' like pure water touching clear sky'\
    " i like to think of a cybernetic  forest"\
    " filled with pines and electronics"\
    " where deer stroll peacefully"\
    " past computers as if they were flowers"\
    " with spinning blosoms"\
    " i like to think of a cybernetic ecology"\
    " where we are free of our labors and joined back to nature"\
    " returned to our mammal brothers and sisters"\
    " and all watched over"\
    " by machines of loving grace"

enc = tiktoken.get_encoding('gpt2')
tokens = torch.tensor(enc.encode(training_data), device=device)
print(enc.decode(tokens.tolist()))

tokens = tokens.unsqueeze(0)
tokens = tokens.repeat(1, 1)      # (B, T)


# Get unique tokens and create mapping
unique_tokens = torch.unique(tokens)
vocab_size = len(unique_tokens)
token_to_id = {t.item(): i for i, t in enumerate(unique_tokens)}
id_to_token = {i: t.item() for i, t in enumerate(unique_tokens)}

# Remap tokens to 0, 1, 2, ... vocab_size-1
tokens_remapped = torch.tensor([[token_to_id[t.item()] for t in tokens[0]]], device=device)
tokens = tokens_remapped.repeat(64, 1)

print(f"Original vocab: 50304, Actual vocab needed: {vocab_size}")






#################
#    training   #
#################
batch_size = 1
n_iters = 10000
log_freq = 1
eval_iters = 50
block_size = 1024

# sampling
sampling_steps = 32
sampling_eps=1e-5
sampling_batch_dims=(1,16)

# noise
sigma_min = 1e-4
sigma_max = 20

# adamW Optimizer
weight_decay = 0
lr = 3e-4
beta1 = 0.9
beta2 = 0.999
eps = 1e-8
warmup = 0.05 * n_iters # 0.2 original
grad_clip = 5.0


#####################
#    build model    #
#####################
config = DLMConfig(
    hidden_size = 64,
    cond_dim = 32,
    length = 1024,
    vocab = vocab_size,
    # vocab = 50304,
    n_blocks = 4,
    n_heads = 4,
    dropout = 0.0
)


print(config)
score_fn = SEDD(config).to(device)
graph = UniformGraph(config.vocab)
noise = GeometricNoise(sigma_min, sigma_max)

num_params = sum(p.numel() for p in score_fn.parameters())
print(f"model params: {num_params/1e6:.3f}M")
print()

#####################
#    optimizer      #
#####################
optimizer = optim.AdamW(score_fn.parameters(), lr=lr, betas=(beta1, beta2), eps=eps, weight_decay=weight_decay)
print(f"optimizer: {optimizer}")
print()

def optimizer_step(optimizer, model, step, lr, warmup, grad_clip):
    # warmup lr
    if warmup > 0:
        lr_scaled = lr * min(step / warmup, 1.0)
        for g in optimizer.param_groups:
            g['lr'] = lr_scaled

    # gradient clipping
    if grad_clip > 0:
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)

    return optimizer, norm


#######################
#    training loop    #
#######################

x0 = tokens
t0 = time.time()

for step in range(n_iters):

    # random sample t -> get noise -> perturb data
    t = (1 - sampling_eps) * torch.rand(x0.shape[0], device=x0.device) + sampling_eps # (B,)
    sigma, dsigma = noise.get_noise(t) # (B,) same shape as t

    # perturb batch
    xt = graph.sample_transition(x0, sigma[:, None]) # sigma[:, None] -> (B, 1)

    # print pct of tokens masked for debugging
    noised_pct = (xt[0] != x0[0]).sum() / x0.size()[1]

    # predict log score
    log_score, _ = score_fn(xt, sigma.reshape(-1), kv_cache=None, prefix_len=0) # (B, S, V)
    loss = graph.score_entropy(log_score, sigma[:, None], xt, x0)
    loss = (dsigma[:, None] * loss).sum(dim=-1).mean()


    # backward pass, f32
    loss.backward()

    # step the optimizer
    optimizer, norm = optimizer_step(optimizer, score_fn, step=step, lr=lr, warmup=warmup, grad_clip=grad_clip)
    optimizer.step()
    optimizer.zero_grad()



    # time and log
    t1 = time.time()
    dt_time = t1 - t0
    t0 = t1

    # Get current lr from optimizer
    current_lr = optimizer.param_groups[0]['lr']
    print(f"step: {step} | t: {t[0]:.2f} | noised: {noised_pct*100:.2f}% | loss: {loss:.4f} | norm: {norm:.4f} | lr: {current_lr:.6f} | time: {dt_time*1000:.2f}ms")





    if step % eval_iters == 0 or step == 0:
        print("="*77)
        with torch.no_grad():

            x = graph.sample_limit(*sampling_batch_dims).to(device)
            timesteps = torch.linspace(1, sampling_eps, sampling_steps+1, device=device)
            dt = (1 - sampling_eps) / sampling_steps

            for i in range(sampling_steps):
                t = timesteps[i] * torch.ones(x.shape[0], 1, device=device)
                sigma, dsigma = noise.get_noise(t)
                log_score, _ = score_fn(x, sigma.reshape(-1), kv_cache=None, prefix_len=0) # (B, S, V)
                score = log_score.exp() # exp for actual score

                rev_rate = dt * dsigma[..., None] * graph.reverse_rate(x, score) # (1) * (B,1,1) * (B,T,V)
                x = graph.sample_rate(x, rev_rate)

                original_tokens = torch.tensor([id_to_token[i.item()] for i in x[0]])
                clipped_x = torch.clamp(original_tokens, 0, enc.n_vocab - 1)
                full_text = enc.decode(clipped_x.tolist())
                print(full_text)
        print("="*77)
