import os
import time

import torch
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from model import SEDD, DLMConfig, GeometricNoise, UniformGraph, print_flops



#################
#    training   #
#################
# ddp
ngpus = 8
accum = 1  # gradient accumulation steps
backend='nccl'

# system
device = "cuda"
dtype=torch.bfloat16
compile=True

# training
batch_size = 512  # total batch size across all GPUs
batch_size_per_gpu = batch_size // (ngpus * accum)  # 512 // 8 = 64 per GPU

n_iters = 600000
log_freq = 1
eval_iters = 100
seq_len = 1024
always_save_checkpoint=True # if True, always save a checkpoint after each eval
init_from="scratch" # "scratch" or "resume"
out_dit = # this should be the lambda labs /mnt persistant

# data
train_dataset = "HuggingFaceFW/fineweb-edu"
train_split = "train"
train_subset = "sample-10BT"  # 10B token sample
valid_dataset = "Rowan/hellaswag"
valid_split = "validation"
cache_dir = None  # or specify path like "~/.cache/huggingface"
num_proc = 8  # parallel processing for tokenization

# geometric noise
sigma_min = 1e-3
sigma_max = 20

# sampling
sampling_steps = 32
sampling_eps=1e-5
sampling_batch_dims=(1,32)

# adamW Optimizer
weight_decay = 0
optimizer="AdamW"
lr = 3e-4
beta1 = 0.9
beta2 = 0.999
eps = 1e-8
warmup = 0.2 * n_iters
grad_clip = 1.0


#####################
#   data loading    #
#####################
"""
TODO:
    - accept None for val dataset so returns none
    - reduces memory usage
    - since for pretraining only need training dataset. no need val

"""
train_loader, val_loader = get_dataloaders(
    train_dataset_name=train_dataset,
    train_split=train_split,
    train_subset=train_subset,
    valid_dataset_name=valid_dataset,
    valid_split=valid_split,
    batch_size=batch_size_per_gpu,
    seq_len=seq_len,
    cache_dir=cache_dir,
    num_proc=num_proc,
    distributed=True,  # Set to True when using DDP
    num_workers=4
)

num_train_tokens = len(train_loader.dataset) * seq_len
num_val_tokens = len(val_loader.dataset) * seq_len

# Compute epoch statistics
tokens_per_iter = batch_size * seq_len
iters_per_epoch = num_train_tokens / tokens_per_iter
num_epochs = n_iters / iters_per_epoch

print("="*60)
print("DATA STATISTICS")
print("="*60)
print(f"Train dataset: {len(train_loader.dataset):,} sequences")
print(f"Val dataset: {len(val_loader.dataset):,} sequences")
print(f"Train tokens: {num_train_tokens:,}")
print(f"Val tokens: {num_val_tokens:,}")
print(f"Total tokens: {num_train_tokens + num_val_tokens:,}")
print()
print(f"Batch size per GPU: {batch_size_per_gpu}")
print(f"Effective batch size: {batch_size_per_gpu * ngpus * accum}")
print(f"Tokens per iteration: {tokens_per_iter:,}")
print(f"Iterations per epoch: {iters_per_epoch:,.1f}")
print(f"Total iterations: {n_iters:,}")
print(f"Total epochs: {num_epochs:.2f}")
print()




#####################
#    build model    #
#####################
config = DLMConfig(
    hidden_size = 512,
    cond_dim = 128,
    length = 1024,
    vocab = 50304,
    n_blocks = 8,
    n_heads = 8,
    dropout = 0.1
)

print(config)
score_fn = SEDD(config).to(device=device, dtype=dtype)
graph = UniformGraph(config.vocab)
noise = GeometricNoise(sigma_min, sigma_max)

num_params = sum(p.numel() for p in score_fn.parameters())
print_flops(config, batch_size, seq_len, num_params, n_iters, ngpus)




###########################################
#                setup ddp                #
###########################################
"""
TODO:
    - ddp
    - autocast
    - wrap model ddp
    - all_reduce
    - loss accum
    - log printing for master process
"""



###########################################
#                dir for logs             #
###########################################
"""
TODO:
    - for sampling
    - for checkpoints
    - for logs (also function for plotting loss curve of logs)
"""



###########################################
#            utils for saving             #
###########################################
"""
save checkpoint:
    - model.state_dict()
    - optimizer.state_dict()
    - model_args
    - iter_num
    - best_val_loss
    - config
    save checkpoint to out_dir, "ckpt.pt

save generated samples

same perplexity

"""



###########################################
#                optimizer                #
###########################################
"""
TODO:
    - add grad scaler in optimizer step
"""
scaler = torch.cuda.amp.GradScaler()
optimizer = optim.AdamW(score_fn.parameters(), lr=lr, betas=(beta1, beta2), eps=eps, weight_decay=weight_decay)
print(f"optimizer: {optimizer}")

# TODO: add scaler
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


############################
#    log score function    #
############################

def log_score(score_fn, x0, xt=None, t=None, kv_cache=None, prefix_len=0):
    if t is None:
        # random sample t -> get noise -> perturb data
        t = (1 - sampling_eps) * torch.rand(x0.shape[0], device=x0.device) + sampling_eps # (B,)

    sigma, dsigma = noise.get_noise(t) # (B,) same shape as t

    if xt is None:
        xt = graph.sample_transition(x0, sigma[:, None]) # sigma[:, None] -> (B, 1)

    # predict log score
    log_score, kv_cache = model(xt, sigma.reshape(-1), kv_cache=None, prefix_len=0)




###############################################
#                training loop                #
###############################################
t0 = time.time()
local_iter_num=0 # number of iters in the lifetime of this process

"""
train iter

"""
while step < n_iters+1:

    # get batch

    #
