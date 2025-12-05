
"""

to add in model.py:
    - autocast mixed precision
    - flash attention


to add in training
    - grad scaler
    - grad accum
    - ddp & distributed sampler
    - sync all reduce gradients

to define at start of training.py
    - hyperparams:
        - lr
        - weight decay
        - decay lr
        - warmup iters

    - compile pytorch
    - dtype = bfloat16


resource accounting:
    - model param size
    - total flops per iteration
        flops per token
        flops per fwd bwd
        flops per iter

    - flops achieved = flops per iter per second
    - flops promised = 312e12 # for A100 GPU bfloat16 peak flops is 312 TFLOPS
    - mfu = flops achieved / flops_promised




"""


"""
need cfg.yaml

"""


"""
Process group initialization:
    - sets default GPU for each process
    - os.get local_rank
    - set master process = ddp_rank = 0
    - backend='nccl'
    device = f"cuda:{ddp_local_rank}"
    print(f"using device: {device}")

"""


"""
wrap model into DDP, device_id = ddp_local_rank
"""



"""
Distributing input data using `DistributedSampler`
    - total_batch = batch_size * nprocs
    - init_process_group(backend="nccl")
    - torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

    train_data = torch.utils.data.DataLoader(
            dataset=train_dataset,
            batch_size=batch_size,
            shuffle=False,
            sampler=DistributedSampler(train_dataset),
        )
"""

"""
autocast mixed precision bf16 for some tensors. followed by scaler

scaler = torch.cuda.amp.GradScaler()


for iter in range(.):

    with torch.cuda.amp.autocast():
        # scale gradients up from bf16 to prevent gradients underflow
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

"""

"""
gradient accum:
    1. normalize loss by num of gradient accum steps
    2. only update optimizer every chunk


    - num_accum_steps =
    - loss / num_accum_steps # normalize the gradients
    - loss.backward
    - if iter % num_accum_steps == 0 or...
            optimizer.step()
            optimizer.zero_grad()

"""


"""
Saving model checkpoint using master_process
if gpu_id == 0
    or
    if master_process:
        ...
        save_checkpoint

"""



#####################
# define the params #
#####################

ngpus=8
tokens=50304


# training
batch_size=512
accum=
n_iters=
snapshot_freq=
log_freq=1
eval_freq=100

# data
train=
valid=
cache_dir=

# eval
batch_size=512
perplexity=True
perplexity_batch_size=32

# optim
weight_decay=0
optimizer="AdamW"
lr=3e-4
beta1=0.9
beta2=0.999
eps=1e-8
warmup=2500
grad_clip=1.0

# model
hidden_size=512
cond_dim=128
length=1024
n_blocks=8
n_heads=8
dropout=0.1
