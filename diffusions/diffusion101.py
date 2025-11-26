
"""
predicts entire noise that was added -> subtract d of it -> repeat
linear schedule: noised at half too early
cosine: half lost later. less steps needed


timestep embedding** how??
text embeding
noisy input


----------- timestep embeddings ----------------
* coarse adjustments -> fine-grained details
* info on what the noise level is
* sinusoidal embd
* each step -> adds embedding about the curernt timestep

latent space
projection to bag of tokens head
projection to token + positosn

"""


import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from einops import rearrange, repeat


# diffusion params
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
T = 1000
beta = torch.linspace(0.0001, 0.02, T).to(device)
alpha = 1.0 - beta
alpha_bar = torch.cumprod(alpha, dim=0)

import matplotlib.pyplot as plt
plt.plot(alpha_bar)
plt.plot(1-alpha_bar)
plt.show()


def forward_noise(x_0, t, noise=None):
    """
    t: (batch_size, )
    gets noise in (B, C, H, W) for images
    applies to x_0

    returns
    x_t: (B,C,H,W)
    noise: (B,C,H,W)

    predict noise
    """
    if noise is None:
        noise = torch.randn_like(x_0).to(device)
    sqrt_alpha_bar = rearrange(alpha_bar[t].sqrt(), 'b -> b 1 1 1') # B,C,H,W
    sqrt_one_minus_alpha_bar = rearrange((1 - alpha_bar[t]).sqrt(), 'b -> b 1 1 1') # B,C,H,W
    x_t = sqrt_alpha_bar * x_0 + sqrt_one_minus_alpha_bar * noise
    return x_t, noise


batch_size = 10
C = 3
H = W = 64
t = torch.randint(0, T, (batch_size,), device=device)

x_0 = torch.rand(batch_size, C, H, W) # random data
print(f"x_0 size: {x_0.size()}")

x_t, noise = forward_noise(x_0, t)

""" ---------------- sinusoidal --------------------
* Use sinusoidal embeddings (inspired by Transformer models)
to encode timestep t into a vector of size embedding_dim.

* Combine sine and cosine functions to create a rich representation of timesteps.


returns (batch, time_embd)

this timestep_embd later goes through time_embd(): linear -> silu -> linear

then add timestep embd to x

then model()...




--------------Sinusoidal Positional Encoding Formula--------------
1. even/odd split
even: PE(pos, 2i)  = sin(pos / 10000^2i/d_model)
odd: PE(pos, 2i+1) = cos(pos / 10000^2i/d_model)

2. half split
pe[:, :half_dim] = torch.sin(positions * div_term)
pe[:, half_dim:] = torch.sin(positions * div_term)

both works


-----------------------timestep embedding------------------------
use timestep instead of position

te[:, :half_dim] = torch.sin(timestep * div_term)
te[:, half_dim:] = torch.sin(timestep * div_term)

but passed through MLP before added to model
why MLP?:
    * x_t is noisy, t is timestep
    * need to match n_dims to add
    * MLP projects t_embd to x_dim size
    * MLP adds nonlinearity to sinusoidal features

full example

class TimeEmbedding(nn.Module):
    def __init__(self, time_embd, n_embd, hidden_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(time_embd, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, n_embd)
        )

    def forward(self, timestep):
        te = timestep_embd(t, time_embd)
        return self.mlp(te)

where:
    time_embd: timestep embeddings
    hidden_dim: 4 * n_embd
    n_embd: token dims

"""

n_embd = 128
timesteps = torch.randint(0, T, (batch_size,))
timesteps
print(timesteps)

half_dim = n_embd // 2
half_dim

embd = torch.log(torch.tensor(10000.0)) / (half_dim-1)
embd = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -embd)
embd = timesteps[:, None] * embd[None, :]
print(f"embd size: {embd.size()}")

embd = torch.cat([torch.sin(embd), torch.cos(embd)], dim=-1)
print(f"concat embd size: {embd.size()}")
timesteps
plt.plot(torch.sin(embd[1]), label=f"t: {timesteps[1]}")
plt.plot(torch.sin(embd[3]), label=f"t: {timesteps[2]}")
plt.legend()
plt.show()




import torch
import math

def positional_encoding(seq_len, embedding_dim):
    """
    seq_len: length of the sequence (number of tokens)
    embedding_dim: dimension of each positional embedding
    """
    half_dim = embedding_dim // 2  # we split embedding_dim in half: sin for first half, cos for second half

    # Compute the "div_term", which scales the positions into different frequencies
    # Explanation:
    #   - math.log(10000) sets the maximum wavelength
    #   - division by (half_dim - 1) spreads frequencies evenly across dimensions
    div_term = torch.exp(
        torch.arange(0, half_dim, dtype=torch.float32) * -(math.log(10000.0) / (half_dim - 1))
    )

    positions = torch.arange(0, seq_len, dtype=torch.float32).unsqueeze(1)  # shape: [seq_len, 1]

    pe = torch.zeros(seq_len, embedding_dim)  # initialize embedding matrix
    pe[:, :half_dim] = torch.sin(positions * div_term)  # sin for first half of dimensions
    pe[:, half_dim:] = torch.cos(positions * div_term)  # cos for second half

    return pe

# Example
pe = positional_encoding(seq_len=5, embedding_dim=8)
print(pe)
