import torch
from einops import rearrange, repeat

"""
https://krasserm.github.io/2022/12/13/rotary-position-embedding/

1. precompute_freq(dim, seq_len): precompute cos & sin arguments
2. rotate_half(u): rearrange embd pairs from (x_k,y_k) -> (-y_k, x_k)
3. apply_rotation(u, pos_enc): u * cos(theta) + (rotate_half(u) * sin(theta))

"""

batch_size = 1
num_heads = 2

dim = 4
seq_len = 3
base = 10000.0

"""-------------------------precompute freqs complex-------------------------"""

# thetha = 1 / (10000^(2k/dim)) where k is the indice pair (dim/2-1)
inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
print("each pair of embd gets rotated by this amount")
print(f"inv_freq shape: {inv_freq.shape}")  # (dim/2,) = (4,)
print(f"inv_freq: {inv_freq}\n")

# Generate position indices
t = torch.arange(seq_len, dtype=torch.float)  # [0, 1, 2, 3]
print("position indices")
print(f"t shape: {t.shape}")  # (seq_len,) = (4,)
print(f"t: {t}\n")

# Outer product: position * freq = rotation angles
# freqs[i,j] = t[i] * inv_freq[j] = position * frequency
freqs = torch.einsum("i,j -> ij", t, inv_freq)
print("rotation of dim pairs")
print(f"freqs shape: {freqs.shape}")  # (seq_len, dim/2) = (4, 4)
print(f"freqs (rotation angles for each position):\n{freqs}\n")
# duplicate each element  # (seq_len, dim)
freqs = repeat(freqs, "... f -> ... (f r)", r=2)
print(f"embd rotation (seq_len, dim): \n{freqs}\n")




# Create sample query/key tensors
query = torch.randn(batch_size, num_heads, seq_len, dim)
key = torch.randn(batch_size, num_heads, seq_len, dim)
print(f"u (query or key) shape: \n{query.shape}\n")
print("(x,y) -> (-y,x)")
print("need a -u2, u1, -u4, ... -ud, ud-1")
print()
print(f"query: \n{query}\n")


""" --------------------- rearrange u to -u2, u1, -u4, ... -ud, ud-1 -----------------------"""
# Rotate Half
u = rearrange(query, '... (d r) -> ... d r', r=2)
print(f"u: \n{u}\n")

u1, u2 = u.unbind(dim=-1)
print(f"u1: \n{u1}\n")
print(f"u2: \n{u2}\n")

u = torch.stack((-u2, u1), dim=-1)
print(f"u: \n{u}\n")

u = rearrange(u, '... d r -> ... (d r)')
print(f"u rearranged: \n{u}")



""" ---------------apply rotation------------------"""
q_rot = query * freqs.cos() + (u * freqs.sin())
print(f"q_rot shape: {q_rot.shape}")
print(f"q_rot: \n{q_rot}\n")
