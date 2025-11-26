# Discrete Diffusion LM Architecture

## Model Configuration

```python
DLMConfig:
    seq_len = 1024
    vocab_size = 50257
    n_layer = 6
    n_head = 8
    n_embd = 512
    cond_dim = 128
    scale_by_sigma = False  # Q_uniform
```

**Parameters: 79.25M**

---

## Architecture Components

### 1. Embedding Layer
```python
vocab_embd: Parameter(vocab_size, n_embd)  # 25.73M params
# Kaiming uniform initialization
```

### 2. Timestep Embedder
```python
Input: σ ∈ ℝ
Sinusoidal: σ → ℝ^256
MLP: ℝ^256 → ℝ^128 → ℝ^128
SiLU activation (twice!)
Output: c ∈ ℝ^128
```

**Math:**
$$\text{freq}_i = \frac{1}{10000^{2i/256}}$$
$$\text{emb} = [\cos(\sigma \cdot \text{freq}), \sin(\sigma \cdot \text{freq})]$$
$$c = \text{SiLU}(\text{MLP}(\text{emb}))$$

### 3. Transformer Block (DDiTBlock) × 6

**Per Block:**
```
Input: x ∈ ℝ^(B,L,512), c ∈ ℝ^(B,128)

AdaLN Modulation:
  c → (shift_attn, scale_attn, gate_attn, shift_mlp, scale_mlp, gate_mlp)

Attention Branch:
  x_norm = RMSNorm(x)
  x_mod = x_norm * (1 + scale_attn) + shift_attn
  QKV = Linear(x_mod) → ℝ^(B,L,1536)
  Apply RoPE to Q,K (NOT V!)
  attn_out = SelfAttention(Q, K, V)
  x = x + gate_attn * attn_out

MLP Branch (SwiGLU):
  x_norm = RMSNorm(x)
  x_mod = x_norm * (1 + scale_mlp) + shift_mlp
  gate = SiLU(W_gate(x_mod))  # 512 → 2048
  up = W_up(x_mod)             # 512 → 2048
  mlp_out = W_down(gate ⊙ up)  # 2048 → 512
  x = x + gate_mlp * mlp_out

Output: x ∈ ℝ^(B,L,512)
```

**Math:**

AdaLN:
$$\text{modulate}(x, s, g) = x \cdot (1 + s) + g$$

RoPE:
$$\text{RoPE}(q_i, k_i) = \begin{bmatrix} \cos(m\theta_j) & -\sin(m\theta_j) \\ \sin(m\theta_j) & \cos(m\theta_j) \end{bmatrix} \begin{bmatrix} q_i \\ k_i \end{bmatrix}$$
where $\theta_j = 10000^{-2j/d}$, applied to Q,K only.

SwiGLU:
$$\text{SwiGLU}(x) = W_{\text{down}}(\text{SiLU}(W_{\text{gate}}(x)) \odot W_{\text{up}}(x))$$

Attention:
$$\text{Attn}(Q,K,V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

### 4. Final Layer
```
x = RMSNorm(x)
x_mod = x * (1 + scale_final) + shift_final
logits = Linear(x_mod) → ℝ^(B,L,50257)

# Mask input positions
logits[i, tok_id[i]] = 0

# Q_uniform: scale_by_sigma = False (skip this)
if scale_by_sigma:
    logits -= log(e^σ - 1) - log(vocab_size - 1)
```

---

## Forward Diffusion (Q_uniform)

**Noise Schedule:**
$$\sigma(t) = \sigma_{\min}^{1-t} \cdot \sigma_{\max}^t, \quad t \in [0,1]$$
$$\sigma_{\min} = 0.001, \quad \sigma_{\max} = 1.0$$

**Forward Process:**
$$p_{\text{move}} = 1 - e^{-\sigma}$$
$$q(x_t | x_0) = \begin{cases}
p_{\text{move}} / V & \text{if } x_t \neq x_0 \\
1 - p_{\text{move}} & \text{if } x_t = x_0
\end{cases}$$

**Implementation:**
```python
move_mask = rand(B,L) < (1 - exp(-σ))
random_tokens = randint(0, vocab_size, (B,L))
x_t = where(move_mask, random_tokens, x_0)
```

---

## Score Entropy Loss (Q_uniform)

**Given:**
- Logits: $\ell \in \mathbb{R}^{B \times L \times V}$
- $x_t, x_0 \in \{0,\ldots,V-1\}^{B \times L}$
- $\sigma \in \mathbb{R}^B$

**Score:**
$$s = \text{softmax}(\ell)$$

**Terms:**
$$e^{\sigma} - 1 = \begin{cases}
\text{expm1}(\sigma) & \text{if } \sigma < 0.5 \\
e^\sigma - 1 & \text{otherwise}
\end{cases}$$

$$r = 1 - \frac{V}{e^{\sigma} - 1 + V}$$

**Negative Term:**
$$n = \mathbb{E}_v[s_v] - \frac{s_{x_t}}{V}$$
$$n = \begin{cases}
r \cdot n & \text{if } x_t = x_0 \\
\frac{s_{x_0}}{e^{\sigma} - 1} + n & \text{otherwise}
\end{cases}$$

**Constant Term:**
$$c = \begin{cases}
\frac{V-1}{V} r (\log r - 1) & \text{if } x_t = x_0 \\
\frac{1}{V}\left(\frac{-\log r - 1}{r} - (V-2)\right) & \text{otherwise}
\end{cases}$$

**Positive Term:**
$$p = \mathbb{E}_v[s_v] - \frac{s_{x_t}}{V}$$
(same as $n$ but using exponential)

**Loss:**
$$\mathcal{L} = \mathbb{E}_{x_0, t, x_t}[p - n + c]$$

---

## Training Pipeline

```python
1. Sample batch: x_0 ∈ {0,...,V-1}^(B,L)
2. Sample timesteps: t ~ Uniform[0,1]^B
3. Compute sigma: σ = sigma_schedule(t)
4. Add noise: x_t = forward_noise(x_0, σ, V)
5. Embed tokens: x = vocab_embd[x_t]
6. Embed timestep: c = SiLU(TimestepEmbedder(σ))
7. Process through 6 transformer blocks with AdaLN
8. Final norm and projection: logits = FinalLayer(x, c)
9. Mask input positions: logits[i, x_t[i]] = 0
10. Compute loss: L = score_entropy_loss(logits, x_t, x_0, σ, V)
11. Backprop and update
```

---

## Sampling Pipeline (Euler Method)

**Denoising from pure noise to clean:**

```python
1. Initialize: x ~ Uniform{0,...,V-1}^L
2. For t in linspace(1, 0, num_steps):
     σ = sigma_schedule(t)
     logits = model(x, σ)
     probs = softmax(logits)
     x ~ Categorical(probs)
3. Return x
```

**With KV Caching (block-based):**
```python
clean_tokens = []
while len(clean_tokens) < seq_len:
    # Cache context K,V
    kv_cache = model.build_kv_cache(clean_tokens, σ)
    
    # Initialize noisy block
    noisy_block = randint(0, V, block_size)
    
    # Denoise (only new tokens, reuse cached K,V)
    for t in linspace(1, 0, num_steps):
        σ = sigma_schedule(t)
        logits = model(noisy_block, σ, kv_cache)
        probs = softmax(logits)
        noisy_block ~ Categorical(probs)
    
    clean_tokens.extend(noisy_block)
```

---

## Key Design Choices

| Component | Choice | Reason |
|-----------|--------|--------|
| Normalization | RMSNorm | Faster, more stable than LayerNorm |
| MLP | SwiGLU | Better than GELU (LLaMA-style) |
| Position Encoding | RoPE | Relative positions, better extrapolation |
| AdaLN Gates | 6 per block | Adaptive modulation for different σ |
| Timestep Embedding | Double SiLU | SEDD design (extra nonlinearity) |
| Output Masking | Always | Prevents trivial solution |
| Graph Type | Q_uniform | Simple, uniform transitions |
| Embedding Init | Kaiming | Better than default |

---

## Model Size Breakdown

```
vocab_embd:     25.73M  (50257 × 512)
sigma_map:       0.05M  (timestep MLP)
blocks (6x):    27.55M
  - attention:   1.57M per block
  - SwiGLU:      3.15M per block
  - AdaLN:       0.02M per block
norm:            0.00M  (512 params)
output:         25.91M  (512 × 50257 + bias)
────────────────────────
Total:          79.25M
```
