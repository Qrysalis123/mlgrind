# Score Entropy Loss for Discrete Diffusion: A Mathematical Derivation

## Motivation: Why This Loss?

In **continuous diffusion models** (like DDPM for images), we learn to predict the noise $\epsilon$ or the score $\nabla_x \log p(x)$. But text is **discrete** - we have tokens from a finite vocabulary $\mathcal{V}$ where $|\mathcal{V}| = V$.

For discrete data, we need a different approach. The **Score Entropy Discrete Diffusion (SEDD)** framework learns to estimate:

$$
\boxed{s(x \mid x_t, \sigma) = \frac{q(x_t \mid x, \sigma) \cdot p_{\text{data}}(x)}{q(x_t \mid \sigma)}}
$$

This is the **ratio** of the joint distribution to the marginal. Why? Because knowing this ratio lets us:
1. Evaluate how likely different clean tokens $x$ are given noised token $x_t$
2. Sample from $p(x \mid x_t, \sigma)$ during generation
3. Perform gradient-based sampling in discrete space

---

## The Forward Process: Q_uniform Noise

We corrupt data using a **uniform replacement** process. At noise level $\sigma \geq 0$:

$$
q(x_t \mid x_0, \sigma) = \begin{cases}
e^{-\sigma} & \text{if } x_t = x_0 \text{ (stay)} \\
\frac{1 - e^{-\sigma}}{V} & \text{if } x_t \neq x_0 \text{ (replace uniformly)}
\end{cases}
$$

**Intuition**: With probability $e^{-\sigma}$, keep the token. Otherwise, replace it with a uniform random token from the vocabulary.

### Key quantity: $e^{\sigma} - 1$

Let's derive a useful identity:

$$
\begin{align}
P(\text{move}) &= 1 - e^{-\sigma} \\
\frac{P(\text{move})}{P(\text{stay})} &= \frac{1 - e^{-\sigma}}{e^{-\sigma}} \\
&= e^{\sigma} - 1
\end{align}
$$

This ratio $e^{\sigma} - 1$ appears everywhere in the loss!

---

## Deriving the Loss: Score Matching

Our goal is to train a model $s_\theta(x \mid x_t, \sigma)$ to approximate the true score ratio. We use a **score matching** objective.

### The True Score (What We Want to Learn)

For a given noised token $x_t$, the true conditional distribution is:

$$
p(x_0 \mid x_t, \sigma) = \frac{q(x_t \mid x_0, \sigma) \cdot p_{\text{data}}(x_0)}{q(x_t \mid \sigma)}
$$

By Bayes' theorem. We want our model to output:

$$
s_\theta(x \mid x_t, \sigma) \approx \frac{q(x_t \mid x, \sigma) \cdot p_{\text{data}}(x)}{q(x_t \mid \sigma)}
$$

### The KL Divergence Objective

We minimize the KL divergence between true and predicted distributions:

$$
\begin{align}
\mathcal{L} &= \mathbb{E}_{x_0 \sim p_{\text{data}}} \mathbb{E}_{x_t \sim q(\cdot \mid x_0, \sigma)} \left[ D_{KL}(p(x \mid x_t, \sigma) \| s_\theta(x \mid x_t, \sigma)) \right] \\
&= \mathbb{E}_{x_0, x_t} \left[ \sum_{x \in \mathcal{V}} p(x \mid x_t, \sigma) \log \frac{p(x \mid x_t, \sigma)}{s_\theta(x \mid x_t, \sigma)} \right]
\end{align}
$$

Expanding:

$$
\mathcal{L} = \mathbb{E}_{x_0, x_t} \left[ \sum_{x} p(x \mid x_t, \sigma) \log p(x \mid x_t, \sigma) - \sum_{x} p(x \mid x_t, \sigma) \log s_\theta(x \mid x_t, \sigma) \right]
$$

The first term (entropy of true distribution) doesn't depend on $\theta$, so we focus on:

$$
\mathcal{L} = - \mathbb{E}_{x_0, x_t} \left[ \sum_{x} p(x \mid x_t, \sigma) \log s_\theta(x \mid x_t, \sigma) \right]
$$

---

## Case Analysis: Perturbed vs Unperturbed

The forward process creates two cases:

### Case 1: $x_t \neq x_0$ (Token was replaced)

This happens with probability $1 - e^{-\sigma}$. Given this occurred:

$$
q(x_t \mid x_0, \sigma) = \frac{1 - e^{-\sigma}}{V}, \quad x_t \neq x_0
$$

The conditional distribution is:

$$
p(x \mid x_t, \sigma) = \begin{cases}
\frac{p_{\text{data}}(x_t)}{q(x_t \mid \sigma)} \cdot \frac{1 - e^{-\sigma}}{V} & \text{if } x = x_t \\
\frac{p_{\text{data}}(x) \cdot e^{-\sigma}}{q(x_t \mid \sigma)} & \text{if } x = x_0 \\
\frac{p_{\text{data}}(x)}{q(x_t \mid \sigma)} \cdot \frac{1 - e^{-\sigma}}{V} & \text{if } x \notin \{x_0, x_t\}
\end{cases}
$$

### Case 2: $x_t = x_0$ (Token stayed same)

This happens with probability $e^{-\sigma}$:

$$
q(x_t \mid x_0, \sigma) = e^{-\sigma}
$$

Or with small probability $\frac{1-e^{-\sigma}}{V}$ if noise randomly picked same token (collision).

---

## The Ratio $r(\sigma)$

Define the **survival ratio**:

$$
r(\sigma) = \frac{e^{\sigma} - 1}{e^{\sigma} - 1 + V} = 1 - \frac{V}{e^{\sigma} - 1 + V}
$$

**Derivation**: This comes from the ratio of probabilities:

$$
\begin{align}
r &= \frac{P(\text{move to specific token})}{P(\text{move to specific token}) + P(\text{could be any clean token})} \\
&= \frac{\frac{1-e^{-\sigma}}{V}}{\frac{1-e^{-\sigma}}{V} + e^{-\sigma} \cdot 1} \\
&= \frac{1 - e^{-\sigma}}{1 - e^{-\sigma} + V \cdot e^{-\sigma}} \\
&= \frac{e^{\sigma}(1 - e^{-\sigma})}{e^{\sigma}(1 - e^{-\sigma} + V \cdot e^{-\sigma})} \\
&= \frac{e^{\sigma} - 1}{e^{\sigma} - 1 + V}
\end{align}
$$

**Interpretation**: $r$ measures how much of the probability mass comes from noise vs clean signal.

---

## Computing the Loss Terms

Now we derive each component of the loss. Let $s_\theta(x \mid x_t, \sigma)$ be our model's softmax output.

### Extracting Score Values

Define:
- $\bar{s} = \frac{1}{V} \sum_{x \in \mathcal{V}} s_\theta(x \mid x_t, \sigma)$ — mean score over vocabulary
- $s_{x_t} = s_\theta(x_t \mid x_t, \sigma)$ — score at noised token
- $s_{x_0} = s_\theta(x_0 \mid x_t, \sigma)$ — score at clean token

**Why mean?** The uniform noise makes all "incorrect" tokens equally likely, so we average over them.

---

## The Cross-Entropy Expansion

The loss involves computing:

$$
-\sum_{x} p(x \mid x_t, \sigma) \log s_\theta(x \mid x_t, \sigma)
$$

We split this sum into parts:

$$
= -p(x_0 \mid x_t, \sigma) \log s_{x_0} - p(x_t \mid x_t, \sigma) \log s_{x_t} - \sum_{x \notin \{x_0, x_t\}} p(x \mid x_t, \sigma) \log s_\theta(x \mid x_t, \sigma)
$$

### Approximating the Sum

For the sum over "other" tokens, we use:

$$
\sum_{x \notin \{x_0, x_t\}} p(x \mid x_t, \sigma) \log s_\theta(x \mid x_t, \sigma) \approx (V-2) \cdot \bar{s} \cdot \log \bar{s}
$$

This is an approximation assuming the model spreads probability relatively uniformly over non-target tokens.

---

## The Negative Term

For **Case 1** ($x_t \neq x_0$):

The conditional probability of the clean token is:

$$
p(x_0 \mid x_t, \sigma) = \frac{p_{\text{data}}(x_0) \cdot e^{-\sigma}}{q(x_t \mid \sigma)}
$$

In the loss, this contributes:

$$
\frac{s_{x_0}}{e^{\sigma} - 1}
$$

**Derivation**: The denominator $e^{\sigma} - 1$ comes from the odds ratio. Specifically:

$$
\frac{p(x_0 \mid x_t, \sigma)}{p(x_t \mid x_t, \sigma)} \approx \frac{e^{-\sigma}}{\frac{1-e^{-\sigma}}{V}} = \frac{V \cdot e^{-\sigma}}{1 - e^{-\sigma}} = \frac{V}{e^{\sigma} - 1}
$$

For **Case 2** ($x_t = x_0$):

No noise was applied, so we scale by $r$:

$$
r \cdot \left(\bar{s} - \frac{s_{x_t}}{V}\right)
$$

Combining both cases:

$$
\text{neg\_term} = \begin{cases}
r \cdot \left(\bar{s} - \frac{s_{x_t}}{V}\right) & \text{if } x_t = x_0 \\
\frac{s_{x_0}}{e^{\sigma} - 1} + \bar{s} - \frac{s_{x_t}}{V} & \text{if } x_t \neq x_0
\end{cases}
$$

**Interpretation**: 
- When $x_t \neq x_0$: Heavily weight the score at clean token $s_{x_0}$
- When $x_t = x_0$: Scale the uniform term by survival ratio $r$

---

## The Constant Term

The constant normalizes the objective and comes from entropy calculations:

For **Case 1** ($x_t \neq x_0$):

$$
\text{const} = \frac{1}{V} \left(\frac{-\log r - 1}{r} - (V - 2)\right)
$$

**Derivation**: From the entropy of the approximate uniform distribution over $V-1$ tokens:

$$
\begin{align}
H &\approx -\frac{V-2}{V} \log \left(\frac{1}{V-2}\right) - \frac{1}{V} \log r \\
&= \frac{V-2}{V} \log(V-2) - \frac{1}{V} \log r
\end{align}
$$

Rearranging and simplifying gives the formula above.

For **Case 2** ($x_t = x_0$):

$$
\text{const} = \frac{V-1}{V} \cdot r \cdot (\log r - 1)
$$

**Derivation**: When no noise was applied, the entropy of staying vs moving is:

$$
H = r \log r + (1-r) \log(1-r)
$$

Scaled by $\frac{V-1}{V}$ to account for vocabulary size.

---

## The Positive Term

This is the simplest component:

$$
\text{pos\_term} = \bar{s} - \frac{s_{x_t}}{V}
$$

**Intuition**: 
- $\bar{s}$: Encourages high average probability (high entropy over vocabulary)
- $-\frac{s_{x_t}}{V}$: Penalizes probability at the noised token (scaled by $V$)

**Why?** Combined with the masking (setting logits at $x_t$ to 0), this forces the model to:
1. Spread probability over many tokens (high entropy)
2. Not assign mass to the noised token (actual denoising)

---

## Final Loss Formula

Combining all three terms:

$$
\boxed{\mathcal{L}(x_0, x_t, \sigma) = \text{pos\_term} - \text{neg\_term} + \text{const}}
$$

Explicitly:

$$
\mathcal{L} = \underbrace{\bar{s} - \frac{s_{x_t}}{V}}_{\text{entropy regularization}} - \underbrace{\begin{cases}
r \cdot \left(\bar{s} - \frac{s_{x_t}}{V}\right) & x_t = x_0 \\
\frac{s_{x_0}}{e^{\sigma} - 1} + \bar{s} - \frac{s_{x_t}}{V} & x_t \neq x_0
\end{cases}}_{\text{score matching term}} + \underbrace{\text{const}(x_t, x_0, r)}_{\text{normalization}}
$$

The final training objective is:

$$
\mathcal{L}_{\text{total}} = \frac{1}{B \cdot L} \sum_{b=1}^{B} \sum_{l=1}^{L} \mathcal{L}(x_0^{(b,l)}, x_t^{(b,l)}, \sigma^{(b)})
$$

Averaged over batch $B$ and sequence length $L$.

---

## Step-by-Step Code-to-Math Mapping

Let's trace through the implementation:

### Step 1: Model Output → Probability Distribution

```python
score = F.softmax(logits, dim=-1)  # (B, L, V)
```

$$
s_\theta(x \mid x_t, \sigma) = \frac{\exp(\text{logits}_x)}{\sum_{x'} \exp(\text{logits}_{x'})}
$$

---

### Step 2: Compute $e^{\sigma} - 1$

```python
esigm1 = torch.where(sigma < 0.5, torch.expm1(sigma), torch.exp(sigma) - 1)
```

$$
e^{\sigma} - 1 = \begin{cases}
\text{expm1}(\sigma) & \sigma < 0.5 \text{ (numerical stability)} \\
e^{\sigma} - 1 & \text{otherwise}
\end{cases}
$$

**Why `expm1`?** For small $\sigma$, computing $e^{\sigma} - 1$ directly loses precision. The function `expm1` computes this accurately.

---

### Step 3: Compute Ratio $r$

```python
ratio = 1 - vocab_size / (esigm1 + vocab_size)
```

$$
r = 1 - \frac{V}{e^{\sigma} - 1 + V} = \frac{e^{\sigma} - 1}{e^{\sigma} - 1 + V}
$$

---

### Step 4: Identify Unperturbed Positions

```python
no_perturb = (x_t == x_0)
```

$$
\mathbb{1}_{x_t = x_0}
$$

Binary indicator: 1 if no noise, 0 if noised.

---

### Step 5: Extract Score Values

```python
score_mean = reduce(score, 'b l v -> b l', 'mean')
score_at_xt = torch.gather(score, -1, x_t.unsqueeze(-1)).squeeze(-1)
score_at_x0 = torch.gather(score, -1, x_0.unsqueeze(-1)).squeeze(-1)
```

$$
\begin{align}
\bar{s} &= \frac{1}{V} \sum_{x \in \mathcal{V}} s_\theta(x \mid x_t, \sigma) \\
s_{x_t} &= s_\theta(x_t \mid x_t, \sigma) \\
s_{x_0} &= s_\theta(x_0 \mid x_t, \sigma)
\end{align}
$$

---

### Step 6: Compute Negative Term

```python
neg_term = score_mean - score_at_xt / vocab_size
neg_term = torch.where(
    no_perturb,
    ratio * neg_term,
    score_at_x0 / esigm1 + neg_term
)
```

$$
\text{neg\_term} = \begin{cases}
r \cdot \left(\bar{s} - \frac{s_{x_t}}{V}\right) & \text{if } \mathbb{1}_{x_t = x_0} = 1 \\
\frac{s_{x_0}}{e^{\sigma} - 1} + \left(\bar{s} - \frac{s_{x_t}}{V}\right) & \text{if } \mathbb{1}_{x_t = x_0} = 0
\end{cases}
$$

---

### Step 7: Compute Constant

```python
const = torch.where(
    no_perturb,
    (vocab_size - 1) / vocab_size * ratio * (ratio.log() - 1),
    ((-ratio.log() - 1) / ratio - (vocab_size - 2)) / vocab_size
)
```

$$
\text{const} = \begin{cases}
\frac{V-1}{V} \cdot r \cdot (\log r - 1) & \text{if } x_t = x_0 \\
\frac{1}{V} \left(\frac{-\log r - 1}{r} - (V - 2)\right) & \text{if } x_t \neq x_0
\end{cases}
$$

---

### Step 8: Compute Positive Term

```python
pos_term = reduce(score, 'b l v -> b l', 'mean') - score_at_xt / vocab_size
```

$$
\text{pos\_term} = \bar{s} - \frac{s_{x_t}}{V}
$$

---

### Step 9: Combine and Average

```python
loss = pos_term - neg_term + const
return reduce(loss, 'b l -> ', 'mean')
```

$$
\begin{align}
\mathcal{L}_{b,l} &= \text{pos\_term}_{b,l} - \text{neg\_term}_{b,l} + \text{const}_{b,l} \\
\mathcal{L} &= \frac{1}{B \cdot L} \sum_{b,l} \mathcal{L}_{b,l}
\end{align}
$$

---

## Intuition: What Does Each Term Do?

### Positive Term: $\bar{s} - \frac{s_{x_t}}{V}$
- Encourages **high entropy** (spread probability over vocabulary)
- Penalizes putting mass on the **noised token** $x_t$
- Forces exploration and prevents mode collapse

### Negative Term: Context-Dependent
- **If $x_t \neq x_0$**: Strongly encourages high score at **clean token** $s_{x_0}$ via $\frac{s_{x_0}}{e^{\sigma}-1}$
- **If $x_t = x_0$**: Scales by ratio $r$ (less aggressive, since no denoising needed)
- Primary **denoising signal**

### Constant Term: Normalization
- Ensures loss is **well-calibrated**
- Accounts for entropy of true distribution
- Different for perturbed vs unperturbed cases

---

## Why This Loss Works

1. **Score Matching**: Minimizes KL divergence between true and predicted score ratios
2. **Handles Discrete Space**: Works directly with categorical distributions over vocabulary
3. **Entropy Regularization**: Prevents model from being overconfident, encourages diverse predictions
4. **Masking Synergy**: Combined with masking (logits at $x_t$ set to 0), forces true denoising
5. **Case Distinction**: Properly handles both noised and unnoised positions

---

## Connection to Sampling

During generation, we use the learned score to sample:

$$
p(x_{t-\Delta t} \mid x_t, \sigma) \propto s_\theta(x_{t-\Delta t} \mid x_t, \sigma)
$$

The score ratio lets us:
1. Compute probabilities over vocabulary at each position
2. Sample new tokens by removing noise gradually
3. Generate coherent sequences by iterative denoising

---

## Summary

The Score Entropy Loss for discrete diffusion:

$$
\boxed{
\mathcal{L} = \underbrace{\left(\bar{s} - \frac{s_{x_t}}{V}\right)}_{\text{Positive}} - \underbrace{\text{Case-dependent term}}_{\text{Negative}} + \underbrace{\text{Normalization}}_{\text{Constant}}
}
$$

**Teaches the model to**:
- ✅ Predict probability ratios (score matching)
- ✅ Assign high probability to clean tokens when $x_t \neq x_0$
- ✅ Avoid assigning probability to noised tokens (with masking)
- ✅ Maintain calibrated distributions (entropy regularization)

This enables **iterative denoising in discrete space** for text generation! 🎯
