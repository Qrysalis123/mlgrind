import torch


def score_entropy(score, sigma, x, x0, vocab_size):
    """
    Official SEDD score entropy loss for uniform graph.
    Based on: https://github.com/louaaron/Score-Entropy-Discrete-Diffusion

    Args:
        score: (B, S, V) log-score predictions (logits)
        sigma: (B,) or (B, 1) noise level
        x: (B, S) perturbed/noisy tokens
        x0: (B, S) clean/target tokens
        vocab_size: int
    Returns:
        loss: (B, S)
    """
    # Ensure sigma is (B, 1) for broadcasting
    if sigma.dim() == 1:
        sigma = sigma.unsqueeze(-1)  # (B, 1)

    dim = vocab_size

    # Numerically stable exp(sigma) - 1
    esigm1 = torch.where(
        sigma < 0.5,
        torch.expm1(sigma),
        torch.exp(sigma) - 1
    )  # (B, 1)

    # Ratio term: 1 - dim / (exp(sigma) - 1 + dim)
    ratio = 1 - dim / (esigm1 + dim)  # (B, 1)

    # NEGATIVE TERM - exactly as in SEDD
    # Mean score across vocab - score at current position / dim
    neg_term = score.mean(dim=-1) - torch.gather(score, -1, x.unsqueeze(-1)).squeeze(-1) / dim  # (B, S)

    # Conditional adjustment based on whether we stayed at x0
    neg_term = torch.where(
        x == x0,
        ratio * neg_term,  # Stayed: scale by ratio
        torch.gather(score, -1, x0.unsqueeze(-1)).squeeze(-1) / esigm1 + neg_term  # Moved: add x0 score term
    )  # (B, S)

    # CONSTANT FACTOR - exactly as in SEDD with numerical stability
    # Clamp ratio to avoid log(0) or division by zero
    ratio_safe = torch.clamp(ratio, min=1e-8, max=1.0)
    const = torch.where(
        x == x0,
        (dim - 1) / dim * ratio_safe * (torch.log(ratio_safe) - 1),  # Stayed
        ((-torch.log(ratio_safe) - 1) / ratio_safe - (dim - 2)) / dim   # Moved
    )  # (B, S) after broadcasting

    # POSITIVE TERM
    # exp(score) mean - exp(score[x]) / dim
    # Clamp score to prevent overflow in exp
    score_clamped = torch.clamp(score, max=20)  # Prevent exp overflow
    sexp = score_clamped.exp()
    pos_term = sexp.mean(dim=-1) - torch.gather(sexp, -1, x.unsqueeze(-1)).squeeze(-1) / dim  # (B, S)

    loss = pos_term - neg_term + const  # (B, S)
    
    # Replace any NaN/Inf with zeros to prevent gradient corruption
    loss = torch.where(torch.isfinite(loss), loss, torch.zeros_like(loss))
    
    return loss

def uniform_noise_schedule(t, sigma_min=1e-3, sigma_max=1.0):
    """
    Uniform noise schedule for SEDD: sigma(t) = t
    For numerical stability, we actually use sigma(t) = sigma_min + t * (sigma_max - sigma_min)
    Returns (sigma, dsigma) both shape (B,)
    """
    sigma = sigma_min + t * (sigma_max - sigma_min)
    dsigma = torch.full_like(sigma, sigma_max - sigma_min)
    return sigma, dsigma


def geometric_noise_schedule(t, sigma_min=1e-3, sigma_max=6.907):
    """
    Geometric schedule: sigma(t) = sigma_min^(1-t) * sigma_max^t
    Returns (sigma, dsigma) both shape (B,)
    """
    sigma_min = torch.tensor(sigma_min, device=t.device, dtype=t.dtype)
    sigma_max = torch.tensor(sigma_max, device=t.device, dtype=t.dtype)
    sigma = sigma_min ** (1 - t) * sigma_max ** t
    dsigma = sigma * torch.log(sigma_max / sigma_min)
    return sigma, dsigma


def staggered_score(score, dsigma, vocab_size):
    """
    Staggered scaling for reverse step - SEDD uniform graph version.
    score: (B, S, V) - probability distribution (not log)
    dsigma: (B,) noise level difference
    returns (B, S, V)
    """
    dim = vocab_size
    
    # Ensure dsigma is broadcastable: (B,) -> (B, 1, 1)
    if dsigma.dim() == 1:
        dsigma = dsigma.unsqueeze(-1).unsqueeze(-1)  # (B, 1, 1)
    elif dsigma.dim() == 2:
        dsigma = dsigma.unsqueeze(-1)  # (B, S, 1) or (B, 1, 1)
    
    epow = (-dsigma).exp()  # (B, 1, 1)
    return ((epow - 1) / (dim * epow)) * score.sum(dim=-1, keepdim=True) + score / epow


def transp_transition(x_t, sigma, vocab_size):
    """
    Uniform graph transition probabilities - exactly as in SEDD.
    x_t: (B, S) current states
    sigma: (B,) or (B, 1) noise level
    returns: (B, S, V) transition probabilities
    
    SEDD formula:
    trans[i,j] = (1 - e^(-sigma)) / dim  for all j
    trans[i,i] = 0  (temporarily)
    trans[i,i] = 1 - sum(trans[i,:])  (final, ensures rows sum to 1)
    """
    # Ensure sigma has shape (B, 1) for broadcasting
    if sigma.dim() == 1:
        sigma = sigma.unsqueeze(-1)  # (B, 1)
    
    B, S = x_t.shape
    V = vocab_size
    
    # Step 1: Initialize all positions to (1 - e^(-sigma)) / V
    # sigma shape: (B, 1), need (B, S, 1) for broadcasting
    trans = torch.ones(B, S, V, device=x_t.device, dtype=sigma.dtype)
    trans = trans * (1 - (-sigma.unsqueeze(-1)).exp()) / V  # (B, S, V)
    
    # Step 2: Zero out current state positions (diagonal)
    trans = trans.scatter(-1, x_t.unsqueeze(-1), torch.zeros_like(trans))
    
    # Step 3: Set diagonal to remaining probability (ensures rows sum to 1)
    trans = trans.scatter(-1, x_t.unsqueeze(-1), 1 - trans.sum(dim=-1, keepdim=True))
    
    return trans



def sample_categorical(probs):
    """
    Gumbel-max sampling from unnormalized weights.
    probs: (B, S, V)
    returns (B, S)
    """
    logp = torch.log(probs + 1e-30)
    g = -torch.log(-torch.log(torch.rand_like(logp) + 1e-30) + 1e-30)
    return (logp + g).argmax(dim=-1)


def sample_transition(x, sigma, vocab_size):
    move_chance = 1 - (-sigma).exp()
    move_indices = torch.rand(*x.shape, device=x.device) < move_chance
    i_pert = torch.where(move_indices, torch.randint_like(x, vocab_size), x)
    return i_pert
