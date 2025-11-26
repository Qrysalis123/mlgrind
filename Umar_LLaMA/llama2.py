import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import math
from dataclasses import dataclass
from typing import Optional


@dataclass
class LLaMAConfig:
    dim: int = 512
    n_layers: int = 32 # layers for the transformer block
    n_heads: int = 32 # Number of heads for Q
    n_kv_heads: Optional[int] = None # Number of heads for K and V
    vocab_size: int = -1 # based on tokenizer
    multiple_of: int = 256
    ffn_dim_multiplier: Optional[float] = None
    num_eps: float = 1e-5

    # KV Cache
    max_batch_size: int = 32
    max_seq_len: int = 2048

    device: str = None

class Transformer(nn.Module):
    """
    -> embd
    -> || rmsnorm -> kv cache -> rope -> rmsnorm -> ffn -> swiglu ||
    -> rmsnorm -> linear -> softmax
    """

    def __init__(self, args = LLaMAConfig):
        super().__init__()
        assert args.vocab_size != -1, "set vocab size"
        self.args = args
        self.vocab_size = args.vocab_size
        self.n_layers = args.n_layers

        self.tok_embeddings = nn.Embedding(self.vocab_size, args.dim)

        self.layers = nn.ModuleList(EncoderBlock(args) for _ in range(self.n_layers))

        self.norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.output = nn.Linear(args.dim, self.vocab_size, bias=False)

        # theta for rope
        self.freqs_complex = precompute_theta_pos_frequencies(
            self.args.dim // self.args.n_heads,
            self.args.max_seq_len * 2,
            device = self.args.device
        )

    def forward(self, tokens: torch.Tensor, start_pos: int):
        # (B, Seq_Len)
        batch_size, seq_len = tokens.shape
        assert seq_len == 1

        # (B, Seq_len) -> (B, Seq_len, Dim)
        h = self.tok_embeddings(tokens)

        # retrive RoPe (n theta)
        freqs_complex = self.freqs_complex[start_pos:start_pos + seq_len]

        for block in self.layers:
            h = block(h, start_pos, freqs_complex)
        h = self.norm(h)

        # (b, s, embd) ->  (b, s, vocab)
        output = self.output(h).float()
        return output


class RMSNorm(nn.Module):
    """
    a: Wx
    RMS(a): sqrt(a^2.mean(dim=embd))
    a_hat: a_i / RMS(a)
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor):
        """
        RMSNorm: x * 1/sqrt(x^2.mean(dim) + eps)
        """
        # rqsrt(): 1 / sqrt(x)
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)

    def forward(self, x:torch.Tensor):
        # (dim) * (b, s, dim) = (b, s, dim)
        return self.weight * self._norm(x.float()).type_as(x)


def precompute_theta_pos_frequencies(
    head_dim: int,
    seq_len: int,
    device: str,
    theta: float = 10000.0
):
    assert head_dim % 2 == 0, "head_dim must be even"

    # θ_i = 10000^(-2(i-1)/d)  ⇒  1 / (theta ** (arange(0, d, 2) / d))
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))

    # (seq_len, head_dim//2)
    t = torch.arange(seq_len, device=device)
    freqs = torch.einsum('i , j -> i j', t, inv_freq)          # outer product

    # polar form: cis(mθ) = cos(mθ) + i·sin(mθ)
    freqs_complex = torch.polar(torch.ones_like(freqs), freqs)  # (seq_len, head_dim//2)

    return freqs_complex

def apply_rotary_embeddings(x: torch.Tensor, freqs_complex:torch.Tensor, device: str):
    """
    # separate the x into pairs of complex
    (b, seq_len, h, head_dim) -> (b, seq_len, h, head_dim/2)

    # reshape freqs_complex to same shape as x
    (seq_len, head_dim / 2) -> (1, seq_len, 1, head_dim/2)

    # multiply
    (b, seq_len, h, head_dim/2) * (1, seq_len, 1, head_dim/2) = (B, seq_len, h, head_dim/2)

    # convert compelx to real
    (b, seq_len, h, head_dim/2) -> (b, seq_len, h, head_dim/2, 2)

    then reshape head_dim/2 to head_dim
    (b, seq_len, h, head_dim/2, 2) -> (b, seq_len, h, head_dim)
    """

    x_complex = torch.view_as_complex(rearrange(x.float(), '..., (d two) -> ... d two', two=2))

    freqs_complex = rearrange(freqs_complex, 's d -> 1, s, 1, d')
    x_rotated = x_complex * freqs_complex
    x_out = torch.view_as_real(x_rotated)
    x_out = rearrange(x_out, '... d two -> ... (d two)')
    return x_out.type_as(x).to(device)


def repeat_kv():
    pass



class EncoderBlock(nn.Module):
    pass



class SelfAttention(nn.Module):


class FeedForward(nn.Module):
    """
    ln_1 = (b s dim) -> (b s 4*dim)
    ln_2 = (b s 4*dim) -> (b s dim)
    ln_3 = (b s dim) -> (b s 4*dim)

    forward:
        (b s dim) -> (b s 4*dim) -> activation -> (b s dim)

        x = ln_2( activation(ln_1(x)) * ln_3(x) )

        activation using SwiGGlu
    """
    pass
