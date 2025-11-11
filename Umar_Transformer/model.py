

import torch
import torch.nn as nn
import math



"""
old. with encoder


* input emb
* (+) pos emb
    * skip conn 1
* MHA
* layer norm
    * (+) skip conn 1
    * skip conn 2
* ff
* layer norm
    * (+) skip conn 2

"""



class InputEmbeddings(nn.Module):
    def __init__(self, d_model:int, vocab_size:int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model:int, seq_len:int, dropout:float) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        # pe (seq_len, d_model)
        pe = torch.zeros(seq_len, d_model)

        # create the positional values
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1) # (seq_len) -> (seq_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)

        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False)
        return self.dropout(x)

class LayerNorm(nn.Module):
    """
    (B, S, D)

    layer norm = norm by D
    x - xmu / stdd + epsilon

    * gamma + beta
    """

    def __init__(self, eps: float = 10e-6) -> None:
        super().__init__()
        self.eps = eps # to prevent too small numbers, avoid div by 0
        self.alpha = nn.Parameter(torch.ones(1)) # mul
        self.bias = nn.Parameter(torch.zeros(1)) # add

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.alpha * (x - mean) / (std + self.eps) + self.bias


class FeedForwardBlock(nn.Module):
    """
    max(0, xW1 + b1) W2 + b2

    d_model = 512
    d_ff = 2048
    """
    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff) # W1 & b1 bias is already by default True
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model) # W2 & b2

    def forward(self, x):
        # (b, s, d) -> (b, s, d_ff) -> (b, s, d)
        return self.linear_2(self.dropout(self.linear_1(x)))

class MHA(nn.Module):

    def __init__(self, d_model:int, h:int, dropout:float) -> None:
        super().__init__()
        self.d_model=d_model,
        self.h=h
        assert d_model % h == 0, "d_model must be divisible by h"

        self.d_k = d_model // h

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        d_k = query.shape[-1]

        # (b, h, s, d) -> (b, h, s, s)
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            attention_scores.masked_fill_(mask == 0, -1e9)
        attention_scores = attention_scores.softmax(dim = -1) # (b, h, s, s)
        if dropout is not None:
            attention_scores = dropout(attention_scores)

        # return attention scores for visualization words attending
        # (b, h, s, d_k), (b, h, s, s)
        return (attention_scores @ value), attention_scores


    def forward(self, q, k, v, mask):
        query = self.w_q(q) # (b, s, d) -> (b, s, d)
        key = self.w_k(k) # (b, s, d) -> (b, s, d)
        value = self.w_v(v) # (b, s, d) -> (b, s, d)

        # split into d_k:
        # (b, s, d) -> (b, s, h, d_k) then transpose -> (b, h, s, d_k)
        # so each head looks at (s, d_k)
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)

        # (b, h, s, d_k), (b, h, s, s)
        x, self.attention_scores = MHA.attention(query, key, mask, self.drouput)

        # transpose then concat: (b, h, s, d_k) -> (b, s, h, d_k) -> (b, s, d_model)
        x.transpose(1,2).contiguous().view(x.shape[0], -1, self.h * self.d_k)

        # (b, s, d_model) -> (b, s, d_model)
        return self.w_o(x)


class ResidualConnection(nn.Module):





class TransformerBlock(nn.Module):
    pass
