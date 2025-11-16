
'''
p1:
    building the model
    running untrained model

p2:
    add the loss function
    optimizer
    data loader

p3:
    gpu training, flash attetnion, quantization,
    hyperparams, scaling laws,

p4:
    grad accum

p5:
    distributed data parallel

p6:
    validation, evaluation

----------------------------------------------



* bpe
* 124M 12 Layers, 768 dim
* layer norm moved to input of each sub-block
* vocab 50257
* context 1024
* batch 512


softmax
linear

-------BLOCK---------
+ skip
mlp
ln 2
skip

+ skip
mha
ln 1
skip
-------BLOCK----------
+ pos
emb


'''


from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
from einops import rearrange
import math
import tiktoken

@dataclass
class GPTConfig:
    block_size: int = 1024 # max seq len
    vocab_size: int = 50257 # vocab size 50257 (50000 bpe + 256 bytes + 1 <eos>)
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        '''
        hf schema:
            wte: (vocab, emb) token embd
            wpe: (seq, emb) pos embd
            h: num heads
            ln_f: layer norm
            lm_head: (embd, vocab_size) the output clasifier
        '''

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)


    def forward(self, idx):
        # idx (B, T)
        B, T = idx.size()
        assert T <= self.config.block_size, "tooooooooooo big"

        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        pos_emb = self.transformer.wpe(pos) # (T) -> (B, T, n_embd)
        tok_emb = self.transformer.wte(idx) # (B, T) -> (B, T, n_embd)
        x = tok_emb + pos_emb

        for block in self.transformer.h:
            x = block(x) # (B, T, n_embd)

        x = self.transformer.ln_f(x)
        logits = self.lm_head(x) # (B,T,n_embd) -> (B,T,vocab_size)
        return logits



    # ===========================================================
    @classmethod
    def from_pretrained(cls, model_type):
        """Loads pretrained GPT-2 model weights from huggingface"""
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model
    # ===========================================================



class Block(nn.Module):
    """
    + skip
    mlp
    ln 2
    skip

    + skip
    mha
    ln 1
    skip
    """

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        '''
        MLP/FFN
        x -> c_fc gelu -> c_proj
        (b, t, embd) -> (b, t, 4 * embd) -> (b, t, embd)
        '''
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)

    def forward(self, x):
        return self.c_proj(self.gelu(self.c_fc(x)))


class CausalSelfAttention(nn.Module):
    '''
    MHA

    x = (b, seq, n_embd)

    init:
    d_k = n_embd / n_head
    c_attn = (n_embd, 3 * n_embd)
    c_proj = (n_embd, n_embd)

    forward:
        - qkv = c_attn(x)
        - (b, seq, n_embd) -> (b, seq, 3 * n_embd)

        - q,k,v = qkv.split(n_embd, dim=2)
        - (b, seq, 3 * n_embd) -> (b, seq, n_embd)

        - split q,k,v into nheads
        - (b, s, d) -> (b, nh, s, dk)

        - attn q @ k.T
        - (b, nh, s, dk) @ (b, nh, s, dk) -> (b, nh, s, s) / sqrt(dk)

        - mask -> softmax -> @ v
        - (b, nh, s, s) @ (b, nh, s, dk) -> (b, nh, s, dk)

        - reassemble to nh * dk
        - (b, nh, s, dk) -> (b, s, nh * dk)

        - proj (b, s, n_embd) -> (b, s, n_embd)
        - c_proj(y)
    '''

    def __init__(self, config):
        super().__init__()
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)

        self.n_head = config.n_head
        self.n_embd = config.n_embd

        assert config.n_embd % config.n_head == 0
        self.dk = config.n_embd // config.n_head # // for int instead of float

        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
            .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        b, s, d = x.size()

        qkv = self.c_attn(x) # (b, s, embd) -> (b, s, 3 * embd)
        q,k,v = qkv.split(self.n_embd, dim=2)

        q = rearrange(q, 'b s (h dk) -> b h s dk', h=self.n_head, dk=self.dk)
        k = rearrange(k, 'b s (h dk) -> b h s dk', h=self.n_head, dk=self.dk)
        v = rearrange(v, 'b s (h dk) -> b h s dk', h=self.n_head, dk=self.dk)

        # ===replace this with flash attention===
        att = torch.einsum('bhsd,bhtd->bhst', q, k) / math.sqrt(self.dk)
        att = att.masked_fill(self.bias[:, :, :s, :s] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        y = torch.einsum('bhst,bhtd->bhsd', att, v)
        # ===replace this with flash attention===

        y = rearrange(y, 'b h s dk -> b s (h dk)')
        y = self.c_proj(y)

        return y # (b, s, embd)






# ----------------------------------------------------------------
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"using device: {device}")


# Get batch data (b * s) -> (b, s) -> shift x = [:-1], y = [1:]
enc = tiktoken.get_encoding('gpt2')
with open('input.txt', 'r') as f:
    text = f.read()
text = text[:1000]
tokens = enc.encode(text)

B, T = 4, 32
buf = torch.tensor(tokens[:B*T+1])
buf.size()
x = rearrange(buf[:-1], '(B T) -> B T', B=B)
y = rearrange(buf[1:], '(B T) -> B T', B=B)


# model = GPT.from_pretrained('gpt2')
model = GPT(GPTConfig())
model.to(device)
logits = model(x)
print(logits.shape) # (B, T, vocab_size)

# count params
total_params = sum(p.numel() for p in model.parameters())
print(f"Total params: {total_params:,}")


# Testing inference
num_return_sequences = 1
max_length = 30
# prefix tokens
tokens = enc.encode("hello, i'm a language model,")
tokens = torch.tensor(tokens, dtype=torch.long) # (t,)
tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1) # (B, T)
x = tokens.to(device) # (B, T)

# Print the initial prompt
print(enc.decode(tokens[0].tolist()), end='', flush=True)

while x.size(1) < max_length:
    with torch.no_grad():
        logits = model(x) # (B, T, vocab_size)
        logits = logits[:, -1, :] # last vocab size (B, vocab_size)
        probs = F.softmax(logits, dim=-1)
        topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)  # (B, 50)
        ix = torch.multinomial(topk_probs, 1) # (B, 1)
        xcol = torch.gather(topk_indices, 1, ix) # (B, 1)
        x = torch.cat((x, xcol), dim=1) # append to sequence

        # Print only the new token
        new_token = xcol[0].item()
        decoded_token = enc.decode([new_token])
        print(decoded_token, end='', flush=True)

print()  # New line at the end
