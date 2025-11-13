
# starting fro mthe top
# loading the hf gpt2 weights
from transformers import GPT2LMHeadModel


# 124M model
model_hf = GPT2LMHeadModel.from_pretrained('gpt2')

# raw tensors of the file
sd_hf = model_hf.state_dict()

'''
wte: token emb (vocab, emb) (50257, 768)
wpe: posit emb (pos, emb)   (1024, 768)

'''

for k, v in sd_hf.items():
    print(k, v.shape)


from transformers import pipeline
gen = pipeline('text-generation', model='gpt2')
gen('bark bark bark woof woof ', max_length=10, num_return_sequences=3)





"""
data batches

(batch * seq len) -> (batch, seq len) -> shift for x, y

"""

with open('input.txt', 'r') as f:
    text = f.read()
data = text[:1000]
print(data[:50])

import tiktoken
enc = tiktoken.get_encoding('gpt2')
tokens = enc.encode(data)


import torch
from einops import rearrange

buf = torch.tensor(tokens[:24+1])
x = rearrange(buf[:-1], '(b t) -> b t', b=4)
y = rearrange(buf[1:], '(b t) -> b t', b=4)

print(x)
print(y)
