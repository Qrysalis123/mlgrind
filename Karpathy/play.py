
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
