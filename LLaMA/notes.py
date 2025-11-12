


"""
LLaMA explained:
    https://www.youtube.com/watch?v=Mn_9W1nCFLo&t=464s

Modern Architectures:
    https://rohitbandaru.github.io/blog/Transformer-Design-Guide-Pt2/#normalization


kv cache
rope
rms norm
gqa
swiglu


- embeddings
---------------------
    conn 1
- rms norm
- rope on QK, not on V
- GQA with kv cache
- (+) conn 1
- rms norm
    conn 2
- rms norm
- ff swiglu
- (+) conn 2
---------------------
- rms norm
- linear
- softmax




RMSNorm:
layer norm internal covariate shift. stabilize
layer normalize by  data items.
batch norm - by features

hypothesis: rescaling > recentering invariance
so RMSNorm focus on rescaling. rescales the features

root mean sqr = x/(x^2 / n)**.5
no mean & stddev needed so less computation

ROPE:
can find g a inner product that depends on:
    1. x_m, x_n  embedding of 2 tokens
    2. relative distance between them (n - m)
    and no otehr info

    rotation x vector x token emb
math mathing maths

SwiGLU:
    gated
    adds another linear trans on Vx acting as gating function


"""
