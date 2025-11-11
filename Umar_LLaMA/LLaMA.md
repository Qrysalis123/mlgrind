
LLaMA

* embeddings
----------------------
* skip conn 1
* RMS Norm
* ROPE of QK
* GQA + KV Cache
* + skip conn 1
* skip conn 2
* RMS Norm
* FF SwiGLU
* + skip conn 2
--------------------
* RMS Norm
* Linear
* Softmax




original transformer
dim: 512
heads: 8
n layers: 6
