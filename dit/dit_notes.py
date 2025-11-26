
"""
DiT paper
Scalable Diffusion Models with Transformers
https://arxiv.org/pdf/2212.09748

https://apxml.com/courses/advanced-diffusion-architectures


---------------traditional layer norm (no conditioning)-----------------
x_norm = (x - mean) / std
x_out = x_norm * gamma + beta
    problem:
        no timestep condition



---------------AdaLN: incorporates `timestamp` condition--------------

Adaptive Layer Normalization (AdaLN)
    AdaLN:
        # makes shift & scale depend on timestep
        # given a c condition (timestep/noise embedding)
        shift, scale, gate = MLP(c).chunk(3)

        then adaptive norm
        x_norm = LayerNorm(x)
        x_mod = x_norm * (1 + scale) + shift

        MLP

        # adaptive residual
        x = x_skip + gate * out


    Shift: moves features in embd space based on timestep
    # different noise levels may need different features centered differently

    Scale: amplify or dampen features based on timestep
    # high noise, model needs stronger feature responses, low noise gentle refinements

    Gate: control how much of the features to add to residual
    # high noise: model needs big changes, low noise: preserve most input




---------------------DiT Block------------------------
* needs additional conditional info: noise timestep. embd separately
* so introduces adaLN-Zero block

* paper did variations of architecture to see which works:
    1. in context conditioning: appending t & c  and treat them same as image tokens
    2. treat t & c separate of len 2. then cross attention.
    3. adaLN (adaptive Layer Norm): learns the mean, std instead of ~(0,1)
    4. adaLN-Zero block: adaLN with additional `scale` alpha to residual connections
        * this helps guide how much info from original is passed through to next layer
        * on the side is:
            * conditioning embd -> MLP -> carries conditional embd info to the other layers
            * like y1, B1, a1, y2, B2, a2
            (scale, shift) scale, scale, shift, scale





-------------------Scalable Diffusion Models with Transformers---------------
* transformer for spatial self attention
* adaptive norm layers (adaLN) to inject conditional info and channel counts

-------------adaLN-Zero-------------
condition embd -> MLP -> 6 * condition embd (2x scale, shift, scale)
MLP outputs 6 params
    scale, shift: applied to input tokens before MHA
    scale: applied to output of MHA
    scale, shift: applied to output of MHA
    scale: applied to output of MLP


---------------DiT: forward pass-------------------
Patchify (tokenizing):
    (3,256,256) -> z (4,32,32) -> patchify by p (T,d) -> sinusoidal positioning

DiT block:
    tokens -> transformer

    conditional embd:
        * noise tinmesteps t
        * class labels c
        * text,.. etc


    1. in-context conditioning t & c. just add 2 extra tokens ViT then remove
    2. concat t & c embd. add heads
    3. adaptive layer norm (adaLN): learn scale & shift params from t & c
    4. adaLN-Zero: zero-initializing the final batch norm scale factor
        * scale & shift
        * scaling params immediately prior to any residual conenctions

Decoder:
    * decode tokens into output noise pred
EMA of DiT weights with decay of 0.9999


------ timestep embd ----
256 dim
2 layer MLP with dim == transformers hidden size
SiLU activations

adaLN -> timestep -> SiLU linear -> 4x or 6x transformer hidden size
GELU in core transformer


c_out: shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp
init with zero to maintain training stability

"""
