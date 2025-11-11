### modern transformer architectures swiglu, rope, rmsnrom, kvcache
* https://rohitbandaru.github.io/blog/Transformer-Design-Guide-Pt2/

### on training
* https://modal.com/gpu-glossary
* quantizationm, post training: https://www.youtube.com/watch?v=0VdNflU08yA
* distributed training: https://www.youtube.com/watch?v=toUSzwR0EV8
* karpathy gpt2: https://www.youtube.com/watch?v=l8pRSuU81PU&t=1632s&pp=ygUMZ3B0MiBzY3JhdGNo




-----------------------------------
**cs336 resource accounting lec 2 yt**
## how long to train 70B param on 15T tokens on 1024 H100s?
* total flops = 6 * 70e9 * 15e12 (6 = 2 forward, 4 backwards)
* mfu = 0.5 (model flop utilization (actual / promised FlOPs). ~ 50% efficiency)
* flops per day = h100 flop per sec * mfu * 1024 * 60 * 60 * 24
* days = total flops / flops per day

## largest num param model to train on 8x H100s using AdamW?
* h100 bytes = 80e9
* bytes per param = 4 + 4 + (4 + 4) params, grad, optimizer state
* num params = (h100 * 8) / bytes per param
