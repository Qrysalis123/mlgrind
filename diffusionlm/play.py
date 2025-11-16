
"""
positionless tokens
gpt can actually take shuffled tokens -> decode -> reorganize
and the text makes sense kinda.

so can learn content first? context last?

(b t) -> (b t c) -> (b h t d) -> (b h t t) -> mask -> (b h t d) -> (b t c) -> (b t v)

(B, t, embd)

positionless loss: cross entropy (B, T, vocab_size) (B, T, vocab_size)

content_loss = F.cross_entropy(
        rearrange(content_logits, 'b t v -> (b t) v'),
        rearrange(content_targets, 'b t v -> (b t) v')
        )


full loss: cross entropy (B, seq_len, vocab_size) (B, vocab_size)

context = F.cross_entropy(
        rearrange(logits, 'b t c -> (b t) c'),
        rearrange(targets, 'b t -> (b t)')
        )


Prompt -> learn dist of tokens for content -> learn to place the tokens in correct position for context

input (B, tokens) -> content dist (B, vocab_size) + previous content dist (B, vocab_size) -> context (B, seq_len, vocab_size)

decaying loss on (B, vocab_size) # the distributon of tokens
increasing loss on (B, seq, T) # the actual tokens

early timesteps: no punishment for generating out of context words. punished for wrong content.
final timestep: punished for wrong context. assumes correct content and just fills up function words for structure/syntax/grammar

# [mask] [expand] [merge] [delete]


T steps
seq len

masking: 10% (includes special tokens)
unmasking: 30%

example:
t:      100% masked
t-1:    70% masked -> remask -> 80% masked
t-2:    50% masked -> remask -> 60% masked
t-3:    30% masked -> remask -> 40% masked
t-4:    10% masked -> remask -> 20% masked
t-5:    0% masked -> done


dont need to shuffle text, just use increasing context loss
decreasing content loss.

* increasing content word masking
* decreasing function word masking
    * generate content words first. so mask content words later.
    * easier to generate function words in the end once content is visible





# linear unmasking
(T - t) / T

# cosine unmasking



"""

import matplotlib.pyplot as plt
import numpy as np

T = 1000
t = np.arange(1000)

masking = (T -t)/T
plt.plot(masking)
