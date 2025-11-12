"""
https://einops.rocks/2-einops-for-deep-learning/

tranposing, view:
        rearrange

        x = (2,3,2,2)
        x.view(x.shape[0], -1)

        einops:
        rearrange(x, 'b c h w -> b (c h w)')

        output = (2,3*2*2) = (2,12)

mean, sum, max...:
        reduce

copy along new axis
        repeat

stack/split/concat
        pack & unpack

"""

from einops import rearrange, reduce
import numpy as np
import torch



x = np.random.RandomState(42).normal(size=[10, 32, 100, 200])
x = torch.from_numpy(x)
x.requires_grad = True



a = rearrange(x, 'b c h w -> b h w c')
a.shape


y0 = x
y1=reduce(y0, 'b c h w -> b c', 'max') # `reduce` axis
y1.shape

y2=rearrange(y1, 'b c -> c b')
y2.shape

y3=reduce(y2, 'c b -> ', 'sum')
y3

y3.backward()

print(reduce(x.grad, 'b c h w -> ', 'sum'))
