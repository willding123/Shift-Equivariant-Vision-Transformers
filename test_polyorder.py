#%% 
from models.poly_utils import *
import torch
import numpy as np
from torch.nn.functional import pad
# %%
H,W = (224,224)
B = 4
C = 3

from torch import autograd

for i in range(100):
    x = torch.randn(B, C, H, W, requires_grad=True).cuda()
    idx = np.random.randint(0,7, size=(2,))
    x[:,:,idx[0]::7,idx[1]::7] = 2
    y1 = PolyOrder.apply(x, (32,32), (7,7), 2, True)

    class F(torch.autograd.Function):
        def forward(self, x):
            return torch.tensor((1.0,), requires_grad=True)
        def backward(self, grad_output):
            return  y1

    y = F().apply(y1)
    grad=autograd.grad(y,x)[0].cuda()
    assert (grad-x).sum() == 0 
# expecting a dense matrix x, which is shifted opposite to how y1 shifted
# got something with zero padding

# %%
