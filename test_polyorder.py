#%% 
from models.poly_utils import *
import torch
import numpy as np

# %%
H,W = (6,6)
B = 1
C = 1

from torch import autograd
x = torch.randn(B, C, H, W, requires_grad=True).cuda()
class F(torch.autograd.Function):
    def forward(self, x):
        return torch.tensor((1.0,), requires_grad=True)
    def backward(self, grad_output):
        return  x
        
x[:,:,1::2,1::2] = 2
y1 = PolyOrder.apply(x, (3,3), (2,2), 2, True)
loss = torch.ones((B, C, H, W), requires_grad=True).cuda()
y = F().apply(y1)
grad=autograd.grad(y,x)[0]
# expecting a dense matrix x, which is shifted opposite to how y1 shifted
# got something with zero padding
# %%
