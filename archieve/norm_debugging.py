#%%
import numpy as np 
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch


norm = torch.nn.LayerNorm(96)
B, L, C = 4, 3136, 96

# 56 * 56 = 3136

x1 = torch.rand(B, L, C)

#%%

x2 = x1.clone().view(B, 56, 56, C)

x2 = torch.roll(x2, 1, 1)

x2 = x2.view(B, L, C)

#%%

x1_norm = norm(x1)
x2_norm = norm(x2)

#%%

import utils

#%%



# utils.confirm_bijective_matches_batch(x1_norm.detach().numpy(), x2_norm.detach().numpy())


# # %%
# norms = []
# for i in range(B):
#     for j in range(L):
#         print(torch.linalg.norm(x1_norm[i, j] - x2_norm[i, j]))
#         norms.append(torch.linalg.norm(x1_norm[i, j] - x2_norm[i, j]))
#         # if j > 10:
#         #     break

# # %%

# print(torch.mean(torch.tensor(norms)))
# print(torch.std(torch.tensor(norms)))
# %%
from models.swin_transformer_poly import *


grid_size = (8, 8)
H, W = 56, 56
x1_norm = torch.permute(x1_norm.view(B,H,W,C), (0,3,1,2))
# # rearrange x based on max polyphase 
x1_norm =  PolyOrder.apply(x1_norm, grid_size, to_2tuple(7)  , 2, False)
x1_ordered = torch.permute(x1_norm, (0,2,3,1)).contiguous()

x2_norm = torch.permute(x2_norm.view(B,H,W,C), (0,3,1,2))
x2_norm =  PolyOrder.apply(x2_norm, grid_size, to_2tuple(7)  , 2, False)
x2_ordered = torch.permute(x2_norm, (0,2,3,1)).contiguous()
# %%

x1_ordered = x1_ordered.view(B, L, C)

x2_ordered = x2_ordered.view(B, L, C)

utils.confirm_bijective_matches_batch(x1_ordered.detach().numpy(), x2_ordered.detach().numpy())

# %%