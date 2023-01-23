# %%
import torch
B = 4
C = 1
feature_size = 56
nums = list(range(1, feature_size**2 + 1))

x = torch.zeros(B, C, feature_size, feature_size)
for i in range(B):
    for j in range(C):
        x[i, j] = torch.tensor(nums).reshape(feature_size, feature_size)

window_size = (7, 7)
window_resolution = (8, 8)

x1 = x.clone()

x2 = x.clone()

x1 = torch.roll(x1, shifts=(-1,-1), dims=(2,3))
x1_rolled = x1.clone()
x2 = torch.roll(x2, shifts=(-3,-1), dims=(2,3))
x2_rolled = x2.clone()
# %%

from models.swin_transformer_poly import window_partition
# %%
x1 = torch.permute(x1, (0, 2, 3, 1))
x2 = torch.permute(x2, (0, 2, 3, 1))

# %%
B, H, W, C = x1.shape

# %%
x1 = x1.view(B, H // window_size[0], window_size[0], W // window_size[0], window_size[0], C)

# %%
w1 = x1.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[0], C)

# %% 
x2 = x2.view(B, H // window_size[0], window_size[0], W // window_size[0], window_size[0], C)
w2 = x2.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[0], C)

# %% 
