# %%
import torch
nums =  torch.arange(1, 37).reshape(6, 6)
# %%
features_size = 6
batch_size = 1
grid_size = (3, 3)
patch_size = (2, 2)
token_size = 3

nums_arr = torch.zeros(batch_size, token_size, features_size, features_size)

for i in range(features_size):
    for j in range(features_size):
        # set token to nums[i, j]
        nums_arr[:, :, i, j] = nums[i, j]

# %%
B, C, H, W = nums_arr.shape
x = nums_arr.clone()
tmp = nums_arr.view(B,C,grid_size[0], patch_size[0], grid_size[0], patch_size[1] )
tmp = torch.permute(tmp, (0,1,2, 4,3,5))
# %%
tmp = torch.permute(tmp.reshape(B,C, grid_size[0]**2, patch_size[0]**2), (0,1,3,2))
tmp = torch.permute(tmp, (0,2,1,3))
tmp = tmp.contiguous()
# %%


# %%
norm = torch.linalg.vector_norm(tmp, dim=(2,3))
del tmp
idx = torch.argmax(norm, dim=1).int()
theta = torch.zeros((B,2,3), requires_grad=False).float()
theta[:,0, 0] = 1; theta[:,1,1] = 1;
# %%

from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from torch.nn.functional import affine_grid, pad, grid_sample
from torchvision.transforms.functional import crop

# %%

l = patch_size[0]-1
x = pad(x, (0,l,0,l) ,"circular").float()
theta[:,1,2]  = (idx/patch_size[0]).int()*2/x.shape[2]
theta[:,0,2] = (idx%patch_size[1])*2/x.shape[3] 
# ctx.theta =  theta; ctx.l = l; ctx.H = H; ctx.W = W; ctx.B = B; ctx.C = C
grid = affine_grid(theta, (B,C,x.shape[2], x.shape[3]), align_corners= False)
x = grid_sample(x, grid, "nearest", align_corners= False)
x = crop(x, 0, 0, H, W)

# %%

