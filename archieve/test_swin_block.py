#%% 
##### Jupyter Notebook for testing Swin Transformer
### Run first cell to set up the environment

# import libraries 
from models.swin_transformer_poly import *
from utils import * 
import matplotlib.pyplot as plt 
import os, sys
kernel_path = os.path.abspath(os.path.join("."))
from kernels.window_process.window_process import WindowProcess, WindowProcessReverse

np.random.seed(123)
torch.manual_seed(123)
# model configuration and input setup for the following tests 
img_size = (224, 224)
patch_size = 4
in_chans = 3
norm_layer = nn.LayerNorm 
patches_resolution = (img_size[0]//patch_size, img_size[1]//patch_size)
drop_rate = 0.0 
attn_drop_rate = 0.0
embed_dim = 96
mlp_ratio = 4
window_size = 7
depths = [2,2]
drop_path_rate = 0.1
num_heads = [3,6]
qkv_bias=True 
qk_scale=None
use_checkpoint=False
fused_window_process = True
# Patch embedding
patch_embed = PolyPatch(input_resolution = img_size, patch_size = patch_size, in_chans = in_chans,
                out_chans = embed_dim, norm_layer=norm_layer).cuda()
# Positional drop embedding (enabled only in training)
pos_drop = nn.Dropout(p=drop_rate).cuda()
# Two successive swin blocks
blocks = nn.ModuleList([
    SwinTransformerBlock(dim=embed_dim, input_resolution=(patches_resolution[0], patches_resolution[1]),
                            num_heads=1, window_size=window_size,
                            shift_size=0 if (i % 2 == 0) else window_size // 2,
                            mlp_ratio=mlp_ratio,
                            qkv_bias=True, qk_scale=None,
                            drop=0.0, attn_drop=0.0,
                            drop_path=0,
                            norm_layer=norm_layer,
                            fused_window_process=True)
    for i in range(2)]).cuda()

# some utility functions specific to this notebook 

# forward function for model (embedding + positional drop + swin blocks)
def pred(x):
    x = patch_embed(x) 
    x = pos_drop(x)
    for blk in blocks:
        x = blk(x)
    return x 

# polyphase reordering function inside the swin block forward function
def reorder(x, idx = 0):
    blk = blocks[idx]
    H, W = blk.input_resolution
    B, L, C = x.shape
    blk.grid_size = (H // blk.window_size, W // blk.window_size)
    assert L == H * W, "input feature has wrong size"
    # idx stands for the index of the block
    x = torch.permute(x.view(B,H,W,C), (0,3,1,2))
    # # rearrange x based on max polyphase 
    x =  PolyOrder.apply(x, blk.grid_size, to_2tuple(blk.window_size))
    x = torch.permute(x, (0,2,3,1)).contiguous()
    return x

# TODO: FIX THIS FUNCTION 
# unit test function for polyphase reordering
def check_polyphase(t, t1, shifts = None):
    # t: original tensor, t1: shifted tensor
    # check if the polyphase is correct 
    # t: B, H, W, C
    # t1: B, H, W, C

    B, H, W, C = t.shape
    patch_reso = (H//7, W//7)
    
    for i, shift in enumerate(shifts):
        p0 = t[i, 0::patch_reso[0] , 0::patch_reso[1], :]
        p1 = t1[i, 0::patch_reso[0] , 0::patch_reso[1], :]
        p0_shifted = torch.roll(p1, shifts = shift, dims = (0, 1))
        assert torch.linalg.norm(p0_shifted - p1) < 1e-4, "polyphase is not correct"
    

def cyclic_shift(x, idx = 0):
    blk = blocks[idx] 
    B, H, W, C = x.shape
    L = H*W
        # cyclic shift
    if blk.shift_size > 0:
        if not blk.fused_window_process:
            shifted_x = torch.roll(x, shifts=(-blk.shift_size, -blk.shift_size), dims=(1, 2))
            # partition windows
            x_windows = window_partition(shifted_x, blk.window_size)  # nW*B, window_size, window_size, C
        else:
            x_windows = WindowProcess.apply(x, B, H, W, C, -blk.shift_size, blk.window_size)
    else:
        shifted_x = x
        # partition windows
        x_windows = window_partition(shifted_x, blk.window_size)  # nW*B, window_size, window_size, C
    
    x_windows = x_windows.view(-1, window_size * window_size, C)  # nW*B, window_size*window_size, C

    return x_windows

def check_window(t, t1):
    count = 0 
    for i in range(t.shape[0]):
        for j in range(t1.shape[0]):
            if torch.linalg.norm(t[i]-t1[j]) == 0:
                count += 1
    print(f"count: {count}")
    assert count  == t.shape[0]

def attention(x_windows, idx = 0):
    C = x_windows.shape[-1]
    blk = blocks[idx]
    attn_windows = blk.attn(x_windows, mask=blk.attn_mask)  # nW*B, window_size*window_size, C
    attn_windows = attn_windows.view(-1, blk.window_size, blk.window_size, C)
    attn_windows = blk.norm1(attn_windows)
    return attn_windows

def reverse_cyclic_shift(attn_windows, shortcut, B, idx = 0):            
    blk = blocks[idx]
    H, W = blk.input_resolution
    C = attn_windows.shape[-1]
    B = B
    # reverse cyclic shift
    if blk.shift_size > 0:
        if not blk.fused_window_process:
            shifted_x = window_reverse(attn_windows, blk.window_size, H, W)  # B H' W' C
            x = torch.roll(shifted_x, shifts=(blk.shift_size, blk.shift_size), dims=(1, 2))
        else:
            x = WindowProcessReverse.apply(attn_windows, B, H, W, C, blk.shift_size, blk.window_size)
    else:
        shifted_x = window_reverse(attn_windows, blk.window_size, H, W)  # B H' W' C
        x = shifted_x
    x = x.view(B, H * W, C)
    x = shortcut + blk.drop_path(x)
    return x

def mlp(x, idx = 0):
    blk = blocks[idx]
    x = blk.mlp(blk.norm2(x)) + x 
    return x

#%% 
### unit test on swin transformer block 
x = torch.rand((1,3,224,224)).cuda()
B, C, H, W = x.shape
num_test = 10
# shifts = (30,13)
for i in range(num_test):
    shifts = tuple(np.random.randint(0,32,2))
    x1 = torch.roll(x, shifts, (2,3)).cuda()
    t = pred(x)
    t1 = pred(x1)
    confirm_bijective_matches_batch(t.cpu().detach().numpy(), t1.cpu().detach().numpy())

#%%
### test individual components of two successive swin blocks: 
##     reorder, cyclic shift, attention, reverse_cyclic_shift, mlp

# setup input and get tokens
x = torch.rand((1,3,224,224)).cuda()
B, C, H, W = x.shape
# shifts = tuple(np.random.randint(0,32,2))
shifts = (30,13)
x1 = torch.roll(x, shifts, (2,3)).cuda()
p = patch_embed(x)
p1 = patch_embed(x1)
# swin block l
t = reorder(p)
t1 = reorder(p1)
shortcut = t.view(1, -1, embed_dim); shortcut1 = t1.view(1, -1, embed_dim) 
shifts = find_shift2d_batch(t, t1, early_break=True)
print(shift_and_compare(t, t1, shifts, (0,1) )) 
# check_polyphase(t, t1, shifts) # confirm polyphase is correct
# confirm_bijective_matches_batch(t.view(t.shape[0], -1 ,t.shape[-1]).cpu().detach().numpy(), t1.view(t.shape[0], -1 ,t.shape[-1]).cpu().detach().numpy())
t = cyclic_shift(t, 0)
t1 = cyclic_shift(t1, 0)
check_window(t, t1)
t = attention(t, 0)
t1 = attention(t1, 0)
check_window(t, t1)
t = reverse_cyclic_shift(t, shortcut, 1, 0)
t1 = reverse_cyclic_shift(t1, shortcut1, 1, 0)
confirm_bijective_matches_batch(t.cpu().detach().numpy(), t1.cpu().detach().numpy())
t = mlp(t); t1 = mlp(t1)
confirm_bijective_matches_batch(t.cpu().detach().numpy(), t1.cpu().detach().numpy())
# test block l+1 
t = reorder(t)
t1 = reorder(t1)
shortcut = t.view(1,-1,embed_dim); shortcut1 = t1.view(1,-1,embed_dim)
shifts = find_shift2d_batch(t, t1, early_break=True)
print(shift_and_compare(t, t1, shifts, (0,1) ))
# check_polyphase(t, t1, shifts)
t = cyclic_shift(t, 1)
t1 = cyclic_shift(t1, 1)
check_window(t, t1)
t = attention(t, 1)
t1 = attention(t1, 1)
check_window(t, t1)
t = reverse_cyclic_shift(t, shortcut, 1, 1)
t1 = reverse_cyclic_shift(t1, shortcut1, 1, 1)
confirm_bijective_matches_batch(t.cpu().detach().numpy(), t1.cpu().detach().numpy())
t = mlp(t); t1 = mlp(t1)
confirm_bijective_matches_batch(t.cpu().detach().numpy(), t1.cpu().detach().numpy())
print("Done")
#%% 
### unit test on one swin transformer stage - BasicLayer
x = torch.rand((1,3,224,224)).cuda()
B, C, H, W = x.shape
shifts = (30,13)
x1 = torch.roll(x, shifts, (2,3)).cuda()
patch_embed = PolyPatch(input_resolution = img_size, patch_size = 4, in_chans = in_chans,
                out_chans = 96, norm_layer=None).cuda()
layer = BasicLayer(dim = 96, input_resolution= (56,56), depth=2, num_heads=3, window_size=7).cuda()

def pred_l(x):
    x = patch_embed(x)
    x = layer(x)
    return x

t = pred_l(x)
t1 = pred_l(x1)
confirm_bijective_matches_batch(t.cpu().detach().numpy(), t1.cpu().detach().numpy())

# %% 
### unittest multi-stage swin transformer  
x = torch.rand((1,3,224,224)).cuda()
shifts = tuple(np.random.randint(0,224,2))
x1 = torch.roll(x, shifts, (2,3)).cuda()
num_layers = 2
dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
layers = nn.ModuleList()
for i_layer in range(num_layers):
    layer = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                        input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                            patches_resolution[1] // (2 ** i_layer)),
                        depth=depths[i_layer],
                        num_heads=num_heads[i_layer],
                        window_size=window_size,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=True, qk_scale=None,
                        drop=drop_rate, attn_drop=attn_drop_rate,
                        drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                        norm_layer=norm_layer,
                        downsample=PolyPatch if (i_layer < num_layers - 1) else None,
                        use_checkpoint=False,
                        fused_window_process=fused_window_process)
    layers.append(layer)
layers = layers.cuda()
def pred_ls(x):
    x = patch_embed(x)
    for layer in layers:
        x = layer(x)
    return x
t = pred_ls(x)
t1 = pred_ls(x1)
confirm_bijective_matches_batch(t.cpu().detach().numpy(), t1.cpu().detach().numpy())

#%% 
### test entire swin model 
x = torch.rand((1,3,224,224)).cuda()
num_test = 1000
model = PolySwin(img_size=img_size, norm_layer=nn.Identity).cuda()
model.eval()
for i in range(num_test):
    shifts = tuple(np.random.randint(0,111,2))
    x1 = torch.roll(x, shifts, (2,3)).cuda()
    t = model(x)
    t1 = model(x1)
    try:
        assert torch.argmax(t).item() == torch.argmax(t1).item()
    except:
        print("Error")
        print(shifts)
        print(torch.argmax(t).item())
        print(torch.argmax(t1).item())
        break
print("Done")

# %% 

