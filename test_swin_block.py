#%% 
from models.swin_transformer_poly import *
from models.swin_transformer import SwinTransformer, SwinTransformerBlock
from utils import * 
import matplotlib.pyplot as plt 

#%% 
img_size = (224, 224)
patch_size = 16
in_chans = 3
norm_layer = nn.LayerNorm 
patches_resolution = (img_size[0]//patch_size, img_size[1]//patch_size)
drop_rate = 0.0 
embed_dim = 1
mlp_ratio = 4
window_size = 7
# test poly swin transformer block
patch_embed = PolyPatch(input_resolution = img_size, patch_size = patch_size, in_chans = in_chans,
                out_chans = embed_dim, norm_layer=None).cuda()
pos_drop = nn.Dropout(p=drop_rate).cuda()
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
# self.model = nn.Sequential(patch_embed, pos_drop, blocks).cuda()

def pred(x):
    x = patch_embed(x)
    x = pos_drop(x)
    for blk in blocks:
        x = blk(x)
    return x 

def reorder(x, idx = 0):
    blk = blocks[idx]
    H, W = blk.input_resolution
    B, L, C = x.shape
    blk.grid_size = (H // blk.window_size, W // blk.window_size)
    assert L == H * W, "input feature has wrong size"
    # idx stands for the index of the block
    # x = blk.norm1(x)
    x = torch.permute(x.view(B,H,W,C), (0,3,1,2))
    # # rearrange x based on max polyphase 
    x =  PolyOrder.apply(x, blk.grid_size, to_2tuple(blk.window_size))
    x = torch.permute(x, (0,2,3,1)).contiguous()
    return x

def check_polyphase(t, t1, shifts = None):
    # t: original tensor, t1: shifted tensor
    # check if the polyphase is correct 
    # t: B, H, W, C
    # t1: B, H, W, C

    B, H, W, C = t.shape
    patch_reso = (H//7, W//7)
    
    norms = []
    for i, shift in enumerate(shifts):
        p0 = t[i:, 0::patch_reso[0] , 0::patch_reso[1], :]
        p1 = t1[i:, 0::patch_reso[0] , 0::patch_reso[1], :]
        p0_shifted = torch.roll(p0, shifts = shift, dims = (0, 1))
        norms.append(torch.linalg.norm(p0_shifted - p1))
    print(norms)
    return norms
    # B, H, W, C = t.shape
    # patch_reso = (H//7, W//7)
    # print(torch.linalg.norm(t[:, 0::patch_reso[0], 0::patch_reso[1], :]) - torch.linalg.norm(t1[:, 0::patch_reso[0], 0::patch_reso[1], :]))
    # assert torch.linalg.norm(t[:, 0::patch_reso[0], 0::patch_reso[1], :]) - torch.linalg.norm(t1[:, 0::patch_reso[0], 0::patch_reso[1], :]) < 1e-4, "polyphase is not correct"

    

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


x = torch.rand((1,3,224,224)).cuda()
B, C, H, W = x.shape
# shifts = tuple(np.random.randint(0,32,2))
shifts = (37,43)
x1 = torch.roll(x, shifts, (2,3)).cuda()
# poly swin output
print("predicting")
p = patch_embed(x)
p1 = patch_embed(x1)
print("reordering")
t = reorder(p)
t1 = reorder(p1)
shifts = find_shift2d_batch(t, t1, early_break=True)
print(shift_and_compare(t, t1, shifts, (0,1) ))
check_polyphase(t, t1, shifts)
# confirm_bijective_matches_batch(t.view(t.shape[0], -1 ,t.shape[-1]).cpu().detach().numpy(), t1.view(t.shape[0], -1 ,t.shape[-1]).cpu().detach().numpy())
t = cyclic_shift(t, 0)
t1 = cyclic_shift(t1, 0)
check_window(t, t1)
t = attention(t, 0)
t1 = attention(t1, 0)
check_window(t, t1)
t = reverse_cyclic_shift(t, p, 1, 0)
t1 = reverse_cyclic_shift(t1, p1, 1, 0)
# confirm_bijective_matches_batch(t.cpu().detach().numpy(), t1.cpu().detach().numpy())

# %%
