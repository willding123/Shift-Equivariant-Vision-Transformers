 
import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from torch.nn.functional import affine_grid, pad, grid_sample
from torchvision.transforms.functional import crop

class PolyOrder(torch.autograd.Function):
    @staticmethod
    def forward (ctx, x, grid_size, patch_size, norm =2, use_gpu = True):
        device = "cuda" if use_gpu else "cpu"
        B, C, H, W = x.shape
        tmp = x.clone()
        tmp = tmp.view(B,C,grid_size[0], patch_size[0], grid_size[0], patch_size[1] )
        tmp = torch.permute(tmp, (0,1,2, 4,3,5))
        tmp = torch.permute(tmp.reshape(B,C, grid_size[0]**2, patch_size[0]**2), (0,1,3,2))
        tmp = torch.permute(tmp, (0,2,1,3))
        tmp = tmp.contiguous()
        norm = torch.linalg.vector_norm(tmp, dim=(2,3))
        del tmp
        idx = torch.argmax(norm, dim=1).int()
        theta = torch.zeros((B,2,3), requires_grad=False).to(device).float()
        theta[:,0, 0] = 1; theta[:,1,1] = 1;
        l = patch_size[0]-1
        x = pad(x, (0,l,0,l) ,"circular").float()
        theta[:,1,2]  = (idx/patch_size[0]).int()*2/x.shape[2]
        theta[:,0,2] = (idx%patch_size[1])*2/x.shape[3] 
        ctx.theta =  theta; ctx.l = l; ctx.H = H; ctx.W = W; ctx.B = B; ctx.C = C
        grid = affine_grid(theta, (B,C,x.shape[2], x.shape[3]), align_corners= False)
        x = grid_sample(x, grid, "nearest", align_corners= False)
        x = crop(x, 0, 0, H, W)
        return  x
    
    @staticmethod
    def backward(ctx,grad_in):
        theta = ctx.theta
        l = ctx.l;  H = ctx.H;  W =ctx.W; B = ctx.B ; C = ctx.C
        grad_in = pad(grad_in, (l,0,l,0) ,"circular").float()
        theta[:,1,2] = -theta[:,1,2]; theta[:,0,2] = -theta[:,0,2]
        grid = affine_grid(theta, (B,C,grad_in.shape[2], grad_in.shape[3]), align_corners= False)
        grad_out = grid_sample(grad_in, grid, "nearest", align_corners= False)
        grad_out = crop(grad_out, 0, 0, H, W)
        return grad_out,None, None, None, None, None, None, None



class PolyPatch(nn.Module): 
    r""" PolyPatch Layer: 
    Args: 
        input_resolution (tuple of int): input resolution i.e img_size 
        patch_size (int): patch size (number of pixels in a patch)
        in_chans (int): input dimensions 
        out_chans (int): output dimensions
        norm_layer : nn.LayerNorm if patch merging layer, None if image patching layer
    """
    def __init__(self, input_resolution, patch_size, in_chans, out_chans, norm_layer= None):
        super().__init__()
        self.input_resolution = input_resolution
        self.patch_size = to_2tuple(patch_size) 
        self.in_chans = in_chans 
        self.out_chans = out_chans
        self.grid_size = [input_resolution[0] // patch_size, input_resolution[1] // patch_size] 
        self.patches_resolution = self.grid_size
        self.num_patches = self.patches_resolution[0] * self.patches_resolution[1]

        self.proj = nn.Conv2d(in_chans, out_chans, kernel_size=patch_size, stride = patch_size, padding_mode = "circular")
        self.norm = norm_layer(out_chans) if norm_layer else nn.Identity()
    
    def forward(self, x, invariance = False):
            B, C, H, W = x.shape
            assert H == self.input_resolution[0] and W == self.input_resolution[1], \
                f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
            x = PolyOrder.apply(x, self.grid_size, self.patch_size)
            x = self.proj(x).flatten(2).transpose(1, 2)
            if self.norm is not None:
                x = self.norm(x)
            return x

    def flops(self):
        Ho, Wo = self.patches_resolution
        flops = Ho * Wo * self.out_chans * self.in_chans * (self.patch_size[0] * self.patch_size[1])
        flops += Ho * Wo * self.in_chans
        if self.norm is not None:
            flops += Ho * Wo * self.out_chans
        return flops        
        
    
      