 
import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from torch.nn.functional import affine_grid, pad, grid_sample
from torchvision.transforms.functional import crop
from timm.models.twins import LocallyGroupedAttn

class PolyOrder(torch.autograd.Function):
    @staticmethod
    def forward (ctx, x, patch_size, norm =2,  invariance = False, use_gpu = True):
        device = "cuda" if use_gpu else "cpu"
        B, C, H, W = x.shape
        grid_size = (H//patch_size[0], W//patch_size[1])
        tmp = x.clone()
        tmp = tmp.view(B,C,grid_size[0], patch_size[0], grid_size[0], patch_size[1] )
        tmp = torch.permute(tmp, (0,1,2, 4,3,5))
        tmp = torch.permute(tmp.reshape(B,C, grid_size[0]**2, patch_size[0]**2), (0,1,3,2))
        tmp = torch.permute(tmp, (0,2,1,3))
        tmp = tmp.contiguous()
        norm = torch.linalg.vector_norm(tmp, dim=(2,3))
        del tmp
        idx = torch.argmax(norm, dim=1).int()
        px = (idx/patch_size[0]).int()
        py = idx%patch_size[1]
        if invariance:
        # choose the polyphase based on its idx and patch_size 
            norm1 = torch.zeros((B,grid_size[0]*grid_size[1]), requires_grad=False).to(device).float()
            ## TODO: fix bug, use tmp to find maximum norm for all images in a batch
            for i in range(B):
                norm1[i] = torch.linalg.norm(x[i,:, px[i]::patch_size[0], py[i]::patch_size[1]].permute(1,2,0).reshape(-1, C), dim=1)
            idx1 = torch.argmax(norm1, dim=1).int()
            px1 = (idx1/grid_size[0]).int()*patch_size[0]
            py1 = idx1%grid_size[1]*patch_size[1]
            px = px1 + px
            py = py1 + py
        theta = torch.zeros((B,2,3), requires_grad=False).to(device).float()
        theta[:,0, 0] = 1; theta[:,1,1] = 1
        if invariance:
            l = H-1
        else:
            l = patch_size[0]-1
        x = pad(x, (0,l,0,l) ,"circular").float()
        theta[:,1,2]  = px*2/x.shape[2]
        theta[:,0,2] = py*2/x.shape[3] 
        ctx.theta =  theta; ctx.l = l; ctx.H = H; ctx.W = W; ctx.B = B; ctx.C = C
        grid = affine_grid(theta, (B,C,x.shape[2], x.shape[3]), align_corners= False)
        x = grid_sample(x, grid, "nearest", align_corners= False)
        x = crop(x, 0, 0, H, W)
        return  x
    
    @staticmethod
    #TODO: implement invariance for backward pass
    def backward(ctx,grad_in):
        # breakpoint()
        # import pdb 
        # pdb.set_trace()
        theta = ctx.theta
        l = ctx.l;  H = ctx.H;  W =ctx.W; B = ctx.B ; C = ctx.C
        grad_in = pad(grad_in, (l,0,l,0) ,"circular").float()
        theta[:,1,2] = -theta[:,1,2]; theta[:,0,2] = -theta[:,0,2]
        grid = affine_grid(theta, (B,C,grad_in.shape[2], grad_in.shape[3]), align_corners= False)
        grad_out = grid_sample(grad_in, grid, "nearest", align_corners= False)
        grad_out = crop(grad_out, l, l, H, W)
        return grad_out,None, None, None, None, None, None, None


class PolyOrderModule(nn.Module):
    def __init__(self, patch_size, norm =2, invariance = False, use_gpu = True):
        super().__init__()
        self.patch_size = patch_size
        self.norm = norm
        self.use_gpu = use_gpu
        self.invariance = invariance
    def forward(self, x):
        grid_size = (x.shape[2]//self.patch_size[0], x.shape[3]//self.patch_size[1])
        return PolyOrder.apply(x, self.patch_size, self.norm, self.invariance, self.use_gpu)

class PolyPatchEmbed(nn.Module): 
    r""" PolyPatchEmbed Layer: 
    Args: 
        input_resolution (tuple of int): input resolution i.e img_size 
        patch_size (int): patch size (number of pixels in a patch)
        in_chans (int): input dimensions 
        out_chans (int): output dimensions
        norm_layer : nn.LayerNorm if patch merging layer, None if image patching layer
    """
    def __init__(self, patch_size, in_chans, out_chans, norm_layer= None, return_size = False):
        super().__init__()
        self.patch_size = to_2tuple(patch_size) 
        self.in_chans = in_chans 
        self.out_chans = out_chans
        self.proj = nn.Conv2d(in_chans, out_chans, kernel_size=patch_size, stride = patch_size, padding_mode = "circular")
        self.norm = norm_layer(out_chans) if norm_layer else nn.Identity()
        self.return_size = return_size

    def forward(self, x, invariance = False):
            B, C, H, W = x.shape
            input_resolution = (H,W) 
            self.grid_size = [input_resolution[0] // self.patch_size[0], input_resolution[1] // self.patch_size[1]] 
            self.patches_resolution = self.grid_size
            x = PolyOrder.apply(x, self.patch_size)
            x = self.proj(x).flatten(2).transpose(1, 2)
            if self.norm is not None:
                x = self.norm(x)
            if self.return_size:
                return x, self.patches_resolution
            return x

    def flops(self):
        Ho, Wo = self.patches_resolution
        flops = Ho * Wo * self.out_chans * self.in_chans * (self.patch_size[0] * self.patch_size[1])
        flops += Ho * Wo * self.in_chans
        if self.norm is not None:
            flops += Ho * Wo * self.out_chans
        return flops        
        
def copy_model_weights(swin_model: torch.nn.Module, swin_poly_model: torch.nn.Module) -> torch.nn.Module:
    # loop only over swin_poly_model keys
    weights_copied = 0
    for k, v in swin_poly_model.state_dict().items():
        if k in swin_model.state_dict().keys():
            # if shapes match
            if v.shape == swin_model.state_dict()[k].shape:
                # if the key is in swin_model_keys, copy the weights
                swin_poly_model.state_dict()[k].copy_(swin_model.state_dict()[k])
                weights_copied += 1
                if k == "patch_embed.proj.weight":
                    print("k: {}".format(k))
                    print("Weights in swin_model")
                    print(swin_model.state_dict()[k])
                    print("Weights in swin_poly_model")
                    print(swin_poly_model.state_dict()[k])

    print("weights_copied: {}".format(weights_copied))
    return swin_poly_model

class PosConv(nn.Module):
    # PEG  from https://arxiv.org/abs/2102.10882
    def __init__(self, in_chans, embed_dim=768, stride=1):
        super(PosConv, self).__init__()
        self.proj = nn.Sequential(nn.Conv2d(in_chans, embed_dim, 3, stride, 1, bias=True, groups=embed_dim, padding_mode='circular'), )
        self.stride = stride

    def forward(self, x):
        B, N, C = x.shape
        cnn_feat_token = x.transpose(1, 2).view(B, C, -1)
        x = self.proj(cnn_feat_token)
        if self.stride == 1:
            x += cnn_feat_token
        x = x.flatten(2).transpose(1, 2)
        return x


def arrange_polyphases(x, patch_size):
    B, C, H, W = x.shape
    grid_size = (H//patch_size[0], W//patch_size[1])
    tmp = x.clone()
    tmp = tmp.view(B,C,grid_size[0], patch_size[0], grid_size[0], patch_size[1] )
    tmp = torch.permute(tmp, (0,1,2, 4,3,5))
    tmp = torch.permute(tmp.reshape(B,C, grid_size[0]**2, patch_size[0]**2), (0,1,3,2))
    tmp = torch.permute(tmp, (0,2,1,3))
    tmp = tmp.contiguous()
    norm = torch.linalg.vector_norm(tmp, dim=(2,3))
    return tmp, norm 



def debug_polyphase(x, patch_size):
    _, n = arrange_polyphases(x, patch_size)
    try:
        assert torch.topk(n, 2).values[0][0] != torch.topk(n, 2).values[0][1]
    except:
        print("polyphases are not unique")
        breakpoint()

debugger = {}