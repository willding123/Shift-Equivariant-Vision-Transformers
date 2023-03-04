#%% 
import torch 
from models.vision_transformer import * 
import numpy as np 
from timm.models.layers import PatchEmbed, Mlp, DropPath, trunc_normal_, lecun_normal_
import timm
from models.poly_utils import *
from config import _C
from models.build import build_model
import matplotlib.pyplot as plt

#%% 
x = torch.rand((1,3,224,224)).cuda()
num_test = 10000
# config  = _C.clone()
# config.MODEL.TYPE = "vit_poly_base"
# config.MODEL.PRETRAIN_PATH = "/home/pding/scratch.cmsc663/poly_vit_base_0227/default/ckpt_epoch_173.pth"
# config.MODEL.PRETRAIN_PATH = "/home/pding/scratch.cmsc663/poly_vit_small_0228/default/ckpt_epoch_299.pth"
# model = build_model(config, is_pretrain=True).cuda()
model = VisionTransformer(embed_layer=PatchEmbed ,weight_init = 'skip').cuda()
model = nn.Sequential(
PolyOrderModule(grid_size=(14,14), patch_size=(16,16)),
model).cuda()
model.eval()
#%%
for i in range(num_test):
    shifts = tuple(np.random.randint(0,16,2))
    x1 = torch.roll(x, shifts, (2,3))
    t = model(x)
    t1 = model(x1)
    try:
        assert torch.argmax(t).item() == torch.argmax(t1).item()
    except:
        print("Error at iteration: ", i)
        print(shifts)
        print(torch.argmax(t).item())
        print(torch.argmax(t1).item())
        break
print("Done")
# %%
x = torch.rand((1,3,224,224)).cuda()
model = timm.models.vision_transformer.vit_tiny_patch16_224(pretrained=True)
model = nn.Sequential(
PolyOrderModule(grid_size=(14,14), patch_size=(16,16)),
model).cuda()
model.eval()
model(x)

# %%
# outliers testsing 
config  = _C.clone()
config.MODEL.TYPE = "vit_poly_base"
config.MODEL.PRETRAIN_PATH = "/home/pding/scratch.cmsc663/poly_vit_base_0227/default/ckpt_epoch_173.pth"
# config.MODEL.PRETRAIN_PATH = "/home/pding/scratch.cmsc663/poly_vit_small_0228/default/ckpt_epoch_299.pth"
model = build_model(config, is_pretrain=True).cuda()


# %%
outliers = torch.load("outliers.pth")
# model =  PolyOrderModule(patch_size=(16,16), grid_size=(14,14))

for i in range(len(outliers)): 
    x = outliers[i]["raw"].cuda()
    x1 = torch.roll(x, outliers[i]["shifts"], (2,3))
    x2 = torch.roll(x, outliers[i]["shifts1"], (2,3))
    # p, norm = arrange_polyphases(x, (16,16))
    # p1, norm1 = arrange_polyphases(x1, (16,16))
    # p2, norm2 = arrange_polyphases(x2, (16,16))
    p = model(x)
    p1 = model(x1)
    p2 = model(x2)
    try:
        assert torch.argmax(p).item() == torch.argmax(p1).item()
        assert torch.argmax(p2).item() == torch.argmax(p1).item()
    except:
        print(i)
        print(torch.argmax(p).item(), torch.argmax(p1).item(), torch.argmax(p2).item())


    # plt.imshow(p[0].permute(1,2,0).cpu())
    # plt.show()
    # plt.imshow(p1[0].permute(1,2,0).cpu())
    # plt.show()
    # plt.imshow(p2[0].permute(1,2,0).cpu())
    # plt.show()


    
# %%
