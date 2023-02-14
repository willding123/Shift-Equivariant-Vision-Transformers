#%% 
import torch 
from models.vision_transformer import * 
import numpy as np 
from timm.models.layers import PatchEmbed, Mlp, DropPath, trunc_normal_, lecun_normal_


#%% 
x = torch.rand((1,3,224,224)).cuda()
num_test = 1000
model = VisionTransformer(embed_layer=PatchEmbed ,weight_init = 'skip').cuda()
model.eval()
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
