#%% 
import torch
import torch.nn as nn
import torch.nn.functional as F

x = torch.rand((1, 768, 197)).cuda()



#%%

# conv layer
# conv hyperparams: 768 768 3 1 1 True 768 circular

#add a circular padding layer
cnn_layer = nn.Conv2d(1, 32, (3, 3), stride=1)  #nn.Conv2d(768, 768, 3, 1, 1, bias=True, groups=768, padding_mode='circular').cuda()
sequential = nn.Sequential(cnn_layer).cuda()
# %%

padded = F.pad(x, (1, 1), mode='circular')
our = sequential(padded)

# %%



import torch
import torch.nn as nn

x = torch.rand((1, 768, 1, 197)).cuda()

# Custom circular padding function
def circular_pad(tensor, pad):
    return torch.cat((tensor[..., -pad:], tensor, tensor[..., :pad]), dim=-1)

# Conv layer
cnn_layer = nn.Conv2d(768, 768, (1, 3), 1, (0, 0), bias=True, groups=768).cuda()
sequential = nn.Sequential(cnn_layer).cuda()

# Apply circular padding and pass input to the sequential model
out = sequential(circular_pad(x, pad=1))