#%% 
from models.poly_utils import *
import torch
import numpy as np

#%%
x = torch.randn(1, 3, 224, 224).cuda()
input_resolution = (224,224)
grid_size = 7
patch_size = 4
in_chans = 3
out_chans = 64
model = nn.Sequential(
    PolyPatch(input_resolution, patch_size, in_chans, out_chans),
    nn.AvgPool1d((64,)),
    nn.Flatten(1),
    nn.Linear(3136, 1000)
).cuda()
criterion = nn.CrossEntropyLoss().cuda()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
model.train()
for i in range(1):
    optimizer.zero_grad()
    y = model(x)
    loss = criterion(y, torch.empty(size=(1,), dtype=torch.long).random_(0, 1000).cuda())
    loss.backward()
    optimizer.step()
    print(loss.item())
# %%
