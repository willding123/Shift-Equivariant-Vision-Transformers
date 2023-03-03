#%% 
import torch 
import torchvision 
from torchvision.datasets import ImageFolder
from data.build import * 
from config import _C
import matplotlib.pyplot as plt

config = _C.clone()
raw_data = ImageFolder(root='~/scratch.cmsc663/train', transform=None)
transform = build_transform(False, config, roll=True)
data = ImageFolder(root='~/scratch.cmsc663/val', transform=transform)


# %%
num = 1 
for i in range(num): 
    # print("raw data")
    # plt.imshow(raw_data[i][0])
    # plt.show()
    print("transformed data")
    plt.imshow(data[10][0][0], cmap='gray')
    plt.show()
    plt.imshow(data[10][0][1], cmap='gray')
    plt.show()
    plt.imshow(data[10][0][2], cmap='gray')
    plt.show()

# %%
