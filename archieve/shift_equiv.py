#%%
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torchvision.transforms import InterpolationMode
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import timm
from torchvision.datasets import ImageFolder
from tqdm import tqdm
from config import _C
from models.build import build_model
from math import sqrt
from utils import find_shift2d_batch, shift_and_compare
## TODO: export figures to a shared folder 
# set random seed for reproducibility
torch.manual_seed(0)
np.random.seed(0)
# clear cache
torch.cuda.empty_cache()

data_path = "~/scratch.cmsc663/train"
device = "cuda" if torch.cuda.is_available() else "cpu"
model_card = "hf_hub:timm/vit_small_patch16_224.augreg_in1k"
# model_card = "twins_svt_base"
# model = timm.create_model(model_card, pretrained=True).to(device)

# Define transform
transform = transforms.Compose([
            transforms.Resize(size = 256, interpolation=InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=IMAGENET_DEFAULT_MEAN,
                std=IMAGENET_DEFAULT_STD
            )
        ])

# Define the shift sizes and number of samples
shift_sizes = range(-5,5)
num_samples = 64

# load data 
imagenet_dataset = ImageFolder(root=data_path, transform=transform)
imagenet_loader = torch.utils.data.DataLoader(imagenet_dataset, batch_size=num_samples, shuffle=True)
images, labels = next(iter(imagenet_loader))
images = images.to(device)
labels = labels.to(device)

config  = _C.clone()
config.MODEL.TYPE = "vit"
config.MODEL.CARD = model_card
config.MODEL.PRETRAIN_PATH = "/home/pding/scratch.cmsc663/vit_s_1kscratch/default/ckpt_epoch_299.pth"
# config.MODEL.PRETRAIN_PATH = "/home/pding/scratch.cmsc663/vit_b_1kscratch/default/ckpt_epoch_86.pth"
# config.MODEL.PRETRAIN_PATH = "/home/pding/scratch.cmsc663/twins_svts_b_scratch/default/ckpt_epoch_270.pth"
model = build_model(config)
ckpt  = torch.load(config.MODEL.PRETRAIN_PATH, map_location=device)
model.load_state_dict(ckpt["model"])
model = model.to(device)
model.eval()


config  = _C.clone()
config.MODEL.TYPE = "polyvit"
config.MODEL.CARD = model_card
config.MODEL.PRETRAIN_PATH = "/home/pding/scratch.cmsc663/pvit_s_1kscratch/default/ckpt_epoch_299.pth"
# config.MODEL.PRETRAIN_PATH = "/home/pding/scratch.cmsc663/pvit_b_1kscratch/default/ckpt_epoch_185.pth"
# config.MODEL.PRETRAIN_PATH = "/home/pding/scratch.cmsc663/ptwins_svts_b_scratch/default/ckpt_epoch_292.pth"
model1 = build_model(config)
ckpt  = torch.load(config.MODEL.PRETRAIN_PATH, map_location=device)
model1.load_state_dict(ckpt["model"])
model1 = model1.to(device)
model1.eval()

shift = (np.random.randint(0,5), np.random.randint(0,5))

shifted_images = torch.roll(images, shifts=shift, dims=(2, 3))

def test_shift_equiv(original, shifted):
    # reshape to (batch, channel, height, width)    
    B = original.shape[0]; C = original.shape[-1]; H = int(sqrt(original.shape[1])); W = int(H)
    original = original.view(original.shape[0], H, W, original.shape[2])
    original = original.permute(0, 3, 1, 2)
    shifted = shifted.view(shifted.shape[0], H, W, shifted.shape[2])
    shifted = shifted.permute(0, 3, 1, 2)
    shifts = find_shift2d_batch(original, shifted, False)
    distance = shift_and_compare(original, shifted, shifts, (0,1))
    return distance

# Pass the original and shifted input images through the model to obtain feature maps
with torch.no_grad():
    if model_card == "twins_svt_base":
        original_features = model.forward_features(images)
        shifted_features = model.forward_features(shifted_images)
        original_features1 = model1.forward_features(images)
        shifted_features1 = model1.forward_features(shifted_images)
    else:
        original_features = model.forward_features(images)[:,1:]
        shifted_features = model.forward_features(shifted_images)[:,1:]
        original_features1 = model1.forward_features(images)[:,1:]
        shifted_features1 = model1.forward_features(shifted_images)[:,1:]


distance = test_shift_equiv(original_features, shifted_features)
distance1 = test_shift_equiv(original_features1, shifted_features1)
print(distance)
print(distance1)
# plot numbers in distance list as an 8 x 8 heat map 
distance = torch.tensor(distance)
distance = distance.view(int(sqrt(len(distance))), -1)
distance1 = torch.tensor(distance1)
distance1 = distance1.view(int(sqrt(len(distance1))), -1)
# create a side by side plot
vmin = 0
vmax = 500
fig, axs = plt.subplots(1, 2)
im1 = axs[0].imshow(distance.cpu().numpy(), cmap='plasma', vmin = vmin, vmax = vmax)
# axs[0].set_title('Twins_B')
axs[0].set_title('ViT_S/16')

im2 = axs[1].imshow(distance1.cpu().numpy(), cmap='plasma', vmin = vmin, vmax = vmax)
# axs[1].set_title('Twins_B-poly')
axs[1].set_title('ViT_S/16-poly')

plt.show()
# add color bar
cbar = fig.colorbar(im1, ax=axs, shrink=0.6, aspect=15)
# cbar = fig.colorbar(im1, ax=axs.ravel().tolist())
cbar.ax.set_title('Shift Equivariance')
# fig.savefig("twins_shift.png")
fig.savefig("vits_shift.png")



#%% 


# # Load the pre-trained vision transformer model on ImageNet
# # model = torchvision.models.vit_large_patch16_224(pretrained=True)

# # Define the input image size
# input_size = (224, 224)

# # Define the number of pixels to shift the input by
# shift = 1

# # Shift the input image by one pixel in the x and y direction
# shifted_images = torch.roll(images, shifts=(shift, shift), dims=(2, 3))

# # Pass the original and shifted input images through the model to obtain feature maps
# with torch.no_grad():
#     original_features = model.forward_features(images)
#     shifted_features = model.forward_features(shifted_images)
#     original_features1 = model1.forward_features(images)
#     shifted_features1 = model1.forward_features(shifted_images)

# # Calculate the feature-wise mean absolute difference (MAD) between the original and shifted feature maps
# feature_mad = torch.mean(torch.abs(original_features - shifted_features), dim=(1, 2))
# feature_mad1 = torch.mean(torch.abs(original_features1 - shifted_features1), dim=(1, 2))

# # # Calculate L2 distance between the original and shifted feature maps
# # feature_l2 = torch.sqrt(torch.sum((original_features - shifted_features)**2, dim=(1, 2)))

# # Create a figure with two subplots
# fig, axs = plt.subplots(1, 2, figsize=(8, 4))
# # Plot the first heat map
# im1 = axs[0].imshow(feature_mad.cpu().numpy().reshape(8,-1), cmap='plasma', vmin=0, vmax=1)
# axs[0].set_title("ViT-B/16")
# # Plot the second heat map
# im2 = axs[1].imshow(feature_mad1.cpu().numpy().reshape(8,-1), cmap='plasma', vmin=0, vmax=1)
# axs[1].set_title('ViT-B/16-poly')
# # Create a color bar for the heat maps
# cbar = fig.colorbar(im1, ax=axs.ravel().tolist())
# # Set the title for the color bar
# cbar.ax.set_title('L1 Distance')
# plt.title('Internal Feature Stability')
# # Show the plot
# plt.show()
# print("saving figure ...")
# fig.savefig("vitb_feature_stability.png")


# # %%
