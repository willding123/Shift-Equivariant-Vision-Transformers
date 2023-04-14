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

# set random seed for reproducibility
torch.manual_seed(0)
np.random.seed(0)

data_path = "~/scratch.cmsc663/train"
device = "cuda" if torch.cuda.is_available() else "cpu"
model_card = "hf_hub:timm/vit_small_patch16_224.augreg_in21k_ft_in1k"
model = timm.create_model(model_card, pretrained=True).to(device)

config  = _C.clone()
config.MODEL.TYPE = "polyvit"
config.MODEL.CARD = model_card
config.MODEL.PRETRAIN_PATH = "/home/pding/scratch.cmsc663/pvit_small_w/default/ckpt_epoch_80.pth"
model1 = build_model(config)
ckpt  = torch.load(config.MODEL.PRETRAIN_PATH, map_location=device)
model1.load_state_dict(ckpt["model"])
model1 = model.to(device)

model.eval()
transform = transforms.Compose([
            transforms.Resize(size = 256, interpolation=InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=IMAGENET_DEFAULT_MEAN,
                std=IMAGENET_DEFAULT_STD
            )
        ])

# Define a function to calculate the variance of output probabilities for each shift size
def get_variances(model, shift_sizes, num_samples):
    probs = []
    imagenet_dataset = ImageFolder(root=data_path, transform=transform)
    imagenet_loader = torch.utils.data.DataLoader(imagenet_dataset, batch_size=num_samples, shuffle=True)
    images, labels = next(iter(imagenet_loader))
    images = images.to(device)
    labels = labels.to(device)
    for shift_size in tqdm(shift_sizes):
        shifted_images = torch.roll(images, (shift_size, shift_size), (2,3))
        # Get the output probabilities for the correct labels of the images
        logits = model(shifted_images)
        probabilities = nn.functional.softmax(logits, dim=1)
        idx = np.arange(num_samples)
        output_probs = probabilities[idx, labels].cpu().detach().numpy().reshape(-1,1)
        probs.append(output_probs)
        # Calculate the variance of the output probabilities
    probs = np.concatenate(probs, axis=1)
    variances = np.var(probs, axis=1)
    return variances

# Define the shift sizes and number of samples
shift_sizes = range(-5,5)
num_samples = 256

# Get the variances for each shift size
variances = get_variances(model, shift_sizes, num_samples)
variances1 = get_variances(model1, shift_sizes, num_samples)

plt.imshow(variances.reshape(16,-1), cmap="hot")
plt.colorbar()
plt.show()

plt.imshow(variances.reshape(16,-1), cmap="hot")
plt.colorbar()
plt.show()
#%% 


transform = transforms.Compose([
            transforms.Resize(size = 256, interpolation=InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=IMAGENET_DEFAULT_MEAN,
                std=IMAGENET_DEFAULT_STD
            )
        ])

# Load the pre-trained vision transformer model on ImageNet
# model = torchvision.models.vit_large_patch16_224(pretrained=True)

# Define the input image size
input_size = (224, 224)

# Define the number of pixels to shift the input by
shift = 1

imagenet_dataset = ImageFolder(root=data_path, transform=transform)
imagenet_loader = torch.utils.data.DataLoader(imagenet_dataset, batch_size=num_samples, shuffle=True)
images, labels = next(iter(imagenet_loader))
# Shift the input image by one pixel in the x and y direction
shifted_images = torch.roll(images, shifts=(shift, shift), dims=(2, 3))

# Pass the original and shifted input images through the model to obtain feature maps
with torch.no_grad():
    original_features = model.forward_features(images)
    shifted_features = model.forward_features(shifted_images)

# Calculate the feature-wise mean absolute difference (MAD) between the original and shifted feature maps
feature_mad = torch.mean(torch.abs(original_features - shifted_features), dim=(2, 3))

# Plot the feature-wise MAD values as a heatmap
plt.imshow(feature_mad.numpy(), cmap='hot')
plt.title('Internal Feature Stability')
plt.xlabel('Feature Map Index')
plt.ylabel('Layer Index')
plt.colorbar()
plt.show()

# %%
