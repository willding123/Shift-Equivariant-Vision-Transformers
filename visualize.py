#%%
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torchvision.transforms import InterpolationMode
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import timm
from torchvision.datasets import ImageFolder
from tqdm import tqdm
# Load the PyTorch model that will be used to get the output probabilities

data_path = "~/scratch.cmsc663/train"
device = "cuda" if torch.cuda.is_available() else "cpu"
model = timm.create_model("hf_hub:timm/vit_relpos_small_patch16_224.sw_in1k", pretrained=True).to(device)
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
def get_variances(shift_sizes, num_samples):
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
variances = get_variances(shift_sizes, num_samples)

plt.imshow(variances.reshape(16,-1), cmap="hot")
plt.colorbar()
plt.show()
#%%

import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt


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
model = torchvision.models.vit_large_patch16_224(pretrained=True)

# Define the input image size
input_size = (224, 224)

# Define the number of pixels to shift the input by
shift = 1

# Generate a random input image
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
