#%%
import torch
import torchvision
import torchvision.transforms as transforms
import random
from torchvision.transforms import InterpolationMode
from tqdm import tqdm
import timm
from torch.utils.data import DataLoader, Subset

# TODO: test on a subset of the dataset
# set the random seed
random.seed(0)
torch.manual_seed(0)

# clear cache 
torch.cuda.empty_cache()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load the pre-trained vision transformer model
model = timm.create_model("hf_hub:timm/vit_base_patch16_224.augreg_in21k_ft_in1k", pretrained=True)
model.to(device)
model.eval()

# Load the ImageNet dataset
transform = transforms.Compose([
            transforms.Resize(size = 256, interpolation=InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5]
            )
        ])
data_path = "/home/pding/scratch.cmsc663/val/"
dataset = torchvision.datasets.ImageFolder(data_path, transform=transform)

# Sample a subset of 10000 images from the dataset
subset_indices = random.sample(range(len(dataset)), 512)
subset = Subset(dataset, subset_indices)
dataloader = DataLoader(subset, batch_size=256, shuffle=True)


# Define a function to perform the data augmentation by randomly shifting the images
def shift_image(image, shift):
    """Shift an image by the given amount."""
    x_shift, y_shift = shift
    image = torch.roll(image, shifts=(x_shift, y_shift), dims=(2, 3))
    return image

# Define a function to evaluate the accuracy of the model on a shifted dataset given shift size, the model, and the dataset
def evaluate(model, dataloader, shift):
    """Evaluate the accuracy of the given model on the given dataset."""
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images = shift_image(images, shift)
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    return accuracy



# Initialize variables to store the minimum accuracy found so far and the corresponding shift size
min_accuracy = float('inf')
min_shift = None


# Iterate through all the shift sizes in the given range and find the one with the lowest accuracy
for x_shift in tqdm(range(-15, 16)):
    for y_shift in range(-15, 16):
        shift = (x_shift, y_shift)
        # Evaluate the accuracy of the model on the augmented subset using the DataLoader
        accuracy = evaluate(model, dataloader, shift)
        # Update the minimum accuracy and the corresponding shift size if necessary
        if accuracy < min_accuracy:
            min_accuracy = accuracy
            min_shift = shift

# Print the shift size that resulted in the lowest accuracy
print("Shift size with the lowest accuracy:", min_shift)
print("Accuracy:", min_accuracy)
#%%