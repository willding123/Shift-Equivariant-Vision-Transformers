#%% 
import torch
import torchvision.models as models
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
# import imagenet default constants from timm
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
from torchvision.transforms import InterpolationMode
import timm 
import time 
import numpy as np 
from tqdm import tqdm
#%%
# variables

for specific_shift in [30, 24, 17, 15, 8, 3, 0]:
    roll = True
    shift_size = 32
    # specific_shift = 40
    model_type = "timm"
    # Define the location of the validation dataset
    data_path = '/fs/cml-datasets/ImageNet/ILSVRC2012/val'
    model = timm.create_model("hf_hub:timm/vit_small_patch16_224.augreg_in21k_ft_in1k", pretrained=True)

    if model_type == "timm":
        # Define the transforms to be applied to the input images
        transforms = transforms.Compose([
            transforms.Resize(size = 256, interpolation=InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=IMAGENET_INCEPTION_MEAN,
                std=IMAGENET_INCEPTION_STD
            )
        ])
    else: 
        # Define the transforms to be applied to the input images
        transforms = transforms.Compose([
            transforms.Resize(size = 256, interpolation=InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=IMAGENET_DEFAULT_MEAN,
                std=IMAGENET_DEFAULT_STD
            )
        ])


    # Load the ImageNet-O dataset using the ImageFolder class
    dataset = ImageFolder(root=data_path, transform=transforms)

    # Define the batch size for the data loader
    batch_size = 200

    # Create the data loader for dataset
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=8)

    # Load the pre-trained model
    # Set the model to evaluate mode
    model.eval()

    # Define the device to use for computation (CPU or GPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Move the model to the device
    model.to(device)

    # Define the criterion (loss function) to use for evaluation
    criterion = torch.nn.CrossEntropyLoss()

    # Define variables to keep track of the total loss and accuracy
    total_loss = 0
    correct = 0
    total = 0
    consistent = 0

    start = time.time()
    # Disable gradient computation for evaluation
    with torch.no_grad():
        # Iterate over batches of data from the ImageNet-O dataset
        for images, labels in tqdm(dataloader):
            # Move the data to the device
            images = images.to(device)
            labels = labels.to(device)

            if roll: 
                # generate a two-element tuple of random integers between -shift_size and shift_size
                if specific_shift:
                    shifts = (specific_shift, specific_shift)
                else:
                    shifts = (np.random.randint(-shift_size, shift_size),np.random.randint(-shift_size, shift_size))
                shifts1 = (np.random.randint(-shift_size, shift_size),np.random.randint(-shift_size, shift_size))
                images = torch.roll(images, shifts, (2,3))
                images1 = torch.roll(images, shifts1, (2,3))

            
            # Compute the model output for the input batch
            outputs = model(images)
            outputs1 = model(images1)
            
            # Compute the loss for the batch
            loss = criterion(outputs, labels)
            
            # Update the total loss
            total_loss += loss.item() * images.size(0)
            
            # Compute the predicted classes for the batch
            _, predicted = torch.max(outputs.data, 1)
            _, predicted1 = torch.max(outputs1.data, 1)

            # Update the number of correct predictions
            correct += (predicted == labels).sum().item()
            consistent += (predicted == predicted1).sum().item()
            
            # Update the total number of images
            total += labels.size(0)

    # Compute the average loss and accuracy over all batches
    average_loss = total_loss / total
    accuracy = correct / total
    consistency = consistent / total
    end_time = time.time() - start

    # Print the results
    print("Time Elapsed {:.4f}".format(end_time))
    print('Average Loss: {:.4f}, Accuracy: {:.4f}, Consistency {:.4f}'.format(average_loss, accuracy, consistency))
    with open("eval_results.txt", "a") as f:
        f.write("Time Elapsed {:.4f}".format(end_time))
        f.write('Average Loss: {:.4f}, Accuracy: {:.4f}, Consistency {:.4f}'.format(average_loss, accuracy, consistency))

    # %%
# Average Loss: 0.8190, Accuracy: 0.7796, Consistency 0.8812 (shift_size = 40)