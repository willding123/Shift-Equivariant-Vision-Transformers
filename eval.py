#%% 
import torch
from torch import nn
import torchvision.models as models
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from timm.models.vision_transformer_relpos import VisionTransformerRelPos
# import imagenet default constants from timm
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
from torchvision.transforms import InterpolationMode
import timm 
import time 
import numpy as np 
from config import _C
from models.build import build_model
import matplotlib.pyplot as plt 
from tqdm import tqdm
from timm.models.layers import PatchEmbed
from models.poly_utils import PolyOrderModule, arrange_polyphases, PolyPatch, PolyOrder
from timm.models.twins import Twins
from timm.models.layers import to_2tuple
from timm.models.twins import LocallyGroupedAttn
from math import sqrt
from models.vision_transformer import PolyViT

# test 100% consistent model (relu, circular pos block, no layernorm for patch embed)
# eliminate relu
# eliminate circular padding
# add layernorm for patch embed

# variables 
batch_size = 128
roll = True
shift_size = 15
model_type = ""
# Define the location of the validation dataset
# data_path = '/home/pding/scratch.cmsc663/val'
data_path = '/fs/cml-datasets/ImageNet/ILSVRC2012/val'
# model = timm.create_model("hf_hub:timm/vit_base_patch16_224.augreg_in21k_ft_in1k", pretrained=True)
# config  = _C.clone()
# config.MODEL.TYPE = "vit_poly_base"
# config.MODEL.PRETRAIN_PATH = "/home/pding/scratch.cmsc663/poly_vit_base_0227/default/ckpt_epoch_173.pth"
# model = build_model(config, is_pretrain=True)
# model = VisionTransformer(weight_init = 'skip')
# model = VisionTransformerRelPos()
# model = timm.create_model("hf_hub:timm/vit_relpos_small_patch16_224.sw_in1k", pretrained=True)
# model = timm.create_model("twins_pcpvt_small")
model = timm.create_model("twins_svt_small", pretrained=True).cuda()
# model = PolyViT("hf_hub:timm/vit_small_patch16_224.augreg_in21k_ft_in1k", pretrained=True)

# model = torch.nn.Sequential(
# PolyOrderModule(patch_size=(16,16)),
# model)

#%%

def debug_polyphasex(x, patch_size):
    _, n = arrange_polyphases(x, patch_size)
    try:
        assert torch.topk(n, 2).values[0][0] != torch.topk(n, 2).values[0][1]
    except:
        print("polyphases are not unique")
        breakpoint()

debugger = {}


# #%% vit with rel_pos
# # set model's parameters to zero if their name contains "rel_pos"
# for name, param in model.named_parameters():
#     if "rel_pos" in name:
#         param.data.zero_()

# # verify the changed parameters 
# for name, param in model.named_parameters():
#     if "rel_pos" in name:
#         print(model.state_dict()[name])

#%%
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

# Create the data loader for dataset
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=1)

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
outliers = []
normal = []
start = time.time()
# Disable gradient computation for evaluation
with torch.no_grad():
    # Iterate over batches of data from the dataset
    for images, labels in tqdm(dataloader):
        # Move the data to the device
        images = images.to(device)
        raw = images.clone()
        labels = labels.to(device)

        if roll: 
            # generate a two-element tuple of random integers between -shift_size and shift_size
            shifts = tuple(np.random.randint(0,shift_size,2))
            shifts1 = tuple(np.random.randint(0,shift_size,2))
            images = torch.roll(raw, shifts, (2,3))
            images1 = torch.roll(raw, shifts1, (2,3))

        
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

        # if predicted != predicted1:
        # outliers.append({"raw":raw.cpu(), "shifts": shifts, "shifts1": shifts1})
        #     if len(outliers) > 0 :
        #         print(f"got  {len(outliers)} outliers")
        #         break 
        
        # if predicted == predicted1:
        #     normal.append({"raw":raw[0].permute(1,2,0).cpu(), "shifts": shifts, "shifts1": shifts1})

        # Update the number of correct predictions
        correct += (predicted == labels).sum().item()
        consistent += (predicted == predicted1).sum().item()
        
            # Update the total number of images
        total += labels.size(0)
        try: 
            # assert the entry difference between two output vectors is less than 1e-5
            assert (predicted == predicted1).all()
        except: 
            # get index of all the different entries
            diff = (predicted != predicted1).nonzero()
            outliers.append({"raw":raw[diff].cpu(), "shifts": shifts, "shifts1": shifts1})
        
        # if outliers.__len__() > 1:
        #     print(f"got  {len(outliers)} outliers")
        #     break

        # Compute the average loss and accuracy over all batches
    average_loss = total_loss / total
    accuracy = correct / total
    consistency = consistent / total
    end_time = time.time() - start
    

# Print the results
print("Time Elapsed {:.4f}".format(end_time))
print('Average Loss: {:.4f}, Accuracy: {:.4f}, Consistency {:.4f}'.format(average_loss, accuracy, consistency))

#%%