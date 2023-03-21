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
from models.vision_transformer import VisionTransformer
from timm.models.layers import PatchEmbed
from models.poly_utils import PolyOrderModule, arrange_polyphases, PolyPatch, PolyOrder
from timm.models.twins import Twins
from timm.models.layers import to_2tuple
from timm.models.twins import LocallyGroupedAttn
from math import sqrt

# test 100% consistent model (relu, circular pos block, no layernorm for patch embed)
# eliminate relu
# eliminate circular padding
# add layernorm for patch embed
job = {"norelu":0}

# variables 
batch_size = 128
roll = True
shift_size = 15
model_type = "timm"
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
# model = timm.create_model("hf_hub:timm/vit_small_patch16_224.augreg_in21k_ft_in1k", pretrained=True)

# model = torch.nn.Sequential(
# PolyOrderModule(grid_size=(56,56), patch_size=(4,4)),
# model).cuda()

#%%

def debug_polyphasex(x, patch_size):
    _, n = arrange_polyphases(x, patch_size)
    try:
        assert torch.topk(n, 2).values[0][0] != torch.topk(n, 2).values[0][1]
    except:
        print("polyphases are not unique")
        breakpoint()

debugger = {}

class PolyTwins(timm.models.twins.Twins):
    def __init__(self, model_type, pretrained = False, **kwargs):
        super().__init__()
        model = timm.create_model(model_type , pretrained=pretrained)
        self.patch_embeds = model.patch_embeds
        self.pos_drops = model.pos_drops
        self.blocks = model.blocks
        self.pos_block = model.pos_block
        self.norm = model.norm
        self.head = model.head
        self.depths = model.depths
        self.num_classes = model.num_classes
        self.num_features = model.num_features
        self.embed_dim = model.embed_dims
        self.global_pool = model.global_pool
        
    
    def forward_features(self, x):
        B = x.shape[0]
        tmp = x.clone()
        for i, (embed, drop, blocks, pos_blk) in enumerate(
                zip(self.patch_embeds, self.pos_drops, self.blocks, self.pos_block)):        
            if i == 0:
                x = PolyOrder.apply(x, embed.proj.kernel_size)
            x, size = embed(x)
            x = drop(x)
            for j, blk in enumerate(blocks):
                x = x.view(B, int(sqrt(x.shape[1])), -1, x.shape[2]).permute(0, 3, 1, 2)
                # if type(blk.attn) == LocallyGroupedAttn:                
                #     x = PolyOrder.apply(x, to_2tuple(blk.attn.ws))
                # else: 
                #     if blk.attn.sr_ratio > 1:
                #         x = PolyOrder.apply(x, to_2tuple(blk.attn.sr_ratio))
                    
                x = x.permute(0, 2, 3, 1)
                x = x.reshape(B, -1, x.shape[3]).contiguous()
                x = blk(x, size)
                if j == 0:
                    x = pos_blk(x, size)  # PEG here
            if i < len(self.depths) - 1:
                x = x.reshape(B, *size, -1).permute(0, 3, 1, 2).contiguous()
        x = self.norm(x)
        return x

tmp = PolyTwins("twins_svt_small", pretrained=True)
tmp.load_state_dict(model.state_dict())
model = tmp 

# for m in model.patch_embeds:
#     m.norm = nn.Identity()

# for bs in model.blocks:
#     for b in bs: 
#         b.mlp.act = nn.ReLU()

cs = [64, 128, 256, 512]
for i, l in enumerate(model.pos_block):
    # l.proj = torch.nn.Sequential(torch.nn.Conv2d(cs[i], cs[i], 3, 1, 1, bias=True, groups=cs[i], padding_mode='circular'), )
    l.proj[0].padding_mode = "circular"

for i, l in enumerate(model.patch_embeds):
    l.proj.padding_mode = "circular"

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

job["consistent"] = {"consistency": consistency, "accuracy": accuracy, "average_loss": average_loss, "time": end_time}
# export job as json file
# with open(f"~/job_consistent.json", "w") as f:
#     json.dump(job, f, indent=4)
###########
###########
### full design: accuracy: 0.0012 consistency 100%
### fully_consistent_model (FCM) - relu: Average Loss: 6.9410, Accuracy: 0.0012, Consistency 1.0000
### FCM - circular - relu: Average Loss: 6.9446, Accuracy: 0.0008, Consistency 0.8746
### FCM - circular - relu + layernorm: Average Loss: 6.9342, Accuracy: 0.0012, Consistency 0.8171
### Pretrain + Polyorder only at patch embedding stage 
###########
###########
#%%

# would it be more consistent if I tune the patch size, and lp norm? 
idx = (predicted != predicted1).nonzero()[0]

# debugging twins 
embed = model.patch_embeds[3]
blocks = model.blocks[3]
pos_blk = model.pos_block[3]
norm = model.norm

# x = images[idx]
# B,C,H,W = x.shape
# xs = torch.roll(x, shifts, (2,3))
# xs1 = torch.roll(x, shifts1, (2,3))

# embedding tests
x = PolyOrder.apply(x, embed.proj.kernel_size)
xs = PolyOrder.apply(xs, embed.proj.kernel_size)
xs1 = PolyOrder.apply(xs1, embed.proj.kernel_size)
x, size = embed(x)
xs, _ = embed(xs)
xs1, _ = embed(xs1)

from utils import find_shift2d_batch, shift_and_compare

t = x.view(B, int(sqrt(x.shape[1])), int(sqrt(x.shape[1])), x.shape[2])
ts = xs.view(B, int(sqrt(x.shape[1])), int(sqrt(x.shape[1])), x.shape[2])
ts1 = xs1.view(B, int(sqrt(x.shape[1])), int(sqrt(x.shape[1])), x.shape[2])


s1 = find_shift2d_batch(t,ts, early_break = False)
s2 = find_shift2d_batch(ts,ts1, early_break = False)
shift_and_compare(t, ts, s1, (0,1) )
shift_and_compare(ts, ts1, s2, (0,1) )

# block tests
def block_forward(x, size):
    for j, blk in enumerate(blocks):
        x = x.view(B, int(sqrt(x.shape[1])), -1, x.shape[2]).permute(0, 3, 1, 2)
        if type(blk.attn) == LocallyGroupedAttn:
            _, n = arrange_polyphases(x, to_2tuple(blk.attn.ws))
            try:
                assert torch.topk(n, 2).values[0][0] != torch.topk(n, 2).values[0][1]
            except:
                print(j, torch.topk(n, 2).values[0])
                break
            x = PolyOrder.apply(x, to_2tuple(blk.attn.ws))
            
        else: 
            x = PolyOrder.apply(x, to_2tuple(blk.attn.sr_ratio))
        x = x.permute(0, 2, 3, 1)
        x = x.reshape(B, -1, x.shape[3]).contiguous()
        x = blk(x, size)
        if j == 0:
            x = pos_blk(x, size) 
    return x

x = block_forward(x, size)
xs = block_forward(xs, size)
xs1 = block_forward(xs1, size)

t = x.view(B, int(sqrt(x.shape[1])), int(sqrt(x.shape[1])), x.shape[2])
ts = xs.view(B, int(sqrt(x.shape[1])), int(sqrt(x.shape[1])), x.shape[2])
ts1 = xs1.view(B, int(sqrt(x.shape[1])), int(sqrt(x.shape[1])), x.shape[2])

s1 = find_shift2d_batch(t,ts, early_break = False)
s2 = find_shift2d_batch(ts,ts1, early_break = False)
shift_and_compare(t, ts, s1, (0,1) )
shift_and_compare(ts, ts1, s2, (0,1) )

# x = x.reshape(B, *size, -1).permute(0, 3, 1, 2).contiguous()
# xs = xs.reshape(B, *size, -1).permute(0, 3, 1, 2).contiguous()
# xs1 = xs1.reshape(B, *size, -1).permute(0, 3, 1, 2).contiguous()


# %%
head = model.head
x = x.mean(dim=1)
xs = xs.mean(dim=1)
xs1 = xs1.mean(dim=1)
x = head(x)
xs = head(xs)
xs1 = head(xs1)

# write a subclass called PolyTwins that inherits from timm.models.twins.Twins

# %%
x = debugger["x"]
model(x)