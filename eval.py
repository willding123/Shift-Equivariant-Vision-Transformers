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
from torchvision.transforms import InterpolationMode, RandomAffine, RandomPerspective, RandomCrop, RandomErasing, RandomHorizontalFlip, RandomVerticalFlip
import timm
import time 
import numpy as np 
from config import _C
from models.build import build_model
import matplotlib.pyplot as plt 
from tqdm import tqdm
from timm.models.layers import PatchEmbed
from models.poly_utils import PolyOrderModule, arrange_polyphases, PolyPatchEmbed, PolyOrder
from timm.models.twins import Twins
from timm.models.layers import to_2tuple
from timm.models.twins import LocallyGroupedAttn
from math import sqrt
from models.vision_transformer import PolyViT
from models.polytwins import PolyTwins
import argparse
import csv
import glob
import os

## TODO: 1. add affine attack, crop attack, and random perspective attack
def parse_args():
    parser = argparse.ArgumentParser('Evaluation script', add_help=False)
    parser.add_argument('--model', type=str, required=True, metavar="FILE", help='model name' )
    parser.add_argument('--data_path', type=str,  default = '/fs/cml-datasets/ImageNet/ILSVRC2012/val', required=False, metavar="FILE", help='dataset path')
    parser.add_argument('--model_card', type=str, required=False, metavar="FILE", help='model name on hugging face used in timm to create models' )
    parser.add_argument('--shift_attack', type=bool, default=True, help='whether enable shift attack')
    parser.add_argument('--shift_size', type=int, default = 15, required=False, metavar="FILE", help='shift size')
    parser.add_argument('--batch_size', type=int, default = 128, required=False, metavar="FILE", help='batch size')
    parser.add_argument('--num_workers', type=int, default = 8, required=False, metavar="FILE", help='number of workers')
    parser.add_argument ("--pretrained_path", type=str, default = None, required=False, metavar="FILE", help="pretrained model path")
    parser.add_argument("--affine", type=bool, default = False, required=False, metavar="FILE", help="whether enable affine attack")
    parser.add_argument("--breaking", action="store_true", required=False, help="break the loop if there are enough inconsistent predictions")
    # add arguments for random perspective attack
    parser.add_argument("--random_perspective", action="store_true", required=False, help="whether enable random perspective attack")
    parser.add_argument("--distortion_scale", type=int, default = 0.5, required=False, metavar="FILE", help='perspective size')
    # add arguments for crop attack
    parser.add_argument("--crop", action="store_true", required=False, help="whether enable crop attack")
    parser.add_argument("--crop_size", type=int, default = 224, required=False, metavar="FILE", help='crop size')
    parser.add_argument("--crop_padding", type=int, default = 10, required=False, metavar="FILE", help='crop ratio')
    # add arguments for random affine attack   
    parser.add_argument("--random_affine", action="store_true", required=False, help="whether enable random affine attack")
    parser.add_argument("--degrees", type=int, default = 30, required=False, metavar="FILE", help='degrees')
    parser.add_argument("--translate", type=int, default = (0.1,0.1), required=False, metavar="FILE", help='translate')
    parser.add_argument("--scale", type=int, default = (0.8,1.2) , required=False, metavar="FILE", help='scale')
    parser.add_argument("--shear", type=int, default = 10, required=False, metavar="FILE", help='shear')
    # add arguments for random erasing
    parser.add_argument("--random_erasing", action="store_true", required=False, help="whether enable random erasing")
    # add argumennts for horizontal flip
    parser.add_argument("--flip", action="store_true", required=False, help="whether enable horizontal flip")
    # add arguments for all attacks
    parser.add_argument("--all_attack", action="store_true", required=False, help="whether enable all attacks")
    args, unparsed = parser.parse_known_args()
    return args

def main(args):
    batch_size = args.batch_size
    if args.shift_attack:
        shift_size = args.shift_size
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_path = args.data_path
    IN_default_mean = True
    
    if args.model == "vit":
        if args.model_card:
            model = timm.create_model(args.model_card, pretrained=True)
        else:
            model = timm.create_model("hf_hub:timm/vit_base_patch16_224.augreg_in21k_ft_in1k", pretrained=True)
        IN_default_mean = False
    elif args.model == "vit_relpos":
        if args.model_card:
            model = timm.create_model(args.model_card, pretrained=True)
        else:
            model = timm.create_model("hf_hub:timm/vit_relpos_small_patch16_224.sw_in1k", pretrained=True)
        IN_default_mean = False
    elif args.model == "polyvit":
        if args.model_card:
            model = PolyViT(args.model_card, pretrained=True)
        else:
            model = PolyViT("hf_hub:timm/vit_base_patch16_224.augreg_in21k_ft_in1k", pretrained=True)
        IN_default_mean = False
    elif args.model == "twins":
        if args.model_card:
            model = timm.create_model(args.model_card, pretrained=True)
        else:
            model = timm.create_model("twins_svt_small", pretrained=True)
    elif args.model == "polytwins":
        if args.model_card:
            model = PolyTwins(args.model_card, pretrained=True)
        else: 
            model = PolyTwins("twins_svt_small", pretrained=True)
    else: 
        raise NotImplementedError
    
    if args.pretrained_path:
        config  = _C.clone()
        config.MODEL.TYPE = args.model
        config.MODEL.CARD = args.model_card
        try: 
            model = build_model(config)
            # find the latest checkpoint in the folder args.pretrained_path/default
            ckpt_list = glob.glob(os.path.join(args.pretrained_path, "default", "*.pth"))
            ckpt_list.sort(key=os.path.getmtime)
            args.pretrained_path = ckpt_list[-1]
            ckpt = torch.load(args.pretrained_path, map_location=device)
            model.load_state_dict(ckpt['model'])
        except:
            raise NotImplementedError

    model = model.to(device)

    if IN_default_mean:
        # Define the transforms to be applied to the input images
        transformation = transforms.Compose([
            transforms.Resize(size = 256, interpolation=InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=IMAGENET_DEFAULT_MEAN,
                std=IMAGENET_DEFAULT_STD
            )
        ])
    else: 
        # Define the transforms to be applied to the input images
        transformation = transforms.Compose([
            transforms.Resize(size = 256, interpolation=InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=IMAGENET_INCEPTION_MEAN,
                std=IMAGENET_INCEPTION_STD
            )
        ])

    # if random perspective attack is enabled, add the random perspective transformation
    if args.random_perspective:
        transformation = transforms.Compose([
        transformation,
        RandomPerspective(distortion_scale = args.distortion_scale)
        ])
    # if random affine attack is enabled, add the random affine transformation
    if args.random_affine:
        transformation = transforms.Compose([
        transformation,
        RandomAffine(degrees = args.degrees, translate = args.translate, scale = args.scale, shear = args.shear)
        ])
    # if crop attack is enabled, add the crop transformation
    if args.crop:
        transformation = transforms.Compose([
        transformation,
        RandomCrop(size = args.crop_size, padding = args.crop_padding)
        ])
    # if random erasing attack is enabled, add the random erasing transformation
    if args.random_erasing:
        transformation = transforms.Compose([
        transformation,
        RandomErasing()
        ])
    # if horizontal flip attack is enabled, add the horizontal flip transformation
    if args.flip:
        transformation = transforms.Compose([
        transformation,
        RandomHorizontalFlip()
        ])
    # if all three attacks are enabled, add the random perspective, random affine and crop transformations
    if args.all_attack:
        transformation = transforms.Compose([
        transformation,
        RandomPerspective(distortion_scale = args.distortion_scale),
        RandomAffine(degrees = args.degrees, translate = args.translate, scale = args.scale, shear = args.shear),
        RandomCrop(size = args.crop_size, padding = args.crop_padding),
        RandomErasing(),
        RandomHorizontalFlip()
        ])
        

    # Load the ImageNet-O dataset using the ImageFolder class
    dataset = ImageFolder(root=data_path, transform=transformation)

    # Define the batch size for the data loader

    # Create the data loader for dataset
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=args.num_workers)

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

            if args.shift_attack: 
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

                # Update the number of correct predictions
                correct += (predicted == labels).sum().item()
                consistent += (predicted == predicted1).sum().item()
                
                # Update the total number of images
                total += labels.size(0)
                if args.breaking:
                    try: 
                        # assert the entry difference between two output vectors is less than 1e-5
                        assert (predicted == predicted1).all()
                    except: 
                        # get index of all the different entries
                        diff = (predicted != predicted1).nonzero()
                        outliers.append({"raw":raw[diff].cpu(), "shifts": shifts, "shifts1": shifts1})
                
                    if outliers.__len__() > 1:
                        print(f"got  {len(outliers)} outliers")
                        break

                # Compute the average loss and accuracy over all batches
        average_loss = total_loss / total
        accuracy = correct / total
        consistency = consistent / total
        end_time = time.time() - start
        
    # print out the attacks used
    if args.random_perspective:
        print("Random Perspective Attack")
    if args.random_affine:
        print("Random Affine Attack")
    if args.crop:
        print("Crop Attack")
    if args.random_erasing:
        print("Random Erasing Attack")
    if args.flip:
        print("Horizontal Flip Attack")
    if args.all_attack:
        print("All Attacks")
    # Print the results
    print("Time Elapsed {:.4f}".format(end_time))
    print(args.model)
    print('Average Loss: {:.4f}, Accuracy: {:.4f}, Consistency {:.4f}'.format(average_loss, accuracy, consistency))
    if args.model_card:
        print("Using model card: ", args.model_card)
    # append accuracy and consistency to a csv file, if no such file exists, create one, in the first row include the column names; if the file exists, append the results to the end of the file
    if not os.path.isfile('eval_results.csv'):
        with open('eval_results.csv', 'w') as f:
            writer = csv.writer(f)
            writer.writerow(["model", "model_card", "accuracy", "consistency", "random_perspective", "random_affine", "crop", "random_erasing", "flip", "all_attack", "shift_attack", "distortion_scale", "degrees", "translate", "scale", "shear", "crop_size", "crop_padding", "average_loss", "time", "pretrained_path"])
            writer.writerow([args.model, args.model_card,  accuracy, consistency, args.random_perspective, args.random_affine, args.crop, args.random_erasing, args.flip, args.all_attack, args.shift_attack, args.distortion_scale, args.degrees, args.translate, args.scale, args.shear, args.crop_size, args.crop_padding, average_loss, end_time, args.pretrained_path.split('/')[-3]])
    else:
        with open('eval_results.csv', 'a') as f:
            writer = csv.writer(f)
            writer.writerow([args.model, args.model_card, accuracy, consistency, args.random_perspective, args.random_affine, args.crop, args.random_erasing, args.flip, args.all_attack, args.shift_attack, args.distortion_scale, args.degrees, args.translate, args.scale, args.shear, args.crop_size, args.crop_padding, average_loss, end_time, args.pretrained_path.split('/')[-3]])

if __name__ == '__main__':
    args = parse_args()
    main(args)