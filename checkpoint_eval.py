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
from torchvision.transforms import InterpolationMode, RandomAffine, RandomPerspective, RandomCrop 
import timm
import time 
import numpy as np 
from config import get_config, _C
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

from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser('Evaluation script', add_help=False)
    parser.add_argument('--cfg', type=str, required=True, metavar="FILE", help='path to config file', ) # This is used to get the model card. It should be the config used during training.
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )
    parser.add_argument("--local_rank", type=int, required=False, help='local rank for DistributedDataParallel')

   
    parser.add_argument('--data_path', type=str,  default = '/fs/cml-datasets/ImageNet/ILSVRC2012/val', required=False, metavar="FILE", help='dataset path')
    parser.add_argument ("--pretrained_path", type=str, default = None, required=False, metavar="FILE", help="pretrained model path")
    parser.add_argument('--batch_size', type=int, default = 128, required=False, metavar="FILE", help='batch size')
    parser.add_argument('--num_workers', type=int, default = 8, required=False, metavar="FILE", help='number of workers')

    parser.add_argument('-k', type=int, default=10, required=False, help='Evaluate every k checkpoints in the given model, where k is given by this argument')

    parser.add_argument('--shift_attack', type=bool, default=True, help='whether enable shift attack')
    parser.add_argument('--shift_size', type=int, default = 15, required=False, metavar="FILE", help='shift size')
    
    
    parser.add_argument("--affine", type=bool, default = False, required=False, metavar="FILE", help="whether enable affine attack")
    parser.add_argument("--breaking", action="store_true", required=False, help="break the loop if there are enough inconsistent predictions")
    # add arguments for random perspective attack
    parser.add_argument("--random_perspective", type=bool, default = False, required=False, metavar="FILE", help="whether enable random perspective attack")
    parser.add_argument("--distortion_scale", type=int, default = 0.5, required=False, metavar="FILE", help='perspective size')
    # add arguments for crop attack
    parser.add_argument("--crop", type=bool, default = False, required=False, metavar="FILE", help="whether enable crop attack")
    parser.add_argument("--crop_size", type=int, default = 224, required=False, metavar="FILE", help='crop size')
    parser.add_argument("--crop_padding", type=int, default = 10, required=False, metavar="FILE", help='crop ratio')
    # add arguments for random affine attack   
    parser.add_argument("--random_affine", type=bool, default = False, required=False, metavar="FILE", help="whether enable random affine attack")
    parser.add_argument("--degrees", type=int, default = 30, required=False, metavar="FILE", help='degrees')
    parser.add_argument("--translate", type=int, default = (0.1,0.1), required=False, metavar="FILE", help='translate')
    parser.add_argument("--scale", type=int, default = (0.8,1.2) , required=False, metavar="FILE", help='scale')
    parser.add_argument("--shear", type=int, default = 10, required=False, metavar="FILE", help='shear')
    # add arguments for all three attacks
    parser.add_argument("--all_three", action="store_true", required=False, help="whether enable all three attacks")
    args, unparsed = parser.parse_known_args()

    config = get_config(args)
    return args, config


# Prepare evaluation data based off of given params - includes adversarial attacks
def prepare_eval_data(args):
    transformation = transforms.Compose([
            transforms.Resize(size = 256, interpolation=InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=IMAGENET_DEFAULT_MEAN,
                std=IMAGENET_DEFAULT_STD
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
    # if all three attacks are enabled, add the random perspective, random affine and crop transformations
    if args.all_three:
        transformation = transforms.Compose([
        transformation,
        RandomPerspective(distortion_scale = args.distortion_scale),
        RandomAffine(degrees = args.degrees, translate = args.translate, scale = args.scale, shear = args.shear),
        RandomCrop(size = args.crop_size, padding = args.crop_padding)
        ])
    
    # Load the ImageNet-O dataset using the ImageFolder class
    dataset = ImageFolder(root=args.data_path, transform=transformation)

    # Define the batch size for the data loader

    # Create the data loader for dataset
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    return dataloader


def main(args, config):
    if args.shift_attack:
        shift_size = args.shift_size

    print(f"Evaluating model: {config.MODEL.CARD}")


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    dataloader = prepare_eval_data(args)


    checkpoint_path = Path(args.pretrained_path)
    if checkpoint_path.parts[-1] != "default" and checkpoint_path.joinpath("default").exists():  # Given main model directory
        checkpoint_path = checkpoint_path.joinpath("default")
    elif checkpoint_path.parts[-1] != "default": # Given main model directory, but no 'default' folder
        print("The checkpoint path was not as expected")
        raise FileNotFoundError


    checkpoints = []
    for child in checkpoint_path.iterdir():
        checkpoints.append(int(child.stem[11:]))

    last_checkpoint = max(checkpoints)

    for ckpt_epoch in range(0, last_checkpoint+1, args.k):
        # Generate full checkpoint .pth file path
        if not checkpoint_path.joinpath(f"ckpt_epoch_{ckpt_epoch}.pth").exists():  # The checkpoint we want for our given k does not exist. Give warning.
            print(f"WARNING: Checkpoint at epoch {ckpt_epoch} not found. Continuing anyway")
        
        else: # Exists.

            # Getting the correct model
            print(f"Evaluating epoch {ckpt_epoch}:")
            model = timm.create_model(model_name=config.MODEL.CARD, checkpoint_path=str(checkpoint_path.joinpath(f"ckpt_epoch_{ckpt_epoch}.pth")))
            model = model.to(device)


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
                

            # Print the results
            
            print("Checkpoint Epoch {} Time Elapsed {:.4f}".format(ckpt_epoch, end_time))
            print('Checkpoint Epoch {} Average Loss: {:.4f}, Accuracy: {:.4f}, Consistency {:.4f}'.format(ckpt_epoch, average_loss, accuracy, consistency))


if __name__ == '__main__':
    args, config = parse_args()
    main(args, config)