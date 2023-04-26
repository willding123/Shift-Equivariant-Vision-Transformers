#!/bin/bash
# bash script for single node gpu training 
#SBATCH --partition=gpu
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --gpus=a100:1
#SBATCH --time=15:00
#SBATCH --job-name=eval
#SBATCH --output=eval.out
#SBATCH --error=eval.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=pding@umd.edu

module load cuda/11.6.2/
source ~/scratch.cmsc663/miniconda3/bin/activate
conda activate swin 
cd ~/scratch.cmsc663/Swin-Transformer 

# eval
# print model name and attack mode
python eval.py --model polyvit --model_card timm/vit_small_patch16_224.augreg_in1k --data_path ~/scratch.cmsc663/val/ --batch_size 512 --pretrained_path  ~/scratch.cmsc663/pvit_small_427_w
python eval.py --model polyvit --model_card timm/vit_small_patch16_224.augreg_in1k --data_path ~/scratch.cmsc663/val/ --batch_size 512 --random_affine --pretrained_path  ~/scratch.cmsc663/pvit_small_427_w
python eval.py --model polyvit --model_card timm/vit_small_patch16_224.augreg_in1k --data_path ~/scratch.cmsc663/val/ --batch_size 512 --random_perspective --pretrained_path  ~/scratch.cmsc663/pvit_small_427_w
python eval.py --model polyvit --model_card timm/vit_small_patch16_224.augreg_in1k --data_path ~/scratch.cmsc663/val/ --batch_size 512 --crop --pretrained_path  ~/scratch.cmsc663/pvit_small_427_w
python eval.py --model polyvit --model_card timm/vit_small_patch16_224.augreg_in1k --data_path ~/scratch.cmsc663/val/ --batch_size 512 --flip --pretrained_path  ~/scratch.cmsc663/pvit_small_427_w
python eval.py --model polyvit --model_card timm/vit_small_patch16_224.augreg_in1k --data_path ~/scratch.cmsc663/val/ --batch_size 512 --random_erasing --pretrained_path  ~/scratch.cmsc663/pvit_small_427_w

# # print model name and attack mode
# python eval.py --model vit --model_card timm/vit_small_patch16_224.augreg_in1k --data_path ~/scratch.cmsc663/val/ --batch_size 512 --pretrained_path ~/scratch.cmsc663/vit_small_411_w/default/ckpt_epoch_206.pth
# python eval.py --model vit --model_card timm/vit_small_patch16_224.augreg_in1k --data_path ~/scratch.cmsc663/val/ --batch_size 512 --random_affine --pretrained_path ~/scratch.cmsc663/vit_small_411_w/default/ckpt_epoch_206.pth
# python eval.py --model vit --model_card timm/vit_small_patch16_224.augreg_in1k --data_path ~/scratch.cmsc663/val/ --batch_size 512 --random_perspective --pretrained_path ~/scratch.cmsc663/vit_small_411_w/default/ckpt_epoch_206.pth
# python eval.py --model vit --model_card timm/vit_small_patch16_224.augreg_in1k --data_path ~/scratch.cmsc663/val/ --batch_size 512 --crop --pretrained_path ~/scratch.cmsc663/vit_small_411_w/default/ckpt_epoch_206.pth
# python eval.py --model vit --model_card timm/vit_small_patch16_224.augreg_in1k --data_path ~/scratch.cmsc663/val/ --batch_size 512 --flip --pretrained_path ~/scratch.cmsc663/vit_small_411_w/default/ckpt_epoch_206.pth
# python eval.py --model vit --model_card timm/vit_small_patch16_224.augreg_in1k --data_path ~/scratch.cmsc663/val/ --batch_size 512 --random_erasing --pretrained_path ~/scratch.cmsc663/vit_small_411_w/default/ckpt_epoch_206.pth
