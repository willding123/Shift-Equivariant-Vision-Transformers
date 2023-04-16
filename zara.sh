#!/bin/bash
# bash script for single node gpu training 
#SBATCH --partition=gpu
#SBATCH -N 1
#SBATCH -c 8
##SBATCH --gpus=a100_1g.5gb:1
#SBATCH --gpus=a100:2
#SBATCH --time=1-12:00:00
#SBATCH --job-name=vit
#SBATCH --output=vit.out
#SBATCH --error=vit.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=pding@umd.edu

nvidia-smi
module load cuda/11.6.2/
source ~/scratch.cmsc663/miniconda3/bin/activate
conda activate swin 
cd ~/scratch.cmsc663/Swin-Transformer 

# ViT
# torchrun  --nproc_per_node 4  main.py --cfg configs/swin/poly_vit_tiny_0217.yaml --data-path /fs/cml-datasets/ImageNet/ILSVRC2012 --output /fs/nexus-projects/shift-equivariant_vision_transformer 
export WANDB_MODE="offline"
# export CUDA_LAUNCH_BLOCKING=1

torchrun --nproc_per_node 2  --nnodes 1 --master_port 22301 main.py --cfg configs/vit_small_w.yaml --data-path ~/scratch.cmsc663 --output ~/scratch.cmsc663 

# eval 
# torchrun --nproc_per_node 1  --nnodes 1 --master_port 41342 main.py --eval --cfg configs/swin/pretrained.yaml --data-path ~/scratch.cmsc663 --output ~/scratch.cmsc663 
