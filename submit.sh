#!/bin/bash
# bash script for single node gpu training 
#SBATCH --account=scavenger
#SBATCH --partition=scavenger
#SBATCH --qos=scavenger
##SBATCH --qos=high
##SBATCH --exclusive
#SBATCH --gres=gpu:8
#SBATCH --time=1-00:00:00  
#SBATCH --job-name=vit_0217
#SBATCH --output=vit_0217.out
#SBATCH --error=vit_0217.err
#SBATCH --mem=128G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=pding@umd.edu

nvidia-smi
module load cuda/11.3.1   
source ~/miniconda3/bin/activate
conda activate swin 
cd /fs/nexus-projects/shift-equivariant_vision_transformer/Swin-Transformer
# torchrun  --nproc_per_node 8  main.py --cfg configs/swin/poly_swin_tiny_0204.yaml --data-path /fs/cml-datasets/ImageNet/ILSVRC2012 --batch-size 128 --output /fs/nexus-projects/shift-equivariant_vision_transformer --fused_window_process 
# torchrun  --nproc_per_node 4  main.py --cfg configs/swin/poly_swin_tiny_0216.yaml --data-path /fs/cml-datasets/ImageNet/ILSVRC2012 --batch-size 128 --output /fs/nexus-projects/shift-equivariant_vision_transformer --fused_window_process 


# ViT
torchrun  --nproc_per_node 8  main.py --cfg configs/swin/poly_vit_tiny_0217.yaml --data-path /fs/cml-datasets/ImageNet/ILSVRC2012 --output /fs/nexus-projects/shift-equivariant_vision_transformer 
