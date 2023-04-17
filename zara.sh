#!/bin/bash
# bash script for single node gpu training 
#SBATCH --partition=gpu
#SBATCH -c 8
#SBATCH --mem=40G
#SBATCH --ntasks=1
##SBATCH --gpus=a100_1g.5gb:1
#SBATCH --gpus=a100:1
#SBATCH --time=1-12:00:00
#SBATCH --job-name=pbvit
#SBATCH --output=pbvit.out
#SBATCH --error=pbvit.err
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
echo "Number of nodes used: $(scontrol show hostname $SLURM_JOB_NODELIST | wc -l)"
torchrun --nproc_per_node 1 --nnodes 1 --master_port 29280 main.py --cfg configs/pvit_base_w.yaml --data-path ~/scratch.cmsc663 --output ~/scratch.cmsc663 

# eval 
# torchrun --nproc_per_node 1  --nnodes 1 --master_port 41342 main.py --eval --cfg configs/swin/pretrained.yaml --data-path ~/scratch.cmsc663 --output ~/scratch.cmsc663 
