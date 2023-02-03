#!/bin/bash
#SBATCH --account=scavenger
#SBATCH --partition=scavenger
#SBATCH --qos=scavenger
#SBATCH --exclusive
#SBATCH --gres=gpu:8
#SBATCH --time=2-00:00:00  
#SBATCH --mem=128G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=pding@umd.edu

module load cuda/11.3.1   
source ~/miniconda3/bin/activate
conda activate swin 
cd /fs/nexus-projects/shift-equivariant_vision_transformer/Swin-Transformer
# batch size for a single gpu is 128
nvidia-smi
torchrun --nproc_per_node 8 --master_port 12345  main.py --cfg configs/swin/poly_swin_tiny_patch4_window7_1k_0202.yaml --data-path /fs/cml-datasets/ImageNet/ILSVRC2012 --batch-size 128 --output /fs/nexus-projects/shift-equivariant_vision_transformer --fused_window_process 
