#!/bin/bash
# bash script for single node gpu training 
#SBATCH --qos=high
#SBATCH --partition=dpart
#SBATCH --account=furongh
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --time=1-12:00:00  
#SBATCH --job-name=brvit
#SBATCH --output=brvit.out
#SBATCH --error=brvit.err
#SBATCH --mem=64G 
#SBATCH --cpus-per-task=16
#SBATCH --mail-type=ALL
#SBATCH --mail-user=pding@umd.edu
nvidia-smi
module load cuda/11.3.1   
source ~/miniconda3/bin/activate
conda activate swin 
cd ~/Swin-Transformer
# torchrun  --nproc_per_node 8  main.py --cfg configs/swin/poly_swin_tiny_0220.yaml --data-path /fs/cml-datasets/ImageNet/ILSVRC2012 --batch-size 128 --output /fs/nexus-projects/shift-equivariant_vision_transformer --fused_window_process 
# torchrun  --nproc_per_node 4  main.py --cfg configs/swin/poly_swin_tiny_0216.yaml --data-path /fs/cml-datasets/ImageNet/ILSVRC2012 --batch-size 128 --output /fs/nexus-projects/shift-equivariant_vision_transformer --fused_window_process 


# ViT
# torchrun  --nproc_per_node 4  main.py --cfg configs/swin/poly_vit_tiny_0217.yaml --data-path /fs/cml-datasets/ImageNet/ILSVRC2012 --output /fs/nexus-projects/shift-equivariant_vision_transformer 

torchrun  --nproc_per_node 4  --master_port 29499  main.py --cfg configs/swin/vit_relpos_b_0304.yaml --data-path /fs/cml-datasets/ImageNet/ILSVRC2012 --output /cmlscratch/pding 

