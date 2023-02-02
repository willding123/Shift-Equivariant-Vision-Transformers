#!/bin/bash
#SBATCH --qos=high
#SBATCH --exclusive
#SBATCH --gres=gpu:4
#SBATCH --time=1-00:00:00  
#SBATCH --mem=64G
#SBATCH --mail-type=END
#SBATCH --mail-user=pding@umd.edu

module load cuda/11.3.1   
source ~/miniconda3/bin/activate
conda activate swin 
cd /fs/nexus-projects/shift-equivariant_vision_transformer/Swin-Transformer
# batch size for a single gpu is 128
nvidia-smi
torchrun --nproc_per_node 4 --master_port 12345  main.py --cfg configs/swin/swin_tiny_patch4_window7_224_22k.yaml --data-path /fs/cml-datasets/ImageNet/ILSVRC2012 --batch-size 128 --output /fs/nexus-projects/shift-equivariant_vision_transformer --fused_window_process 

# python main.py --cfg configs/swin/my_swin_tiny_patch4_window7_1k.yaml --data-path /fs/cml-datasets/ImageNet/ILSVRC2012 --batch-size 8 --output /fs/nexus-scratch/pding/output --fused_window_process 
