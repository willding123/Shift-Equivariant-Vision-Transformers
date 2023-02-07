#!/bin/bash
JOB_NAME="swin_0203"
# bash script for single node gpu training 
#SBATCH --account=scavenger
#SBATCH --partition=scavenger
#SBATCH --qos=scavenger
##SBATCH --exclusive
#SBATCH --gres=gpu:8
#SBATCH --time=2-00:00:00  
#SBATCH --job-name=$JOB_NAME
#SBATCH --output=$JOB_NAME.out
#SBATCH --error=$JOB_NAME.err
#SBATCH --cpus-per-task=32
#SBATCH --mem=255
#SBATCH --mail-type=ALL
#SBATCH --mail-user=pding@umd.edu

nodes=($(scontrol show hostnames $SLURM_JOB_NODELIST))
node_array=($nodes)
head_node=${node_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

module load cuda/11.3.1   
source ~/miniconda3/bin/activate
conda activate swin 
cd /fs/nexus-projects/shift-equivariant_vision_transformer/Swin-Transformer
nvidia-smi
torchrun  --nproc_per_node 8  main.py --cfg configs/swin/poly_swin_tiny_0203.yaml --data-path /fs/cml-datasets/ImageNet/ILSVRC2012 --batch-size 512 --output /fs/nexus-projects/shift-equivariant_vision_transformer --fused_window_process 
