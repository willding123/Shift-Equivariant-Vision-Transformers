#!/bin/bash
#SBATCH --account=scavenger
#SBATCH --partition=scavenger
#SBATCH --qos=scavenger
##SBATCH --exclusive
#SBATCH --gres=gpu:8
##SBATCH --nodes=2
##SBATCH --gpus-per-node=4
#SBATCH --time=2-00:00:00  
#SBATCH --job-name=swin_0204
#SBATCH --output=swin_0204_n1.out
#SBATCH --error=swin_0204_n1.err
##SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=pding@umd.edu
nodes=($(scontrol show hostnames $SLURM_JOB_NODELIST))
node_array=($nodes)
head_node=${node_array[0]}
head_node-ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

module load cuda/11.3.1   
source ~/miniconda3/bin/activate
conda activate swin 
cd /fs/nexus-projects/shift-equivariant_vision_transformer/Swin-Transformer
# batch size for a single gpu is 128
nvidia-smi
# torchrun --nnodes 2 --nproc_per_node 4 rdzv_id=12345 --rdzv_backend=c10d --rdzv_endpoint=$head_node-ip:29500  main.py --cfg configs/swin/poly_swin_tiny_0203.yaml --data-path /fs/cml-datasets/ImageNet/ILSVRC2012 --batch-size 128 --output /fs/nexus-projects/shift-equivariant_vision_transformer --fused_window_process 

torchrun  --nproc_per_node 8  main.py --cfg configs/swin/poly_swin_tiny_0203.yaml --data-path /fs/cml-datasets/ImageNet/ILSVRC2012 --batch-size 512 --output /fs/nexus-projects/shift-equivariant_vision_transformer --fused_window_process 
