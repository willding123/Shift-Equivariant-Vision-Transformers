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
echo "model: polytwins"
echo "attack mode: random_affine"
python eval.py --model polytwins --data_path ~/scratch.cmsc663/val/ --batch_size 256 --random_affine --pretrained_path  ~/scratch.cmsc663/twins_402/default/ckpt_epoch_113.pth

echo "attack mode: random_perspective"
python eval.py --model polytwins --data_path ~/scratch.cmsc663/val/ --batch_size 256 --random_perspective --pretrained_path  ~/scratch.cmsc663/twins_402/default/ckpt_epoch_113.pth

echo "attack mode: crop"
python eval.py --model polytwins --data_path ~/scratch.cmsc663/val/ --batch_size 256 --crop --pretrained_path  ~/scratch.cmsc663/twins_402/default/ckpt_epoch_113.pth

echo "attack mode: flip"
python eval.py --model polytwins --data_path ~/scratch.cmsc663/val/ --batch_size 256 --flip --pretrained_path  ~/scratch.cmsc663/twins_402/default/ckpt_epoch_113.pth

echo "attack mode: random_erasing"
python eval.py --model polytwins --data_path ~/scratch.cmsc663/val/ --batch_size 256 --random_erasing --pretrained_path  ~/scratch.cmsc663/twins_402/default/ckpt_epoch_113.pth

# print model name and attack mode
echo "model: twins"
echo "attack mode: random_affine"
python eval.py --model twins --data_path ~/scratch.cmsc663/val/ --batch_size 256 --random_affine

echo "attack mode: random_perspective"
python eval.py --model twins --data_path ~/scratch.cmsc663/val/ --batch_size 256 --random_perspective

echo "attack mode: crop"
python eval.py --model twins --data_path ~/scratch.cmsc663/val/ --batch_size 256 --crop

echo "attack mode: flip"
python eval.py --model twins --data_path ~/scratch.cmsc663/val/ --batch_size 256 --flip

echo "attack mode: random_erasing"
python eval.py --model twins --data_path ~/scratch.cmsc663/val/ --batch_size 256 --random_erasing
