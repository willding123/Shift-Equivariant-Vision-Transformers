# Polyphase Swin Transformer

Cloned from Swin Transformer [repo](https://github.com/microsoft/Swin-Transformer). Updated Swin transformer (with polyphase implementations) is in models/swin_transformer_poly.py.

## Introduction

This is the official implementation of Polyphase Swin Transformer which proposes an adaptive polyphase anchoring algorithm that can be seamlessly integrated into vision transformer models to ensure shift-equivariance in patch embedding and subsampled attention modules, such as window attention and global subsampled attention. Our algorithms enable ViT, and its variants such as Twins to achieve 100% consistency with respect to input shift.

<!-- include image local path imgs/PolyModels.png -->

<img src="imgs/PolyModels.png" width="800" />  


## Installation


## Usage

PolyPatch and PolyOrder from models/swin_transformer_poly.py can be used as drop-in  modules for helping achieve shift-equivariance in patch embedding and subsampled attention modules. 

Integration of them in common ViT - polyVIT and Twin - PolyTwins architectures can be found in models/vision_transformer.py and models/polytwins.py respectively.

## validation

Pre-trained weights can be supplied to eval.py to evaluate the performance, consistency, and robustness of the models.

Some common arguments are:

```
model : model name
data-path : path to dataset
model_card : path to model card

```

## Training
Models can be trained using train.py. confi files in configs directory can be used to set training and model hyperparameters and architecture.
Sample coomand for training polyVIT on ImageNet dataset is:

```
python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --cfg configs/swin/polyvit_b_0304.yaml --data-path ImageNet/ILSVRC2012 --output out --batch-size 128 --num_workers 8 --model polyvit --model_card polyvit_b_0304 --dist-eval --eval --pretrained
```