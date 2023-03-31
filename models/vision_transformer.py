""" Vision Transformer (ViT) in PyTorch

A PyTorch implement of Vision Transformers as described in:

'An Image Is Worth 16 x 16 Words: Transformers for Image Recognition at Scale'
    - https://arxiv.org/abs/2010.11929

`How to train your ViT? Data, Augmentation, and Regularization in Vision Transformers`
    - https://arxiv.org/abs/2106.10270

`FlexiViT: One Model for All Patch Sizes`
    - https://arxiv.org/abs/2212.08013

The official jax code is released and available at
  * https://github.com/google-research/vision_transformer
  * https://github.com/google-research/big_vision

Acknowledgments:
  * The paper authors for releasing code and weights, thanks!
  * I fixed my class token impl based on Phil Wang's https://github.com/lucidrains/vit-pytorch
  * Simple transformer style inspired by Andrej Karpathy's https://github.com/karpathy/minGPT
  * Bert reference code checks against Huggingface Transformers and Tensorflow Bert

Hacked together by / Copyright 2020, Ross Wightman
"""


import logging
import math
from collections import OrderedDict
from functools import partial
from typing import Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from models.poly_utils import *
from timm.models.layers import PatchEmbed, Mlp, DropPath, trunc_normal_, lecun_normal_
import timm 
class PolyViT(timm.models.vision_transformer.VisionTransformer):
    def __init__(self, model_type, pretrained = False, **kwargs):
        super().__init__()
        model = timm.create_model(model_type, pretrained=pretrained)
        # copy all model's attributes to self
        for k, v in model.__dict__.items():
                self.__setattr__(k, v)
        self.no_embed_class = kwargs.get("no_embed_class", False)
        # self.patch_embed = PolyPatchEmbed(
        #     patch_size = self.patch_embed.patch_size,
        #     in_chans = self.patch_embed.proj.in_channels, 
        #     out_chans = self.patch_embed.proj.out_channels,
        #     norm_layer = self.patch_embed.norm
        #     )
        self.pos_embed = None
        self.pos_conv = PosConv(self.embed_dim, self.embed_dim)
        
    def _pos_embed(self, x):
        if self.no_embed_class:
            # deit-3, updated JAX (big vision)
            # position embedding does not overlap with class token, add then concat
            x = self.pos_conv(x)
            if self.cls_token is not None:
                x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        else:
            # original timm, JAX, and deit vit impl
            # pos_embed has entry for class token, concat then add
            if self.cls_token is not None:
                x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
            x = self.pos_conv(x)
        return self.pos_drop(x)

    