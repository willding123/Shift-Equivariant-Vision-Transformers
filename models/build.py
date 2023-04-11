# --------------------------------------------------------
#  code copied from:
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------
import torch 
from .swin_transformer import SwinTransformer
from .swin_transformer_v2 import SwinTransformerV2
from .swin_transformer_moe import SwinTransformerMoE
from .swin_transformer_poly import PolySwin
from .swin_mlp import SwinMLP
from .simmim import build_simmim
import timm 
from .poly_utils import *
from .vision_transformer import PolyViT
from .polytwins import PolyTwins


def build_model(config, is_pretrain=False):
    model_type = config.MODEL.TYPE

    # accelerate layernorm
    if config.FUSED_LAYERNORM:
        try:
            import apex as amp
            layernorm = amp.normalization.FusedLayerNorm
        except:
            layernorm = None
            print("To use FusedLayerNorm, please install apex.")
    else:
        import torch.nn as nn
        layernorm = nn.LayerNorm

    if model_type == 'swin':
        model = SwinTransformer(img_size=config.DATA.IMG_SIZE,
                                patch_size=config.MODEL.SWIN.PATCH_SIZE,
                                in_chans=config.MODEL.SWIN.IN_CHANS,
                                num_classes=config.MODEL.NUM_CLASSES,
                                embed_dim=config.MODEL.SWIN.EMBED_DIM,
                                depths=config.MODEL.SWIN.DEPTHS,
                                num_heads=config.MODEL.SWIN.NUM_HEADS,
                                window_size=config.MODEL.SWIN.WINDOW_SIZE,
                                mlp_ratio=config.MODEL.SWIN.MLP_RATIO,
                                qkv_bias=config.MODEL.SWIN.QKV_BIAS,
                                qk_scale=config.MODEL.SWIN.QK_SCALE,
                                drop_rate=config.MODEL.DROP_RATE,
                                drop_path_rate=config.MODEL.DROP_PATH_RATE,
                                ape=config.MODEL.SWIN.APE,
                                norm_layer=layernorm,
                                patch_norm=config.MODEL.SWIN.PATCH_NORM,
                                use_checkpoint=config.TRAIN.USE_CHECKPOINT,
                                fused_window_process=config.FUSED_WINDOW_PROCESS)
    elif model_type == "swin_poly": 
        model = PolySwin(img_size=(config.DATA.IMG_SIZE, config.DATA.IMG_SIZE),
                            patch_size=config.MODEL.SWIN.PATCH_SIZE,
                            in_chans=config.MODEL.SWIN.IN_CHANS,
                            num_classes=config.MODEL.NUM_CLASSES,
                            embed_dim=config.MODEL.SWIN.EMBED_DIM,
                            depths=config.MODEL.SWIN.DEPTHS,
                            num_heads=config.MODEL.SWIN.NUM_HEADS,
                            window_size=config.MODEL.SWIN.WINDOW_SIZE,
                            mlp_ratio=config.MODEL.SWIN.MLP_RATIO,
                            qkv_bias=config.MODEL.SWIN.QKV_BIAS,
                            qk_scale=config.MODEL.SWIN.QK_SCALE,
                            drop_rate=config.MODEL.DROP_RATE,
                            drop_path_rate=config.MODEL.DROP_PATH_RATE,
                            ape=config.MODEL.SWIN.APE,
                            norm_layer=layernorm,
                            patch_norm=config.MODEL.SWIN.PATCH_NORM,
                            use_checkpoint=config.TRAIN.USE_CHECKPOINT,
                            fused_window_process=config.FUSED_WINDOW_PROCESS)
    

    elif model_type == "polytwins":
        if config.MODEL.CARD:
            model = PolyTwins(config.MODEL.CARD, pretrained=config.MODEL.PRETRAINED)
        else:
            model = PolyTwins("twins_svt_small", pretrained=config.MODEL.PRETRAINED)

    elif model_type == "polyvit":
        if config.MODEL.CARD:
            model = PolyViT(config.MODEL.CARD, pretrained=config.MODEL.PRETRAINED)
        else:
            model = PolyViT("hf_hub:timm/vit_base_patch16_224.augreg_in21k_ft_in1k", pretrained=config.MODEL.PRETRAINED)
    elif model_type == "vit":
        if config.MODEL.CARD:
            model = timm.create_model(config.MODEL.CARD, pretrained=config.MODEL.PRETRAINED)
        else:
            model = timm.create_model("timm/vit_base_patch16_224.augreg_in21k_ft_in1k", pretrained=config.MODEL.PRETRAINED)
            
    elif model_type == "twins":
        if config.MODEL.CARD:
            model = timm.create_model(config.MODEL.CARD, pretrained=config.MODEL.PRETRAINED)
        else:
            model = timm.create_model("twins_svt_small", pretrained=config.MODEL.PRETRAINED)
    
    else:
        raise NotImplementedError(f"Unkown model: {model_type}")

    return model
