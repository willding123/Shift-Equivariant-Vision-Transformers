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
        model = PolyTwins("twins_svt_small", pretrained=True)

    elif model_type == "polyvit":
        if config.MODEL.CARD:
            model = PolyViT(config.MODEL.CARD, pretrained=True)
        else:
            model = PolyViT("hf_hub:timm/vit_base_patch16_224.augreg_in21k_ft_in1k", pretrained=True)
            
    elif model_type == "swin_poly_pretrained":
        swin_model = SwinTransformer( 
        img_size=224, 
        patch_size=4, 
        in_chans=3, 
        num_classes=1000, 
        embed_dim=96, 
        depths=[2, 2, 6, 2], 
        num_heads=[3, 6, 12, 24], 
        window_size=7, 
        drop_path_rate=0.1, 
        )
        swin_model.load_state_dict(torch.load("swin_tiny_patch4_window7_224_22kto1k_finetune.pth")["model"])
        # swin_poly_model = PolySwin( img_size=(224, 224), patch_size=4, in_chans=3, num_classes=1000, embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24], window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None, attn_drop_rate=0.1)
        # swin_poly_model = copy_model_weights(swin_model, swin_poly_model)
        model = nn.Sequential(
            PolyOrderModule(grid_size=(56,56), patch_size=(4,4)),
            swin_model
        )

    else:
        raise NotImplementedError(f"Unkown model: {model_type}")

    return model
