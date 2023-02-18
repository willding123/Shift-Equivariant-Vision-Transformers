#%%
from models.swin_transformer import SwinTransformer
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.swin_transformer_poly import PolySwin as SwinTransformerPoly

from models.poly_utils import *

#%%

# %%

# Create swin_original_model with following parameters
# MODEL:
#   TYPE: swin
#   NAME: swin_tiny_patch4_window7_224_22kto1k_finetune
#   DROP_PATH_RATE: 0.1
#   SWIN:
#     EMBED_DIM: 96
#     DEPTHS: [ 2, 2, 6, 2 ]
#     NUM_HEADS: [ 3, 6, 12, 24 ]
#     WINDOW_SIZE: 7
# TRAIN:
#   EPOCHS: 30
#   WARMUP_EPOCHS: 5
#   WEIGHT_DECAY: 1e-8
#   BASE_LR: 2e-05
#   WARMUP_LR: 2e-08
#   MIN_LR: 2e-07



# make sure to follow parameters in the config comment. Don't specify any parameters that are not in the config comment
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
# %%

swin_poly_model = SwinTransformerPoly( img_size=(224, 224), patch_size=4, in_chans=3, num_classes=1000, embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24], window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None, attn_drop_rate=0.1)

# %%


# iterate though state_dict of swin_model and swin_poly_model simultaneously
for (k1, v1), (k2, v2) in zip(swin_model.state_dict().items(), swin_poly_model.state_dict().items()):
    print("K1 is: {} and K2 is: {}".format(k1, k2))
    if k1 == k2:
        # check if the key is the same
        if v1.shape == v2.shape:
            print("name and shape the same")
        else:
            print("name the same, shape not the same")

    # check if shape is the same
    if v1.shape == v2.shape:
        print("name, not the same, shape the same")
    else:
        print("name, not the same, shape not the same")
        print("Error: shape not the same")
        # raise ValueError("shape not the same")
# %%

# find names in swin_model but not in swin_poly_model and vice versa
swin_model_keys = set(swin_model.state_dict().keys())
swin_poly_model_keys = set(swin_poly_model.state_dict().keys())
print("swin_model_keys - swin_poly_model_keys: {}".format(swin_model_keys - swin_poly_model_keys))
print("swin_poly_model_keys - swin_model_keys: {}".format(swin_poly_model_keys - swin_model_keys))
print("len(swin_model_keys): {}".format(len(swin_model_keys)))
print("len(swin_poly_model_keys): {}".format(len(swin_poly_model_keys)))

# %%

# unpaired keys and values in swin_poly_model

unpaired_keys_values_swin_poly_model = {}
for k, v in swin_poly_model.state_dict().items():
    if k not in swin_model_keys:
        unpaired_keys_values_swin_poly_model[k] = v
        
print("unpaired_keys_values_swin_poly_model: {}".format(unpaired_keys_values_swin_poly_model))

unpaired_keys_values_swin_model = {}
for k, v in swin_model.state_dict().items():
    if k not in swin_poly_model_keys:
        unpaired_keys_values_swin_model[k] = v

print("unpaired_keys_values_swin_model: {}".format(unpaired_keys_values_swin_model))

# %%

# check if any unpaired weights match shapes
for k, v in unpaired_keys_values_swin_poly_model.items():
    for k2, v2 in unpaired_keys_values_swin_model.items():
        if v.shape == v2.shape:
            print("v.shape: {} and v2.shape: {}".format(v.shape, v2.shape))
            print("v: {}".format(v))
            print("v2: {}".format(v2))
            print("k: {}".format(k))
            print("k2: {}".format(k2))
            print("")

# %%

def copy_model_weights(swin_model: torch.nn.Module, swin_poly_model: torch.nn.Module) -> torch.nn.Module:
    # loop only over swin_poly_model keys
    weights_copied = 0
    for k, v in swin_poly_model.state_dict().items():
        if k in swin_model.state_dict().keys():
            # if shapes match
            if v.shape == swin_model.state_dict()[k].shape:
                # if the key is in swin_model_keys, copy the weights
                swin_poly_model.state_dict()[k].copy_(swin_model.state_dict()[k])
                weights_copied += 1
                if k == "patch_embed.proj.weight":
                    print("k: {}".format(k))
                    print("Weights in swin_model")
                    print(swin_model.state_dict()[k])
                    print("Weights in swin_poly_model")
                    print(swin_poly_model.state_dict()[k])

    print("weights_copied: {}".format(weights_copied))
    return swin_poly_model

# %%

swin_poly_model = copy_model_weights(swin_model, swin_poly_model)
# %%
def test_copied_model( swin_poly_model: torch.nn.Module, swin_model: torch.nn.Module):
    # iterate though state_dict of swin_model and swin_poly_model simultaneously
    for k, v in swin_poly_model.state_dict().items():
        if k in swin_model.state_dict().keys():
            if v.shape == swin_model.state_dict()[k].shape:
                assert torch.allclose(v, swin_model.state_dict()[k]), "v and swin_model.state_dict()[k] not equal"


test_copied_model(swin_poly_model, swin_model)
# %%

