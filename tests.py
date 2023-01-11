import unittest
import os
import time
import json
import random
import argparse
import datetime
import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP

# from models.swin_transformer_poly import poly_order
from models.swin_transformer_poly import *
# from models.swin_transformer import *
from models.swin_transformer import SwinTransformer, SwinTransformerBlock
from torchvision.transforms.functional import affine
from torch.profiler import profile, record_function, ProfilerActivity

import matplotlib.pyplot as plt 

np.random.seed(123)
torch.random.manual_seed(123)
class TestShift(unittest.TestCase):
    def setUp(self):
        img_size = (224, 224)
        patch_size = 2
        in_chans = 3
        norm_layer = nn.LayerNorm
        patches_resolution = (112,112)
        drop_rate = 0.1 
        embed_dim = 96
        dim = 96
        # poly swin transformer block 
        self.model = nn.Sequential(
            PolyPatch(input_resolution = img_size, patch_size = patch_size, in_chans = in_chans, out_chans = embed_dim, norm_layer=norm_layer),
            nn.Dropout(p=drop_rate),
            BasicLayer(dim=dim,
                               input_resolution=(patches_resolution[0],
                                                 patches_resolution[1]),
                               depth=2,
                               num_heads=3,
                               window_size=7,
                               norm_layer=norm_layer,
                               downsample=PolyPatch
        )
        )
        self.model.cuda()
        # swin transformer block 
        self.model1 = nn.Sequential(
            PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer),
            nn.Dropout(p=drop_rate),
            BasicLayer(dim=int(96),
                               input_resolution=(patches_resolution[0],
                                                 patches_resolution[1]),
                               depth=2,
                               num_heads=3,
                               window_size=7,
                               norm_layer=norm_layer,
                               downsample=PatchMerging
            )

        )
        self.model1.cuda()

    def show_features(self): 
        x = torch.rand((4,3,224,224)).cuda()
        shifts = tuple(np.random.randint(0,32,2))
        x1 = torch.roll(x, shifts, (2,3)).cuda()
        print(shifts)
        # poly swin output 
        y = self.model(x)
        y = y.cpu().detach().numpy()
        y1 = self.model(x1).cpu().detach().numpy()
        # swin output 
        z = self.model1(x).cpu().detach().numpy()
        z1 = self.model1(x1).cpu().detach().numpy()

        np.save("original.npy", y)
        np.save("shifted.npy", y1)
        np.save("swin.npy", z)
        np.save("swin1.npy", z1)


        # print(y.shape)
        # self.model(x1[0])


    # def test_model(self): 
    #     x = torch.rand((4,3,224,224)).cuda()
    #     x1 = torch.roll(x, (1,1), (2,3)).cuda()
    #     label = torch.tensor((1,1,0,0)).cuda()
    #     # model = SwinTransformer(3, (224,224), 1, qk_scale=1).cuda()
    #     model1 = PolySwin(img_size=(224,224)).cuda()
    #     criterion = torch.nn.CrossEntropyLoss()
    #     print(model(x))
    #     print(model1(x1))
    #     torch.save(model(x), "tensor.pt")
    #     torch.save(model1(x), "tensor1.pt")

    #     # for i in range(1):
    #     #     s1 = torch.randint(5, (1,)); s2 = torch.randint(5, (1,)); 
    #     #     x1 = torch.roll(x, (s1,s2), (2,3))
            
    #         # loss = criterion(output, label)
    #         # loss1 = criterion(output1, label)
    #         # assertAlmostEqual(loss, loss1, 5)

    #         # print(output)
    #         # print(output1)
    #         # print(torch.linalg.norm(output-output1))
    #         # print(f"loss: {loss}, loss1: {loss1}")
            
    #     # output1 = model(x1)
    #     # loss = criterion(output, label)
    #     # loss1 = criterion(output1, label)
    #     # loss.backward()
    #     # loss1.backward()

    #     # print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
    #     # print(loss)
    #     # print(loss1)

    # def test_polyorder(self):
    #     x = torch.rand(128).cuda()
    #     x = x.view(2,1, 8,8)
    #     x[0,:,1::4, 0::4] = 1 
    #     x[1,:,2::4, 2::4] = 1
    #     print(x)
    #     patch_size = (2,2); grid_size = (4,4)
    #     x = PolyOrder.apply(x, grid_size, patch_size)
    #     print(x)

    # def test_window_atten(self): 
    #     x = torch.rand(64)
    #     x = x.view(1,1, 8,8)
    #     x[0,:,1::2, 0::2] = 1 
    #     B,C,H,W = x.shape
    #     x = x.view(B, H, W, C)
    #     x = x.view(B, C, H, W)
    #     xr = torch.roll(x, (1,1), (1,2))
    #     x = poly_order(x, (2,2), (4,4))
    #     xr =  poly_order(xr, (2,2), (4,4))
    #     x = x.view(B, H, W, C)
    #     xr = x.view(B, H, W, C)

    #     print(x-xr)
    #     x = x.view(B*4,-1, C)
    #     xr = xr.view(B*4, -1 , C)
    #     model = WindowAttention( 1, (4,4), 1)
    #     print(model(x))
    #     print(model(xr))

if __name__ == "__main__":
    test = TestShift()
    test.setUp()
    test.show_features()
    # unittest.main()






