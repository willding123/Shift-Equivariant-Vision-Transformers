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
from models.swin_transformer_poly import WindowAttention
from models.swin_transformer_poly import PolySwin
from models.swin_transformer_poly import PolyOrder
from models.swin_transformer import SwinTransformer, SwinTransformerBlock
from torchvision.transforms.functional import affine
from torch.profiler import profile, record_function, ProfilerActivity


class TestShift(unittest.TestCase):
    def test_model(self): 
        x = torch.rand((1,3,224,224)).cuda()
        x1 = torch.roll(x, (1,1), (2,3)).cuda()
        label = torch.tensor((1,1,0,0)).cuda()
        model = SwinTransformerBlock(3, (224,224), 1, qk_scale=1).cuda()
        model1 = PolySwin(img_size=(224,224)).cuda()
        criterion = torch.nn.CrossEntropyLoss()

        for i in range(1):
            s1 = torch.randint(5, (1,)); s2 = torch.randint(5, (1,)); 
            x1 = torch.roll(x, (s1,s2), (2,3))
            x = torch.permute(x, (0,2,3,1)).view(4,-1,3).contiguous()
            x1 = torch.permute(x1, (0,2,3,1)).view(4,-1,3).contiguous()
            model(x)
            model(x1)
            # loss = criterion(output, label)
            # loss1 = criterion(output1, label)
            # assertAlmostEqual(loss, loss1, 5)

            # print(output)
            # print(output1)
            # print(torch.linalg.norm(output-output1))
            # print(f"loss: {loss}, loss1: {loss1}")
            
        # output1 = model(x1)
        # loss = criterion(output, label)
        # loss1 = criterion(output1, label)
        # loss.backward()
        # loss1.backward()

        # print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
        # print(loss)
        # print(loss1)

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
    unittest.main()






