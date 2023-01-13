import unittest
import os
import time
import json
import random
import argparse
import datetime
import numpy as np
import traceback

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
from utils import * 


np.random.seed(111)
torch.random.manual_seed(111)
class TestShift(unittest.TestCase):
    def setUp(self):
        img_size = (224, 224)
        patch_size = 7
        in_chans = 3
        norm_layer = nn.LayerNorm
        patches_resolution = (32,32)
        drop_rate = 0.1 
        embed_dim = 96
        dim = 96
        # poly swin transformer block 
        # self.model = nn.Sequential(
            # PolyPatch(input_resolution = img_size, patch_size = patch_size, in_chans = in_chans, out_chans = embed_dim, norm_layer=norm_layer)
            # nn.Dropout(p=drop_rate),
            # BasicLayer(dim=dim,
            #                    input_resolution=(patches_resolution[0],
            #                                      patches_resolution[1]),
            #                    depth=2,
            #                    num_heads=3,
            #                    window_size=7,
            #                    norm_layer=norm_layer,
            #                    downsample=PolyPatch
        # )
        # )
        print("polypatch")
        self.model = PolyPatch(input_resolution = img_size, patch_size = patch_size, in_chans = in_chans, out_chans = embed_dim, norm_layer=norm_layer)
        # swin transformer block 
        # self.model1 = nn.Sequential(
        #     PatchEmbed(
        #     img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
        #     norm_layer=norm_layer),
        #     nn.Dropout(p=drop_rate),
        #     BasicLayer(dim=int(96),
        #                        input_resolution=(patches_resolution[0],
        #                                          patches_resolution[1]),
        #                        depth=2,
        #                        num_heads=3,
        #                        window_size=7,
        #                        norm_layer=norm_layer,
        #                        downsample=PatchMerging
        #     )

        # )
        print("patch embed")
        self.model1 = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,norm_layer=norm_layer)
        print("move to gpu")
        self.model = self.model.cuda()
        self.model1 = self.model1.cuda()


    def show_features(self): 
        print("starts...")
        x = torch.rand((4,3,224,224)).cuda()
        # shifts = tuple(np.random.randint(0,32,2))
        shifts = (37,43)
        x1 = torch.roll(x, shifts, (2,3)).cuda()
        # poly swin output 
        print("prediction")
        y = self.model(x).cpu().detach().numpy()
        y1 = self.model(x1).cpu().detach().numpy()
        # swin output 
        z = self.model1(x).cpu().detach().numpy()
        z1 = self.model1(x1).cpu().detach().numpy()
        print("starts for loop")
        for img_id in range(y.shape[0]):
            print(f"Image {img_id}")
            try:
                y1_img = y1[img_id]
                y_img = y[img_id]
                shift_size = find_shift(y_img, y1_img, early_break=False)
                dist = shift_and_compare(y_img, y1_img, shift_size)
                print(f"shift size: {shift_size}")
                print(f"Distance using poly: {dist}")
                dist_orig = np.linalg.norm(y1_img - y_img)
                print(f"Distance w.o: {dist_orig}")
            except:
                print(f"Failed for image {img_id}")




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

    def test_polyorder(self):
        patch_sizes = [(2,2), (4,4), (7,7), (8,8), (14,14)]
        patches_resolutions = [(int(224/x[0]),int(224/x[1])) for x in patch_sizes]
        # initial setup: for each patch size experiment
        # we set first polyphase of each image to highest energy, roll each image randomly 
        for i in range(len(patch_sizes)):
            x = torch.rand((4,3,224,224))
            for j in range(x.shape[0]):
                x[j, :, 0::patches_resolutions[i][0], 0::patches_resolutions[i][1]] = 2 
                x = np.roll(x, tuple(np.random.randint(0,patches_resolutions[i], 2)), axis = (2,3))
            x = torch.tensor(x)
            y = PolyOrder.apply(x, patches_resolutions[i], patch_sizes[i], 2, False)
            assert torch.all(y[:,:,0::patches_resolutions[i][0], 0::patches_resolutions[i][1]]==2).item()
        
        

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
    start = time.time()
    try: 
        test.setUp()
        test.show_features()
    except:
        traceback.print_exc()
    finally: 
        end = time.time() - start 
        print(f"Time elapsed: {end}")
    # unittest.main()






