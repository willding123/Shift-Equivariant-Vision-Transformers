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
        self.img_size = (224, 224)
        self.patch_size = 7
        self.in_chans = 3
        self.norm_layer = nn.LayerNorm
        self.patches_resolution = (32,32)
        self.drop_rate = 0.1 
        self.embed_dim = 96
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
        self.model = self.model.cuda()
        self.model1 = self.model1.cuda()


    def test_patch_embed(self): 
        """
        Should expect embeddings of an original image and its shifted copy to have a bijective matching
        """
        self.model = PolyPatch(input_resolution = self.img_size, patch_size = self.patch_size, in_chans = self.in_chans, 
                                out_chans = self.embed_dim, norm_layer=self.norm_layer)
        self.model1 = PatchEmbed(img_size=self.img_size, patch_size=self.patch_size, in_chans=self.in_chans, embed_dim=self.embed_dim,norm_layer=self.norm_layer)
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

        # np.save("original.npy", y)
        # np.save("shifted.npy", y1)
        # np.save("swin.npy", z)
        # np.save("swin1.npy", z1)
        for img_id in range(y.shape[0]):

            # image 1 cannot find a candidate
            print(f"Image {img_id}")
            try:
                y1_img = y1[img_id]
                y_img = y[img_id]
                confirm_bijective_matches(y_img, y1_img)
                print("There is a bijection between y_img and y1_img")
                
            except: 
                print("Failed")


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


        
    def test_polyorder(self):
        patches_resolutions = [(2,2), (4,4), (7,7), (8,8), (14,14)]
        patch_sizes = [(int(224/x[0]),int(224/x[1])) for x in patches_resolutions]
        # initial setup: for each patch size experiment
        # we set first polyphase of each image to highest energy, roll each image randomly 
        for i in range(len(patch_sizes)):
            x = torch.rand((4,3,224,224))
            for j in range(x.shape[0]):
                x[j, :, 0::patch_sizes[i][0], 0::patch_sizes[i][1]] = 2 
                x = np.roll(x, tuple(np.random.randint(0,patches_resolutions[i], 2)), axis = (2,3))
            x = torch.tensor(x)
            y = PolyOrder.apply(x, patches_resolutions[i], patch_sizes[i], 2, False)
            assert torch.all(y[:,:,0::patch_sizes[i][0], 0::patch_sizes[i][1]]==2).item()
        print("Done!")
    
    def test_polyorder2(self): 
        """Test whether poly order will yield the same predictions for an image and its shifted copy"""
        x = torch.rand((1,1,14,14)).cuda()
        # shifts = tuple(np.random.randint(0,32,2))
        shifts = (37,43)
        x1 = torch.roll(x, shifts, (2,3)).cuda()
        grid_size = (2,2); patch_size = (7,7); 
        y = PolyOrder.apply(x, grid_size, patch_size)
        y1 = PolyOrder.apply(x1, grid_size, patch_size)
        # assert torch.linalg.norm(y-y1) < 1e-3
        assert torch.linalg.norm(y[:,:,0::patch_size[0]]) == torch.linalg.norm(y1[:,:,0::patch_size[0]]) 
        print(y)
        print(y1)

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
        print("Exception!")
        traceback.print_exc()
    finally: 
        end = time.time() - start 
        print(f"Time elapsed: {end}")
    # unittest.main()






