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
        self.patch_size = 4
        self.in_chans = 3
        self.norm_layer = nn.LayerNorm
        self.patches_resolution = (self.img_size[0]//self.patch_size, self.img_size[1]//self.patch_size)
        self.drop_rate = 0.0 
        self.embed_dim = 96
        self.mlp_ratio = 4
        self.window_size = 7


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

    def test_swin_block(self):
        # test poly swin transformer block
        patch_embed = PolyPatch(input_resolution = self.img_size, patch_size = self.patch_size, in_chans = self.in_chans,
                        out_chans = self.embed_dim, norm_layer=self.norm_layer).cuda()
        pos_drop = nn.Dropout(p=self.drop_rate).cuda()
        blocks = nn.ModuleList([
            SwinTransformerBlock(dim=self.embed_dim, input_resolution=(self.patches_resolution[0], self.patches_resolution[1]),
                                 num_heads=3, window_size=self.window_size,
                                 shift_size=0 if (i % 2 == 0) else self.window_size // 2,
                                 mlp_ratio=self.mlp_ratio,
                                 qkv_bias=True, qk_scale=None,
                                 drop=0.0, attn_drop=0.0,
                                 drop_path=0,
                                 norm_layer=self.norm_layer,
                                 fused_window_process=True)
            for i in range(2)]).cuda()
        # self.model = nn.Sequential(patch_embed, pos_drop, blocks).cuda()

        def pred(x):
            x = patch_embed(x)
            x = pos_drop(x)
            for blk in blocks:
                x = blk(x)
            return x 

        def reorder(x, idx = 0):
            blk = blocks[idx]
            H, W = blk.input_resolution
            B, L, C = x.shape
            blk.grid_size = (H // blk.window_size, W // blk.window_size)
            assert L == H * W, "input feature has wrong size"
            # idx stands for the index of the block
            x = blk.norm1(x)
            x = torch.permute(x.view(B,H,W,C), (0,3,1,2))
            # # rearrange x based on max polyphase 
            x =  PolyOrder.apply(x, blk.grid_size, to_2tuple(blk.window_size))
            x = torch.permute(x, (0,2,3,1)).contiguous()
            return x
        
        def check_polyphase(t, t1, shifts = None):
            # t: original tensor, t1: shifted tensor
            # check if the polyphase is correct 
            # t: B, H, W, C
            # t1: B, H, W, C

            B, H, W, C = t.shape
            patch_reso = (H//7, W//7)
            
            norms = []
            for i, shift in enumerate(shifts):
                p0 = t[i:, 0::patch_reso[0] , 0::patch_reso[1], :]
                p1 = t1[i:, 0::patch_reso[0] , 0::patch_reso[1], :]
                p0_shifted = torch.roll(p0, shifts = shift, dims = (0, 1))
                norms.append(torch.linalg.norm(p0_shifted - p1))
            print(norms)
            return norms
            # B, H, W, C = t.shape
            # patch_reso = (H//7, W//7)
            # print(torch.linalg.norm(t[:, 0::patch_reso[0], 0::patch_reso[1], :]) - torch.linalg.norm(t1[:, 0::patch_reso[0], 0::patch_reso[1], :]))
            # assert torch.linalg.norm(t[:, 0::patch_reso[0], 0::patch_reso[1], :]) - torch.linalg.norm(t1[:, 0::patch_reso[0], 0::patch_reso[1], :]) < 1e-4, "polyphase is not correct"

            
        
        def cyclic_shift(x, idx = 0):
            blk = blocks[idx] 
            B, H, W, C = x.shape
            L = H*W
                # cyclic shift
            if blk.shift_size > 0:
                if not blk.fused_window_process:
                    shifted_x = torch.roll(x, shifts=(-blk.shift_size, -blk.shift_size), dims=(1, 2))
                    # partition windows
                    x_windows = window_partition(shifted_x, blk.window_size)  # nW*B, window_size, window_size, C
                else:
                    x_windows = WindowProcess.apply(x, B, H, W, C, -blk.shift_size, blk.window_size)
            else:
                shifted_x = x
                # partition windows
                x_windows = window_partition(shifted_x, blk.window_size)  # nW*B, window_size, window_size, C
            
            x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

            return x_windows
        
        def check_window(t, t1):
            count = 0 
            for i in range(t.shape[0]):
                for j in range(t1.shape[0]):
                    if torch.linalg.norm(t[i]-t1[j]) < 0.1:
                        count += 1
            print(f"count: {count}")
            assert count  == t.shape[0]

        def attention(x_windows, idx = 0):
            C = x_windows.shape[-1]
            blk = blocks[idx]
            attn_windows = blk.attn(x_windows, mask=blk.attn_mask)  # nW*B, window_size*window_size, C
            attn_windows = attn_windows.view(-1, blk.window_size, blk.window_size, C)
            return attn_windows

        def reverse_cyclic_shift(attn_windows, shortcut, idx = 0):            
            blk = blocks[idx]
            H, W = blk.input_resolution
            C = attn_windows.shape[-1]
            B = 4
            # reverse cyclic shift
            if blk.shift_size > 0:
                if not blk.fused_window_process:
                    shifted_x = window_reverse(attn_windows, blk.window_size, H, W)  # B H' W' C
                    x = torch.roll(shifted_x, shifts=(blk.shift_size, blk.shift_size), dims=(1, 2))
                else:
                    x = WindowProcessReverse.apply(attn_windows, B, H, W, C, blk.shift_size, blk.window_size)
            else:
                shifted_x = window_reverse(attn_windows, blk.window_size, H, W)  # B H' W' C
                x = shifted_x
            x = x.view(B, H * W, C)
            x = shortcut + blk.drop_path(x)
            return x
        
       

        x = torch.rand((4,3,224,224)).cuda()
        # shifts = tuple(np.random.randint(0,32,2))
        shifts = (37,43)
        x1 = torch.roll(x, shifts, (2,3)).cuda()
        # poly swin output
        print("predicting")
        p = patch_embed(x)
        p1 = patch_embed(x1)
        print("reordering")
        t = reorder(p)
        t1 = reorder(p1)
        shifts = find_shift2d_batch(t, t1, early_break=True)
        print(shift_and_compare(t, t1, shifts, (0,1) ))
        check_polyphase(t, t1, shifts)
        # confirm_bijective_matches_batch(t.view(t.shape[0], -1 ,t.shape[-1]).cpu().detach().numpy(), t1.view(t.shape[0], -1 ,t.shape[-1]).cpu().detach().numpy())
        t = cyclic_shift(t, 0)
        t1 = cyclic_shift(t1, 0)
        check_window(t, t1)
        t = attention(t, 0)
        t1 = attention(t1, 0)
        check_window(t, t1)
        t = reverse_cyclic_shift(t, p, 0)
        t1 = reverse_cyclic_shift(t1, p1, 0)
        confirm_bijective_matches_batch(t.cpu().detach().numpy(), t1.cpu().detach().numpy())

    def test_model(self): 
        x = torch.rand((4,3,224,224)).cuda()
        x1 = torch.roll(x, (1,1), (2,3)).cuda()
        label = torch.tensor((1,1,0,0)).cuda()
        # model = SwinTransformer(3, (224,224), 1, qk_scale=1).cuda()
        model1 = PolySwin(img_size=(224,224)).cuda()
        criterion = torch.nn.CrossEntropyLoss()
        print(model(x))
        print(model1(x1))
        torch.save(model(x), "tensor.pt")
        torch.save(model1(x), "tensor1.pt")

        for i in range(1):
            s1 = torch.randint(5, (1,)); s2 = torch.randint(5, (1,)); 
            x1 = torch.roll(x, (s1,s2), (2,3))
            
            loss = criterion(output, label)
            loss1 = criterion(output1, label)
            assertAlmostEqual(loss, loss1, 5)
            print(output)
            print(output1)
            print(torch.linalg.norm(output-output1))
            print(f"loss: {loss}, loss1: {loss1}")
        output1 = model(x1)
        loss = criterion(output, label)
        loss1 = criterion(output1, label)
        loss.backward()
        loss1.backward()
        print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
        print(loss)
        print(loss1)
        
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

    def test_polyorder_tokens(self):
        # B, C, H, W
        feature_size = (56,56)
        patches_resolution = (8, 8)
        patch_size = (7, 7)
        token_size = 96

        x = torch.rand((4,token_size, feature_size[0], feature_size[1]))

        for j in range(x.shape[0]):
            x[j, :, 0::patch_size[0], 0::patch_size[1]] = 2 
            x = torch.roll(x, tuple(torch.randint(0,patches_resolution[0], (2,))), dims = (2,3))
        x = torch.tensor(x)
        y = PolyOrder.apply(x, patches_resolution, patch_size, 2, False)

        assert torch.all(y[:,:,0:: patch_size[0], 0:: patch_size[1]]==2).item()
        print("Done Token Polyphase")
    
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

    def test_window_atten(self): 
        x = torch.rand(64)
        x = x.view(1,1, 8,8)
        x[0,:,1::2, 0::2] = 1 
        B,C,H,W = x.shape
        x = x.view(B, H, W, C)
        x = x.view(B, C, H, W)
        xr = torch.roll(x, (1,1), (1,2))
        x = poly_order(x, (2,2), (4,4))
        xr =  poly_order(xr, (2,2), (4,4))
        x = x.view(B, H, W, C)
        xr = x.view(B, H, W, C)

        print(x-xr)
        x = x.view(B*4,-1, C)
        xr = xr.view(B*4, -1 , C)
        model = WindowAttention( 1, (4,4), 1)
        print(model(x))
        print(model(xr))

if __name__ == "__main__":
    test = TestShift()
    start = time.time()
    try: 
        test.setUp()
        test.test_polyorder_tokens()
        test.test_swin_block()
    except:
        print("Exception!")
        traceback.print_exc()
    finally: 
        end = time.time() - start 
        print(f"Time elapsed: {end}")
    # unittest.main()






