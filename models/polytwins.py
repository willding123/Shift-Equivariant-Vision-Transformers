import timm 
from timm.models.twins import LocallyGroupedAttn
from models.poly_utils import PolyOrderModule, arrange_polyphases, PolyPatchEmbed, PolyOrder
from timm.models.twins import Twins
from timm.models.layers import to_2tuple
from math import sqrt
 
class PolyTwins(timm.models.twins.Twins):
    def __init__(self, model_type, pretrained = False, **kwargs):
        super().__init__()
        model = timm.create_model(model_type , pretrained=pretrained)
        self.patch_embeds = model.patch_embeds
        self.pos_drops = model.pos_drops
        self.blocks = model.blocks
        self.pos_block = model.pos_block
        self.norm = model.norm
        self.head = model.head
        self.depths = model.depths
        self.num_classes = model.num_classes
        self.num_features = model.num_features
        self.embed_dim = model.embed_dims
        self.global_pool = model.global_pool
        cs = [64, 128, 256, 512]
        for i, l in enumerate(self.pos_block):
            l.proj[0].padding_mode = "circular"

        for i, l in enumerate(self.patch_embeds):
            l.proj.padding_mode = "circular"
                
    
    def forward_features(self, x):
        B = x.shape[0]
        tmp = x.clone()
        for i, (embed, drop, blocks, pos_blk) in enumerate(
                zip(self.patch_embeds, self.pos_drops, self.blocks, self.pos_block)):        
            if i == 0:
                x = PolyOrder.apply(x, embed.proj.kernel_size)
            x, size = embed(x)
            x = drop(x)
            for j, blk in enumerate(blocks):
                x = x.view(B, int(sqrt(x.shape[1])), -1, x.shape[2]).permute(0, 3, 1, 2)
                if type(blk.attn) == LocallyGroupedAttn:                
                    x = PolyOrder.apply(x, to_2tuple(blk.attn.ws))
                else: 
                    if blk.attn.sr_ratio > 1:
                        x = PolyOrder.apply(x, to_2tuple(blk.attn.sr_ratio))
                    
                x = x.permute(0, 2, 3, 1)
                x = x.reshape(B, -1, x.shape[3]).contiguous()
                x = blk(x, size)
                if j == 0:
                    x = pos_blk(x, size)  # PEG here
            if i < len(self.depths) - 1:
                x = x.reshape(B, *size, -1).permute(0, 3, 1, 2).contiguous()
        x = self.norm(x)
        return x

