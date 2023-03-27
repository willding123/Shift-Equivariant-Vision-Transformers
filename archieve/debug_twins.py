

# would it be more consistent if I tune the patch size, and lp norm? 
idx = (predicted != predicted1).nonzero()[0]

# debugging twins 
embed = model.patch_embeds[3]
blocks = model.blocks[3]
pos_blk = model.pos_block[3]
norm = model.norm

# x = images[idx]
# B,C,H,W = x.shape
# xs = torch.roll(x, shifts, (2,3))
# xs1 = torch.roll(x, shifts1, (2,3))

# embedding tests
x = PolyOrder.apply(x, embed.proj.kernel_size)
xs = PolyOrder.apply(xs, embed.proj.kernel_size)
xs1 = PolyOrder.apply(xs1, embed.proj.kernel_size)
x, size = embed(x)
xs, _ = embed(xs)
xs1, _ = embed(xs1)

from utils import find_shift2d_batch, shift_and_compare

t = x.view(B, int(sqrt(x.shape[1])), int(sqrt(x.shape[1])), x.shape[2])
ts = xs.view(B, int(sqrt(x.shape[1])), int(sqrt(x.shape[1])), x.shape[2])
ts1 = xs1.view(B, int(sqrt(x.shape[1])), int(sqrt(x.shape[1])), x.shape[2])


s1 = find_shift2d_batch(t,ts, early_break = False)
s2 = find_shift2d_batch(ts,ts1, early_break = False)
shift_and_compare(t, ts, s1, (0,1) )
shift_and_compare(ts, ts1, s2, (0,1) )

# block tests
def block_forward(x, size):
    for j, blk in enumerate(blocks):
        x = x.view(B, int(sqrt(x.shape[1])), -1, x.shape[2]).permute(0, 3, 1, 2)
        if type(blk.attn) == LocallyGroupedAttn:
            _, n = arrange_polyphases(x, to_2tuple(blk.attn.ws))
            try:
                assert torch.topk(n, 2).values[0][0] != torch.topk(n, 2).values[0][1]
            except:
                print(j, torch.topk(n, 2).values[0])
                break
            x = PolyOrder.apply(x, to_2tuple(blk.attn.ws))
            
        else: 
            x = PolyOrder.apply(x, to_2tuple(blk.attn.sr_ratio))
        x = x.permute(0, 2, 3, 1)
        x = x.reshape(B, -1, x.shape[3]).contiguous()
        x = blk(x, size)
        if j == 0:
            x = pos_blk(x, size) 
    return x

x = block_forward(x, size)
xs = block_forward(xs, size)
xs1 = block_forward(xs1, size)

t = x.view(B, int(sqrt(x.shape[1])), int(sqrt(x.shape[1])), x.shape[2])
ts = xs.view(B, int(sqrt(x.shape[1])), int(sqrt(x.shape[1])), x.shape[2])
ts1 = xs1.view(B, int(sqrt(x.shape[1])), int(sqrt(x.shape[1])), x.shape[2])

s1 = find_shift2d_batch(t,ts, early_break = False)
s2 = find_shift2d_batch(ts,ts1, early_break = False)
shift_and_compare(t, ts, s1, (0,1) )
shift_and_compare(ts, ts1, s2, (0,1) )

# x = x.reshape(B, *size, -1).permute(0, 3, 1, 2).contiguous()
# xs = xs.reshape(B, *size, -1).permute(0, 3, 1, 2).contiguous()
# xs1 = xs1.reshape(B, *size, -1).permute(0, 3, 1, 2).contiguous()


# %%
head = model.head
x = x.mean(dim=1)
xs = xs.mean(dim=1)
xs1 = xs1.mean(dim=1)
x = head(x)
xs = head(xs)
xs1 = head(xs1)

# write a subclass called PolyTwins that inherits from timm.models.twins.Twins

# %%
x = debugger["x"]
model(x)