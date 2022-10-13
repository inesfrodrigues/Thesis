import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms

import torch.optim as optim

import matplotlib.pyplot as plt
import numpy as np


class CNN(nn.Module):
    def __init__(self, kernel_size, stride, padding): # kernel = 3, stride = 2, padding = 1
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 16, kernel_size = kernel_size, stride = stride, padding = padding)
        #put bias = false
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(in_channels = 16, out_channels = 64, kernel_size = kernel_size, stride = stride, padding = padding)
        self.bn2 = nn.BatchNorm2d(64)
        self.silu = nn.SiLU()
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.silu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.silu(x)
        #x = torch.flatten(x, 1)
        
        return x


class Downsample(nn.Module): #page 4, paragraph 2
    def __init__(self, in_channels, out_channels, kernel_size): #kernel = 3
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding = 1, stride = 2)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        
        return x


class HMHSA(nn.Module):
    def __init__(self, dim, head, grid_size, ds_ratio, drop): # we have an embedding and we are going to split it in (heads) =! parts
        super(SelfAttention, self).__init__()
        self.dim = dim
        self.head = head
        self.num_heads = dim // head
        self.grid_size = grid_size
        
        assert (self.num_heads * head == dim), "Dim needs to be divisible by Head."
        
        self.qkv = nn.Linear(self.dim, self.dim, bias = False) # q, k, v are all the same; change to conv
        #self.qkv = nn.Conv2d(dim, dim * 3, 1)
        # put bias = False
        self.proj = nn.Conv2d(dim, dim, 1) # DUV
        self.drop = nn.Dropout2d(drop, inplace = True)

        if grid_size > 1:
            self.norm1 = nn.GroupNorm(1, dim)
            self.avg_pool = nn.AvgPool2d(ds_ratio, stride = ds_ratio) # DUV - average pooling with both the kernel size and stride of G2
            self.q = nn.Linear(self.dim, self.dim, bias = False) # change to conv
            #self.q = nn.Conv2d(dim, dim, 1)
            self.kv = nn.Linear(self.dim, self.dim, bias = False) # change to conv
            #self.kv = nn.Conv2d(dim, dim * 2, 1)
        
    def forward(self, x, mask): # DUV mask
        N, C, H, W = x.shape # N - nº samples, C, H, W - feature dimension, height, width of x (see paper)
        qkv = self.qkv(x) # do linear
        #qkv = self.qkv(self.norm(x))

        if self.grid_size > 1:

            # formula (6)
            grid_h, grid_w = H // self.grid_size, W // self.grid_size # grid_h - H/G_1; grid_w - W/G_1 -> paper(6)
            qkv = qkv.reshape(N, 3, self.num_heads, self.head, grid_h, self.grid_size, grid_w, self.grid_size) # 3 bc qkv; head=C; grid_h*grid_size=H... -> paper(6) --- DUV NUM_HEADS
            qkv = qkv.permute(1, 0, 2, 4, 6, 5, 7, 3) # (3, N, num_heads, grid_h, grid_w, grid_size, grid_size, head) -> paper(6) 2nd eq.
            qkv = qkv.reshape(3, -1, self.grid_size * self.grid_size, self.head) # -1 -> single dim --- DUV WHY --- -> reshape to paper(6) 2nd eq.
            query, key, value = qkv[0], qkv[1], qkv[2]
        
            # eq. (2)
            attention = query @ key.transpose(-2, -1) 

            if mask is not None:
                attention = attention.masked_fill(mask = 0, value = float("-1e20"))

            attention = torch.softmax(attention / (self.dim ** (1/2)), dim = -1) 

            # formula (8)
            attention_x = (attention @ value).reshape(N, self.num_heads, grid_h, grid_w, self.grid_size, self.grid_size, self.head)
            attention_x = attention_x.permute(0, 1, 6, 2, 4, 3, 5).reshape(N, C, H, W) # (N, num_heads, head, grid_h, grid_size, grid_w, grid_size)

            # DUV DONT CONCATENAT AFTER ATTENTION?

            #formula (9)
            attention_x = self.norm1(attention_x + x)
            #grid_x = self.grid_norm(x + grid_x)

            # DUV DONT FEED FORWARD?

            # formula (10)
            kv = self.kv(self.avg_pool(attention_x))
            #kv = self.kv(self.ds_norm(self.avg_pool(grid_x)))

            # formula (11)(12)
            query = self.q(attention_x).reshape(N, self.num_heads, self.head, -1) #DUV -1
            query = query.transpose(-2, -1) # (N, num_heads, -1, head) 
            kv = kv.reshape(N, 2, self.num_heads, self.head, -1) # DUV -1
            kv = kv.permute(1, 0, 2, 4, 3) # (2, N, num_heads, -1, head)
            key, value = kv[0], kv[1]

        else:
            qkv = qkv.reshape(N, 3, self.num_heads, self.head, -1)
            qkv = qkv.permute(1, 0, 2, 4, 3) # (2, N, num_heads, -1, head)
            query, key, value = qkv[0], qkv[1], qkv[2]  

        # eq. (2)
        attention = query @ key.transpose(-2, -1)

        if mask is not None:
                attention = attention.masked_fill(mask = 0, value = float("-1e20"))

        attention = torch.softmax(attention / (self.dim ** (1/2)), dim = -1)

        # formula (13)
        global_attention_x = (attention @ value).transpose(-2, -1).reshape(N, C, H, W)

        # DUV DONT CONCATENAT?
        # DUV DONT NORM AND FEED FORWARD?

        # formula (14)
        if self.grid_size > 1:
            global_attention_x = global_attention_x + attention_x

        x = self.drop(self.proj(global_attention_x)) # DUV DROP, why not add x 
        
        return x


class MBConv(nn.Module):
    def __init__(self, in_channels, out_channels, expansion, kernel_size):
        expanded_channels = in_channels * expansion
        padding = (kernel_size - 1) // 2
        # use ResidualAdd if dims match, otherwise a normal Sequential
        super().__init__()
        # narrow -> wide
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, expanded_channels, kernel_size = 1, padding = 0, bias = False),
            nn.BatchNorm2d(expanded_dim),
            nn.SiLU()
        )                
        # wide -> wide
        self.conv2 = nn.Sequential(
            nn.Conv2d(expanded_channels, expanded_channels, kernel_size = kernel_size, padding = padding, groups = expanded_channels, bias = False),
            nn.BatchNorm2d(expanded_channels),
            nn.SiLU()
        )
        # wide -> narrow
        self.conv3 = nn.Sequential(
            nn.Conv2d(expanded_channels, out_channels, kernel_size = 1, padding = 0),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x + self.conv3(x)

        return x
    
    
 class Block(nn.Module):
    def __init__(self, dim, head, kernel_size, expansion, grid_size, ds_ratio, drop):
        super().__init__()
        self.HMSHA = HMHSA(dim = dim, head = head, grid_size = grid_size, ds_ratio = ds_ratio, drop = drop)
        self.MLP = MBConv(in_channels = dim, out_channels = dim, expansion = expansion, kernel_size = kernel_size)

    def forward(self, x):
        x = self.HMSHA(x)
        x = self.MLP(x)

        return x
    
    
    class HAT_Net(nn.Module):
    def __init__(self, dims, head, kernel_sizes, expansions, grid_sizes, ds_ratios, drop, depths, fc):
        super().__init__()
        self.depths = depths

        # two sequential vanilla 3 × 3 convolutions - first downsample
        self.CNN = CNN(dim = dims[0])

        # block - H-MSHA + MLP
        self.blocks = []
        for stage in range(len(dims)):
            self.blocks.append(nn.ModuleList([Block(
                dim = dims[stage], head = head, kernel_size = kernel_sizes[stage], expansion = expansions[stage],
                grid_size = grid_sizes[stage], ds_ratio = ds_ratios[stage], drop = drop)
                for i in range(depths[stage])])) # will calculate which block depth times
        self.blocks = nn.ModuleList(self.blocks)

        # downsamples
        self.ds1 = Downsample(in_channels = dim[0], out_channels = dim[1])
        self.ds2 = Downsample(in_channels = dim[1], out_channels = dim[2])
        self.ds3 = Downsample(in_channels = dim[2], out_channels = dim[3])

        # fully connected layer -> 1000
        self.fullyconnected = nn.Sequential(
           # nn.Dropout(0.2, inplace=True),
            nn.Linear(dims[3], fc),

        # DUV

    def forward(self, x):
        x = self.CNN(x)
        for block in self.blocks[0]:
            x = block(x)
        x = self.ds1(x)
        for block in self.blocks[1]:
            x = block(x)
        x = self.ds2(x)
        for block in self.blocks[2]:
            x = block(x)
        x = self.ds3(x)
        for block in self.blocks[3]:
            x = block(x)
        x = F.adaptive_avg_pool2d(x, (1, 1)) #.flatten(1) DUV  # F so we specify the input
        # In adaptive_avg_pool2d, we define the output size we require at the end of the pooling operation, and pytorch infers what pooling parameters to use to do that.
        x = self.fullyconnected(x)

        # DUV paper - softmax ??

    return x
