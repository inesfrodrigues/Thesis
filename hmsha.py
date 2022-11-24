import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.layers import DropPath, trunc_normal_
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg



#class CNN(nn.Module):
#    def __init__(self, dim, kernel_size=3, stride=2, padding=1): # kernel = 3, stride = 2, padding = 1
#        super().__init__()
#        self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 16, kernel_size = kernel_size, stride = stride, padding = padding, bias = False)
#        self.bn1 = nn.GroupNorm(1, 16),
#        self.conv2 = nn.Conv2d(in_channels = 16, out_channels = dim, kernel_size = kernel_size, stride = stride, padding = padding, bias = False)
#        self.bn2 = nn.GroupNorm(1, dim),
#        self.silu = nn.SiLU()
        
#    def forward(self, x):
#        x = self.conv1(x)
#        x = self.bn1(x)
#        x = self.silu(x)
#        x = self.conv2(x)
#        x = self.bn2(x) #ver
#        x = self.silu(x) #ver
        
#        return x


class Upsample(nn.Module): 
    def __init__(self, in_channels, out_channels): 
        super().__init__()
        self.convT = nn.ConvTranspose2d(in_channels, out_channels, kernel_size = 4, padding = 1, stride = 2)
        self.bn = nn.GroupNorm(1, out_channels, eps = 1e-6)

    def forward(self, x):
        x = self.convT(x)
        x = self.bn(x)
        
        return x



class Downsample(nn.Module): #page 4, paragraph 2
    def __init__(self, in_channels, out_channels): 
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size = 3, padding = 1, stride = 2)
        self.bn = nn.GroupNorm(1, out_channels, eps = 1e-6)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        
        return x



class HMHSA(nn.Module):
    def __init__(self, dim, head, grid_size, ds_ratio, drop = 0.): # we have an embedding and we are going to split it in (heads) =! parts
        super().__init__()
        self.num_heads = dim // head
        self.grid_size = grid_size
        self.head = head
        self.dim = dim
        
        assert (self.num_heads * head == dim), "Dim needs to be divisible by Head."
        
        self.qkv = nn.Conv2d(dim, dim * 3, 1) # q, k, v are all the same; * 3 bc 3 vectors qkv
        self.proj = nn.Conv2d(dim, dim, 1) 
        self.norm = nn.GroupNorm(1, dim, eps = 1e-6)
        self.drop = nn.Dropout2d(drop, inplace = True)

        if self.grid_size > 1:
            self.attention_norm = nn.GroupNorm(1, dim, eps = 1e-6)
            self.avg_pool = nn.AvgPool2d(ds_ratio, stride = ds_ratio)
            self.ds_norm = nn.GroupNorm(1, dim, eps = 1e-6)
            self.q = nn.Conv2d(dim, dim, 1)
            self.kv = nn.Conv2d(dim, dim * 2, 1) # * 2 bc kv
        
    def forward(self, x): # mask
        N, C, H, W = x.shape # N - nº samples, C, H, W - feature dimension, height, width of x (see paper)
        # qkv = self.qkv(x) # do "linear"
        qkv = self.qkv(self.norm(x))

        if self.grid_size > 1:

            # formula (6)
            grid_h, grid_w = H // self.grid_size, W // self.grid_size # grid_h - H/G_1; grid_w - W/G_1 -> paper(6)
            qkv = qkv.reshape(N, 3, self.num_heads, self.head, grid_h, self.grid_size, grid_w, self.grid_size) # 3 bc qkv; head=C; grid_h*grid_size=H... -> paper(6)
            qkv = qkv.permute(1, 0, 2, 4, 6, 5, 7, 3) # (3, N, num_heads, grid_h, grid_w, grid_size, grid_size, head) -> paper(6) 2nd eq.
            qkv = qkv.reshape(3, -1, self.grid_size * self.grid_size, self.head) # -1 -> single dim --- DUV WHY --- -> reshape to paper(6) 2nd eq.
            query, key, value = qkv[0], qkv[1], qkv[2]
        
            # eq. (2)
            attention = (query / (self.dim ** (1/2))) @ key.transpose(-2, -1)

            #if mask is not None:
                #attention = attention.masked_fill(mask = 0, value = float("-1e20"))

            attention = attention.softmax(dim = -1)

            # formula (8)
            attention_x = (attention @ value).reshape(N, self.num_heads, grid_h, grid_w, self.grid_size, self.grid_size, self.head)
            attention_x = attention_x.permute(0, 1, 6, 2, 4, 3, 5).reshape(N, C, H, W) # (N, num_heads, head, grid_h, grid_size, grid_w, grid_size); reshape -> concatenate


            #formula (9)
            attention_x = self.attention_norm(x + attention_x)

            # formula (10)
            #kv = self.kv(self.avg_pool(attention_x))
            kv = self.kv(self.ds_norm(self.avg_pool(attention_x)))

            # formula (11)(12)
            query = self.q(attention_x).reshape(N, self.num_heads, self.head, -1)
            query = query.transpose(-2, -1) # (N, num_heads, -1, head) 
            kv = kv.reshape(N, 2, self.num_heads, self.head, -1)
            kv = kv.permute(1, 0, 2, 4, 3) # (2, N, num_heads, -1, head)
            key, value = kv[0], kv[1]

        else:
            qkv = qkv.reshape(N, 3, self.num_heads, self.head, -1)
            qkv = qkv.permute(1, 0, 2, 4, 3) # (2, N, num_heads, -1, head)
            query, key, value = qkv[0], qkv[1], qkv[2]  

        # eq. (2)
        attention = (query / (self.dim ** (1/2))) @ key.transpose(-2, -1)

        #if mask is not None:
        #        attention = attention.masked_fill(mask = 0, value = float("-1e20"))
        # do masks

        attention = attention.softmax(dim = -1)

        # formula (13)
        global_attention_x = (attention @ value).transpose(-2, -1).reshape(N, C, H, W) # concatenate


        # formula (14)
        if self.grid_size > 1:
            global_attention_x = global_attention_x + attention_x

        x = self.drop(self.proj(global_attention_x)) # x + ...
        
        return x



class MBConv(nn.Module):
    def __init__(self, in_channels, out_channels, expansion, kernel_size, drop = 0., act_layer=nn.SiLU):
        expanded_channels = in_channels * expansion
        out_channels = out_channels
        padding = (kernel_size - 1) // 2
        # use ResidualAdd if dims match, otherwise a normal Sequential
        super().__init__()
        # narrow -> wide
        self.conv1 = nn.Sequential(
            nn.GroupNorm(1, in_channels, eps = 1e-6),
            nn.Conv2d(in_channels, expanded_channels, kernel_size = 1, padding = 0, bias = False),
            act_layer(inplace = True)
        )                
        # wide -> wide
        self.conv2 = nn.Sequential(
            nn.Conv2d(expanded_channels, expanded_channels, kernel_size = kernel_size, padding = padding, groups = expanded_channels, bias = False),
            act_layer(inplace = True)
        )
        # wide -> narrow
        self.conv3 = nn.Sequential(
            nn.Conv2d(expanded_channels, out_channels, kernel_size = 1, padding = 0, bias = False),
            nn.GroupNorm(1, out_channels, eps = 1e-6)
        )

        self.drop = nn.Dropout2d(drop, inplace = True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.drop(x)
        x = self.conv3(x)
        x = self.drop(x)

        return x



class Block(nn.Module):
    def __init__(self, dim, head, grid_size = 1, ds_ratio = 1, expansion = 4, drop = 0., drop_path = 0., kernel_size = 3, act_layer = nn.SiLU):
        super().__init__()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.attn = HMHSA(dim, head, grid_size = grid_size, ds_ratio = ds_ratio, drop = drop)
        self.conv = MBConv(in_channels = dim, out_channels = dim, expansion = expansion, kernel_size = kernel_size, drop = drop, act_layer = act_layer)

    def forward(self, x):
        x = x + self.drop_path(self.attn(x))
        x = x + self.drop_path(self.conv(x))
        return x




class HAT_Net(nn.Module):
    def __init__(self, dims, head, kernel_sizes, expansions, grid_sizes, ds_ratios, drop_rate, depths, drop_path_rate, img_size = 224, in_chans = 3, num_classes = 1000, act_layer = nn.SiLU):
        super().__init__()
        self.depths = depths

        # two sequential vanilla 3 × 3 convolutions - first downsample
        #self.CNN = CNN(dim = dims[0])
        self.CNN = nn.Sequential(
            nn.Conv2d(in_channels = 3, out_channels = 16, kernel_size = 3, stride = 2, padding = 1),
            nn.GroupNorm(1, 16, eps = 1e-6),
            act_layer(inplace = True),
            nn.Conv2d(in_channels = 16, out_channels = dims[0], kernel_size = 3, stride = 2, padding = 1),
        )

        # block - H-MSHA + MLP
        self.blocks = []
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        for stage in range(len(dims)):
            self.blocks.append(nn.ModuleList([Block(
                dim = dims[stage], head = head, kernel_size = kernel_sizes[stage], expansion = expansions[stage],
                grid_size = grid_sizes[stage], ds_ratio = ds_ratios[stage], drop = drop_rate, drop_path = dpr[sum(depths[:stage]) + i])
                for i in range(depths[stage])])) # will calculate each block depth times
        self.blocks = nn.ModuleList(self.blocks)

        # downsamples
        self.ds1 = Downsample(in_channels = dims[0], out_channels = dims[1])
        self.ds2 = Downsample(in_channels = dims[1], out_channels = dims[2])
        self.ds3 = Downsample(in_channels = dims[2], out_channels = dims[3])

        # upsamples
        self.us1 = Upsample(in_channels = dims[3], out_channels = dims[2])
        self.us2 = Upsample(in_channels = dims[2], out_channels = dims[1])
        self.us3 = Upsample(in_channels = dims[1], out_channels = dims[0])

        self.TCNN = nn.Sequential(
            nn.ConvTranspose2d(in_channels = dims[0], out_channels = 16, kernel_size = 4, stride = 2, padding = 1),
            nn.GroupNorm(1, 16, eps = 1e-6),
            act_layer(inplace = True),
            nn.ConvTranspose2d(in_channels = 16, out_channels = 1, kernel_size = 4, stride = 2, padding = 1),
        )

        # fully connected layer -> 1000
        #self.fullyconnected = nn.Sequential(
        #    nn.Dropout(0.2, inplace = True),
        #    nn.Linear(dims[3], num_classes),
        #)
                
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            trunc_normal_(m.weight, std = .02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

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
        x =self.us1(x)
        x =self.us2(x)
        x =self.us3(x)
        x =self.TCNN(x)
        #x = F.adaptive_avg_pool2d(x, (1, 1)).flatten(1) # F so we specify the input
        # In adaptive_avg_pool2d, we define the output size we require at the end of the pooling operation, and pytorch infers what pooling parameters to use to do that.
        #x = self.fullyconnected(x)

        return x





@register_model
def HAT_Net_medium(pretrained = False, **kwargs):
    model = HAT_Net(dims = [64, 128, 320, 512], head = 64, kernel_sizes = [5, 3, 5, 3], expansions = [8, 8, 4, 4], grid_sizes = [8, 7, 7, 1], ds_ratios = [8, 4, 2, 1], depths = [3, 6, 18, 3],  **kwargs)
    model.default_cfg = _cfg()
    return model

#model = HAT_Net(dims = [64, 128, 320, 512], head = 64, kernel_sizes = [5, 3, 5, 3], expansions = [8, 8, 4, 4],
        #grid_sizes = [8, 7, 7, 1], ds_ratios = [8, 4, 2, 1], drop = 0, depths = [3, 6, 18, 3], fc = 1000, drop_path_rate = 0)