import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
from myyololib.basic_blocks import Conv, Bottleneck, C2f, SPPF, Detect, DFL, autopad

# npu simulation: int8->int8
class NConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, 
                 num_bits=8, fw=6, fx=4, fy=4):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias).requires_grad_(False)
        self.num_bits = num_bits
        # shifting value for fixed point convolution
        self.shift = fx + fw - fy
        # quantization range
        self.min_val = -2 ** (num_bits - 1)
        self.max_val = 2 ** (num_bits - 1) - 1

    def forward(self, x, vis=False):
        x = self.conv(x)
        x = torch.floor(x / (2 ** self.shift))
        return torch.clamp(x, self.min_val, self.max_val)

# NPU Convolution
class NConv(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, num_bits=8, fw=6, fx=4, fy=4,vis=False):
        super().__init__()
        self.conv = NConv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=True, 
                            num_bits=num_bits, fw=fw, fx=fx, fy=fy)
        self.act = nn.ReLU()
        self.vis = vis

    def forward(self, x):
        return self.act(self.conv(x, vis=self.vis))

# NPU Bottleneck
class NBottleneck(Bottleneck):
    def __init__(
        self, c1: int, c2: int, shortcut: bool = True, g: int = 1, k: Tuple[int, int] = (3, 3), e: float = 0.5
    ):
        super(Bottleneck, self).__init__()        
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = NConv(c1, c_, k[0], 1)
        self.cv2 = NConv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

# NPU C2f
class NC2f(C2f):
    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = False, g: int = 1, e: float = 0.5):
        super(C2f, self).__init__()        
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = NConv(c1, 2 * self.c, 1, 1)
        self.cv2 = NConv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(NBottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

# NPU SPPF
class NSPPF(SPPF):
    def __init__(self, c1: int, c2: int, k: int = 5):
        super(SPPF, self).__init__()        
        c_ = c1 // 2  # hidden channels
        self.cv1 = NConv(c1, c_, 1, 1)
        self.cv2 = NConv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

class Dequantize(nn.Module):
    def __init__(self, num_bits=8, fraction_bits=4):
        super(Dequantize, self).__init__()
        self.scale = 2 ** fraction_bits
        self.min_val = -2 ** (num_bits - 1)
        self.max_val = 2 ** (num_bits - 1) - 1

    def forward(self, x):
        x = torch.clamp(x, self.min_val, self.max_val)
        return x / self.scale
    
# NPU Detect
class NDetect(Detect):
    def __init__(self, ch: Tuple = ()):
        super(Detect, self).__init__()        
        self.nc = 80  # number of classes
        self.nl = 3  # number of detection layers
        self.reg_max = 16  # DFL channels (ch[0] // 16 to scale 4/8/12/16/20 for n/s/m/l/x)
        self.no = self.nc + self.reg_max * 4  
        self.anchors = None
        self.strides= None
        self.stride = [8, 16, 32]

        c2, c3 = max((16, ch[0] // 4, self.reg_max * 4)), max(ch[0], min(self.nc, 100))  # channels
        self.cv2 = nn.ModuleList(
            nn.Sequential(NConv(x, c2, 3), 
                          NConv(c2, c2, 3, num_bits=8, fw=6, fx=4, fy=4), 
                          NConv2d(c2, 4 * self.reg_max, 1, bias=True, num_bits=8, fw=7, fx=4, fy=6),
                          Dequantize(num_bits=8, fraction_bits=6)
                          ) for x in ch
        )
        self.cv3 = (
            nn.ModuleList(
                nn.Sequential(
                    NConv(x, c3, 3), 
                    NConv(c3, c3, 3, num_bits=8, fw=6, fx=4, fy=3), 
                    NConv2d(c3, self.nc, 1, bias=True, num_bits=8, fw=7, fx=3, fy=6),
                    Dequantize(num_bits=8, fraction_bits=6)
                    ) for x in ch)
        )
        self.dfl = DFL(self.reg_max) 
