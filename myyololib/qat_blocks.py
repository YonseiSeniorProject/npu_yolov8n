import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
from myyololib.basic_blocks import Conv, Bottleneck, C2f, SPPF, Detect, DFL, autopad


# STE module for QAT
class STE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, q_min, q_max):
        # Save the input tensor and quantization parameters for backward pass
        ctx.save_for_backward(x)
        ctx.q_min = q_min
        ctx.q_max = q_max

        # forward pass: clamp and round
        temp = x.clone()
        x_clamped = torch.clamp(temp, q_min, q_max)
        x_rounded = x_clamped.round()

        return x_rounded
    
    @staticmethod
    def backward(ctx, grad_out):
        # Retrieve the saved tensor and quantization parameters
        x, = ctx.saved_tensors
        q_min, q_max = ctx.q_min, ctx.q_max

        # clipping range mask
        out_of_range_mask = (x < q_min) | (x > q_max)

        # Gradient is passed through only for values within the clipping range
        grad_input = grad_out.clone()
        grad_input[out_of_range_mask] = 0

        return grad_input, None, None
    
def custom_clip_round(x, q_min, q_max):
    return STE.apply(x, q_min, q_max)

# 8bit fixed point quantization for Conv2d
class QConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, 
                 qcfg=None):
        super(QConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        (num_bits, w_fraction_bits, a_fraction_bits) = [8, 6, 4] if qcfg is None else qcfg
        # print(qcfg, num_bits, w_fraction_bits, a_fraction_bits) # debug
        self.clip_round = custom_clip_round
        self.num_bits = num_bits
        # fixed point negative scale factor
        self.w_scale = 2 ** w_fraction_bits
        self.a_scale = 2 ** a_fraction_bits
        self.b_scale = self.w_scale * self.a_scale
        # quantization range
        self.min_val = -2 ** (num_bits - 1)
        self.max_val = 2 ** (num_bits - 1) - 1
        # bias quantization range is twice wider than weight/activation since y = b + w*a
        self.b_min_val = -2 ** (2*num_bits - 1)
        self.b_max_val = 2 ** (2*num_bits - 1) - 1

    # weight, bias, and activation quantizers
    def weight_quantizer(self, w):
        quant_w = self.discretizer(w, self.w_scale, self.min_val, self.max_val)
        return quant_w
    
    def act_quantizer(self, x):
        quant_x = self.discretizer(x, self.a_scale, self.min_val, self.max_val)
        return quant_x

    def bias_quantizer(self, b):
        quant_b = self.discretizer(b, self.b_scale, self.b_min_val, self.b_max_val)
        return quant_b
    
    def discretizer(self, v, scale, min_val, max_val):
        return self.clip_round(v * scale, min_val, max_val) / scale

    def forward(self, x):
            
        quant_w = self.weight_quantizer(self.weight)
        quant_x = self.act_quantizer(x)
        quant_b = self.bias_quantizer(self.bias) if self.bias is not None else None

        Qout = F.conv2d(quant_x, quant_w, quant_b, self.stride, self.padding, self.dilation, self.groups)

        return Qout
    
# Quantized Convolution
class QConv(nn.Module):
    """
    A Convolutional layer followed by ReLU activation, both quantized using QConv2d.
    Args:
    qcfg (tuple): A tuple containing quantization parameters (num_bits, w_fraction_bits, a_fraction_bits).    
    """
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, qcfg: tuple =  None):
        super().__init__()
        self.conv = QConv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=True, 
                            qcfg=qcfg)
        self.act = nn.ReLU()

    def forward(self, x):
        return self.act(self.conv(x))

# Quantized Bottleneck
class QBottleneck(Bottleneck):
    def __init__(
        self, c1: int, c2: int, shortcut: bool = True, g: int = 1, k: Tuple[int, int] = (3, 3), e: float = 0.5,
        block_qcfg: dict = None
    ):
        super(Bottleneck, self).__init__()        
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = QConv(c1, c_, k[0], 1, 
                         qcfg=None if block_qcfg is None else block_qcfg.get("cv1"))
        self.cv2 = QConv(c_, c2, k[1], 1, g=g, 
                         qcfg=None if block_qcfg is None else block_qcfg.get("cv2"))
        self.add = shortcut and c1 == c2

# Quantized C2f
class QC2f(C2f):
    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = False, g: int = 1, e: float = 0.5,
                 layer_qcfg: dict = None):
        super(C2f, self).__init__()        
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = QConv(c1, 2 * self.c, 1, 1,
                         qcfg=None if layer_qcfg is None else layer_qcfg.get("cv1"))
        self.cv2 = QConv((2 + n) * self.c, c2, 1,
                         qcfg=None if layer_qcfg is None else layer_qcfg.get("cv2"))

        self.m = nn.ModuleList(QBottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0
                                           , block_qcfg=None if layer_qcfg is None else layer_qcfg.get(f"m.{i}"))
                                             for i in range(n))

# Quantized SPPF
class QSPPF(SPPF):
    def __init__(self, c1: int, c2: int, k: int = 5,
                 layer_qcfg: dict = None):
        super(SPPF, self).__init__()        
        c_ = c1 // 2  # hidden channels
        self.cv1 = QConv(c1, c_, 1, 1, 
                         qcfg=None if layer_qcfg is None else layer_qcfg.get("cv1"))
        self.cv2 = QConv(c_ * 4, c2, 1, 1,
                         qcfg=None if layer_qcfg is None else layer_qcfg.get("cv2"))
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

# Quantized Detect
class QDetect(Detect):
    def __init__(self, ch: Tuple = (),
                  layer_qcfg: dict = None):
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
            nn.Sequential(QConv(x, c2, 3, qcfg=None if layer_qcfg is None else layer_qcfg.get("cv2.0")),
                          QConv(c2, c2, 3, qcfg=None if layer_qcfg is None else layer_qcfg.get("cv2.1")),
                          QConv2d(c2, 4 * self.reg_max, 1, bias=True, qcfg=None if layer_qcfg is None else layer_qcfg.get("cv2.2"))
                          ) for x in ch
        )
        self.cv3 = (
            nn.ModuleList(
                nn.Sequential(
                    QConv(x, c3, 3, qcfg=None if layer_qcfg is None else layer_qcfg.get("cv3.0")),
                    QConv(c3, c3, 3, qcfg=None if layer_qcfg is None else layer_qcfg.get("cv3.1")),
                    QConv2d(c3, self.nc, 1, bias=True, qcfg=None if layer_qcfg is None else layer_qcfg.get("cv3.2"))
                    ) for x in ch)
        )
        self.dfl = DFL(self.reg_max) 
