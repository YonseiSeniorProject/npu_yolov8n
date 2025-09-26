from typing import Tuple
from torchvision.ops import nms

import torch
import torch.nn as nn

from myyololib.utils import cxcywh2xyxy

from ultralytics.utils.tal import dist2bbox, make_anchors
from typing import Any, Dict, List, Tuple, Union

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

class Conv(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1):
        """
        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size.
            s (int): Stride.
            p (int, optional): Padding.
            g (int): Groups.
        """
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        # self.act = nn.SiLU()
        self.act = nn.ReLU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class Bottleneck(nn.Module):
    def __init__(
        self, c1: int, c2: int, shortcut: bool = True, g: int = 1, k: Tuple[int, int] = (3, 3), e: float = 0.5
    ):
        """
        Initialize a standard bottleneck module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            shortcut (bool): Whether to use shortcut connection.
            g (int): Groups for convolutions.
            k (tuple): Kernel sizes for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply bottleneck with optional shortcut connection."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

class C2f(nn.Module):
    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = False, g: int = 1, e: float = 0.5):
        """
        Initialize a CSP bottleneck with 2 convolutions.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of Bottleneck blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x: torch.Tensor) -> torch.Tensor:
        y = self.cv1(x).split((self.c, self.c), 1)
        y = [y[0], y[1]]
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher."""

    def __init__(self, c1: int, c2: int, k: int = 5):
        """
        Initialize the SPPF layer with given input/output channels and kernel size.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            k (int): Kernel size.

        Notes:
            This module is equivalent to SPP(k=(5, 9, 13)).
        """
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = [self.cv1(x)]
        y.extend(self.m(y[-1]) for _ in range(3))
        return self.cv2(torch.cat(y, 1))

class Concat(nn.Module):

    def __init__(self, dimension=1):
        super().__init__()
        self.d = dimension

    def forward(self, x: List[torch.Tensor]):
        return torch.cat(x, self.d)

class DFL(nn.Module):
    """
    Integral module of Distribution Focal Loss (DFL).
    """

    def __init__(self, c1: int = 16):
        super().__init__()
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(c1, dtype=torch.float)
        self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1))
        self.c1 = c1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, a = x.shape  # batch, channels, anchors
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)

class Detect(nn.Module):
    """
    YOLO Detect head for object detection models.

    This class implements the detection head used in YOLO models for predicting bounding boxes and class probabilities.
    It supports both training mode.

    Attributes:
        cv2 (nn.ModuleList): Convolution layers for box regression.
        cv3 (nn.ModuleList): Convolution layers for classification.
        dfl (nn.Module): Distribution Focal Loss layer.

    Examples:
        Create a detection head for 80 classes
        >>> detect = Detect(nc=80, ch=(256, 512, 1024))
        >>> x = [torch.randn(1, 256, 80, 80), torch.randn(1, 512, 40, 40), torch.randn(1, 1024, 20, 20)]
        >>> outputs = detect(x)
    """
    def __init__(self, ch: Tuple = ()):
        super().__init__()
        self.nc = 80  # number of classes
        self.nl = 3  # number of detection layers
        self.reg_max = 16  # DFL channels (ch[0] // 16 to scale 4/8/12/16/20 for n/s/m/l/x)
        self.no = self.nc + self.reg_max * 4  
        self.anchors = None
        self.strides= None
        self.stride = [8, 16, 32]

        c2, c3 = max((16, ch[0] // 4, self.reg_max * 4)), max(ch[0], min(self.nc, 100))  # channels
        self.cv2 = nn.ModuleList(
            nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 4 * self.reg_max, 1)) for x in ch
        )
        self.cv3 = (
            nn.ModuleList(nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, self.nc, 1)) for x in ch)
        )
        self.dfl = DFL(self.reg_max) 

    def forward(self, x: List[torch.Tensor]) -> Union[List[torch.Tensor], Tuple]:
        """Concatenate and return predicted bounding boxes and class probabilities."""
        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)
        return x
    
    def _inference(self, x: List[torch.Tensor]) -> Union[List[torch.Tensor], Tuple]:
        shape = x[0].shape  # BCHW
        x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2) # [b, 4*16+80, 80*80 + 40*40 + 20*20]

        self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))
        self.shape = shape

        box, cls = x_cat.split((self.reg_max * 4, self.nc), 1) # [b, 4*16, 80*80 + 40*40 + 20*20], [b, 80, 80*80 + 40*40 + 20*20]

        dbox = self.decode_bboxes(self.dfl(box), self.anchors.unsqueeze(0)) * self.strides

        return torch.cat((dbox, cls.sigmoid()), 1)
    
    def decode_bboxes(self, bboxes: torch.Tensor, anchors: torch.Tensor, xywh: bool = True) -> torch.Tensor:
        """Decode bounding boxes from predictions."""
        return dist2bbox(bboxes, anchors, xywh=True, dim=1)

    @staticmethod
    def postprocess(preds: torch.Tensor, max_det: int, nc: int = 80) -> torch.Tensor:
        """
        Post-process YOLO model predictions.

        Args:
            preds (torch.Tensor): Raw predictions with shape (batch_size, num_anchors, 4 + nc) with last dimension
                format [x, y, w, h, class_probs].
            max_det (int): Maximum detections per image.
            nc (int, optional): Number of classes.

        Returns:
            (torch.Tensor): Processed predictions with shape (batch_size, min(max_det, num_anchors), 6) and last
                dimension format [x, y, w, h, max_class_prob, class_index].
        """
        batch_size, anchors, _ = preds.shape  # i.e. shape(16,8400,84)
        boxes, scores = preds.split([4, nc], dim=-1) # boxes: (batch_size, 8400, 4), scores: (batch_size, 8400, 80)
        index = scores.amax(dim=-1).topk(min(max_det, anchors))[1].unsqueeze(-1) # (b, 8400). score of predicted class 
                                                                                 # -> (b, max_det). index of anchors with top {max_det} confidence score
                                                                                 # -> unsqueeze to (b, max_det, 1)
                                                                                 # note: index.repeat(1, 1, 4) -> (b, max_det, 4). used for index of anchors using torch.gather
        boxes = boxes.gather(dim=1, index=index.repeat(1, 1, 4))        # boxes: (b, max_det, 4). bounding boxes of anchors with top {max_det} confidence score
        scores = scores.gather(dim=1, index=index.repeat(1, 1, nc))     # scores: (b, max_det, 80). class probabilities of anchors with top {max_det} confidence score
        scores, index = scores.flatten(1).topk(min(max_det, anchors))   # -(flatten)-> (b, 80 * max_det) -(topk)-> (b, max_det). top {max_det} class scores among all anchors in the image
                                                                        # scores: (b, max_det), index: (b, max_det)
        i = torch.arange(batch_size)[..., None]  # batch indices [[0], [1], [2], ..., [batch_size-1]]
        return torch.cat([boxes[i, index // nc], scores[..., None], (index % nc)[..., None].float()], dim=-1) # return (b, max_det, 6)

    def detect(self, preds: torch.Tensor, max_det: int = 300, nc: int = 80, conf_threshold: float = 0.25, iou_threshold: float = 0.7) -> torch.Tensor:
        
        all_detections = self.postprocess(preds, max_det, nc)

        results = []
        for image_detections in all_detections:
            mask = image_detections[:, 4] > conf_threshold # check confidence score
            filtered_detections = image_detections[mask] # filter out low confidence detections
            valid_anchors = nms(cxcywh2xyxy(filtered_detections[:, :4]), filtered_detections[:, 4], iou_threshold=iou_threshold)
            valid_detections = filtered_detections[valid_anchors]
            results.append(valid_detections) 
        return results # result is list of tensors. each tensor: (num_detections, 6) for each image in the batch. 6 = cxcywh(4), score(1), class_id(1)