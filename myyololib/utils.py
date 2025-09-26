import torch

def cxcywh2xyxy(boxes: torch.Tensor) -> torch.Tensor:

    cx, cy, w, h = boxes.unbind(-1)
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    return torch.stack((x1, y1, x2, y2), dim=-1)

def coco2xyxy(boxes: torch.Tensor) -> torch.Tensor:

    x, y, w, h = boxes.unbind(-1)
    x1 = x
    y1 = y 
    x2 = x + w
    y2 = y + h
    return torch.stack((x1, y1, x2, y2), dim=-1)

def xywh2cxcywh(boxes: torch.Tensor) -> torch.Tensor:

    x1, y1, w, h = boxes.unbind(-1)
    cx = x1 + w / 2
    cy = y1 + h / 2
    return torch.stack((cx, cy, w, h), dim=-1)
