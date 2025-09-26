import torch
import cv2
import numpy as np
from myyololib.utils import cxcywh2xyxy

def visualize(tensor_image, boxes, classes, scores, orig_shape, label2coco_category, name = 'vis2.jpg'):
    tensor_image = tensor_image[[2,1,0],:,:] # BGR to RGB
    img = tensor_image.permute(1, 2, 0).cpu().numpy()*255
    img = img.astype(np.uint8).copy()
    orig_h, orig_w = orig_shape

    scale_h = orig_h / 640.0
    scale_w = orig_w / 640.0

    # 클래스별 고유 색상 팔레트 (RGB 형식)
    color_palette = [
        (255, 0, 0),    # 빨간색
        (0, 255, 0),    # 초록색 
        (0, 0, 255),    # 파란색 
        (255, 255, 0),  # 노란색 
        (255, 0, 255),  # 마젠타 
        (0, 255, 255),  # 청록색 
    ]

    # 색상 팔레트가 클래스 수보다 적을 경우, 나머지는 무작위 색상으로 채움
    while len(color_palette) < len(label2coco_category):
        color_palette.append(tuple(np.random.randint(0, 256, 3).tolist()))

    for box, cls_idx, score in zip(boxes, classes, scores):
        x1, y1, x2, y2 = box.cpu().numpy()
        x1 *= scale_w
        y1 *= scale_h
        x2 *= scale_w
        y2 *= scale_h
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        # 클래스별 고유 색상 사용
        color = color_palette[int(cls_idx) % len(color_palette)]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        class_name = label2coco_category[int(cls_idx)] if 0 <= int(cls_idx) < len(label2coco_category) else f'class_{int(cls_idx)}'
        label = f'{class_name}: {score.item():.2f}'
        cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    cv2.imwrite(name, img)

def demo(train_loader, model, device, coco_id2label, label2coco_category, conf_threshold=0.25, max_det=300, iou_threshold=0.5):
    orig_shape = (640, 640) 

    images, annotations = next(iter(train_loader))

    input = images[0].unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input, inference=True, do_postprocessing=True, conf_threshold=conf_threshold, max_det=max_det, iou_threshold=iou_threshold)
        
    boxes, scores, classes = outputs[0].split((4, 1, 1), 1)
    visualize(input[0], cxcywh2xyxy(boxes), classes, scores, orig_shape, label2coco_category, name = 'predict.jpg')

    b = []
    c = []
    s = []

    for an in annotations[0]:
        bbox = an['bbox']
        bbox = torch.Tensor(bbox)
        bbox = cxcywh2xyxy(bbox)*640

        category_id = an['category_id']
        cls = coco_id2label[category_id]

        b.append(bbox.unsqueeze(0))
        c.append(torch.Tensor([cls]).unsqueeze(0))
        s.append(torch.Tensor([1.0]).unsqueeze(0))

    boxes_label = torch.cat(b, dim=0)
    classes_label = torch.cat(c, dim=0)
    scores_label = torch.cat(s, dim=0)

    visualize(input[0], boxes_label, classes_label, scores_label, orig_shape, label2coco_category, name = 'label.jpg')
