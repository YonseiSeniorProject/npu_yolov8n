import torch
from myyololib.utils import cxcywh2xyxy
from torchmetrics.detection import MeanAveragePrecision

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate(model, val_loader, coco_id2label, map_only=False, return_results=False):
    model.eval()
    model.to(device)

    preds = []
    targets = []

    for i, (images, annotations) in enumerate(val_loader): 
        input = torch.stack([img for img in images]).to(device)
        with torch.no_grad():
            outputs = model(input, inference=True, do_postprocessing=True, conf_threshold=0.0001, max_det=2000)

        for out in outputs:
            out
            preds.append(
                dict(
                    boxes=cxcywh2xyxy(out[:, :4].cpu()), # bbox [x1, y1, x2, y2]
                    scores=out[:, 4].cpu(), # conf score
                    labels=out[:, 5].int().cpu() # class label
                )
            )

        for an in annotations: # an (list): annotation of the sigle image. an is a list of dict that contains "category_id", "bbox", ... of each object in the image
            labels = torch.stack(
                [torch.tensor(coco_id2label[obj_dict["category_id"]]) for obj_dict in an]
                ) if len(an) > 0 else torch.zeros((0,))
            boxes = torch.stack(
                [torch.tensor(obj_dict["bbox"])*640 for obj_dict in an]
                ) if len(an) > 0 else torch.zeros((0, 4))
            
            targets.append(
                dict(
                    boxes=cxcywh2xyxy(boxes.cpu()), # convert (x,y,w,h) to xyxy
                    labels=labels.int().cpu() # class label
                )
            )

        if i % 100 == 99: 
            print(f"eval batch: {i+1}/{len(val_loader)}")

    metric = MeanAveragePrecision(box_format='xyxy', class_metrics=True)

    metric.update(preds, targets)

    results = metric.compute()
    
    print("evaluation results:")
    for key, value in results.items():
        print(f"{key}: {value}")
        if map_only:
            break
    
    # return results
    if return_results:
        return results
    return None

