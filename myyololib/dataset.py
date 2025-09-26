from torch.utils.data import DataLoader
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch
import numpy as np
import cv2
import os
import json
import copy
from PIL import Image

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# custom CocoDetection with bbox resize
class CocoDetectionWithResize(dset.CocoDetection):
    def __init__(self, root, annFile, img_size=640, train=True):
        super().__init__(root, annFile)
        self.img_size = img_size
        self.train = train
        self.to_tensor = transforms.ToTensor()

    def __getitem__(self, index):
        img, target = super().__getitem__(index)
        target_cpy = copy.deepcopy(target)
        
        # --- YOLOv8 style resize with padding (keeping aspect ratio) ---
        img = np.array(img.copy())  # PIL to numpy
        img, r, (dw, dh)= self.letterbox(img, new_shape=(self.img_size, self.img_size))
        img = Image.fromarray(img) # numpy to PIL
        # -----------------------------------------------------
        img = self.to_tensor(img)

        # resize bbox of annotations
        for obj in target_cpy:
            x, y, w, h = obj['bbox']
            # resize bbox
            x = x * r + int(round(dw - 0.1))
            y = y * r + int(round(dh - 0.1))
            w *= r
            h *= r
            # convert xywh to cxcywh
            cx = x + w / 2
            cy = y + h / 2
            # normalize bbox
            obj['bbox'] = [cx/self.img_size, cy/self.img_size, w/self.img_size, h/self.img_size] 

        return img, target_cpy
    
    @staticmethod
    def letterbox(im, new_shape=(640, 640), color=(114, 114, 114)):
        """YOLOv8 style resize with padding (keeping aspect ratio)."""
        shape = im.shape[:2]  # (h, w)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])  # resize ratio
        new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # padding
        dw /= 2
        dh /= 2

        im_resized = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        im_padded = cv2.copyMakeBorder(im_resized, top, bottom, left, right,
                                    cv2.BORDER_CONSTANT, value=color)
        return im_padded, r, (dw, dh)

def get_dataloader(data_root: str = './datasets/coco', batch_size: int = 16, input_size: int = 640):
    # dataset directory
    annotations_dir = os.path.join(data_root, 'annotations')
    train_images_path = os.path.join(data_root, 'train2017')
    val_images_path = os.path.join(data_root, 'val2017')
    train_ann_path = os.path.join(annotations_dir, 'instances_train2017.json')
    val_ann_path = os.path.join(annotations_dir, 'instances_val2017.json')

    # datasets
    train_dataset = CocoDetectionWithResize(root=train_images_path,
                                            annFile=train_ann_path,
                                            img_size=input_size,
                                            train=True)

    val_dataset = CocoDetectionWithResize(root=val_images_path,
                                        annFile=val_ann_path,
                                        img_size=input_size,
                                        train=False)

    # dataloader
    def collate_fn(batch):
        return tuple(zip(*batch))

    train_loader = DataLoader(dataset=train_dataset,
                            batch_size=batch_size,
                            shuffle=True,
                            collate_fn=collate_fn)

    val_loader = DataLoader(dataset=val_dataset,
                            batch_size=batch_size,
                            shuffle=False,
                            collate_fn=collate_fn)

    with open(val_ann_path, 'r') as f: # load coco annotations 
        coco_data = json.load(f) 

    coco_id2label = {} # key: coco id, value: 0~79 index
    label2coco_category = {} # key: 0~79 index, value: coco category name

    for i, data in enumerate(coco_data['categories']):
        if i == 80: # num_classes: 80
            break
        coco_id2label[data['id']] = i
        label2coco_category[i] = data['name']

    return train_loader, val_loader, coco_id2label, label2coco_category

def preprocess_dataset(images, annotations, coco_id2label, device):
    input = torch.stack([img for img in images]).to(device)

    batch_idx_list = []
    cls_list = []
    bboxes_list = []

    for i, an in enumerate(annotations):
        for obj_dict in an:
            batch_idx_list.append(i)
            cls_list.append(coco_id2label[obj_dict["category_id"]])
            bboxes_list.append(obj_dict["bbox"])

    batch = {
        "batch_idx": torch.tensor(batch_idx_list, dtype=torch.long, device=device),
        "cls": torch.tensor(cls_list, dtype=torch.long, device=device),
        "bboxes": torch.tensor(bboxes_list, dtype=torch.float32, device=device)
    }

    return input, batch