# npu_yolov8n
compressed version of yolov8n model for npu

# Setup
```
apt-get update
apt-get install -y libgl1-mesa-glx
pip install pycocotools
pip install torchmetrics
```
# Dataset
```
import torch
from torch.utils.data import DataLoader
import torchvision.datasets as dset
import torchvision.transforms as transforms
import os
import urllib.request
import zipfile

data_root = './datasets/coco'
annotations_dir = os.path.join(data_root, 'annotations')
train_images_path = os.path.join(data_root, 'train2017')
val_images_path = os.path.join(data_root, 'val2017')

if not (os.path.exists(train_images_path) and os.path.exists(val_images_path) and os.path.exists(annotations_dir)):
    print("Downloading COCO 2017 dataset. It takes time...")

    urls = {
        'annotations': 'http://images.cocodataset.org/annotations/annotations_trainval2017.zip',
        'train2017': 'http://images.cocodataset.org/zips/train2017.zip',
        'val2017': 'http://images.cocodataset.org/zips/val2017.zip'
    }

    def download_and_unzip(url, download_root):
        zip_path = os.path.join(download_root, os.path.basename(url))
        if not os.path.exists(zip_path.replace('.zip', '')):
            os.makedirs(download_root, exist_ok=True)
            print(f"{os.path.basename(url)} downloading...")
            urllib.request.urlretrieve(url, zip_path)
            print("unzip...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(download_root)
            os.remove(zip_path) 
            print("done.")
        else:
            print(f"{os.path.basename(zip_path).replace('.zip','')}already exists.")

    download_and_unzip(urls['annotations'], data_root)
    download_and_unzip(urls['train2017'], data_root)
    download_and_unzip(urls['val2017'], data_root)

else:
    print("dataset already exists.")
```
# train
```
python train.py -eval
python train.py -eval --base_model 'checkpoints/relu_091214.pth' --quant_config 'config/finetune_config.yaml'

```
