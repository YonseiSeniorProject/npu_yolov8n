# npu_yolov8n
compressed version of yolov8n model for npu

# Setup
```
!pip install pycocotools
!pip install ultralytics
!pip install torchmetrics
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

# --- 데이터셋 다운로드 및 준비 (추가된 부분) ---
data_root = './datasets/coco'
annotations_dir = os.path.join(data_root, 'annotations')
train_images_path = os.path.join(data_root, 'train2017')
val_images_path = os.path.join(data_root, 'val2017')

# 데이터셋이 이미 준비되었는지 확인
if not (os.path.exists(train_images_path) and os.path.exists(val_images_path) and os.path.exists(annotations_dir)):
    print("COCO 2017 데이터셋을 다운로드합니다. 시간이 오래 걸릴 수 있습니다...")

    # 필요한 URL
    urls = {
        'annotations': 'http://images.cocodataset.org/annotations/annotations_trainval2017.zip',
        'train2017': 'http://images.cocodataset.org/zips/train2017.zip',
        'val2017': 'http://images.cocodataset.org/zips/val2017.zip'
    }

    # 다운로드 및 압축 해제 함수
    def download_and_unzip(url, download_root):
        zip_path = os.path.join(download_root, os.path.basename(url))
        if not os.path.exists(zip_path.replace('.zip', '')):
            os.makedirs(download_root, exist_ok=True)
            print(f"{os.path.basename(url)} 다운로드 중...")
            urllib.request.urlretrieve(url, zip_path)
            print("압축 해제 중...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(download_root)
            os.remove(zip_path) # 압축 해제 후 zip 파일 삭제
            print("완료.")
        else:
            print(f"{os.path.basename(zip_path).replace('.zip','')}가 이미 존재합니다.")

    # 각 파일 다운로드 실행
    download_and_unzip(urls['annotations'], data_root)
    download_and_unzip(urls['train2017'], data_root)
    download_and_unzip(urls['val2017'], data_root)

else:
    print("데이터셋이 이미 존재합니다. 다운로드를 건너뜁니다.")
```


