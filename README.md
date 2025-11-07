# npu_yolov8n
compressed version of yolov8n model for npu

# Setup
```
pip install pycocotools
pip install ultralytics
pip install torchmetrics
```
# Dataset
```
python download_dataset.py
```
# QAT fine-tuning
```
python train.py -eval
python train.py -eval --base_model 'checkpoints/relu_091214.pth' --quant_config 'config/quantization_config.yaml'

```
