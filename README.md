# npu_yolov8n
Compressed and quantization-aware YOLOv8n model for NPU inference.

This repository provides a lightweight version of YOLOv8n optimized for deployment on NPUs, with a focus on Quantization-Aware Training (QAT) and NPU-friendly model structure.

# Features

- 🚀 YOLOv8n lightweight model for NPU inference
- 🔧 Quantization-Aware Training (QAT) support
- 📉 Model compression & NPU-friendly architecture
- 🧠 Trained and evaluated on COCO 2017 dataset
- 🔥 Based on [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)

# Installation
1.Install requirements
```
pip install pycocotools
pip install ultralytics
pip install torchmetrics
```
2. Download Dataset
```
python download_dataset.py
```

# Directory Structure
```
.
├── train.py                 # QAT evaluation script
├── download_dataset.py      # COCO2017 download script
├── checkpoints/             # Pretrained and fine-tuned model weights (Original YOLOv8n, ReLU retrained YOLOv8n, quantized YOLOv8n)
├── config/                  # Quantization configuration files
├── notebooks/               # Tutorial notebooks
└── npu_yolov8n/
    ├── data/                # Dataset loading and preprocessing
    ├── models/              # Model definitions and QAT/NPU related utilities
    ├── pruning/             # Model pruning (not implemented yet)
    ├── training/            # Training and QAT logic
    └── utils/               # Common utilities
```

# Quantization-Aware Training (QAT)
```
python train.py -eval
python train.py -eval --base_model 'checkpoints/relu_091214.pth' --quant_config 'config/q8_signed.yaml'

```
# experiments
![QuantizationexperimentresultsfortheYOLOv8n
model](assets/experiment.png)

