
import os
import torch
import numpy as np
import yaml

from myyololib.dataset import get_dataloader

from myyololib.train_config import TrainConfig, FinetuneConfig
from myyololib.trainer import train
from myyololib.evaluator import evaluate
from myyololib.load_model import load_model, load_QAT_model, load_NPU_model

import argparse
from datetime import datetime

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='for QAT model fine-tuning') # TODO: add full model finetuning
    parser.add_argument('-b', '--base_model', type=str, default='./checkpoints/relu_091214.pth', help='Path to the base model checkpoint')
    parser.add_argument('-q', '--quant_config', type=str, default='./config/quantization_config.yaml', help='Path to the config file')
    parser.add_argument('-eval', '--evaluate', help='whether to evaluate the model', action='store_true', default=False)
    args = parser.parse_args()

    # load dataset
    train_loader, val_loader, coco_id2label, label2coco_category = get_dataloader()

    #load quantization config
    with open(args.quant_config) as f:
        loaded_cfg = yaml.safe_load(f)

    Qmodel = load_QAT_model(args.base_model, device, model_qcfg=loaded_cfg)
    train(Qmodel, train_loader, val_loader, coco_id2label, FinetuneConfig)

    # Evaluate
    if args.evaluate:
        evaluate(Qmodel, val_loader, coco_id2label, map_only=True, return_results=False)

    # Model Save
    now = datetime.now()
    formatted = now.strftime("%m%d%H%M")
    save_model_name = f"qat_{formatted}"
    # torch.save(Qmodel.state_dict(), save_model_name)

    os.makedirs('./checkpoints', exist_ok=True)
    ckpt_path = f'./checkpoints/{save_model_name}.pt'
    torch.save({
        'qcfg': loaded_cfg,
        'train_config': FinetuneConfig,
        'model_state_dict': Qmodel.state_dict(),
    }, ckpt_path)
    print(f"💾 Checkpoint saved: {ckpt_path}")

