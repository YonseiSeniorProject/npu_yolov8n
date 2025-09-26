import torch
import torch.nn as nn
from myyololib.basic_blocks import Conv, C2f, SPPF, Detect, Concat
from myyololib.qat_blocks import QConv, QC2f, QSPPF, QDetect, custom_clip_round
from myyololib.npu_blocks import NConv, NC2f, NSPPF, NDetect


class MyYOLOv8n(nn.Module):
    def __init__(self, ch=3, nc=80):
        super().__init__()
        self.model = nn.Sequential(
            Conv(ch, 16, 3, 2),                         # 0
            Conv(16, 32, 3, 2),                         # 1
            C2f(32, 32, 1, True),                       # 2
            Conv(32, 64, 3, 2),                         # 3
            C2f(64, 64, 2, True),                       # 4
            Conv(64, 128, 3, 2),                        # 5
            C2f(128, 128, 2, True),                     # 6
            Conv(128, 256, 3, 2),                       # 7
            C2f(256, 256, 1, True),                     # 8
            SPPF(256, 256, 5),                          # 9
            nn.Upsample(scale_factor=2, mode='nearest'),# 10
            Concat(dimension=1),                        # 11
            C2f(384, 128, 1, False),                    # 12
            nn.Upsample(scale_factor=2, mode='nearest'),# 13
            Concat(dimension=1),                        # 14
            C2f(192, 64, 1, False),                     # 15
            Conv(64, 64, 3, 2),                         # 16
            Concat(dimension=1),                        # 17
            C2f(192, 128, 1, False),                    # 18
            Conv(128, 128, 3, 2),                       # 19
            Concat(dimension=1),                        # 20
            C2f(384, 256, 1, False),                    # 21
            Detect((64, 128, 256))                  # 22, [P3, P4, P5] 
        )

    def forward(self, x, inference=False, do_postprocessing=False, conf_threshold=0.25, max_det=300, iou_threshold=0.7):    
        skip_connection = {
            6: 11,
            4: 14,
            12: 17,
            9: 20
        }
        memory = {}

        # print(f"Model input shape: {x.shape}")
        feature_maps = []                               # feature maps [P3, P4, P5]
        for layer_idx, layer in enumerate(self.model):
            if isinstance(layer, Concat):             
                x = layer([x, memory[layer_idx]])
            elif layer_idx in skip_connection.keys():             
                x = layer(x)
                memory[skip_connection[layer_idx]] = x
            elif layer_idx in [15, 18, 21]:             # save P3, P4, P5 
                x = layer(x)
                feature_maps.append(x)
            elif isinstance(layer, Detect):
                features = layer(feature_maps)                  
                if inference:
                    features = layer._inference(features)
                    if do_postprocessing:
                        features = layer.detect(features.permute(0, 2, 1), conf_threshold=conf_threshold, max_det=max_det, iou_threshold=iou_threshold)
            else:
                x = layer(x)

        return features
    

class QYOLOv8n(nn.Module):
    def __init__(self, ch=3, nc=80):
        super().__init__()
        self.model = nn.Sequential(
            QConv(ch, 16, 3, 2, w_fraction_bits=3, a_fraction_bits=7),                         # 0
            QConv(16, 32, 3, 2),                         # 1
            QC2f(32, 32, 1, True),                       # 2
            QConv(32, 64, 3, 2),                         # 3
            QC2f(64, 64, 2, True),                       # 4
            QConv(64, 128, 3, 2),                        # 5
            QC2f(128, 128, 2, True),                     # 6
            QConv(128, 256, 3, 2),                       # 7
            QC2f(256, 256, 1, True),                     # 8
            QSPPF(256, 256, 5),                          # 9
            nn.Upsample(scale_factor=2, mode='nearest'),# 10
            Concat(dimension=1),                        # 11
            QC2f(384, 128, 1, False),                    # 12
            nn.Upsample(scale_factor=2, mode='nearest'),# 13
            Concat(dimension=1),                        # 14
            QC2f(192, 64, 1, False),                     # 15
            QConv(64, 64, 3, 2),                         # 16
            Concat(dimension=1),                        # 17
            QC2f(192, 128, 1, False),                    # 18
            QConv(128, 128, 3, 2),                       # 19
            Concat(dimension=1),                        # 20
            QC2f(384, 256, 1, False),                    # 21
            QDetect((64, 128, 256))                  # 22, [P3, P4, P5] 
        )

    def forward(self, x, inference=False, do_postprocessing=False, conf_threshold=0.25, max_det=300, iou_threshold=0.7):    
        # input quantization
        self.input_precision = 8
        self.input_fraction_bits = 7
        self.input_scale = 2 ** self.input_fraction_bits
        self.input_min = -2 ** (self.input_precision - 1)
        self.input_max = 2 ** (self.input_precision - 1) - 1
        x = custom_clip_round(x*self.input_scale, self.input_min, self.input_max)/self.input_scale

        skip_connection = {
            6: 11,
            4: 14,
            12: 17,
            9: 20
        }
        memory = {}

        # print(f"Model input shape: {x.shape}")
        feature_maps = []                               # feature maps [P3, P4, P5]
        for layer_idx, layer in enumerate(self.model):
            if isinstance(layer, Concat):             
                x = layer([x, memory[layer_idx]])
            elif layer_idx in skip_connection.keys():             
                x = layer(x)
                memory[skip_connection[layer_idx]] = x
            elif layer_idx in [15, 18, 21]:             # save P3, P4, P5 
                x = layer(x)
                feature_maps.append(x)
            elif isinstance(layer, Detect):
                features = layer(feature_maps)                  
                if inference:
                    features = layer._inference(features)
                    if do_postprocessing:
                        features = layer.detect(features.permute(0, 2, 1), conf_threshold=conf_threshold, max_det=max_det, iou_threshold=iou_threshold)
            else:
                x = layer(x)

        return features
    

class NYOLOv8n(nn.Module):
    def __init__(self, ch=3, nc=80):
        super().__init__()
        self.model = nn.Sequential(
            NConv(ch, 16, 3, 2, fx=7, fw=3, fy=4, vis=True),                         # 0
            NConv(16, 32, 3, 2),                         # 1
            NC2f(32, 32, 1, True),                       # 2
            NConv(32, 64, 3, 2),                         # 3
            NC2f(64, 64, 2, True),                       # 4
            NConv(64, 128, 3, 2),                        # 5
            NC2f(128, 128, 2, True),                     # 6
            NConv(128, 256, 3, 2),                       # 7
            NC2f(256, 256, 1, True),                     # 8
            NSPPF(256, 256, 5),                          # 9
            nn.Upsample(scale_factor=2, mode='nearest'),# 10
            Concat(dimension=1),                        # 11
            NC2f(384, 128, 1, False),                    # 12
            nn.Upsample(scale_factor=2, mode='nearest'),# 13
            Concat(dimension=1),                        # 14
            NC2f(192, 64, 1, False),                     # 15
            NConv(64, 64, 3, 2),                         # 16
            Concat(dimension=1),                        # 17
            NC2f(192, 128, 1, False),                    # 18
            NConv(128, 128, 3, 2),                       # 19
            Concat(dimension=1),                        # 20
            NC2f(384, 256, 1, False),                    # 21
            NDetect((64, 128, 256))                  # 22, [P3, P4, P5] 
        )

    def forward(self, x, inference=False, do_postprocessing=False, conf_threshold=0.25, max_det=300, iou_threshold=0.7):    
        # input quantization
        self.input_precision = 8
        self.input_fraction_bits = 7
        self.input_scale = 2 ** self.input_fraction_bits
        self.input_min = -2 ** (self.input_precision - 1)
        self.input_max = 2 ** (self.input_precision - 1) - 1
        x = custom_clip_round(x*self.input_scale, self.input_min, self.input_max) # convert to int8 range

        skip_connection = {
            6: 11,
            4: 14,
            12: 17,
            9: 20
        }
        memory = {}

        # print(f"Model input shape: {x.shape}")
        feature_maps = []                               # feature maps [P3, P4, P5]
        for layer_idx, layer in enumerate(self.model):
            if isinstance(layer, Concat):             
                x = layer([x, memory[layer_idx]])
            elif layer_idx in skip_connection.keys():             
                x = layer(x)
                memory[skip_connection[layer_idx]] = x
            elif layer_idx in [15, 18, 21]:             # save P3, P4, P5 
                x = layer(x)
                feature_maps.append(x)
            elif isinstance(layer, Detect):
                features = layer(feature_maps)                  
                if inference:
                    features = layer._inference(features)
                    if do_postprocessing:
                        features = layer.detect(features.permute(0, 2, 1), conf_threshold=conf_threshold, max_det=max_det, iou_threshold=iou_threshold)
            else:
                x = layer(x)

        return features