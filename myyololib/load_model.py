import torch
from myyololib.models import MyYOLOv8n, QYOLOv8n, NYOLOv8n

def match_keys_sequential(pretrained_dict, model_dict, print_info=False):
    """
    Match the keys of pretrained model and my model
    """
    # Check if the number of keys in pretrained model and my model are equal
    try:
        assert len(pretrained_dict) == len(model_dict)
    except AssertionError as DictLengthNotMatchError:
        print("The number of keys in pretrained model and my model are not equal.")
        return None
    
    # match the keys of pretrained model and my model
    print("Start matching keys...\n")
    for i, key in enumerate(pretrained_dict.keys()):
        if print_info:
            print(f'matching keys: {key} -> {list(model_dict.keys())[i]}')
        model_dict[list(model_dict.keys())[i]] = pretrained_dict[key]
    print("Matching keys done.\n")
    return model_dict
    
def scale_clip_round(x, b, f):
    qmin = -2 ** (b - 1)
    qmax = 2 ** (b - 1) - 1
    scale = 2 ** f
    return torch.clamp(torch.round(x * scale), qmin, qmax) 

def load_model(checkpoint_path, device, model_type="base"):
    if model_type == "base":
        model = MyYOLOv8n()
    elif model_type == "qat":
        model = QYOLOv8n()
    elif model_type == "npu":
        model = NYOLOv8n()
    model.to(device)

    state_dict = torch.load(checkpoint_path, map_location=device)
    state_dict = match_keys_sequential(state_dict, model.state_dict(), print_info=False)
    model.load_state_dict(state_dict)

    model.eval()
    return model

def load_QAT_model(checkpoint_path, device):

    state_dict = torch.load(checkpoint_path, map_location=device)

    # BatchNorm + Conv weight fusion
    fused_tensor_dict = {}

    for k in state_dict.keys():
        if 'conv.weight' in k:
            conv_weight = state_dict[k]
            conv_name = k.replace('conv.weight', '')

            # search for corresponding BatchNorm
            bn_weight = state_dict.get(conv_name + 'bn.weight', None)
            bn_bias   = state_dict.get(conv_name + 'bn.bias', None)
            bn_mean   = state_dict.get(conv_name + 'bn.running_mean', None)
            bn_var    = state_dict.get(conv_name + 'bn.running_var', None)
            bn_eps    = 1e-5 

            if bn_weight is not None:
                # BatchNorm fusion formula
                std = torch.sqrt(bn_var + bn_eps)
                fused_weight = conv_weight * (bn_weight / std).reshape(-1, 1, 1, 1)
                if state_dict.get(conv_name + 'bias') is not None:
                    conv_bias = state_dict[conv_name + 'bias']
                else:
                    conv_bias = torch.zeros_like(bn_mean)
                fused_bias = bn_bias + (conv_bias - bn_mean) * (bn_weight / std)

                fused_tensor_dict[conv_name + 'fused.weight'] = fused_weight
                fused_tensor_dict[conv_name + 'fused.bias'] = fused_bias
            else:
                # if BatchNorm doesn't exist, just add weight and bias
                fused_tensor_dict[conv_name + 'weight'] = conv_weight
                if state_dict.get(conv_name + 'bias') is not None:
                    fused_tensor_dict[conv_name + 'bias'] = state_dict[conv_name + 'bias']
        elif ("model.22.cv2.0.2" in k or "model.22.cv2.1.2" in k or "model.22.cv2.2.2" in k
            or "model.22.cv3.0.2" in k or "model.22.cv3.1.2" in k or "model.22.cv3.2.2" in k):
            fused_tensor_dict[k] = state_dict[k]

    Qmodel = QYOLOv8n()
    Qmodel.to(device)

    qstate_dict = match_keys_sequential(fused_tensor_dict, Qmodel.state_dict(), print_info=False)
    Qmodel.load_state_dict(qstate_dict)

    Qmodel.eval()
    return Qmodel

def load_NPU_model(checkpoint_path, device): # NOTE: use quantization config!!!
    q_state_dict = torch.load(checkpoint_path, map_location=device)

    bit_precision = 8

    special = {
        'model.0.conv.weight' : 3,
        'model.0.conv.bias' : 10,

        'model.22.cv2.0.2.weight' : 7,
        'model.22.cv2.0.2.bias' : 11,
        'model.22.cv2.1.2.weight' : 7,
        'model.22.cv2.1.2.bias' : 11,
        'model.22.cv2.2.2.weight' : 7,
        'model.22.cv2.2.2.bias' : 11,

        'model.22.cv3.0.2.weight' : 7,
        'model.22.cv3.0.2.bias' : 10,
        'model.22.cv3.1.2.weight' : 7,
        'model.22.cv3.1.2.bias' : 10,
        'model.22.cv3.2.2.weight' : 7,
        'model.22.cv3.2.2.bias' : 10,
    }

    n_state_dict = {}
    for k in q_state_dict.keys():
        if k in special.keys():
            if "weight" in k:
                n_state_dict[k] = scale_clip_round(q_state_dict[k], bit_precision, special[k])
            elif "bias" in k:
                n_state_dict[k] = scale_clip_round(q_state_dict[k], 2*bit_precision, special[k])
            continue
        if "dfl" in k:
            n_state_dict[k] = q_state_dict[k] # do not quantize
            continue
        if 'weight' in k:
            n_state_dict[k] = scale_clip_round(q_state_dict[k], bit_precision, 6) # default: fw=6, fx=4, fy=4]
        elif 'bias' in k:
            n_state_dict[k] = scale_clip_round(q_state_dict[k], 2*bit_precision, 10) # default: fw=6, fx=4, fy=4]

    Nmodel = NYOLOv8n()
    Nmodel.to(device)
    npu_state_dict = match_keys_sequential(n_state_dict, Nmodel.state_dict(), print_info=False)

    Nmodel.load_state_dict(npu_state_dict)
    Nmodel.eval()
    return Nmodel