import torch
import os
from myyololib.models import MyYOLOv8n, QYOLOv8n, NYOLOv8n
import networkx as nx
from collections import defaultdict
import yaml


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

def get_state_dict(checkpoint_path, device):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"{checkpoint_path} does not exist.")

    # extension check
    ext = os.path.splitext(checkpoint_path)[1].lower()

    try:
        if ext == ".pth":
            # Load state_dict
            state_dict = torch.load(checkpoint_path, map_location=device)
            print("Loaded .pth state_dict")
        elif ext == ".pt":
            # Load state dict directly or state dict from full model
            try:
                state_dict = torch.load(checkpoint_path, map_location=device, weights_only=False)['model_state_dict']
            except:
                state_dict = torch.load(checkpoint_path, map_location=device, weights_only=False)['model'].state_dict()
        else:
            raise ValueError("Unsupported file extension. Only (.pt, .pth) are allowed.")
    except Exception as e:
        print("Failed to load model:", e)

    return state_dict

def load_model(checkpoint_path, device, model_type="base", **kwargs):
    if model_type == "base":
        model = MyYOLOv8n()
    elif model_type == "qat":
        model = QYOLOv8n(model_qcfg=kwargs.pop('model_qcfg', None))
    elif model_type == "npu":
        model = NYOLOv8n(model_ncfg=kwargs.pop('model_ncfg', None))
    model.to(device)

    state_dict = get_state_dict(checkpoint_path, device)
    state_dict = match_keys_sequential(state_dict, model.state_dict(), print_info=False)
    model.load_state_dict(state_dict)

    model.eval()
    return model

def load_QAT_model(checkpoint_path, device, **kwargs):

    state_dict = get_state_dict(checkpoint_path, device)

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

    Qmodel = QYOLOv8n(model_qcfg=kwargs.pop('model_qcfg', None))
    Qmodel.to(device)

    qstate_dict = match_keys_sequential(fused_tensor_dict, Qmodel.state_dict(), print_info=False)
    Qmodel.load_state_dict(qstate_dict)

    Qmodel.eval()
    return Qmodel

def load_NPU_model(checkpoint_path, device, qcfg, **kwargs): 
    q_state_dict = get_state_dict(checkpoint_path, device)
    ncfg = qcfg2ncfg(qcfg) 
    # weight dequantization
    n_state_dict = {}
    for k in q_state_dict.keys():
        if 'model.22.dfl.conv' in k:
            ncfg_key = k
            config = ncfg[ncfg_key]
            # n_state_dict[k] = scale_clip_round(q_state_dict[k], config['num_bits'], config['fw']) 
            n_state_dict[k] = q_state_dict[k] 
        elif 'weight' in k:
            ncfg_key = k.removesuffix('.weight')
            config = ncfg[ncfg_key]
            n_state_dict[k] = scale_clip_round(q_state_dict[k], config['num_bits'], config['fw']) 
        elif 'bias' in k:
            ncfg_key = k.removesuffix('.bias')
            config = ncfg[ncfg_key]
            n_state_dict[k] = scale_clip_round(q_state_dict[k], 2*config['num_bits'], config['fw'] + config['fx']) 
        else:
            print(f'--- Warning: key {k} is not weight or bias ---')
        # print(f'ncfg_key: {ncfg_key}, num_bits: {config["num_bits"]}, fw: {config["fw"]}, fx: {config["fx"]}')

    Nmodel = NYOLOv8n(model_ncfg=ncfg_to_dictcfg(ncfg))
    
    Nmodel.to(device)
    npu_state_dict = match_keys_sequential(n_state_dict, Nmodel.state_dict(), print_info=False)
    # print(n_state_dict.keys(), Nmodel.state_dict().keys())
    Nmodel.load_state_dict(npu_state_dict)
    Nmodel.eval()
    if kwargs.get('debug_ret', False):
        return Nmodel, ncfg, npu_state_dict
    return Nmodel

# ----------------------- dependency graph generation code -----------------------
# TODO: move this part to another file

def rename_node(node):
    return node.replace('layer_', 'model.')

def smart_connect(graph, prev, current):
    if isinstance(prev, (list, tuple, set)):
        for p in prev:
            graph.add_edge(p, current)
    else:
        graph.add_edge(prev, current)

def depg_gen_bottleneck(name, graph, prev, cfg, shortcut=True):
    input_nodes = prev
    for k, v in cfg.items():
        current = f'{name}.{k}'
        graph.add_node(current, value=v)
        smart_connect(graph, prev, current)
        prev = current
    if shortcut:
        return input_nodes + [prev] if isinstance(input_nodes, list) else [input_nodes, prev]
    return [prev]

def depg_gen_c2f(name, graph, prev, cfg, shortcut=True):
    cv2_prev_nodes = [] # prev nodes of cv2
    current = f'{name}.cv1'
    graph.add_node(current, value=cfg['cv1'])
    smart_connect(graph, prev, current)
    prev = current
    cv2_prev_nodes.append(prev) # since cv1 output, prev is not a list

    for key, mcfg in cfg.items():
        if 'm' in key:
            prev = depg_gen_bottleneck(f'{name}.{key}', graph, prev, mcfg, shortcut=shortcut)
            cv2_prev_nodes  = cv2_prev_nodes + prev

    current = f'{name}.cv2'
    graph.add_node(current, value=cfg['cv2'])
    smart_connect(graph, cv2_prev_nodes, current)
    prev = current
    return prev

def depg_gen_sppf(name, graph, prev, cfg):
    current = f'{name}.cv1'
    graph.add_node(current, value=cfg['cv1'])
    smart_connect(graph, prev, current)
    prev = current

    current = f'{name}.cv2'
    graph.add_node(current, value=cfg['cv2'])
    smart_connect(graph, prev, current)
    prev = current

    return prev

def depg_gen_detect(name, graph, prev, cfg):
    prev_copy = prev
    current = f'{name}.cv2.0'
    graph.add_node(current, value=cfg['cv2.0'])
    smart_connect(graph, prev, current)
    prev = current

    current = f'{name}.cv2.1'
    graph.add_node(current, value=cfg['cv2.1'])
    smart_connect(graph, prev, current)
    prev = current

    current = f'{name}.cv2.2'
    graph.add_node(current, value=cfg['cv2.2'])
    smart_connect(graph, prev, current)
    prev = current

    prev = prev_copy
    current = f'{name}.cv3.0'
    graph.add_node(current, value=cfg['cv3.0'])
    smart_connect(graph, prev, current)
    prev = current

    current = f'{name}.cv3.1'
    graph.add_node(current, value=cfg['cv3.1'])
    smart_connect(graph, prev, current)
    prev = current

    current = f'{name}.cv3.2'
    graph.add_node(current, value=cfg['cv3.2'])
    smart_connect(graph, prev, current)
    prev = current




def qcfg2ncfg(qcfg):
    # TODO: move this layer connection configuration part to another file
    layer_type_map = {
        'conv2d' : ['layer_0', 'layer_1', 'layer_3', 'layer_5', 'layer_7',
                    'layer_16', 'layer_19'],
        'c2f' : ['layer_2', 'layer_4', 'layer_6', 'layer_8', 
                'layer_12', 'layer_15', 
                'layer_18', 'layer_21'],
        'sppf': ['layer_9'],
        'detect': ['layer_22'],
    }
    out2msin_layers = {
        # concat input
        'layer_4': 'layer_15',
        'layer_6': 'layer_12',
        'layer_12': 'layer_18',
        'layer_9': 'layer_21',
        # anchor input
        'layer_15': 'layer_22',
        'layer_18': 'layer_22',
    }
    shortcut_layers = ['layer_2', 'layer_4', 'layer_6', 'layer_8',]
    dfl_input_fraction_bits = 2

    # directed graph
    graph = nx.DiGraph()
    ms_prevs = defaultdict(list)

    prev = None
    for layer, cfg in qcfg.items():
        # check multi-scale input layer
        if layer in ms_prevs.keys():
            prev2 = ms_prevs[layer]
            prev = prev2 + [prev]

        if layer in layer_type_map['conv2d']:
            current = layer
            graph.add_node(current, value=cfg)
            if prev is not None:
                smart_connect(graph, prev, current)
            prev = current

        elif layer in layer_type_map['c2f']:
            if layer in shortcut_layers:
                shortcut = True
            else:
                shortcut = False
            prev = depg_gen_c2f(layer, graph, prev, cfg)

        elif layer in layer_type_map['sppf']:
            prev = depg_gen_sppf(layer, graph, prev, cfg)

        elif layer in layer_type_map['detect']:
            prev = depg_gen_detect(layer, graph, prev, cfg)

        # check whether the output of the layer goes into for multi-scale input layer
        if layer in out2msin_layers.keys():
            p = prev if isinstance(prev, list) else [prev]
            ms_prevs[out2msin_layers[layer]].extend(p)

    ncfg = {}

    for node in graph.nodes:
        # sanity check: is output activation scale is consistent?
        output_act_scale = [graph.nodes[next_node]['value'][2] for next_node in list(graph.successors(node))]
        if len(set(output_act_scale)) > 1:
            print(f'\n--- Warning: inconsistent output activation scales at node {node}: {output_act_scale} @ node{list(graph.successors(node))}\n')
        # elif len(output_act_scale) > 1:
        #     print(f'\n+++ Note: multiple successors at node {node}: {output_act_scale}\n')
        try:
            next = list(graph.successors(node))[0]
            num_bits = graph.nodes[next]['value'][0]
            fy = graph.nodes[next]['value'][2]
            fx = graph.nodes[node]['value'][2]
            fw = graph.nodes[node]['value'][1]
        except:
            # print(f'(end node)')
            num_bits = graph.nodes[node]['value'][0]
            fy = dfl_input_fraction_bits
            fx = graph.nodes[node]['value'][2]
            fw = graph.nodes[node]['value'][1]
        shift = fx + fw - fy
        # print(f'node {node} - fw: {fw}, fx: {fx}, fy: {fy}, num_bits: {num_bits}, shift: {shift}')
        # print(f'node {rename_node(node)} - fw: {fw}, fx: {fx}, fy: {fy}, num_bits: {num_bits}, shift: {shift}')
        if 'model.22.cv2' in rename_node(node) or 'model.22.cv3' in rename_node(node):
            conv_name = rename_node(node)
            if conv_name[-1] == '2':
                ncfg[conv_name[:-1] + '0.' + conv_name[-1]] = {
                    'fx': fx,
                    'fw': fw,
                    'fy': fy,
                    'num_bits': num_bits,
                    'shift': shift
                }
                ncfg[conv_name[:-1] + '1.' + conv_name[-1]] = {
                    'fx': fx,
                    'fw': fw,
                    'fy': fy,
                    'num_bits': num_bits,
                    'shift': shift
                }
                ncfg[conv_name[:-1] + '2.' + conv_name[-1]] = {
                    'fx': fx,
                    'fw': fw,
                    'fy': fy,
                    'num_bits': num_bits,
                    'shift': shift
                }
            else:
                ncfg[conv_name[:-1] + '0.' + conv_name[-1] + '.conv'] = {
                    'fx': fx,
                    'fw': fw,
                    'fy': fy,
                    'num_bits': num_bits,
                    'shift': shift
                }
                ncfg[conv_name[:-1] + '1.' + conv_name[-1] + '.conv'] = {
                    'fx': fx,
                    'fw': fw,
                    'fy': fy,
                    'num_bits': num_bits,
                    'shift': shift
                }
                ncfg[conv_name[:-1] + '2.' + conv_name[-1] + '.conv'] = {
                    'fx': fx,
                    'fw': fw,
                    'fy': fy,
                    'num_bits': num_bits,
                    'shift': shift
                }
        else:
            ncfg[rename_node(node) + '.conv'] = {
                'fx': fx,
                'fw': fw,
                'fy': fy,
                'num_bits': num_bits,
                'shift': shift
            }
    ncfg['model.22.dfl.conv.weight'] = {
        'fx': fy,
        'fw': 0,
        'fy': 3, # 4 for unsigned, 3 for signed
        'num_bits': num_bits,
        'shift': fx + fw - fy
    }
    return ncfg

# ---------------------------------------------
def bottleneck_qcfg_gen(cfg):
    return {
        "cv1": cfg.pop(0),
        "cv2": cfg.pop(0),
    }

def c2f_qcfg_gen(cfg, n):
    ret = {
        "cv1": cfg.pop(0),
    }
    ret["cv2"] = cfg.pop(0)
    for i in range(n):
        ret[f"m.{i}"] = bottleneck_qcfg_gen(cfg)
    return ret

def sppf_qcfg_gen(cfg):
    return {
        "cv1": cfg.pop(0),
        "cv2": cfg.pop(0),
    }

def detect_qcfg_gen(cfg):
    return {
        "cv2.0": cfg.pop(0),
        "cv2.1": cfg.pop(0),
        "cv2.2": cfg.pop(0),

        "cv3.0": cfg.pop(0),
        "cv3.1": cfg.pop(0),
        "cv3.2": cfg.pop(0),
    }

def ncfg_to_dictcfg(ncfg):
    npu_config = {
        "layer_0": ncfg['model.0.conv'],
        "layer_1": ncfg['model.1.conv'],
        "layer_2": c2f_qcfg_gen([ncfg['model.2.cv1.conv'], # cv1
                                ncfg['model.2.cv2.conv'], # cv2
                                ncfg['model.2.m.0.cv1.conv'], # bottleneck 0.cv1
                                ncfg['model.2.m.0.cv2.conv'], # bottleneck 0.cv2
                                ], 1),
        "layer_3": ncfg['model.3.conv'],
        "layer_4": c2f_qcfg_gen([ncfg['model.4.cv1.conv'], # cv1
                                ncfg['model.4.cv2.conv'], # cv2
                                ncfg['model.4.m.0.cv1.conv'], # bottleneck 0.cv1
                                ncfg['model.4.m.0.cv2.conv'], # bottleneck 0.cv2
                                ncfg['model.4.m.1.cv1.conv'], # bottleneck 1.cv1
                                ncfg['model.4.m.1.cv2.conv'], # bottleneck 1.cv2
                                ], 2),
        "layer_5": ncfg['model.5.conv'],
        "layer_6": c2f_qcfg_gen([ncfg['model.6.cv1.conv'], # cv1
                                ncfg['model.6.cv2.conv'], # cv2
                                ncfg['model.6.m.0.cv1.conv'], # bottleneck 0.cv1
                                ncfg['model.6.m.0.cv2.conv'], # bottleneck 0.cv2
                                ncfg['model.6.m.1.cv1.conv'], # bottleneck 1.cv1
                                ncfg['model.6.m.1.cv2.conv'], # bottleneck 1.cv2
                                ], 2),
        "layer_7": ncfg['model.7.conv'],
        "layer_8": c2f_qcfg_gen([ncfg['model.8.cv1.conv'], # cv1
                                ncfg['model.8.cv2.conv'], # cv2
                                ncfg['model.8.m.0.cv1.conv'], # bottleneck 0.cv1
                                ncfg['model.8.m.0.cv2.conv'], # bottleneck 0.cv2
                                ], 1),
        "layer_9": sppf_qcfg_gen([ncfg['model.9.cv1.conv'], # cv1
                                ncfg['model.9.cv2.conv'], # cv2
                                ]),
        "layer_12": c2f_qcfg_gen([ncfg['model.12.cv1.conv'], # cv1
                                ncfg['model.12.cv2.conv'], # cv2
                                ncfg['model.12.m.0.cv1.conv'], # bottleneck 0.cv1
                                ncfg['model.12.m.0.cv2.conv'], # bottleneck 0.cv2
                                ], 1),
        "layer_15": c2f_qcfg_gen([ncfg['model.15.cv1.conv'], # cv1
                                ncfg['model.15.cv2.conv'], # cv2
                                ncfg['model.15.m.0.cv1.conv'], # bottleneck 0.cv1
                                ncfg['model.15.m.0.cv2.conv'], # bottleneck 0.cv2
                                ], 1),
        "layer_16": ncfg['model.16.conv'],
        "layer_18": c2f_qcfg_gen([ncfg['model.18.cv1.conv'], # cv1
                                ncfg['model.18.cv2.conv'], # cv2
                                ncfg['model.18.m.0.cv1.conv'], # bottleneck 0.cv1
                                ncfg['model.18.m.0.cv2.conv'], # bottleneck 0.cv2
                                ], 1),
        "layer_19": ncfg['model.19.conv'],
        "layer_21": c2f_qcfg_gen([ncfg['model.21.cv1.conv'], # cv1
                                ncfg['model.21.cv2.conv'], # cv2
                                ncfg['model.21.m.0.cv1.conv'], # bottleneck 0.cv1
                                ncfg['model.21.m.0.cv2.conv'], # bottleneck 0.cv2
                                ], 1),

        "layer_22": detect_qcfg_gen([ncfg['model.22.cv2.0.0.conv'], # cv2.0
                                    ncfg['model.22.cv2.0.1.conv'], # cv2.1
                                    ncfg['model.22.cv2.0.2'], # cv2.2
                                    ncfg['model.22.cv3.0.0.conv'], # cv3.0
                                    ncfg['model.22.cv3.0.1.conv'], # cv3.1
                                    ncfg['model.22.cv3.0.2'], # cv3.2
                                    ]),
    }

    return npu_config