import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def visualize_multiple_tensors(tensor_dict, bins=50, n_cols=3):
    """
    여러 텐서를 한 화면에 가로로 subplot으로 시각화, 최대/최솟값 맞춰서 축 설정. float 텐서.
    
    Args:
        tensor_dict (dict): {이름: tensor} 형태
        bins (int): 히스토그램 bins
        n_cols (int): 한 행에 그릴 subplot 개수
    """
    names = list(tensor_dict.keys())
    n = len(names)
    n_rows = (n + n_cols - 1) // n_cols  # 필요한 행 개수 계산

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    axes = np.array(axes).reshape(-1)  # 1D 배열로 변환, subplot 개수가 1행일 때도 안전

    for i, name in enumerate(names):
        tensor = tensor_dict[name].flatten()
        min_val = tensor.min().item()
        max_val = tensor.max().item()

        max_val = max(abs(min_val), abs(max_val))
        min_val = -max_val

        axes[i].hist(tensor.cpu().numpy(), bins=bins, color='skyblue')
        axes[i].set_title(name)
        axes[i].set_xlim(min_val, max_val)
        axes[i].grid(axis='y', linestyle='--', alpha=0.7)

    # 남는 subplot 제거
    for j in range(i+1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()

def visualize_float_tensor_distribution(tensor, bins=50, title='Float Tensor Value Distribution'):
    """
    float 텐서의 값 분포를 시각화하고, 최소/최댓값 출력.
    
    Args:
        tensor (torch.Tensor): 시각화할 float 텐서
        bins (int): 히스토그램 bin 개수
    """
    # flatten
    tensor_flat = tensor.flatten()

    # 최소, 최대
    min_val = tensor_flat.min().item()
    max_val = tensor_flat.max().item()

    # 히스토그램 생성
    plt.figure(figsize=(12, 3))
    plt.hist(tensor_flat.cpu().numpy(), bins=bins, color='skyblue')
    
    # 제목, 라벨
    plt.title(title)
    plt.xlabel('Value')
    plt.ylabel('n')

    # x축 범위
    plt.xlim(min_val, max_val)

    # 그리드
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()


def visualize_tensor_distribution(tensor, bit_width=8, fraction_bits=4, title='Tensor Value Distribution'):
    # 표현 가능한 실제 값의 전체 범위
    values = torch.arange(-2**(bit_width-1), 2**(bit_width-1)) / (2**fraction_bits)

    # 실제 텐서 값 → scaling 적용
    tensor_scaled = tensor.to(torch.float32) / (2**fraction_bits)

    # unique + count
    unique_values, counts = torch.unique(tensor_scaled, return_counts=True)
    counts_dict = dict(zip(unique_values.tolist(), counts.tolist()))

    # values 순서대로 카운트
    counts = torch.tensor([counts_dict.get(float(x), 0) for x in values])

    # plot
    plt.figure(figsize=(12, 3))
    plt.bar(values.numpy(), counts.numpy(), color='skyblue', width=1/(2**fraction_bits))

    # x축 범위
    plt.xlim(values.min().item(), values.max().item())

    # x축 눈금: 4칸 간격으로 설정
    step = 4
    tick_idx = np.arange(0, len(values), step)  # 0, 4, 8, ...
    plt.xticks(values[tick_idx].numpy(), rotation=90, fontsize=8)

    # 제목, 라벨
    plt.title(title)
    plt.xlabel('values')
    plt.ylabel('n')

    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

def visualize_multiple_poz(tensor_dict, n_cols=3):
    names = list(tensor_dict.keys())
    n = len(names)
    n_rows = (n + n_cols - 1) // n_cols  # needed rows

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    axes = np.array(axes).reshape(-1)  

    for i, name in enumerate(names):
        td_zero = (tensor_dict[name] == 0)
        # 각 채널(index)별 0 비율 계산
        # tensor: [num_channels] 각 채널에 대한 배치 평균 0 비율
        tensor = td_zero.float().mean(dim=1) * 100

        num_channels = tensor_dict[name].size(0)
        indices = np.arange(num_channels)  # 채널 index

        # 히스토그램 대신 bar plot으로 index별 표시
        axes[i].bar(indices, tensor.cpu().numpy(), color='skyblue')
        axes[i].set_title(f"Percentage of Zeros in activation '{name}'")
        axes[i].set_xlabel("Channel Index")
        axes[i].set_ylabel("Percentage of Zeros")
        axes[i].set_xticks(indices)
        axes[i].set_ylim(0, 100)
        axes[i].grid(axis='y', linestyle='--', alpha=0.7)

    # remove unused subplots
    for j in range(i+1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()

# activation hooking
class ActivationStatsCollector:
    """
    collect activation statistics using forward hooks.
    Args:
        model (nn.Module): model
        layer_class (nn.Module): class to hook. (single class or tuple of classes)
    """
    def __init__(self, model, layer_class=nn.Conv2d):
        self.activation_stats = {}
        self._register_hooks(model, layer_class)

    def _get_hook(self, name):
        def hook(module, input, output):
            x = input[0].detach().cpu()
            self.activation_stats[name].append(x)
        return hook

    def _register_hooks(self, model, layer_class):
        for name, module in model.named_modules():
            if isinstance(module, layer_class):
                self.activation_stats[name] = []
                module.register_forward_hook(self._get_hook(name))

    def get_stats(self):
        return self.activation_stats

    def clear_stats(self):
        for key in self.activation_stats.keys():
            self.activation_stats[key] = []