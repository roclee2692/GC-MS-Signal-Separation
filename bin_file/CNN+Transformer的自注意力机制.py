import os
os.environ["OMP_NUM_THREADS"] = "8"
os.environ["OPENBLAS_NUM_THREADS"] = "8"
os.environ["MKL_NUM_THREADS"] = "8"
os.environ["VECLIB_MAXIMUM_THREADS"] = "8"
os.environ["NUMEXPR_NUM_THREADS"] = "8"
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torch.amp import autocast, GradScaler  # 替换旧版cuda.amp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from scipy.interpolate import interp1d
from scipy.signal import find_peaks, savgol_filter
import matplotlib.pyplot as plt
import math
import warnings
import random
from multiprocessing import cpu_count
from numba import jit, prange
import time

warnings.filterwarnings('ignore')
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# ==================== GPU环境优化（新增） ====================
# 设置CUDA基准模式，提升性能
torch.backends.cudnn.benchmark = True
# 禁用CUDA扩展日志，减少开销
os.environ["CUDA_LAUNCH_BLOCKING"] = "0"
# 设置GPU显存预分配（避免碎片化）
torch.cuda.empty_cache()


# ==================== 分布式训练配置（GPU多卡） ====================
def setup_ddp():
    """初始化分布式训练环境"""
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        init_process_group(backend="nccl")
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        return local_rank
    return -1


def cleanup_ddp():
    """销毁分布式训练环境"""
    if torch.distributed.is_initialized():
        destroy_process_group()


# ==================== 配置参数（集中管理，GPU优化） ====================
class Config:
    # 数据路径
    dataset_A_dir = os.environ.get('DATASET_A_DIR', os.path.join(PROJECT_ROOT, 'GCMS_单个样本数据'))
    output_dir = os.environ.get('OUTPUT_DIR', os.path.join(PROJECT_ROOT, 'outputs', 'cnn_transformer'))

    # 色谱图网格
    rt_min = 4
    rt_max = 50
    n_rt_points = 1000

    # 数据增强参数
    prominence_scale = 0.15
    min_distance_denom = 500
    min_distance_min = 2
    drift_amplitude = (0.08, 0.2)
    drift_direction = "random"
    drift_percentage = 0.05
    smooth_transition_ratio = 0.5
    peak_noise_ratio = 0.02
    baseline_noise_ratio = 0.005
    smooth_window_ratio = 0.005
    min_peak_intensity_ratio = 0.85

    # 训练数据生成
    num_augmentations = 5  # 每个原始样本的增强次数
    num_ratios_per_pair = 8  # 每对样本生成的比例数

    # 训练参数（GPU显存优化）
    test_ratio = 0.2
    batch_size = 128  # RTX 5060 8GB最优批次
    learning_rate = 5e-4
    epochs = int(os.environ.get('EPOCHS', '100'))
    lambda_reg = 0.05  # 比例损失的权重
    grad_accum_steps = 1  # 显存充足时设为1
    seed = int(os.environ.get('SEED', '42'))

    # DataLoader 参数（GPU优化）
    num_workers = 0  # Windows+GPU禁用多进程，避免冲突
    prefetch_factor = None  # 禁用预取因子（num_workers=0时必须）
    pin_memory = True  # 固定内存页（GPU加速）
    persistent_workers = False  # Windows+GPU禁用持久化进程

    # 设备加速配置（RTX 5060适配）
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    use_amp = True  # 混合精度训练（必须开启，节省显存）
    use_compile = False  # 关闭TorchCompile（避免编译器兼容问题）
    use_ddp = False  # 单卡禁用DDP
    num_threads = 8  # CPU线程数

    # 模型参数（GPU显存优化）
    d_model = 64
    nhead = 8
    num_layers = 4
    dim_feedforward = 256


# 设置CPU线程数
torch.set_num_threads(Config.num_threads)


# ==================== Numba加速CPU密集型函数 ====================
@jit(nopython=True, parallel=True, fastmath=True)
def detect_peak_windows_numba(intensity, rt_step, prominence, distance, threshold):
    """Numba加速的峰检测（CPU并行）"""
    n = len(intensity)
    peaks_idx = []
    # 简化版find_peaks（Numba兼容）
    for i in prange(1, n - 1):
        if intensity[i] > intensity[i - 1] and intensity[i] > intensity[i + 1]:
            # 距离过滤
            if len(peaks_idx) == 0 or i - peaks_idx[-1] >= distance:
                # 突出度过滤
                left_base = np.min(intensity[peaks_idx[-1] + 1:i] if peaks_idx else intensity[:i])
                right_base = np.min(
                    intensity[i + 1:peaks_idx[-1] + distance + 1] if peaks_idx else intensity[i + 1:i + distance + 1])
                peak_prom = intensity[i] - max(left_base, right_base)
                if peak_prom >= prominence:
                    peaks_idx.append(i)

    peak_windows = []
    peak_max_intensities = []
    for idx in peaks_idx:
        left = idx
        while left > 0 and intensity[left] > threshold * 0.5:
            left -= 1
        right = idx
        while right < n - 1 and intensity[right] > threshold * 0.5:
            right += 1
        peak_windows.append((left, right))
        peak_max_intensities.append(np.max(intensity[left:right + 1]))

    baseline = intensity.copy()
    for l, r in peak_windows:
        baseline[l:r + 1] = 0
    return np.array(peaks_idx), peak_windows, baseline, peak_max_intensities


@jit(nopython=True, parallel=True, fastmath=True)
def add_peak_drift_numba(intensity, baseline, peak_windows, rt_step, peak_max_intensities,
                         drift_amplitude, drift_percentage, smooth_transition_ratio):
    """Numba加速的峰漂移（CPU并行）"""
    n = len(intensity)
    out_int = baseline.copy()
    peak_indices = np.arange(len(peak_windows))
    np.random.shuffle(peak_indices)
    keep = int(len(peak_windows) * drift_percentage)
    selected = set(peak_indices[:keep])

    for i in prange(len(peak_windows)):
        l, r = peak_windows[i]
        if i in selected:
            peak_data = intensity[l:r + 1].copy()
            dv = np.random.uniform(drift_amplitude[0], drift_amplitude[1]) * np.random.choice([-1, 1])
            shift = int(round(dv / rt_step))
            if abs(shift) < 2:
                shift = 2 if shift >= 0 else -2
            nl, nr = l + shift, r + shift

            if nl < 0 or nr >= n:
                out_int[l:r + 1] = peak_data
                continue

            out_int[nl:nr + 1] = peak_data
            trans = max(3, int((nr - nl) * smooth_transition_ratio))

            a, b = max(0, nl - trans), nl
            if a < b:
                w = np.linspace(0, 1, b - a)
                out_int[a:b] = baseline[a:b] + peak_data[0] * w

            a, b = nr, min(n - 1, nr + trans)
            if a < b:
                w = np.linspace(1, 0, b - a)
                out_int[a:b] = baseline[a:b] + peak_data[-1] * w
        else:
            out_int[l:r + 1] = intensity[l:r + 1]
    return np.maximum(out_int, 0)


# ==================== 核心函数优化 ====================
def read_chromatogram(file_path):
    """读取Excel色谱图（优化IO速度）"""
    # 使用快速引擎读取Excel
    df = pd.read_excel(file_path, engine='openpyxl' if file_path.endswith('.xlsx') else 'xlrd')
    df.columns = [c.strip().lower() for c in df.columns]
    rt_col = next((c for c in df.columns if 'rt' in c or '保留时间(分钟)' in c), None)
    int_col = next((c for c in df.columns if '归一化强度(10^8)' in c or 'intensity' in c), None)
    if rt_col is None or int_col is None:
        raise ValueError(f"无法识别RT或强度列: {file_path}")
    # 向量化操作，减少循环
    df = df[[rt_col, int_col]].dropna()
    df = df.sort_values(by=rt_col)
    # 转换为numpy数组（减少pandas开销）
    rt_vals = df[rt_col].values.astype(np.float32)
    int_vals = df[int_col].values.astype(np.float32)
    return rt_vals, int_vals


def interpolate_to_grid(rt, intensity, grid):
    """插值优化（向量化）"""
    f = interp1d(rt, intensity, kind='linear', bounds_error=False, fill_value=0, assume_sorted=True)
    return f(grid).astype(np.float32)


def detect_peak_windows(intensity, rt_vals, prominence_scale=0.15, min_distance_denom=500, min_distance_min=2):
    """峰检测（CPU加速版）"""
    std = np.std(intensity)
    distance = max(min_distance_min, len(intensity) // min_distance_denom)
    prominence = std * prominence_scale
    threshold = std * 1.5
    rt_step = np.mean(np.diff(rt_vals)) if len(rt_vals) > 1 else 0.01

    # 调用Numba加速函数
    peaks_idx, peak_windows, baseline, peak_max_intensities = detect_peak_windows_numba(
        intensity, rt_step, prominence, distance, threshold
    )
    peak_centers = rt_vals[peaks_idx]
    return peak_centers, peak_windows, baseline, rt_step, peak_max_intensities


def add_peak_drift_only(intensity, rt_vals, peak_centers, peak_windows, baseline, rt_step, peak_max_intensities,
                        drift_amplitude, drift_direction, drift_percentage, smooth_transition_ratio):
    """峰漂移（CPU加速版）"""
    return add_peak_drift_numba(
        intensity, baseline, peak_windows, rt_step, peak_max_intensities,
        drift_amplitude, drift_percentage, smooth_transition_ratio
    )


def add_peak_only_noise(intensity, rt_vals, peak_windows, peak_max_intensities,
                        peak_noise_ratio, baseline_noise_ratio, smooth_window_ratio, min_peak_intensity_ratio,
                        seed=None):
    """加噪优化（向量化+Numba）"""
    if seed is not None:
        np.random.seed(seed)
    out = intensity.copy()
    n = len(out)

    # 向量化mask生成
    mask = np.zeros(n, dtype=bool)
    for l, r in peak_windows:
        mask[l:r + 1] = True

    # 向量化噪声生成
    std = np.std(out[mask]) if np.any(mask) else 0
    noise = np.zeros(n, dtype=np.float32)
    noise[mask] = np.random.normal(0, std * peak_noise_ratio, size=np.sum(mask)).astype(np.float32)
    noise[~mask] = np.random.normal(0, std * baseline_noise_ratio, size=np.sum(~mask)).astype(np.float32)

    # 平滑窗口优化
    win = max(7, int(n * smooth_window_ratio))
    win = win if win % 2 == 1 else win + 1
    noise = savgol_filter(noise, win, 3)

    out += noise
    out = np.maximum(out, 0)

    # 向量化峰强度保护
    for i, (l, r) in enumerate(peak_windows):
        min_peak_intensity = peak_max_intensities[i] * min_peak_intensity_ratio
        core_start = l + int((r - l) * 0.2)
        core_end = r - int((r - l) * 0.2)
        if core_start < core_end:
            out[core_start:core_end] = np.maximum(out[core_start:core_end], min_peak_intensity)

    final_win = max(5, win // 2)
    final_win = final_win if final_win % 2 == 1 else final_win + 1
    out = savgol_filter(out, final_win, 2)
    return np.maximum(out, 0).astype(np.float32)


def apply_drift_and_noise(intensity, rt_vals, **kwargs):
    """数据增强主函数（流水线优化）"""
    # 峰检测
    peak_centers, peak_windows, baseline, rt_step, peak_max_intensities = detect_peak_windows(
        intensity, rt_vals,
        **{k: v for k, v in kwargs.items() if k in ['prominence_scale', 'min_distance_denom', 'min_distance_min']}
    )
    # 漂移
    drifted = add_peak_drift_only(intensity, rt_vals, peak_centers, peak_windows, baseline, rt_step,
                                  peak_max_intensities,
                                  **{k: v for k, v in kwargs.items() if
                                     k in ['drift_amplitude', 'drift_direction', 'drift_percentage',
                                           'smooth_transition_ratio']})
    # 重新检测峰
    if len(peak_windows) > 0:
        _, new_windows, _, _, new_max_intensities = detect_peak_windows(
            drifted, rt_vals,
            **{k: v for k, v in kwargs.items() if k in ['prominence_scale', 'min_distance_denom', 'min_distance_min']}
        )
        # 加噪
        noised = add_peak_only_noise(drifted, rt_vals, new_windows, new_max_intensities,
                                     **{k: v for k, v in kwargs.items() if
                                        k in ['peak_noise_ratio', 'baseline_noise_ratio', 'smooth_window_ratio',
                                              'min_peak_intensity_ratio']})
        return noised
    return drifted


def generate_training_data_continuous(intensities, num_ratios_per_pair=5):
    """训练数据生成（向量化优化）"""
    X_list, y1_list, y2_list, y_ratio_list = [], [], [], []
    n = len(intensities)
    print(f"开始生成连续比例数据，每对样本生成 {num_ratios_per_pair} 个比例...")

    # 预分配内存（减少列表append开销）
    total_pairs = n * (n + 1) // 2 * num_ratios_per_pair * 2
    X = np.zeros((total_pairs, intensities.shape[1]), dtype=np.float32)
    y1 = np.zeros(total_pairs, dtype=np.int64)
    y2 = np.zeros(total_pairs, dtype=np.int64)
    y_ratio = np.zeros(total_pairs, dtype=np.float32)

    idx = 0
    for i in range(n):
        for j in range(i, n):
            # 向量化生成比例
            ratios = np.random.uniform(0.1 if i != j else 0, 0.9 if i != j else 0.5, num_ratios_per_pair)
            for r1 in ratios:
                r2 = 1 - r1
                # 向量化叠加
                X[idx] = r1 * intensities[i] + r2 * intensities[j]
                y1[idx] = i
                y2[idx] = j
                y_ratio[idx] = r1
                idx += 1

                X[idx] = r2 * intensities[i] + r1 * intensities[j]
                y1[idx] = j
                y2[idx] = i
                y_ratio[idx] = r1  # j的实际权重为r1
                idx += 1

    # 裁剪到实际大小
    X = X[:idx]
    y1 = y1[:idx]
    y2 = y2[:idx]
    y_ratio = y_ratio[:idx].reshape(-1, 1)

    print(f"生成样本总数: {len(X)}")
    return X, y1, y2, y_ratio


# ==================== 数据集类优化（GPU适配） ====================
class OverlayDataset(Dataset):
    """数据集类（内存映射+预加载）"""

    def __init__(self, X_norm, y_class1, y_class2, y_ratio):
        # 内存映射（大数据集时用np.memmap）
        self.X = X_norm.astype(np.float32)
        self.y_class1 = torch.tensor(y_class1, dtype=torch.long)
        self.y_class2 = torch.tensor(y_class2, dtype=torch.long)
        self.y_ratio = torch.tensor(y_ratio, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # 直接返回Tensor（减少转换开销）
        x = torch.from_numpy(self.X[idx]).float().unsqueeze(0)
        return x, self.y_class1[idx], self.y_class2[idx], self.y_ratio[idx]


# ==================== 模型优化（GPU显存优化） ====================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        # 修正 float32() → float()，适配GPU
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)  # (max_len, 1, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0)]


class OverlayTransformer(nn.Module):
    def __init__(self, n_classes, input_length=1000, d_model=64, nhead=8, num_layers=4, dim_feedforward=256):
        super().__init__()
        # 卷积嵌入层（优化初始化，GPU适配）
        self.conv_embed = nn.Sequential(
            nn.Conv1d(1, d_model, kernel_size=7, stride=4, padding=3),
            nn.BatchNorm1d(d_model),  # BN层加速GPU训练
            nn.ReLU(inplace=True)  # 原地操作节省GPU显存
        )
        self.seq_len = (input_length + 3 * 2 - 7) // 4 + 1
        self.pos_encoder = PositionalEncoding(d_model)

        # Transformer编码器（GPU优化）
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=0.1,
            batch_first=False,
            activation='gelu',  # GELU比ReLU更适合GPU加速
            norm_first=True,  # 前置归一化加速GPU收敛
            device=Config.device  # 直接指定GPU设备
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 全连接层（优化初始化，GPU适配）
        self.fc_shared = nn.Linear(d_model * self.seq_len, 512)
        nn.init.xavier_uniform_(self.fc_shared.weight)

        self.class_head1 = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, n_classes)
        )
        self.class_head2 = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, n_classes)
        )
        self.ratio_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # 前向传播优化（减少GPU维度转换次数）
        x = self.conv_embed(x)  # (batch, d_model, seq_len)
        x = x.permute(2, 0, 1)  # (seq_len, batch, d_model)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)  # (seq_len, batch, d_model)
        x = x.permute(1, 0, 2).flatten(1)  # (batch, d_model*seq_len) - 合并reshape和permute
        feat = self.fc_shared(x)
        out1 = self.class_head1(feat)
        out2 = self.class_head2(feat)
        out_ratio = self.ratio_head(feat)
        return out1, out2, out_ratio


# ==================== 评估函数优化（GPU批处理） ====================
@torch.no_grad()
def evaluate_model(model, test_loader, device):
    """评估函数（GPU批处理加速）"""
    model.eval()
    total = 0
    correct1 = 0
    correct2 = 0
    correct_both = 0
    total_ratio_error = 0.0

    # 批处理评估（减少GPU-CPU传输）
    for X_batch, y1, y2, y_ratio_batch in test_loader:
        X_batch = X_batch.to(device, non_blocking=True)
        y1 = y1.to(device, non_blocking=True)
        y2 = y2.to(device, non_blocking=True)
        y_ratio_batch = y_ratio_batch.to(device, non_blocking=True)

        # 混合精度推理（新版amp API）
        if Config.use_amp and device.type == 'cuda':
            with autocast('cuda'):
                out1, out2, out_ratio = model(X_batch)
        else:
            out1, out2, out_ratio = model(X_batch)

        # 向量化计算准确率（GPU上直接计算）
        pred1 = torch.argmax(out1, dim=1)
        pred2 = torch.argmax(out2, dim=1)
        batch_size = y1.size(0)
        total += batch_size
        correct1 += (pred1 == y1).sum().item()
        correct2 += (pred2 == y2).sum().item()
        correct_both += ((pred1 == y1) & (pred2 == y2)).sum().item()
        total_ratio_error += torch.abs(out_ratio.squeeze() - y_ratio_batch.squeeze()).sum().item()

    acc1 = correct1 / total
    acc2 = correct2 / total
    acc_both = correct_both / total
    mae_ratio = total_ratio_error / total
    print("\n===== 模型评价 =====")
    print(f"测试集样本数: {total}")
    print(f"分类准确率 - 样本1: {acc1:.4f} ({correct1}/{total})")
    print(f"分类准确率 - 样本2: {acc2:.4f} ({correct2}/{total})")
    print(f"两个样本同时正确率: {acc_both:.4f} ({correct_both}/{total})")
    print(f"比例预测平均绝对误差 (MAE): {mae_ratio:.4f}")
    return acc1, acc2, acc_both, mae_ratio


# ==================== 预测函数优化（GPU推理） ====================
@torch.no_grad()
def predict_overlay_top3(overlay_intensity, model, A_names, device, apply_augment=False, **kwargs):
    """预测函数（GPU推理加速）"""
    # 数据增强（可选）
    if apply_augment:
        overlay_intensity = apply_drift_and_noise(overlay_intensity, kwargs['rt_grid'], **kwargs)

    # 标准化（向量化，CPU端完成）
    mean = overlay_intensity.mean()
    std = overlay_intensity.std() + 1e-8
    overlay_intensity = (overlay_intensity - mean) / std

    # 批量推理（即使单样本也用batch维度，GPU批处理优化）
    x = torch.tensor(overlay_intensity, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device, non_blocking=True)

    # 混合精度推理（新版amp API）
    if Config.use_amp and device.type == 'cuda':
        with autocast('cuda'):
            out1, out2, out_ratio = model(x)
    else:
        out1, out2, out_ratio = model(x)

    # 向量化获取top3（GPU上直接计算）
    prob1 = torch.softmax(out1, dim=1).squeeze()
    prob2 = torch.softmax(out2, dim=1).squeeze()
    top3_prob1, top3_idx1 = torch.topk(prob1, 3)
    top3_prob2, top3_idx2 = torch.topk(prob2, 3)
    pred_ratio = out_ratio.item()

    # 减少GPU-CPU传输（批量转换）
    top3_idx1_cpu = top3_idx1.cpu().numpy()
    top3_idx2_cpu = top3_idx2.cpu().numpy()
    top3_names1 = [A_names[idx] for idx in top3_idx1_cpu]
    top3_names2 = [A_names[idx] for idx in top3_idx2_cpu]
    top3_probs1 = top3_prob1.cpu().numpy()
    top3_probs2 = top3_prob2.cpu().numpy()

    return top3_names1, top3_probs1, top3_names2, top3_probs2, pred_ratio


# ==================== 主程序入口（GPU核心优化） ====================
def main():
    # 初始化分布式训练（单卡禁用）
    local_rank = setup_ddp() if Config.use_ddp else -1
    cfg = Config()

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    # 设置设备（强制使用RTX 5060）
    if local_rank >= 0:
        device = torch.device(f'cuda:{local_rank}')
    else:
        device = cfg.device
    print(f"使用设备: {device}")
    if device.type == 'cuda':
        print(f"GPU名称: {torch.cuda.get_device_name(0)}")
        print(f"GPU显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")

    # 创建输出目录
    os.makedirs(cfg.output_dir, exist_ok=True)

    # 读取A库数据（批量读取优化）
    rt_grid = np.linspace(cfg.rt_min, cfg.rt_max, cfg.n_rt_points, dtype=np.float32)
    A_files = [f for f in os.listdir(cfg.dataset_A_dir) if f.endswith(('.xlsx', '.xls'))]
    A_files.sort()
    n_classes = len(A_files)
    if n_classes == 0:
        raise FileNotFoundError(
            f"在数据目录未找到Excel样本文件: {cfg.dataset_A_dir}。"
            "请确认目录下包含 .xlsx 或 .xls 文件，或通过环境变量 DATASET_A_DIR 指定正确路径。"
        )
    print(f"A库共有 {n_classes} 个样本")

    # 批量读取色谱图（CPU端完成）
    A_intensities = []
    A_names = []
    for fname in A_files:
        path = os.path.join(cfg.dataset_A_dir, fname)
        rt, inten = read_chromatogram(path)
        inten_grid = interpolate_to_grid(rt, inten, rt_grid)
        A_intensities.append(inten_grid)
        A_names.append(os.path.splitext(fname)[0])
    A_intensities = np.array(A_intensities, dtype=np.float32)

    # 保存原始数据（仅主进程）
    if local_rank <= 0:
        np.save(os.path.join(cfg.output_dir, 'A_intensities.npy'), A_intensities)
        np.save(os.path.join(cfg.output_dir, 'A_names.npy'), A_names)

    # 生成/加载原始数据
    raw_data_path = os.path.join(cfg.output_dir, 'X_raw.npy')
    if os.path.exists(raw_data_path) and local_rank <= 0:
        print("加载已存在的原始叠加数据...")
        X_raw = np.load(raw_data_path)
        y_class1 = np.load(os.path.join(cfg.output_dir, 'y_class1.npy'))
        y_class2 = np.load(os.path.join(cfg.output_dir, 'y_class2.npy'))
        y_ratio = np.load(os.path.join(cfg.output_dir, 'y_ratio.npy'))
    else:
        if local_rank <= 0:
            print("生成原始连续比例叠加数据...")
            X_raw, y_class1, y_class2, y_ratio = generate_training_data_continuous(A_intensities,
                                                                                   num_ratios_per_pair=cfg.num_ratios_per_pair)
            np.save(os.path.join(cfg.output_dir, 'X_raw.npy'), X_raw)
            np.save(os.path.join(cfg.output_dir, 'y_class1.npy'), y_class1)
            np.save(os.path.join(cfg.output_dir, 'y_class2.npy'), y_class2)
            np.save(os.path.join(cfg.output_dir, 'y_ratio.npy'), y_ratio)
        # DDP同步（单卡跳过）
        if Config.use_ddp:
            torch.distributed.barrier()
            X_raw = np.load(raw_data_path)
            y_class1 = np.load(os.path.join(cfg.output_dir, 'y_class1.npy'))
            y_class2 = np.load(os.path.join(cfg.output_dir, 'y_class2.npy'))
            y_ratio = np.load(os.path.join(cfg.output_dir, 'y_ratio.npy'))

    # 划分数据集（CPU端完成）
    dataset_raw = OverlayDataset(X_raw, y_class1, y_class2, y_ratio)
    train_size = int((1 - cfg.test_ratio) * len(dataset_raw))
    test_size = len(dataset_raw) - train_size
    train_idx, test_idx = random_split(range(len(dataset_raw)), [train_size, test_size],
                                       generator=torch.Generator().manual_seed(cfg.seed))

    # 生成增强数据并标准化（CPU端完成）
    # 测试集增强
    test_aug_path = os.path.join(cfg.output_dir, 'X_test_aug_norm.npy')
    if os.path.exists(test_aug_path) and local_rank <= 0:
        print("加载已存在的测试集增强数据...")
        X_test_aug_norm = np.load(test_aug_path)
        y1_test = np.load(os.path.join(cfg.output_dir, 'y1_test.npy'))
        y2_test = np.load(os.path.join(cfg.output_dir, 'y2_test.npy'))
        y_ratio_test = np.load(os.path.join(cfg.output_dir, 'y_ratio_test.npy')).reshape(-1, 1)
    else:
        if local_rank <= 0:
            print("生成测试集增强数据并标准化...")
            X_test_aug_raw = []
            y1_test = []
            y2_test = []
            y_ratio_test = []
            np.random.seed(cfg.seed)
            for idx in test_idx:
                x = X_raw[idx]
                y1 = y_class1[idx]
                y2 = y_class2[idx]
                yr = y_ratio[idx]
                x_aug = apply_drift_and_noise(x, rt_grid,
                                              prominence_scale=cfg.prominence_scale,
                                              min_distance_denom=cfg.min_distance_denom,
                                              min_distance_min=cfg.min_distance_min,
                                              drift_amplitude=cfg.drift_amplitude,
                                              drift_direction=cfg.drift_direction,
                                              drift_percentage=cfg.drift_percentage,
                                              smooth_transition_ratio=cfg.smooth_transition_ratio,
                                              peak_noise_ratio=cfg.peak_noise_ratio,
                                              baseline_noise_ratio=cfg.baseline_noise_ratio,
                                              smooth_window_ratio=cfg.smooth_window_ratio,
                                              min_peak_intensity_ratio=cfg.min_peak_intensity_ratio)
                X_test_aug_raw.append(x_aug)
                y1_test.append(y1)
                y2_test.append(y2)
                y_ratio_test.append(yr)
            X_test_aug_raw = np.array(X_test_aug_raw, dtype=np.float32)
            # 逐样本标准化（向量化）
            mean_test = X_test_aug_raw.mean(axis=1, keepdims=True)
            std_test = X_test_aug_raw.std(axis=1, keepdims=True) + 1e-8
            X_test_aug_norm = (X_test_aug_raw - mean_test) / std_test
            y1_test = np.array(y1_test, dtype=np.int64)
            y2_test = np.array(y2_test, dtype=np.int64)
            y_ratio_test = np.array(y_ratio_test, dtype=np.float32).reshape(-1, 1)
            # 保存
            np.save(test_aug_path, X_test_aug_norm)
            np.save(os.path.join(cfg.output_dir, 'y1_test.npy'), y1_test)
            np.save(os.path.join(cfg.output_dir, 'y2_test.npy'), y2_test)
            np.save(os.path.join(cfg.output_dir, 'y_ratio_test.npy'), y_ratio_test)
        # DDP同步（单卡跳过）
        if Config.use_ddp:
            torch.distributed.barrier()
            X_test_aug_norm = np.load(test_aug_path)
            y1_test = np.load(os.path.join(cfg.output_dir, 'y1_test.npy'))
            y2_test = np.load(os.path.join(cfg.output_dir, 'y2_test.npy'))
            y_ratio_test = np.load(os.path.join(cfg.output_dir, 'y_ratio_test.npy')).reshape(-1, 1)

    # 创建测试数据集和DataLoader（GPU适配）
    test_dataset = OverlayDataset(X_test_aug_norm, y1_test, y2_test, y_ratio_test)
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        persistent_workers=cfg.persistent_workers,
        prefetch_factor=cfg.prefetch_factor
    )

    # 训练集增强
    train_aug_path = os.path.join(cfg.output_dir, 'X_train_aug_norm.npy')
    if os.path.exists(train_aug_path) and local_rank <= 0:
        print("加载已存在的训练集增强数据...")
        X_train_aug_norm = np.load(train_aug_path)
        y1_train = np.load(os.path.join(cfg.output_dir, 'y1_train.npy'))
        y2_train = np.load(os.path.join(cfg.output_dir, 'y2_train.npy'))
        y_ratio_train = np.load(os.path.join(cfg.output_dir, 'y_ratio_train.npy')).reshape(-1, 1)
    else:
        if local_rank <= 0:
            print("生成训练集增强数据并标准化...")
            X_train_aug_raw = []
            y1_train = []
            y2_train = []
            y_ratio_train = []
            for idx in train_idx.indices:
                x = X_raw[idx]
                for aug in range(cfg.num_augmentations):
                    x_aug = apply_drift_and_noise(x, rt_grid,
                                                  prominence_scale=cfg.prominence_scale,
                                                  min_distance_denom=cfg.min_distance_denom,
                                                  min_distance_min=cfg.min_distance_min,
                                                  drift_amplitude=cfg.drift_amplitude,
                                                  drift_direction=cfg.drift_direction,
                                                  drift_percentage=cfg.drift_percentage,
                                                  smooth_transition_ratio=cfg.smooth_transition_ratio,
                                                  peak_noise_ratio=cfg.peak_noise_ratio,
                                                  baseline_noise_ratio=cfg.baseline_noise_ratio,
                                                  smooth_window_ratio=cfg.smooth_window_ratio,
                                                  min_peak_intensity_ratio=cfg.min_peak_intensity_ratio)
                    X_train_aug_raw.append(x_aug)
                    y1_train.append(y_class1[idx])
                    y2_train.append(y_class2[idx])
                    y_ratio_train.append(y_ratio[idx])
            X_train_aug_raw = np.array(X_train_aug_raw, dtype=np.float32)
            # 逐样本标准化（向量化）
            mean_train = X_train_aug_raw.mean(axis=1, keepdims=True)
            std_train = X_train_aug_raw.std(axis=1, keepdims=True) + 1e-8
            X_train_aug_norm = (X_train_aug_raw - mean_train) / std_train
            y1_train = np.array(y1_train, dtype=np.int64)
            y2_train = np.array(y2_train, dtype=np.int64)
            y_ratio_train = np.array(y_ratio_train, dtype=np.float32).reshape(-1, 1)
            # 保存
            np.save(train_aug_path, X_train_aug_norm)
            np.save(os.path.join(cfg.output_dir, 'y1_train.npy'), y1_train)
            np.save(os.path.join(cfg.output_dir, 'y2_train.npy'), y2_train)
            np.save(os.path.join(cfg.output_dir, 'y_ratio_train.npy'), y_ratio_train)
        # DDP同步（单卡跳过）
        if Config.use_ddp:
            torch.distributed.barrier()
            X_train_aug_norm = np.load(train_aug_path)
            y1_train = np.load(os.path.join(cfg.output_dir, 'y1_train.npy'))
            y2_train = np.load(os.path.join(cfg.output_dir, 'y2_train.npy'))
            y_ratio_train = np.load(os.path.join(cfg.output_dir, 'y_ratio_train.npy')).reshape(-1, 1)

    # 创建训练数据集和DataLoader（GPU适配）
    train_dataset = OverlayDataset(X_train_aug_norm, y1_train, y2_train, y_ratio_train)

    # DDP采样器（单卡禁用）
    if Config.use_ddp:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        train_loader = DataLoader(
            train_dataset,
            batch_size=cfg.batch_size,
            sampler=train_sampler,
            num_workers=cfg.num_workers,
            pin_memory=cfg.pin_memory,
            persistent_workers=cfg.persistent_workers,
            prefetch_factor=cfg.prefetch_factor
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=cfg.num_workers,
            pin_memory=cfg.pin_memory,
            persistent_workers=cfg.persistent_workers,
            prefetch_factor=cfg.prefetch_factor
        )

    if local_rank <= 0:
        print(f"训练集样本数: {len(train_dataset)}, 测试集样本数: {len(test_dataset)}")

    # 初始化模型（直接部署到GPU）
    model = OverlayTransformer(
        n_classes=n_classes,
        input_length=cfg.n_rt_points,
        d_model=cfg.d_model,
        nhead=cfg.nhead,
        num_layers=cfg.num_layers,
        dim_feedforward=cfg.dim_feedforward
    ).to(device)

    # 优化器和损失函数（GPU适配）
    criterion_class = nn.CrossEntropyLoss().to(device)  # 损失函数部署到GPU
    criterion_ratio = nn.HuberLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate, betas=(0.9, 0.999), eps=1e-8)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-6)

    # 混合精度训练（新版amp API）
    scaler = GradScaler('cuda') if cfg.use_amp and device.type == 'cuda' else None

    # 训练循环（GPU核心优化）
    train_losses, test_losses = [], []
    best_test_loss = float('inf')
    start_time = time.time()

    for epoch in range(cfg.epochs):
        if Config.use_ddp:
            train_sampler.set_epoch(epoch)  # DDP打乱数据（单卡跳过）
        model.train()
        running_loss = 0.0

        # 梯度累积（GPU显存优化）
        optimizer.zero_grad()
        for batch_idx, (X_batch, y1, y2, y_ratio_batch) in enumerate(train_loader):
            # 数据异步传输到GPU（non_blocking=True）
            X_batch = X_batch.to(device, non_blocking=True)
            y1 = y1.to(device, non_blocking=True)
            y2 = y2.to(device, non_blocking=True)
            y_ratio_batch = y_ratio_batch.to(device, non_blocking=True)

            # 混合精度前向（新版amp API）
            if scaler is not None:
                with autocast('cuda'):
                    out1, out2, out_ratio = model(X_batch)
                    loss_class = criterion_class(out1, y1) + criterion_class(out2, y2)
                    loss_reg = criterion_ratio(out_ratio.squeeze(), y_ratio_batch.squeeze())
                    loss = (loss_class + cfg.lambda_reg * loss_reg) / cfg.grad_accum_steps
                scaler.scale(loss).backward()
            else:
                out1, out2, out_ratio = model(X_batch)
                loss_class = criterion_class(out1, y1) + criterion_class(out2, y2)
                loss_reg = criterion_ratio(out_ratio.squeeze(), y_ratio_batch.squeeze())
                loss = (loss_class + cfg.lambda_reg * loss_reg) / cfg.grad_accum_steps
                loss.backward()

            # 梯度累积更新
            if (batch_idx + 1) % cfg.grad_accum_steps == 0:
                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()

            running_loss += loss.item() * cfg.grad_accum_steps

        # 计算平均损失
        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # 验证（仅主进程）
        if local_rank <= 0:
            model.eval()
            test_loss = 0.0
            with torch.no_grad():
                for X_batch, y1, y2, y_ratio_batch in test_loader:
                    X_batch = X_batch.to(device, non_blocking=True)
                    y1 = y1.to(device, non_blocking=True)
                    y2 = y2.to(device, non_blocking=True)
                    y_ratio_batch = y_ratio_batch.to(device, non_blocking=True)

                    # 混合精度验证
                    if scaler is not None:
                        with autocast('cuda'):
                            out1, out2, out_ratio = model(X_batch)
                            loss_class = criterion_class(out1, y1) + criterion_class(out2, y2)
                            loss_reg = criterion_ratio(out_ratio.squeeze(), y_ratio_batch.squeeze())
                            loss = loss_class + cfg.lambda_reg * loss_reg
                    else:
                        out1, out2, out_ratio = model(X_batch)
                        loss_class = criterion_class(out1, y1) + criterion_class(out2, y2)
                        loss_reg = criterion_ratio(out_ratio.squeeze(), y_ratio_batch.squeeze())
                        loss = loss_class + cfg.lambda_reg * loss_reg

                    test_loss += loss.item()

            avg_test_loss = test_loss / len(test_loader)
            test_losses.append(avg_test_loss)

            # 学习率调度
            scheduler.step(avg_test_loss)

            # 保存最佳模型（GPU格式）
            if avg_test_loss < best_test_loss:
                best_test_loss = avg_test_loss
                # 保存DDP的原始模型（单卡直接保存）
                if Config.use_ddp:
                    torch.save(model.module.state_dict(), os.path.join(cfg.output_dir, 'best_model.pth'))
                else:
                    torch.save(model.state_dict(), os.path.join(cfg.output_dir, 'best_model.pth'))

            # 打印日志
            lr = optimizer.param_groups[0]['lr']
            epoch_time = time.time() - start_time
            print(
                f"Epoch {epoch + 1:3d}/{cfg.epochs} | Train Loss: {avg_train_loss:.4f} | Test Loss: {avg_test_loss:.4f} | LR: {lr:.2e} | Time: {epoch_time:.2f}s")

    # 加载最佳模型
    if local_rank <= 0:
        if Config.use_ddp:
            model.module.load_state_dict(torch.load(os.path.join(cfg.output_dir, 'best_model.pth')))
        else:
            model.load_state_dict(torch.load(os.path.join(cfg.output_dir, 'best_model.pth')))

        # 保存最终模型（GPU兼容格式）
        if Config.use_ddp:
            torch.save(model.module.state_dict(), os.path.join(cfg.output_dir, 'overlay_transformer.pth'))
        else:
            torch.save(model.state_dict(), os.path.join(cfg.output_dir, 'overlay_transformer.pth'))

        # 绘制损失曲线
        plt.plot(train_losses, label='Train')
        plt.plot(test_losses, label='Test')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(os.path.join(cfg.output_dir, 'loss_curve.png'))
        plt.show()

        # 模型评价（GPU加速）
        evaluate_model(model, test_loader, device)

        # 测试示例
        if len(A_files) >= 2:
            test_file1 = os.path.join(cfg.dataset_A_dir, A_files[-2])
            test_file2 = os.path.join(cfg.dataset_A_dir, A_files[-1])
            test_ratio = 0.3

            rt1, int1 = read_chromatogram(test_file1)
            rt2, int2 = read_chromatogram(test_file2)
            int1_grid = interpolate_to_grid(rt1, int1, rt_grid)
            int2_grid = interpolate_to_grid(rt2, int2, rt_grid)
            overlay = test_ratio * int1_grid + (1 - test_ratio) * int2_grid

            print(f"\n===== 测试示例 =====")
            print(f"测试叠加谱由以下两个样本生成：")
            print(f"  样本1: {os.path.basename(test_file1)}")
            print(f"  样本2: {os.path.basename(test_file2)}")
            print(f"  真实比例: {test_ratio:.2f} : {1 - test_ratio:.2f}")

            top3_names1, top3_probs1, top3_names2, top3_probs2, pred_ratio = predict_overlay_top3(
                overlay, model, A_names, device, apply_augment=False,
                rt_grid=rt_grid,
                prominence_scale=cfg.prominence_scale,
                min_distance_denom=cfg.min_distance_denom,
                min_distance_min=cfg.min_distance_min,
                drift_amplitude=cfg.drift_amplitude,
                drift_direction=cfg.drift_direction,
                drift_percentage=cfg.drift_percentage,
                smooth_transition_ratio=cfg.smooth_transition_ratio,
                peak_noise_ratio=cfg.peak_noise_ratio,
                baseline_noise_ratio=cfg.baseline_noise_ratio,
                smooth_window_ratio=cfg.smooth_window_ratio,
                min_peak_intensity_ratio=cfg.min_peak_intensity_ratio)

            print(f"\n模型预测比例: {pred_ratio:.4f} : {1 - pred_ratio:.4f}")
            print("\n第一个样本的top3预测:")
            for name, prob in zip(top3_names1, top3_probs1):
                print(f"  {name}: {prob:.4f}")
            print("\n第二个样本的top3预测:")
            for name, prob in zip(top3_names2, top3_probs2):
                print(f"  {name}: {prob:.4f}")
        else:
            print("A库样本不足两个，无法测试。")

    # 清理分布式环境（单卡跳过）
    if Config.use_ddp:
        cleanup_ddp()


if __name__ == '__main__':
    cfg = Config()
    # 用环境变量设置线程数
    os.environ["OMP_NUM_THREADS"] = str(cfg.num_threads)
    main()