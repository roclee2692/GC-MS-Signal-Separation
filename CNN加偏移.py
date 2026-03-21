import os
import json
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from scipy.interpolate import interp1d
from scipy.signal import find_peaks, savgol_filter
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ==================== 配置参数 ====================
workspace_dir = os.path.dirname(os.path.abspath(__file__))
dataset_A_dir = os.environ.get('DATASET_A_DIR', os.path.join(workspace_dir, 'GCMS_单个样本数据'))
output_dir = os.environ.get('OUTPUT_DIR', os.path.join(workspace_dir, 'outputs', 'cnn_offset'))
os.makedirs(output_dir, exist_ok=True)

# 保留时间范围及插值点数
rt_min, rt_max = 4, 50
n_rt_points = 1000

# 训练参数
test_ratio = 0.2
batch_size = 32
learning_rate = 5e-4
epochs = int(os.environ.get('EPOCHS', '100'))
lambda_reg = float(os.environ.get('LAMBDA_REG', '0.1'))
seed = int(os.environ.get('SEED', '42'))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

# ========== 增强参数（漂移幅度保持原范围，但大幅降低比例和噪声） ==========
# 峰检测
PROMINENCE_SCALE = 0.15
MIN_DISTANCE_DENOMINATOR = 500
MIN_DISTANCE_MIN = 2

# 漂移控制（漂移幅度保持原范围 0.08~0.2 分钟）
DRIFT_AMPLITUDE = (0.08, 0.2)        # 漂移幅度范围（分钟）
DRIFT_DIRECTION = "random"
DRIFT_PERCENTAGE = 0.1                # 大幅降低漂移峰比例（10%）
SMOOTH_TRANSITION_RATIO = 0.5

# 噪声控制（大幅降低噪声）
PEAK_NOISE_RATIO = 0.02                # 峰区域噪声比例（2%）
BASELINE_NOISE_RATIO = 0.005           # 基线噪声比例（0.5%）
SMOOTH_WINDOW_RATIO = 0.005
MIN_PEAK_INTENSITY_RATIO = 0.85         # 提高保护阈值（85%）

# 数据增强方式
DYNAMIC_AUGMENT = False                # 静态增强（预先生成多个副本）
NUM_AUGMENTATIONS = 3                   # 每个原始样本生成的副本数
NUM_RATIOS_PER_PAIR = 5                 # 每对样本生成的连续比例个数

# ==================== 读取A库所有样本 ====================
def read_chromatogram(file_path):
    """读取Excel色谱图，返回RT和强度"""
    df = pd.read_excel(file_path)
    df.columns = [c.strip().lower() for c in df.columns]
    rt_col = next((c for c in df.columns if 'rt' in c or '保留时间(分钟)' in c), None)
    int_col = next((c for c in df.columns if '归一化强度(10^8)' in c or 'intensity' in c), None)
    if rt_col is None or int_col is None:
        raise ValueError(f"无法识别RT或强度列: {file_path}")
    df = df[[rt_col, int_col]].dropna().sort_values(by=rt_col)
    return df[rt_col].values, df[int_col].values

def interpolate_to_grid(rt, intensity, grid):
    f = interp1d(rt, intensity, kind='linear', bounds_error=False, fill_value=0)
    return f(grid)

# 统一网格
rt_grid = np.linspace(rt_min, rt_max, n_rt_points)
rt_step = rt_grid[1] - rt_grid[0]

# 获取A库所有文件
A_files = [f for f in os.listdir(dataset_A_dir) if f.endswith(('.xlsx','.xls'))]
A_files.sort()
n_classes = len(A_files)
if n_classes == 0:
    raise FileNotFoundError(
        f"在数据目录未找到Excel样本文件: {dataset_A_dir}。"
        "请确认目录下包含 .xlsx 或 .xls 文件，或通过环境变量 DATASET_A_DIR 指定正确路径。"
    )
print(f"A库共有 {n_classes} 个样本")

# 存储强度向量和文件名
A_intensities = []
A_names = []
for fname in A_files:
    path = os.path.join(dataset_A_dir, fname)
    rt, inten = read_chromatogram(path)
    inten_grid = interpolate_to_grid(rt, inten, rt_grid)
    A_intensities.append(inten_grid)
    A_names.append(os.path.splitext(fname)[0])
A_intensities = np.array(A_intensities)   # (n_classes, n_rt_points)

# 保存原始数据
np.save(os.path.join(output_dir, 'A_intensities.npy'), A_intensities)
np.save(os.path.join(output_dir, 'A_names.npy'), A_names)

# ==================== 增强函数（保留漂移和噪声） ====================
def detect_peak_windows(intensity, rt_vals):
    """
    检测峰窗口
    输入：
        intensity: 强度数组 (n,)
        rt_vals: 保留时间数组 (n,)
    返回：
        peak_centers: 峰中心位置（保留时间）
        peak_windows: 列表，每个元素为(left_index, right_index)
        baseline: 基线强度数组（去除峰区域后的值）
        rt_step: 保留时间步长
        peak_max_intensities: 每个峰的原始最大强度
    """
    std = np.std(intensity)
    distance = max(MIN_DISTANCE_MIN, len(intensity) // MIN_DISTANCE_DENOMINATOR)
    prominence = std * PROMINENCE_SCALE

    peaks_idx, _ = find_peaks(intensity, prominence=prominence, distance=distance)
    peak_centers = rt_vals[peaks_idx]
    rt_step = np.mean(np.diff(rt_vals)) if len(rt_vals) > 1 else 0.01

    # 确定峰窗口
    peak_windows = []
    peak_max_intensities = []
    threshold = std * 1.5
    for idx in peaks_idx:
        left = idx
        while left > 0 and intensity[left] > threshold * 0.5:
            left -= 1
        right = idx
        while right < len(intensity) - 1 and intensity[right] > threshold * 0.5:
            right += 1
        peak_windows.append((left, right))
        peak_max_intensities.append(np.max(intensity[left:right+1]))

    # 基线：非峰区域强度设为0（但保留原始基线值用于后续处理）
    baseline = intensity.copy()
    for l, r in peak_windows:
        baseline[l:r+1] = 0

    return peak_centers, peak_windows, baseline, rt_step, peak_max_intensities

def add_peak_drift_only(intensity, rt_vals, peak_centers, peak_windows, baseline, rt_step, peak_max_intensities):
    """
    对部分峰进行漂移（漂移幅度保持原范围 0.08~0.2 分钟）
    输入：
        intensity: 原始强度数组
        rt_vals: 保留时间数组
        peak_centers, peak_windows, baseline, rt_step, peak_max_intensities: 峰检测结果
    返回：
        漂移后的强度数组
    """
    n = len(intensity)
    out_int = baseline.copy()

    # 随机选择部分峰进行漂移（比例由 DRIFT_PERCENTAGE 控制）
    peak_indices = list(range(len(peak_windows)))
    np.random.shuffle(peak_indices)
    keep = int(len(peak_windows) * DRIFT_PERCENTAGE)
    selected = peak_indices[:keep]

    for i in selected:
        l, r = peak_windows[i]
        peak_data = intensity[l:r+1].copy()

        # 漂移方向
        if DRIFT_DIRECTION == "left":
            dv = -np.random.uniform(*DRIFT_AMPLITUDE)
        elif DRIFT_DIRECTION == "right":
            dv = np.random.uniform(*DRIFT_AMPLITUDE)
        else:  # random
            dv = np.random.uniform(*DRIFT_AMPLITUDE) * np.random.choice([-1, 1])

        shift = int(round(dv / rt_step))
        if abs(shift) < 2:
            shift = 2 if shift >= 0 else -2

        nl, nr = l + shift, r + shift
        if nl < 0 or nr >= n:
            # 越界则不漂移，放回原处
            out_int[l:r+1] = peak_data
            continue

        # 将峰移到新位置
        out_int[nl:nr+1] = peak_data

        # 平滑过渡
        trans = max(3, int((nr - nl) * SMOOTH_TRANSITION_RATIO))
        # 左过渡
        a, b = max(0, nl - trans), nl
        if a < b:
            w = np.linspace(0, 1, b - a)
            out_int[a:b] = baseline[a:b] + peak_data[0] * w
        # 右过渡
        a, b = nr, min(n - 1, nr + trans)
        if a < b:
            w = np.linspace(1, 0, b - a)
            out_int[a:b] = baseline[a:b] + peak_data[-1] * w

    return np.maximum(out_int, 0)

def add_peak_only_noise(intensity, rt_vals, peak_windows, peak_max_intensities, seed=None):
    """
    给峰区域加噪声，并保护峰核心区域强度（噪声强度由 PEAK_NOISE_RATIO 控制）
    输入：
        intensity: 漂移后的强度数组
        rt_vals: 保留时间数组
        peak_windows: 峰窗口列表
        peak_max_intensities: 每个峰的原始最大强度
        seed: 随机种子（用于可重复性）
    返回：
        加噪后的强度数组
    """
    if seed is not None:
        np.random.seed(seed)

    out = intensity.copy()
    n = len(out)

    # 构建峰掩码
    mask = np.zeros(n, dtype=bool)
    for l, r in peak_windows:
        mask[l:r+1] = True

    std = np.std(out[mask]) if np.any(mask) else 0

    # 生成噪声
    noise = np.zeros(n)
    noise[mask] = np.random.normal(0, std * PEAK_NOISE_RATIO, size=np.sum(mask))
    noise[~mask] = np.random.normal(0, std * BASELINE_NOISE_RATIO, size=np.sum(~mask))

    # 平滑噪声
    win = max(7, int(n * SMOOTH_WINDOW_RATIO))
    if win % 2 == 0:
        win += 1
    noise = savgol_filter(noise, win, 3)

    out += noise
    out = np.maximum(out, 0)

    # 峰核心区域强度保护
    for i, (l, r) in enumerate(peak_windows):
        min_peak_intensity = peak_max_intensities[i] * MIN_PEAK_INTENSITY_RATIO
        core_start = l + int((r - l) * 0.2)
        core_end = r - int((r - l) * 0.2)
        if core_start < core_end:
            out[core_start:core_end] = np.maximum(out[core_start:core_end], min_peak_intensity)

    # 最终整体轻平滑
    final_win = max(5, win // 2)
    if final_win % 2 == 0:
        final_win += 1
    out = savgol_filter(out, final_win, 2)
    out = np.maximum(out, 0)

    return out

def apply_drift_and_noise(intensity, rt_vals, seed=None):
    """
    对输入强度向量依次应用漂移和噪声增强
    输入：
        intensity: 原始强度数组 (n_rt_points,)
        rt_vals: 保留时间数组
        seed: 随机种子（可选）
    返回：
        增强后的强度数组
    """
    # 峰检测
    peak_centers, peak_windows, baseline, rt_step, peak_max_intensities = detect_peak_windows(intensity, rt_vals)
    # 漂移
    drifted = add_peak_drift_only(intensity, rt_vals, peak_centers, peak_windows, baseline, rt_step, peak_max_intensities)
    # 加噪（重新检测峰以获得新窗口）
    _, new_windows, _, _, new_max_intensities = detect_peak_windows(drifted, rt_vals)
    if len(new_windows) == 0:
        return drifted
    noised = add_peak_only_noise(drifted, rt_vals, new_windows, new_max_intensities, seed=seed)
    return noised

# ==================== 生成训练数据（连续比例） ====================
def generate_training_data_continuous(intensities, num_ratios_per_pair=5):
    """
    生成连续比例的叠加数据，包含两种顺序
    每对样本生成 num_ratios_per_pair 个随机比例
    """
    X_list, y1_list, y2_list, y_ratio_list = [], [], [], []
    n = len(intensities)
    total_pairs = n * (n + 1) // 2  # 包括相同样本对
    print(f"开始生成连续比例数据，每对样本生成 {num_ratios_per_pair} 个比例...")
    for i in range(n):
        for j in range(i, n):
            for _ in range(num_ratios_per_pair):
                if i == j:
                    # 相同样本，比例在 [0, 0.5] 内随机（确保唯一性）
                    r1 = np.random.uniform(0, 0.5)
                else:
                    # 不同样本，比例在 [0.1, 0.9] 内随机
                    r1 = np.random.uniform(0.1, 0.9)
                r2 = 1 - r1
                # 顺序1: (i, j) 比例 r1
                overlay = r1 * intensities[i] + r2 * intensities[j]
                X_list.append(overlay)
                y1_list.append(i)
                y2_list.append(j)
                y_ratio_list.append(r1)
                # 顺序2: (j, i) — j为sample1，其在overlay2中的实际权重为r1
                overlay2 = r2 * intensities[i] + r1 * intensities[j]
                X_list.append(overlay2)
                y1_list.append(j)
                y2_list.append(i)
                y_ratio_list.append(r1)
    X = np.array(X_list)
    y_class1 = np.array(y1_list)
    y_class2 = np.array(y2_list)
    y_ratio = np.array(y_ratio_list).reshape(-1, 1)
    print(f"生成样本总数: {len(X)}")
    return X, y_class1, y_class2, y_ratio

# 生成原始叠加谱（未增强）
print("生成原始连续比例叠加数据...")
X_raw, y_class1, y_class2, y_ratio = generate_training_data_continuous(A_intensities, num_ratios_per_pair=NUM_RATIOS_PER_PAIR)

# 保存原始数据
np.save(os.path.join(output_dir, 'X_raw.npy'), X_raw)
np.save(os.path.join(output_dir, 'y_class1.npy'), y_class1)
np.save(os.path.join(output_dir, 'y_class2.npy'), y_class2)
np.save(os.path.join(output_dir, 'y_ratio.npy'), y_ratio)

# ==================== 数据标准化 ====================
# 计算训练集的均值和标准差（用于后续标准化）
# 注意：这里使用原始叠加谱计算统计量，增强后的数据会应用相同的标准化参数
mean = X_raw.mean(axis=1, keepdims=True)  # 每个样本的均值
std = X_raw.std(axis=1, keepdims=True) + 1e-8
X_raw_norm = (X_raw - mean) / std

# 存储标准化参数以便测试时使用
np.save(os.path.join(output_dir, 'mean.npy'), mean)
np.save(os.path.join(output_dir, 'std.npy'), std)

# ==================== 数据集定义（支持标准化和增强） ====================
class OverlayDataset(Dataset):
    def __init__(self, X, y_class1, y_class2, y_ratio, rt_grid=None, transform=None, mean=None, std=None):
        """
        X: 原始强度数组 (N, n_rt_points) - 未标准化
        y_class1, y_class2, y_ratio: 标签
        rt_grid: 保留时间网格
        transform: 增强函数
        mean, std: 标准化参数（如果提供，则应用标准化）
        """
        self.X = X  # 保持numpy格式，以便增强时处理
        self.y_class1 = torch.tensor(y_class1, dtype=torch.long)
        self.y_class2 = torch.tensor(y_class2, dtype=torch.long)
        self.y_ratio = torch.tensor(y_ratio, dtype=torch.float32)
        self.rt_grid = rt_grid
        self.transform = transform
        self.mean = mean
        self.std = std

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx].copy()  # 防止修改原数据
        if self.transform is not None:
            x = self.transform(x, self.rt_grid)
        # 标准化
        if self.mean is not None and self.std is not None:
            x = (x - self.mean[idx]) / self.std[idx]
        x = torch.tensor(x, dtype=torch.float32).unsqueeze(0)  # (1, n_rt_points)
        return x, self.y_class1[idx], self.y_class2[idx], self.y_ratio[idx]

# 划分训练集和测试集索引
dataset_raw = OverlayDataset(X_raw, y_class1, y_class2, y_ratio)  # 仅用于划分
train_size = int((1 - test_ratio) * len(dataset_raw))
test_size = len(dataset_raw) - train_size
split_generator = torch.Generator().manual_seed(seed)
train_idx, test_idx = random_split(range(len(dataset_raw)), [train_size, test_size], generator=split_generator)

# 构建测试集（预先增强，固定）
print("生成测试集增强数据...")
X_test_aug = []
y1_test = []
y2_test = []
y_ratio_test = []
mean_test = []
std_test = []
np.random.seed(42)  # 固定种子确保测试集可重复
for idx in test_idx:
    x = X_raw[idx]
    y1 = y_class1[idx]
    y2 = y_class2[idx]
    yr = y_ratio[idx]
    x_aug = apply_drift_and_noise(x, rt_grid)
    X_test_aug.append(x_aug)
    y1_test.append(y1)
    y2_test.append(y2)
    y_ratio_test.append(yr)
    mean_test.append(mean[idx])
    std_test.append(std[idx])
X_test_aug = np.array(X_test_aug)
y1_test = np.array(y1_test)
y2_test = np.array(y2_test)
y_ratio_test = np.array(y_ratio_test).reshape(-1, 1)
mean_test = np.array(mean_test)
std_test = np.array(std_test)

# 测试数据集（无变换，但应用标准化）
test_dataset = OverlayDataset(X_test_aug, y1_test, y2_test, y_ratio_test, transform=None, mean=mean_test, std=std_test)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 构建训练数据集（静态增强）
print("生成训练集静态增强数据...")
X_train_aug = []
y1_train = []
y2_train = []
y_ratio_train = []
mean_train = []
std_train = []
for idx in train_idx.indices:
    x = X_raw[idx]
    for aug in range(NUM_AUGMENTATIONS):
        x_aug = apply_drift_and_noise(x, rt_grid)
        X_train_aug.append(x_aug)
        y1_train.append(y_class1[idx])
        y2_train.append(y_class2[idx])
        y_ratio_train.append(y_ratio[idx])
        mean_train.append(mean[idx])
        std_train.append(std[idx])
X_train_aug = np.array(X_train_aug)
y1_train = np.array(y1_train)
y2_train = np.array(y2_train)
y_ratio_train = np.array(y_ratio_train).reshape(-1, 1)
mean_train = np.array(mean_train)
std_train = np.array(std_train)

train_dataset = OverlayDataset(X_train_aug, y1_train, y2_train, y_ratio_train, transform=None, mean=mean_train, std=std_train)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
print(f"训练集样本数: {len(train_dataset)}, 测试集样本数: {len(test_dataset)}")

# ==================== 定义模型（增强版，含残差块） ====================
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

class OverlayCNN(nn.Module):
    def __init__(self, n_classes, input_length=1000):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            ResidualBlock(32, 64, stride=2),
            ResidualBlock(64, 128, stride=2),
            nn.AdaptiveAvgPool1d(1)
        )
        self.fc_shared = nn.Linear(128, 512)
        self.class_head1 = nn.Sequential(nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, n_classes))
        self.class_head2 = nn.Sequential(nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, n_classes))
        self.ratio_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        feat = self.conv_layers(x).squeeze(-1)
        feat = self.fc_shared(feat)
        out1 = self.class_head1(feat)
        out2 = self.class_head2(feat)
        out_ratio = self.ratio_head(feat)
        return out1, out2, out_ratio

model = OverlayCNN(n_classes=n_classes).to(device)

# ==================== 损失与优化器 ====================
criterion_class = nn.CrossEntropyLoss()
criterion_ratio = nn.HuberLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
# 移除了 verbose 参数，因为新版 PyTorch 可能不支持
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

# ==================== 训练循环 ====================
train_losses, test_losses = [], []
best_test_loss = float('inf')
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for X_batch, y1, y2, y_ratio_batch in train_loader:
        X_batch = X_batch.to(device)
        y1 = y1.to(device)
        y2 = y2.to(device)
        y_ratio_batch = y_ratio_batch.to(device)

        optimizer.zero_grad()
        out1, out2, out_ratio = model(X_batch)
        loss_class = criterion_class(out1, y1) + criterion_class(out2, y2)
        loss_reg = criterion_ratio(out_ratio.squeeze(), y_ratio_batch.squeeze())
        loss = loss_class + lambda_reg * loss_reg
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    avg_train_loss = running_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    # 测试
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for X_batch, y1, y2, y_ratio_batch in test_loader:
            X_batch = X_batch.to(device)
            y1 = y1.to(device)
            y2 = y2.to(device)
            y_ratio_batch = y_ratio_batch.to(device)
            out1, out2, out_ratio = model(X_batch)
            loss_class = criterion_class(out1, y1) + criterion_class(out2, y2)
            loss_reg = criterion_ratio(out_ratio.squeeze(), y_ratio_batch.squeeze())
            loss = loss_class + lambda_reg * loss_reg
            test_loss += loss.item()
    avg_test_loss = test_loss / len(test_loader)
    test_losses.append(avg_test_loss)

    # 学习率调度
    scheduler.step(avg_test_loss)

    # 保存最佳模型
    if avg_test_loss < best_test_loss:
        best_test_loss = avg_test_loss
        torch.save(model.state_dict(), os.path.join(output_dir, 'best_model.pth'))

    print(f"Epoch {epoch+1:3d}/{epochs} | Train Loss: {avg_train_loss:.4f} | Test Loss: {avg_test_loss:.4f} | LR: {optimizer.param_groups[0]['lr']:.2e}")

# 加载最佳模型
model.load_state_dict(torch.load(os.path.join(output_dir, 'best_model.pth')))
print("最佳模型已加载。")

# 保存最终模型
torch.save(model.state_dict(), os.path.join(output_dir, 'overlay_cnn_fullA.pth'))

# 绘制损失曲线
plt.plot(train_losses, label='Train')
plt.plot(test_losses, label='Test')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig(os.path.join(output_dir, 'loss_curve.png'))
plt.show()

# ==================== 模型评价 ====================
def evaluate_model(model, test_loader, device):
    model.eval()
    total = 0
    correct1 = 0
    correct2 = 0
    correct_both = 0
    total_ratio_error = 0.0
    with torch.no_grad():
        for X_batch, y1, y2, y_ratio_batch in test_loader:
            X_batch = X_batch.to(device)
            y1 = y1.to(device)
            y2 = y2.to(device)
            y_ratio_batch = y_ratio_batch.to(device)
            out1, out2, out_ratio = model(X_batch)
            pred1 = torch.argmax(out1, dim=1)
            pred2 = torch.argmax(out2, dim=1)
            total += y1.size(0)
            correct1 += (pred1 == y1).sum().item()
            correct2 += (pred2 == y2).sum().item()
            correct_both += ((pred1 == y1) & (pred2 == y2)).sum().item()
            ratio_error = torch.abs(out_ratio.squeeze() - y_ratio_batch.squeeze())
            total_ratio_error += ratio_error.sum().item()
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

acc1, acc2, acc_both, mae_ratio = evaluate_model(model, test_loader, device)

run_summary = {
    'dataset_A_dir': dataset_A_dir,
    'output_dir': output_dir,
    'n_classes': int(n_classes),
    'n_samples_total': int(len(X_raw)),
    'train_size': int(train_size),
    'test_size': int(test_size),
    'epochs': int(epochs),
    'batch_size': int(batch_size),
    'learning_rate': float(learning_rate),
    'lambda_reg': float(lambda_reg),
    'seed': int(seed),
    'metrics': {
        'acc_sample1': float(acc1),
        'acc_sample2': float(acc2),
        'acc_both': float(acc_both),
        'mae_ratio': float(mae_ratio)
    }
}
with open(os.path.join(output_dir, 'run_summary.json'), 'w', encoding='utf-8') as f:
    json.dump(run_summary, f, ensure_ascii=False, indent=2)

# ==================== 预测函数 ====================
def predict_overlay_top3(overlay_intensity, model, A_names, device, mean_val=None, std_val=None, apply_augment=False):
    """
    输入叠加谱向量（未增强），可选择是否先应用增强（测试时一般设为False）
    预测两个样本的top3候选及比例
    """
    if apply_augment:
        overlay_intensity = apply_drift_and_noise(overlay_intensity, rt_grid)
    if mean_val is not None and std_val is not None:
        overlay_intensity = (overlay_intensity - mean_val) / std_val
    model.eval()
    with torch.no_grad():
        x = torch.tensor(overlay_intensity, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
        out1, out2, out_ratio = model(x)
        prob1 = torch.softmax(out1, dim=1).squeeze()
        prob2 = torch.softmax(out2, dim=1).squeeze()
        top3_prob1, top3_idx1 = torch.topk(prob1, 3)
        top3_prob2, top3_idx2 = torch.topk(prob2, 3)
        pred_ratio = out_ratio.item()
    top3_names1 = [A_names[idx] for idx in top3_idx1.cpu().numpy()]
    top3_names2 = [A_names[idx] for idx in top3_idx2.cpu().numpy()]
    top3_probs1 = top3_prob1.cpu().numpy()
    top3_probs2 = top3_prob2.cpu().numpy()
    return top3_names1, top3_probs1, top3_names2, top3_probs2, pred_ratio

# ==================== 测试示例 ====================
if len(A_files) >= 2:
    test_file1 = os.path.join(dataset_A_dir, A_files[-2])
    test_file2 = os.path.join(dataset_A_dir, A_files[-1])
    test_ratio = 0.3

    # 读取并插值
    rt1, int1 = read_chromatogram(test_file1)
    rt2, int2 = read_chromatogram(test_file2)
    int1_grid = interpolate_to_grid(rt1, int1, rt_grid)
    int2_grid = interpolate_to_grid(rt2, int2, rt_grid)
    overlay = test_ratio * int1_grid + (1 - test_ratio) * int2_grid

    # 获取该样本对应的标准化参数（使用输入谱自身的均值和标准差）
    mean_pred = overlay.mean()
    std_pred = overlay.std() + 1e-8

    print(f"\n===== 测试示例 =====")
    print(f"测试叠加谱由以下两个样本生成：")
    print(f"  样本1: {os.path.basename(test_file1)}")
    print(f"  样本2: {os.path.basename(test_file2)}")
    print(f"  真实比例: {test_ratio:.2f} : {1-test_ratio:.2f}")

    # 预测（不应用额外增强，但应用标准化）
    top3_names1, top3_probs1, top3_names2, top3_probs2, pred_ratio = predict_overlay_top3(
        overlay, model, A_names, device, mean_val=mean_pred, std_val=std_pred, apply_augment=False)

    print(f"\n模型预测比例: {pred_ratio:.4f} : {1-pred_ratio:.4f}")
    print("\n第一个样本的top3预测:")
    for name, prob in zip(top3_names1, top3_probs1):
        print(f"  {name}: {prob:.4f}")
    print("\n第二个样本的top3预测:")
    for name, prob in zip(top3_names2, top3_probs2):
        print(f"  {name}: {prob:.4f}")
else:
    print("A库样本不足两个，无法测试。")