"""
CNN 训练脚本 — B组数据版
基于 CNN.py，改动：
  1. 默认数据源 → 单个色谱/单个色谱图-B/ (87个样本)
  2. 固定10种比例（非随机）
  3. 跳过 i==j（同样本自叠加无意义）
  4. plt.show() → plt.close()
"""
import os
import json
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from scipy.interpolate import interp1d
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ==================== 配置参数 ====================
workspace_dir = os.path.dirname(os.path.abspath(__file__))
dataset_A_dir = os.environ.get(
    'DATASET_A_DIR',
    os.path.join(workspace_dir, '单个色谱', '单个色谱图-B')
)
output_dir = os.environ.get(
    'OUTPUT_DIR',
    os.path.join(workspace_dir, 'outputs', 'cnn_B_baseline')
)
os.makedirs(output_dir, exist_ok=True)

# 固定比例（10种）
FIXED_RATIOS = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

# RT漂移增强参数
drift_aug_enabled = os.environ.get('DRIFT_AUG', '0') == '1'
DRIFT_MAX_POINTS = int(os.environ.get('DRIFT_MAX_POINTS', '5'))

# 噪声增强参数
noise_aug_enabled = os.environ.get('NOISE_AUG', '0') == '1'
NOISE_LEVEL = float(os.environ.get('NOISE_LEVEL', '0.02'))

# 基线漂移增强
baseline_wander_enabled = os.environ.get('BASELINE_WANDER_AUG', '0') == '1'
BASELINE_WANDER_AMP = float(os.environ.get('BASELINE_WANDER_AMP', '0.05'))

# 强度缩放增强
intensity_scale_enabled = os.environ.get('INTENSITY_SCALE_AUG', '0') == '1'
INTENSITY_SCALE_RANGE = (
    float(os.environ.get('INTENSITY_SCALE_MIN', '0.5')),
    float(os.environ.get('INTENSITY_SCALE_MAX', '2.0')),
)

# 峰展宽增强
peak_broaden_enabled = os.environ.get('PEAK_BROADEN_AUG', '0') == '1'
PEAK_BROADEN_MAX_SIGMA = float(os.environ.get('PEAK_BROADEN_MAX_SIGMA', '3.0'))

# 保留时间范围及插值点数
rt_min, rt_max = 4, 50
n_rt_points = 1000

# 训练参数
test_ratio = 0.2
batch_size = 64  # B组数据量大，增大batch
learning_rate = 5e-4
epochs = int(os.environ.get('EPOCHS', '100'))
lambda_reg = float(os.environ.get('LAMBDA_REG', '5.0'))
seed = int(os.environ.get('SEED', '42'))
early_stopping_patience = int(os.environ.get('EARLY_STOPPING_PATIENCE', '15'))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

print(f"设备: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# ==================== 读取样本 ====================
def read_chromatogram(file_path):
    df = pd.read_excel(file_path)
    df.columns = [c.strip().lower() for c in df.columns]
    rt_col = next((c for c in df.columns if 'rt' in c or '保留时间' in c), None)
    int_col = next((c for c in df.columns if '归一化强度' in c or 'intensity' in c), None)
    if rt_col is None or int_col is None:
        raise ValueError(f"无法识别RT或强度列: {file_path}, 列名: {list(df.columns)}")
    df = df[[rt_col, int_col]].dropna().sort_values(by=rt_col)
    return df[rt_col].values, df[int_col].values

def interpolate_to_grid(rt, intensity, grid):
    f = interp1d(rt, intensity, kind='linear', bounds_error=False, fill_value=0)
    return f(grid)

rt_grid = np.linspace(rt_min, rt_max, n_rt_points)

# 获取所有样本文件
A_files = [f for f in os.listdir(dataset_A_dir) if f.endswith(('.xlsx', '.xls'))]
A_files.sort()
n_classes = len(A_files)
if n_classes == 0:
    raise FileNotFoundError(f"数据目录为空: {dataset_A_dir}")
print(f"样本库共 {n_classes} 个样本")
print(f"数据目录: {dataset_A_dir}")
print(f"输出目录: {output_dir}")

# 读取所有样本
A_intensities = []
A_names = []
for idx, fname in enumerate(A_files):
    path = os.path.join(dataset_A_dir, fname)
    try:
        rt, inten = read_chromatogram(path)
        inten_grid = interpolate_to_grid(rt, inten, rt_grid)
        A_intensities.append(inten_grid)
        A_names.append(os.path.splitext(fname)[0])
    except Exception as e:
        print(f"  跳过文件 {fname}: {e}")
A_intensities = np.array(A_intensities)
n_classes = len(A_intensities)
print(f"成功加载 {n_classes} 个样本")

np.save(os.path.join(output_dir, 'A_intensities.npy'), A_intensities)
np.save(os.path.join(output_dir, 'A_names.npy'), A_names)

# ==================== 增强函数 ====================
def apply_rt_drift(intensity, max_shift_points=5):
    """RT方向整体平移"""
    shift = np.random.uniform(-max_shift_points, max_shift_points)
    n = len(intensity)
    x_orig = np.arange(n, dtype=np.float64)
    x_shifted = x_orig - shift
    return np.interp(x_shifted, x_orig, intensity, left=0.0, right=0.0)

def apply_noise(intensity, noise_level=0.02):
    """加性高斯噪声"""
    sigma = noise_level * np.std(intensity)
    return np.maximum(intensity + np.random.normal(0, sigma, len(intensity)), 0.0)

def apply_baseline_wander(intensity, amplitude=0.05):
    """低频基线漂移：用1~3个随机正弦波模拟"""
    n = len(intensity)
    x = np.linspace(0, 1, n)
    wander = np.zeros(n)
    n_waves = np.random.randint(1, 4)
    for _ in range(n_waves):
        freq = np.random.uniform(0.5, 3.0)
        phase = np.random.uniform(0, 2 * np.pi)
        wander += np.sin(2 * np.pi * freq * x + phase)
    wander = wander / n_waves  # normalize to [-1, 1]
    scale = amplitude * np.max(np.abs(intensity))
    return np.maximum(intensity + scale * wander, 0.0)

def apply_intensity_scale(intensity, scale_range=(0.5, 2.0)):
    """整体强度随机缩放"""
    factor = np.random.uniform(*scale_range)
    return intensity * factor

def apply_peak_broadening(intensity, max_sigma=3.0):
    """高斯卷积峰展宽"""
    sigma = np.random.uniform(0.5, max_sigma)
    kernel_size = int(6 * sigma) | 1  # 确保奇数
    x = np.arange(kernel_size) - kernel_size // 2
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    kernel = kernel / kernel.sum()
    return np.convolve(intensity, kernel, mode='same')

# ==================== 生成训练数据（固定比例） ====================
def generate_training_data_fixed(intensities, ratios):
    X_list, y1_list, y2_list, y_ratio_list = [], [], [], []
    n = len(intensities)
    total_pairs = n * (n - 1) // 2
    print(f"  组合数: C({n},2) = {total_pairs} 对")
    print(f"  比例数: {len(ratios)} 种")
    print(f"  预计样本: {total_pairs * len(ratios) * 2} 个")
    for i in range(n):
        for j in range(i + 1, n):  # 跳过 i==j
            for r1 in ratios:
                r2 = 1 - r1
                for order, (ri, rj, label_i, label_j) in enumerate([
                    (r1, r2, i, j),  # 顺序1
                    (r2, r1, j, i),  # 顺序2
                ]):
                    overlay = ri * intensities[i] + rj * intensities[j]
                    if drift_aug_enabled:
                        overlay = apply_rt_drift(overlay, DRIFT_MAX_POINTS)
                    if noise_aug_enabled:
                        overlay = apply_noise(overlay, NOISE_LEVEL)
                    if baseline_wander_enabled:
                        overlay = apply_baseline_wander(overlay, BASELINE_WANDER_AMP)
                    if intensity_scale_enabled:
                        overlay = apply_intensity_scale(overlay, INTENSITY_SCALE_RANGE)
                    if peak_broaden_enabled:
                        overlay = apply_peak_broadening(overlay, PEAK_BROADEN_MAX_SIGMA)
                    X_list.append(overlay)
                    y1_list.append(label_i)
                    y2_list.append(label_j)
                    y_ratio_list.append(r1)
    X = np.array(X_list, dtype=np.float32)
    y_class1 = np.array(y1_list)
    y_class2 = np.array(y2_list)
    y_ratio = np.array(y_ratio_list, dtype=np.float32).reshape(-1, 1)
    return X, y_class1, y_class2, y_ratio

aug_status = [
    ('RT漂移', drift_aug_enabled, f'max_shift={DRIFT_MAX_POINTS}点, ~{DRIFT_MAX_POINTS * (rt_max-rt_min)/n_rt_points:.2f}min'),
    ('噪声', noise_aug_enabled, f'level={NOISE_LEVEL}'),
    ('基线漂移', baseline_wander_enabled, f'amplitude={BASELINE_WANDER_AMP}'),
    ('强度缩放', intensity_scale_enabled, f'range={INTENSITY_SCALE_RANGE}'),
    ('峰展宽', peak_broaden_enabled, f'max_sigma={PEAK_BROADEN_MAX_SIGMA}'),
]
for name, enabled, params in aug_status:
    print(f"{name}增强: {'开启 (' + params + ')' if enabled else '关闭'}")

print("生成训练数据（固定比例）...")
X, y_class1, y_class2, y_ratio = generate_training_data_fixed(A_intensities, FIXED_RATIOS)
print(f"实际生成样本数: {len(X)}")

np.save(os.path.join(output_dir, 'X.npy'), X)
np.save(os.path.join(output_dir, 'y_class1.npy'), y_class1)
np.save(os.path.join(output_dir, 'y_class2.npy'), y_class2)
np.save(os.path.join(output_dir, 'y_ratio.npy'), y_ratio)

# ==================== 数据集 ====================
class OverlayDataset(Dataset):
    def __init__(self, X, y_class1, y_class2, y_ratio):
        self.X = torch.tensor(X, dtype=torch.float32).unsqueeze(1)
        self.y_class1 = torch.tensor(y_class1, dtype=torch.long)
        self.y_class2 = torch.tensor(y_class2, dtype=torch.long)
        self.y_ratio = torch.tensor(y_ratio, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y_class1[idx], self.y_class2[idx], self.y_ratio[idx]

dataset = OverlayDataset(X, y_class1, y_class2, y_ratio)
train_size = int((1 - test_ratio) * len(dataset))
test_size = len(dataset) - train_size
split_generator = torch.Generator().manual_seed(seed)
train_dataset, test_dataset = random_split(dataset, [train_size, test_size], generator=split_generator)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=0)

print(f"训练集: {train_size}, 测试集: {test_size}")

# ==================== 模型 ====================
class OverlayCNN(nn.Module):
    def __init__(self, n_classes, input_length=1000):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128), nn.ReLU(), nn.AdaptiveAvgPool1d(1),
        )
        self.fc_shared = nn.Linear(128, 256)
        self.class_head1 = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, n_classes))
        self.class_head2 = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, n_classes))
        self.ratio_head = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 1), nn.Sigmoid(),
        )

    def forward(self, x):
        feat = self.conv_layers(x).squeeze(-1)
        feat = self.fc_shared(feat)
        return self.class_head1(feat), self.class_head2(feat), self.ratio_head(feat)

model = OverlayCNN(n_classes=n_classes).to(device)
total_params = sum(p.numel() for p in model.parameters())
print(f"模型参数量: {total_params:,}")

# ==================== 训练 ====================
criterion_class = nn.CrossEntropyLoss()
criterion_ratio = nn.HuberLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

train_losses, test_losses = [], []
best_test_loss = float('inf')
best_epoch = 0
no_improve_epochs = 0

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
        ratio_pred = out_ratio.squeeze()
        ce1_y1 = F.cross_entropy(out1, y1, reduction='none')
        ce2_y2 = F.cross_entropy(out2, y2, reduction='none')
        ce1_y2 = F.cross_entropy(out1, y2, reduction='none')
        ce2_y1 = F.cross_entropy(out2, y1, reduction='none')
        ratio_loss_orig = F.huber_loss(ratio_pred, y_ratio_batch, reduction='none')
        ratio_loss_swap = F.huber_loss(ratio_pred, 1.0 - y_ratio_batch, reduction='none')
        loss_order1 = ce1_y1 + ce2_y2 + lambda_reg * ratio_loss_orig
        loss_order2 = ce1_y2 + ce2_y1 + lambda_reg * ratio_loss_swap
        loss = torch.min(loss_order1, loss_order2).mean()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    avg_train_loss = running_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for X_batch, y1, y2, y_ratio_batch in test_loader:
            X_batch = X_batch.to(device)
            y1 = y1.to(device)
            y2 = y2.to(device)
            y_ratio_batch = y_ratio_batch.to(device)
            out1, out2, out_ratio = model(X_batch)
            ratio_pred = out_ratio.squeeze()
            ce1_y1 = F.cross_entropy(out1, y1, reduction='none')
            ce2_y2 = F.cross_entropy(out2, y2, reduction='none')
            ce1_y2 = F.cross_entropy(out1, y2, reduction='none')
            ce2_y1 = F.cross_entropy(out2, y1, reduction='none')
            ratio_loss_orig = F.huber_loss(ratio_pred, y_ratio_batch, reduction='none')
            ratio_loss_swap = F.huber_loss(ratio_pred, 1.0 - y_ratio_batch, reduction='none')
            loss_order1 = ce1_y1 + ce2_y2 + lambda_reg * ratio_loss_orig
            loss_order2 = ce1_y2 + ce2_y1 + lambda_reg * ratio_loss_swap
            loss = torch.min(loss_order1, loss_order2).mean()
            test_loss += loss.item()
    avg_test_loss = test_loss / len(test_loader)
    test_losses.append(avg_test_loss)

    if avg_test_loss < best_test_loss:
        best_test_loss = avg_test_loss
        best_epoch = epoch + 1
        no_improve_epochs = 0
        torch.save(model.state_dict(), os.path.join(output_dir, 'best_model.pth'))
    else:
        no_improve_epochs += 1

    print(f"Epoch {epoch+1:3d}/{epochs} | Train Loss: {avg_train_loss:.4f} | Test Loss: {avg_test_loss:.4f}")

    if no_improve_epochs >= early_stopping_patience:
        print(f"Early stopping: 连续 {early_stopping_patience} 个epoch未改善")
        break

if os.path.exists(os.path.join(output_dir, 'best_model.pth')):
    model.load_state_dict(torch.load(os.path.join(output_dir, 'best_model.pth'), map_location=device, weights_only=True))
    print(f"已加载最佳模型（Epoch {best_epoch}, Test Loss: {best_test_loss:.4f}）")

torch.save(model.state_dict(), os.path.join(output_dir, 'overlay_cnn_fullA.pth'))

# 损失曲线
plt.figure(figsize=(10, 4))
plt.plot(train_losses, label='Train')
plt.plot(test_losses, label='Test')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig(os.path.join(output_dir, 'loss_curve.png'))
plt.close()

# ==================== 评价 ====================
def evaluate_model(model, test_loader, device):
    model.eval()
    total = correct1 = correct2 = correct_both = correct_either = 0
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
            strict_match = (pred1 == y1) & (pred2 == y2)
            swapped_match = (pred1 == y2) & (pred2 == y1)
            correct_both += strict_match.sum().item()
            correct_either += (strict_match | swapped_match).sum().item()
            ratio_error = torch.abs(out_ratio.squeeze() - y_ratio_batch.squeeze())
            total_ratio_error += ratio_error.sum().item()
    acc1 = correct1 / total
    acc2 = correct2 / total
    acc_both = correct_both / total
    acc_either = correct_either / total
    mae_ratio = total_ratio_error / total
    print(f"\n===== 模型评价 =====")
    print(f"测试集样本数: {total}")
    print(f"分类准确率 - 样本1: {acc1:.4f} ({correct1}/{total})")
    print(f"分类准确率 - 样本2: {acc2:.4f} ({correct2}/{total})")
    print(f"两个样本同时正确率 (strict): {acc_both:.4f} ({correct_both}/{total})")
    print(f"两个样本同时正确率 (either-order): {acc_either:.4f} ({correct_either}/{total})")
    print(f"比例预测MAE: {mae_ratio:.4f}")
    return acc1, acc2, acc_both, acc_either, mae_ratio

acc1, acc2, acc_both, acc_either, mae_ratio = evaluate_model(model, test_loader, device)

run_summary = {
    'dataset_dir': dataset_A_dir,
    'output_dir': output_dir,
    'n_classes': int(n_classes),
    'fixed_ratios': FIXED_RATIOS,
    'n_samples_total': int(len(X)),
    'train_size': int(train_size),
    'test_size': int(test_size),
    'epochs': int(epochs),
    'early_stopping_patience': int(early_stopping_patience),
    'best_epoch': int(best_epoch),
    'best_test_loss': float(best_test_loss),
    'batch_size': int(batch_size),
    'learning_rate': float(learning_rate),
    'lambda_reg': float(lambda_reg),
    'seed': int(seed),
    'device': str(device),
    'drift_aug': drift_aug_enabled,
    'drift_max_points': DRIFT_MAX_POINTS if drift_aug_enabled else 0,
    'noise_aug': noise_aug_enabled,
    'noise_level': NOISE_LEVEL if noise_aug_enabled else 0,
    'baseline_wander_aug': baseline_wander_enabled,
    'baseline_wander_amp': BASELINE_WANDER_AMP if baseline_wander_enabled else 0,
    'intensity_scale_aug': intensity_scale_enabled,
    'intensity_scale_range': list(INTENSITY_SCALE_RANGE) if intensity_scale_enabled else [],
    'peak_broaden_aug': peak_broaden_enabled,
    'peak_broaden_max_sigma': PEAK_BROADEN_MAX_SIGMA if peak_broaden_enabled else 0,
    'metrics': {
        'acc_sample1': float(acc1),
        'acc_sample2': float(acc2),
        'acc_both_strict': float(acc_both),
        'acc_both_either_order': float(acc_either),
        'mae_ratio': float(mae_ratio),
    }
}
with open(os.path.join(output_dir, 'run_summary.json'), 'w', encoding='utf-8') as f:
    json.dump(run_summary, f, ensure_ascii=False, indent=2)

print("\n训练完成！")
