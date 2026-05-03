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
import matplotlib.pyplot as plt
import math
import warnings
warnings.filterwarnings('ignore')

# ==================== 配置参数 ====================
workspace_dir = os.path.dirname(os.path.abspath(__file__))
dataset_A_dir = os.environ.get(
    'DATASET_A_DIR',
    os.path.join(workspace_dir, 'GCMS_单个样本数据')
)
_default_output_tag = os.environ.get('NUM_RATIOS_PER_PAIR', '20')
output_dir = os.environ.get(
    'OUTPUT_DIR',
    os.path.join(workspace_dir, 'outputs', f'cnn_lc_r{_default_output_tag}')
)
os.makedirs(output_dir, exist_ok=True)

# 连续比例生成参数（可通过环境变量控制，用于学习曲线实验）
num_ratios_per_pair = int(os.environ.get('NUM_RATIOS_PER_PAIR', '20'))

# RT漂移增强参数（通过环境变量 DRIFT_AUG 控制开关）
drift_aug_enabled = os.environ.get('DRIFT_AUG', '0') == '1'
DRIFT_MAX_POINTS = int(os.environ.get('DRIFT_MAX_POINTS', '5'))  # 最大漂移点数

# 噪声增强参数（通过环境变量 NOISE_AUG 控制开关）
noise_aug_enabled = os.environ.get('NOISE_AUG', '0') == '1'
NOISE_LEVEL = float(os.environ.get('NOISE_LEVEL', '0.02'))  # 噪声强度（信号std的比例）

# 保留时间范围及插值点数（根据实际数据调整）
rt_min, rt_max = 4, 50
n_rt_points = 1000

# 训练参数
test_ratio = 0.2
batch_size = 32
learning_rate = 5e-4
epochs = int(os.environ.get('EPOCHS', '100'))
lambda_reg = float(os.environ.get('LAMBDA_REG', '5.0'))  # 回归损失权重
seed = int(os.environ.get('SEED', '42'))
early_stopping_patience = int(os.environ.get('EARLY_STOPPING_PATIENCE', '10'))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

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
print(f"数据目录: {dataset_A_dir}")
print(f"输出目录: {output_dir}")

# 存储强度向量和文件名（不带扩展名）
A_intensities = []
A_names = []
for fname in A_files:
    path = os.path.join(dataset_A_dir, fname)
    rt, inten = read_chromatogram(path)
    inten_grid = interpolate_to_grid(rt, inten, rt_grid)
    A_intensities.append(inten_grid)
    A_names.append(os.path.splitext(fname)[0])
A_intensities = np.array(A_intensities)   # (n_classes, n_rt_points)

# 保存（可选）
np.save(os.path.join(output_dir, 'A_intensities.npy'), A_intensities)
np.save(os.path.join(output_dir, 'A_names.npy'), A_names)

# ==================== RT漂移增强 ====================
def apply_rt_drift(intensity, max_shift_points=5):
    """对整条谱做全局RT漂移（时间轴随机平移），模拟GC-MS仪器校准漂移。
    用插值实现亚像素偏移，边界填零。"""
    shift = np.random.uniform(-max_shift_points, max_shift_points)
    n = len(intensity)
    x_orig = np.arange(n, dtype=np.float64)
    x_shifted = x_orig - shift
    return np.interp(x_shifted, x_orig, intensity, left=0.0, right=0.0)

def apply_noise(intensity, noise_level=0.02):
    """对整条谱加高斯噪声，强度为信号标准差的 noise_level 倍。"""
    sigma = noise_level * np.std(intensity)
    return np.maximum(intensity + np.random.normal(0, sigma, len(intensity)), 0.0)

# ==================== 生成训练数据（连续比例，包含两种顺序） ====================
def generate_training_data_continuous(intensities, num_ratios_per_pair=20):
    """
    生成连续比例的叠加数据，包含两种顺序：
    - 对于 i != j，每对生成 num_ratios_per_pair 个随机比例，两种顺序
    - 对于 i == j，比例在 [0, 0.5] 内随机
    """
    X_list, y1_list, y2_list, y_ratio_list = [], [], [], []
    n = len(intensities)
    for i in range(n):
        for j in range(i, n):
            for _ in range(num_ratios_per_pair):
                if i == j:
                    r1 = np.random.uniform(0, 0.5)
                else:
                    r1 = np.random.uniform(0.1, 0.9)
                r2 = 1 - r1
                # 顺序1: (i, j) 比例 r1
                overlay = r1 * intensities[i] + r2 * intensities[j]
                if drift_aug_enabled:
                    overlay = apply_rt_drift(overlay, DRIFT_MAX_POINTS)
                if noise_aug_enabled:
                    overlay = apply_noise(overlay, NOISE_LEVEL)
                X_list.append(overlay)
                y1_list.append(i)
                y2_list.append(j)
                y_ratio_list.append(r1)
                # 顺序2: (j, i) — j为sample1，其在overlay2中的实际权重为r1
                overlay2 = r2 * intensities[i] + r1 * intensities[j]
                if drift_aug_enabled:
                    overlay2 = apply_rt_drift(overlay2, DRIFT_MAX_POINTS)
                if noise_aug_enabled:
                    overlay2 = apply_noise(overlay2, NOISE_LEVEL)
                X_list.append(overlay2)
                y1_list.append(j)
                y2_list.append(i)
                y_ratio_list.append(r1)
    X = np.array(X_list)
    y_class1 = np.array(y1_list)
    y_class2 = np.array(y2_list)
    y_ratio = np.array(y_ratio_list).reshape(-1, 1)
    return X, y_class1, y_class2, y_ratio

if drift_aug_enabled:
    print(f"RT漂移增强: 开启 (max_shift={DRIFT_MAX_POINTS}点, ~{DRIFT_MAX_POINTS * (rt_max-rt_min)/n_rt_points:.2f}分钟)")
else:
    print("RT漂移增强: 关闭")
if noise_aug_enabled:
    print(f"噪声增强: 开启 (noise_level={NOISE_LEVEL})")
else:
    print("噪声增强: 关闭")
print("生成训练数据（连续比例）...")
X, y_class1, y_class2, y_ratio = generate_training_data_continuous(A_intensities, num_ratios_per_pair)
print(f"生成样本数: {len(X)}")

# 保存原始训练数据
np.save(os.path.join(output_dir, 'X.npy'), X)
np.save(os.path.join(output_dir, 'y_class1.npy'), y_class1)
np.save(os.path.join(output_dir, 'y_class2.npy'), y_class2)
np.save(os.path.join(output_dir, 'y_ratio.npy'), y_ratio)

# ==================== 数据集定义 ====================
class OverlayDataset(Dataset):
    def __init__(self, X, y_class1, y_class2, y_ratio):
        self.X = torch.tensor(X, dtype=torch.float32).unsqueeze(1)  # (N,1,1000)
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
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# ==================== 定义模型（CNN + Transformer） ====================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)  # (max_len, 1, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0)]


class OverlayTransformer(nn.Module):
    def __init__(self, n_classes, input_length=1000, d_model=64, nhead=8,
                 num_layers=4, dim_feedforward=256):
        super().__init__()
        self.conv_embed = nn.Sequential(
            nn.Conv1d(1, d_model, kernel_size=7, stride=4, padding=3),
            nn.BatchNorm1d(d_model),
            nn.ReLU(),
        )
        self.seq_len = (input_length + 3 * 2 - 7) // 4 + 1  # 250
        self.pos_encoder = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=0.1, batch_first=False, activation='gelu', norm_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.fc_shared = nn.Linear(d_model * self.seq_len, 512)
        self.class_head1 = nn.Sequential(
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, n_classes))
        self.class_head2 = nn.Sequential(
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, n_classes))
        self.ratio_head = nn.Sequential(
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 64), nn.ReLU(),
            nn.Linear(64, 1), nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.conv_embed(x)           # (batch, d_model, seq_len)
        x = x.permute(2, 0, 1)           # (seq_len, batch, d_model)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)  # (seq_len, batch, d_model)
        x = x.permute(1, 0, 2).flatten(1)  # (batch, d_model*seq_len)
        feat = self.fc_shared(x)
        return self.class_head1(feat), self.class_head2(feat), self.ratio_head(feat)

model = OverlayTransformer(n_classes=n_classes).to(device)

# ==================== 损失与优化器 ====================
criterion_class = nn.CrossEntropyLoss()
criterion_ratio = nn.HuberLoss()      # 使用Huber损失，对异常值更鲁棒
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# ==================== 训练循环 ====================
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
        # Order-invariant loss: compute both orderings, take per-sample min
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
            # Order-invariant loss (same as training)
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
        print(f"Early stopping触发: 连续 {early_stopping_patience} 个epoch测试损失未改善。")
        break

if os.path.exists(os.path.join(output_dir, 'best_model.pth')):
    model.load_state_dict(torch.load(os.path.join(output_dir, 'best_model.pth'), map_location=device, weights_only=True))
    print(f"已加载最佳模型（Epoch {best_epoch}, Test Loss: {best_test_loss:.4f}）")

# 保存模型
torch.save(model.state_dict(), os.path.join(output_dir, 'overlay_cnn_fullA.pth'))
print("模型训练完成并已保存。")

# 绘制损失曲线
plt.plot(train_losses, label='Train')
plt.plot(test_losses, label='Test')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig(os.path.join(output_dir, 'loss_curve.png'))
plt.close()

# ==================== 模型评价 ====================
def evaluate_model(model, test_loader, device):
    model.eval()
    total = 0
    correct1 = 0
    correct2 = 0
    correct_both = 0
    correct_either = 0
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
            # 比例绝对误差
            ratio_error = torch.abs(out_ratio.squeeze() - y_ratio_batch.squeeze())
            total_ratio_error += ratio_error.sum().item()
    acc1 = correct1 / total
    acc2 = correct2 / total
    acc_both = correct_both / total
    acc_either = correct_either / total
    mae_ratio = total_ratio_error / total
    print("\n===== 模型评价 =====")
    print(f"测试集样本数: {total}")
    print(f"分类准确率 - 样本1: {acc1:.4f} ({correct1}/{total})")
    print(f"分类准确率 - 样本2: {acc2:.4f} ({correct2}/{total})")
    print(f"两个样本同时正确率 (strict): {acc_both:.4f} ({correct_both}/{total})")
    print(f"两个样本同时正确率 (either-order): {acc_either:.4f} ({correct_either}/{total})")
    print(f"比例预测平均绝对误差 (MAE): {mae_ratio:.4f}")
    return acc1, acc2, acc_both, acc_either, mae_ratio

acc1, acc2, acc_both, acc_either, mae_ratio = evaluate_model(model, test_loader, device)

run_summary = {
    'dataset_A_dir': dataset_A_dir,
    'output_dir': output_dir,
    'n_classes': int(n_classes),
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
    'drift_aug': drift_aug_enabled,
    'drift_max_points': DRIFT_MAX_POINTS if drift_aug_enabled else 0,
    'noise_aug': noise_aug_enabled,
    'noise_level': NOISE_LEVEL if noise_aug_enabled else 0,
    'metrics': {
        'acc_sample1': float(acc1),
        'acc_sample2': float(acc2),
        'acc_both_strict': float(acc_both),
        'acc_both_either_order': float(acc_either),
        'mae_ratio': float(mae_ratio)
    }
}
with open(os.path.join(output_dir, 'run_summary.json'), 'w', encoding='utf-8') as f:
    json.dump(run_summary, f, ensure_ascii=False, indent=2)

# ==================== 预测函数（输出top3） ====================
def predict_overlay_top3(overlay_intensity, model, A_names, device):
    """输入叠加谱向量，预测两个样本的top3候选及比例"""
    model.eval()
    with torch.no_grad():
        x = torch.tensor(overlay_intensity, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
        out1, out2, out_ratio = model(x)
        prob1 = torch.softmax(out1, dim=1).squeeze()
        prob2 = torch.softmax(out2, dim=1).squeeze()
        # 获取top3
        top3_prob1, top3_idx1 = torch.topk(prob1, 3)
        top3_prob2, top3_idx2 = torch.topk(prob2, 3)
        pred_ratio = out_ratio.item()
    # 转换为名称列表
    top3_names1 = [A_names[idx] for idx in top3_idx1.cpu().numpy()]
    top3_names2 = [A_names[idx] for idx in top3_idx2.cpu().numpy()]
    top3_probs1 = top3_prob1.cpu().numpy()
    top3_probs2 = top3_prob2.cpu().numpy()
    return top3_names1, top3_probs1, top3_names2, top3_probs2, pred_ratio

# ==================== 测试示例：用A中最后两个样本 ====================
if len(A_files) >= 2:
    test_file1 = os.path.join(dataset_A_dir, A_files[-2])
    test_file2 = os.path.join(dataset_A_dir, A_files[-1])
    demo_ratio = 0.3   # 可自定义

    # 读取并插值
    rt1, int1 = read_chromatogram(test_file1)
    rt2, int2 = read_chromatogram(test_file2)
    int1_grid = interpolate_to_grid(rt1, int1, rt_grid)
    int2_grid = interpolate_to_grid(rt2, int2, rt_grid)
    overlay = demo_ratio * int1_grid + (1 - demo_ratio) * int2_grid

    print(f"\n===== 测试示例 =====")
    print(f"测试叠加谱由以下两个样本生成：")
    print(f"  样本1: {os.path.basename(test_file1)}")
    print(f"  样本2: {os.path.basename(test_file2)}")
    print(f"  真实比例: {demo_ratio:.2f} : {1-demo_ratio:.2f}")

    # 预测top3
    top3_names1, top3_probs1, top3_names2, top3_probs2, pred_ratio = predict_overlay_top3(overlay, model, A_names, device)

    print(f"\n模型预测比例: {pred_ratio:.4f} : {1-pred_ratio:.4f}")
    print("\n第一个样本的top3预测:")
    for name, prob in zip(top3_names1, top3_probs1):
        print(f"  {name}: {prob:.4f}")
    print("\n第二个样本的top3预测:")
    for name, prob in zip(top3_names2, top3_probs2):
        print(f"  {name}: {prob:.4f}")
else:
    print("A库样本不足两个，无法测试。")