"""
严格对照评测脚本：在完全相同的数据 + 划分下，对比两个 checkpoint。
评测指标同时覆盖 train set 和 test set，明确区分。
"""
import os
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

# ==================== 配置 ====================
seed = 42
test_ratio = 0.2
batch_size = 32
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 两个实验的目录
baseline_dir = 'f:/样本叠加测试/outputs/cnn_lc_r80_bugfix'
orderinv_dir = 'f:/样本叠加测试/outputs/cnn_r80_orderinv'

# ==================== 模型定义（必须与训练一致） ====================
class OverlayCNN(nn.Module):
    def __init__(self, n_classes, input_length=1000):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128), nn.ReLU(), nn.AdaptiveAvgPool1d(1)
        )
        self.fc_shared = nn.Linear(128, 256)
        self.class_head1 = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, n_classes))
        self.class_head2 = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, n_classes))
        self.ratio_head = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 1), nn.Sigmoid()
        )

    def forward(self, x):
        feat = self.conv_layers(x).squeeze(-1)
        feat = self.fc_shared(feat)
        out1 = self.class_head1(feat)
        out2 = self.class_head2(feat)
        out_ratio = self.ratio_head(feat)
        return out1, out2, out_ratio

# ==================== Dataset ====================
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

# ==================== 评估函数 ====================
def evaluate(model, loader, device, split_name):
    model.eval()
    total = 0
    correct1 = correct2 = correct_both = correct_either = 0
    total_ratio_error = 0.0
    # 也计算 order-aware MAE (取两种顺序中更小的ratio error)
    total_ratio_error_best = 0.0
    with torch.no_grad():
        for X_batch, y1, y2, y_ratio_batch in loader:
            X_batch = X_batch.to(device)
            y1, y2 = y1.to(device), y2.to(device)
            y_ratio_batch = y_ratio_batch.to(device).squeeze()
            out1, out2, out_ratio = model(X_batch)
            pred1 = torch.argmax(out1, dim=1)
            pred2 = torch.argmax(out2, dim=1)
            ratio_pred = out_ratio.squeeze()

            total += y1.size(0)
            correct1 += (pred1 == y1).sum().item()
            correct2 += (pred2 == y2).sum().item()

            strict_match = (pred1 == y1) & (pred2 == y2)
            swapped_match = (pred1 == y2) & (pred2 == y1)
            correct_both += strict_match.sum().item()
            correct_either += (strict_match | swapped_match).sum().item()

            # strict MAE (ratio vs y_ratio as labeled)
            ratio_error = torch.abs(ratio_pred - y_ratio_batch)
            total_ratio_error += ratio_error.sum().item()

            # best-order MAE: min(|pred - y_ratio|, |pred - (1-y_ratio)|)
            ratio_error_swap = torch.abs(ratio_pred - (1.0 - y_ratio_batch))
            ratio_error_best = torch.min(ratio_error, ratio_error_swap)
            total_ratio_error_best += ratio_error_best.sum().item()

    results = {
        'split': split_name,
        'n_samples': total,
        'acc_sample1': correct1 / total,
        'acc_sample2': correct2 / total,
        'acc_both_strict': correct_both / total,
        'acc_both_either_order': correct_either / total,
        'mae_ratio_strict': total_ratio_error / total,
        'mae_ratio_best_order': total_ratio_error_best / total,
    }
    return results

def print_results(name, train_res, test_res):
    print(f"\n{'='*60}")
    print(f"  Model: {name}")
    print(f"{'='*60}")
    for res in [train_res, test_res]:
        print(f"\n  [{res['split']}] (n={res['n_samples']})")
        print(f"    acc_sample1:          {res['acc_sample1']:.4f}")
        print(f"    acc_sample2:          {res['acc_sample2']:.4f}")
        print(f"    acc_both_strict:      {res['acc_both_strict']:.4f}")
        print(f"    acc_both_either:      {res['acc_both_either_order']:.4f}")
        print(f"    strict-either gap:    {res['acc_both_either_order'] - res['acc_both_strict']:.4f}")
        print(f"    mae_ratio (strict):   {res['mae_ratio_strict']:.4f}")
        print(f"    mae_ratio (best-ord): {res['mae_ratio_best_order']:.4f}")

# ==================== 加载数据（用 baseline 目录的数据，两者生成的一样） ====================
X = np.load(os.path.join(baseline_dir, 'X.npy'))
y_class1 = np.load(os.path.join(baseline_dir, 'y_class1.npy'))
y_class2 = np.load(os.path.join(baseline_dir, 'y_class2.npy'))
y_ratio = np.load(os.path.join(baseline_dir, 'y_ratio.npy'))

n_classes = len(np.unique(y_class1))
print(f"Data: {len(X)} samples, {n_classes} classes")

# 用和训练完全相同的 seed 划分
dataset = OverlayDataset(X, y_class1, y_class2, y_ratio)
train_size = int((1 - test_ratio) * len(dataset))
test_size = len(dataset) - train_size
split_gen = torch.Generator().manual_seed(seed)
train_dataset, test_dataset = random_split(dataset, [train_size, test_size], generator=split_gen)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print(f"Split: train={train_size}, test={test_size} (seed={seed})")

# ==================== 验证数据一致性 ====================
X_inv = np.load(os.path.join(orderinv_dir, 'X.npy'))
data_match = np.allclose(X, X_inv)
print(f"Data identity check (baseline vs orderinv): {'MATCH' if data_match else 'MISMATCH!'}")

# ==================== 评测两个模型 ====================
for name, model_dir in [('r80_baseline', baseline_dir), ('r80_orderinv', orderinv_dir)]:
    model = OverlayCNN(n_classes=n_classes).to(device)
    ckpt_path = os.path.join(model_dir, 'best_model.pth')
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))

    train_res = evaluate(model, train_loader, device, 'TRAIN')
    test_res = evaluate(model, test_loader, device, 'TEST')
    print_results(name, train_res, test_res)

# ==================== 对比表 ====================
print(f"\n{'='*60}")
print("  SIDE-BY-SIDE COMPARISON (TEST SET)")
print(f"{'='*60}")
print(f"{'Metric':<28} {'baseline':>10} {'orderinv':>10} {'delta':>10}")
print(f"{'-'*58}")

# Re-evaluate for comparison table
results = {}
for name, model_dir in [('baseline', baseline_dir), ('orderinv', orderinv_dir)]:
    model = OverlayCNN(n_classes=n_classes).to(device)
    model.load_state_dict(torch.load(os.path.join(model_dir, 'best_model.pth'),
                                     map_location=device, weights_only=True))
    results[name] = evaluate(model, test_loader, device, 'TEST')

for key in ['acc_both_strict', 'acc_both_either_order', 'mae_ratio_strict', 'mae_ratio_best_order']:
    b = results['baseline'][key]
    o = results['orderinv'][key]
    d = o - b
    sign = '+' if d > 0 else ''
    print(f"{key:<28} {b:>10.4f} {o:>10.4f} {sign}{d:>9.4f}")

gap_b = results['baseline']['acc_both_either_order'] - results['baseline']['acc_both_strict']
gap_o = results['orderinv']['acc_both_either_order'] - results['orderinv']['acc_both_strict']
print(f"{'strict-either gap':<28} {gap_b:>10.4f} {gap_o:>10.4f} {gap_o-gap_b:>+10.4f}")

print(f"\nNote: test set was used for early stopping (model selection).")
print(f"These results are from a single split (seed={seed}), not cross-validated.")
print(f"They are valid for controlled A/B comparison but not for absolute generalization claims.")
