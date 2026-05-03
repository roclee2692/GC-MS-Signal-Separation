"""
GC-MS 混合物推理模块
从 CNN.py 提取的最小推理流程：预处理 + 模型定义 + 预测
"""
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.interpolate import interp1d

# ==================== 参数 ====================
RT_MIN, RT_MAX = 4, 50
N_RT_POINTS = 1000

# ==================== 预处理 ====================

def read_chromatogram(file_path):
    """读取 Excel 色谱图，返回 (rt_array, intensity_array)"""
    df = pd.read_excel(file_path)
    df.columns = [c.strip().lower() for c in df.columns]
    rt_col = next((c for c in df.columns if 'rt' in c or '保留时间(分钟)' in c), None)
    int_col = next((c for c in df.columns if '归一化强度(10^8)' in c or 'intensity' in c), None)
    if rt_col is None or int_col is None:
        raise ValueError(
            f"无法识别 RT 或强度列。"
            f"找到的列: {list(df.columns)}。"
            f"需要包含 'rt' 的列和包含 '归一化强度(10^8)' 或 'intensity' 的列。"
        )
    df = df[[rt_col, int_col]].dropna().sort_values(by=rt_col)
    return df[rt_col].values, df[int_col].values


def read_chromatogram_from_bytes(file_bytes, filename):
    """从上传的文件字节流读取色谱图，兼容多种列名格式"""
    import io
    buf = io.BytesIO(file_bytes)
    df = pd.read_excel(buf, engine='openpyxl' if filename.endswith('.xlsx') else None)
    df.columns = [c.strip().lower() for c in df.columns]
    # 宽泛匹配 RT 列
    rt_col = next((c for c in df.columns if any(k in c for k in
        ['rt', '保留时间', 'retention', 'time', '时间'])), None)
    # 宽泛匹配强度列
    int_col = next((c for c in df.columns if any(k in c for k in
        ['归一化强度', 'intensity', '强度', 'abundance', 'signal', 'tic'])), None)
    # 如果只有两列且匹配失败，假设第一列是RT，第二列是强度
    if (rt_col is None or int_col is None) and len(df.columns) == 2:
        rt_col, int_col = df.columns[0], df.columns[1]
    if rt_col is None or int_col is None:
        raise ValueError(
            f"无法识别 RT 或强度列。"
            f"找到的列: {list(df.columns)}。"
            f"需要包含时间/RT相关列和强度/Intensity相关列。"
        )
    df = df[[rt_col, int_col]].dropna().sort_values(by=rt_col)
    return df[rt_col].values, df[int_col].values


def interpolate_to_grid(rt, intensity):
    """将色谱数据插值到统一网格"""
    grid = np.linspace(RT_MIN, RT_MAX, N_RT_POINTS)
    f = interp1d(rt, intensity, kind='linear', bounds_error=False, fill_value=0)
    return grid, f(grid)


# ==================== 模型定义（与 CNN.py 完全一致） ====================

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
        self.class_head1 = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, n_classes))
        self.class_head2 = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, n_classes))
        self.ratio_head = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 1), nn.Sigmoid(),
        )

    def forward(self, x):
        feat = self.conv_layers(x).squeeze(-1)
        feat = self.fc_shared(feat)
        return self.class_head1(feat), self.class_head2(feat), self.ratio_head(feat)


# ==================== 推理引擎 ====================

class GCMSPredictor:
    """封装模型加载和推理的完整流程"""

    def __init__(self, model_path, sample_dir):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 优先从模型目录的 A_names.npy 加载样本名（与训练时一致）
        model_dir = os.path.dirname(model_path)
        names_path = os.path.join(model_dir, 'A_names.npy')
        if os.path.exists(names_path):
            self.sample_names = list(np.load(names_path, allow_pickle=True))
        else:
            # 回退：从样本目录读取
            sample_files = sorted(
                f for f in os.listdir(sample_dir)
                if f.endswith(('.xlsx', '.xls'))
            )
            if not sample_files:
                raise FileNotFoundError(f"样本目录为空: {sample_dir}")
            self.sample_names = [os.path.splitext(f)[0] for f in sample_files]
        self.n_classes = len(self.sample_names)

        # 加载模型
        self.model = OverlayCNN(n_classes=self.n_classes).to(self.device)
        state = torch.load(model_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state)
        self.model.eval()

    def predict(self, intensity_grid):
        """
        输入: intensity_grid — 长度为 N_RT_POINTS 的一维数组
        输出: dict，包含 top3 预测、比例、置信度
        """
        with torch.no_grad():
            x = torch.tensor(intensity_grid, dtype=torch.float32)
            x = x.unsqueeze(0).unsqueeze(0).to(self.device)  # (1, 1, 1000)
            out1, out2, out_ratio = self.model(x)

            prob1 = torch.softmax(out1, dim=1).squeeze()
            prob2 = torch.softmax(out2, dim=1).squeeze()
            pred_ratio = out_ratio.item()

            top3_prob1, top3_idx1 = torch.topk(prob1, min(3, self.n_classes))
            top3_prob2, top3_idx2 = torch.topk(prob2, min(3, self.n_classes))

        # 最终预测：取两个 head 的 top1
        best1_idx = top3_idx1[0].item()
        best2_idx = top3_idx2[0].item()
        best1_name = self.sample_names[best1_idx]
        best2_name = self.sample_names[best2_idx]
        conf1 = top3_prob1[0].item()
        conf2 = top3_prob2[0].item()

        # 整理 top3
        top3_1 = [
            {"name": self.sample_names[idx.item()], "prob": prob.item()}
            for idx, prob in zip(top3_idx1, top3_prob1)
        ]
        top3_2 = [
            {"name": self.sample_names[idx.item()], "prob": prob.item()}
            for idx, prob in zip(top3_idx2, top3_prob2)
        ]

        return {
            "component1": best1_name,
            "component2": best2_name,
            "ratio1": pred_ratio,
            "ratio2": 1 - pred_ratio,
            "confidence1": conf1,
            "confidence2": conf2,
            "top3_head1": top3_1,
            "top3_head2": top3_2,
        }
