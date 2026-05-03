# GC-MS 混合物识别项目 — 阶段性总结

> 最后更新: 2026-03-24

---

## 一、项目概述

利用深度学习模型，从 GC-MS 一维色谱数据（RT-TIC）中自动识别混合物的两种组成成分及其比例。
当前基于 9 种已知样本，进行"十选二"组合识别。

---

## 二、已完成的实验

共 7 组正式实验，全部模型已保存。

### CNN 系列（4 组）

| 实验 | 目录 | either-order | strict | MAE | best_epoch | 模型大小 |
|------|------|:---:|:---:|:---:|:---:|:---:|
| CNN 基线 | `outputs/cnn_r80_orderinv/` | **99.44%** | 60.1% | 0.203 | 34 | 0.7 MB |
| CNN + 漂移 | `outputs/cnn_r80_orderinv_drift/` | 98.0% | 59.1% | **0.185** | 73 | 0.7 MB |
| CNN + 噪声 | `outputs/cnn_r80_orderinv_noise/` | 98.8% | 59.8% | 0.192 | 33 | 0.7 MB |
| CNN + 漂移 + 噪声 | `outputs/cnn_r80_orderinv_drift_noise/` | 93.3% | 56.5% | 0.204 | 46 | 0.7 MB |

### CNN+Transformer 系列（3 组）

| 实验 | 目录 | either-order | strict | MAE | best_epoch | 模型大小 |
|------|------|:---:|:---:|:---:|:---:|:---:|
| Transformer 基线 | `outputs/transformer_r80_baseline/` | **99.86%** | 58.8% | 0.202 | 17 | 34.8 MB |
| Transformer + 漂移 | `outputs/transformer_r80_drift/` | 99.2% | **61.6%** | 0.223 | 25 | 34.8 MB |
| Transformer + 漂移 + 噪声 | `outputs/transformer_r80_drift_noise/` | 98.9% | 59.2% | 0.229 | 28 | 34.8 MB |

### 归档实验（早期探索）

位于 `outputs/_archived/`，包含 label bug 修复前的实验和学习曲线实验（r=8/15/20/30/40），仅供参考。

---

## 三、统一训练参数

所有正式实验使用以下固定参数：

- 样本数: 7200（9 种样本, C(9,2)=36 对, 每对 80 个随机比例, 每个比例 2 种排列）
- 训练/测试集: 80%/20%（5760/1440）
- Seed: 42
- Batch size: 32
- Learning rate: 5e-4
- Lambda_reg（比例 loss 权重）: 5.0
- Early stopping patience: 10
- 损失函数: order-invariant loss（对两种排列取 min）
- 漂移增强: max_shift=5 点（~0.23 分钟全局平移）
- 噪声增强: noise_level=0.02（高斯噪声）

---

## 四、核心结论

### 1. 成分识别（either-order）已接近上限
- 最佳: Transformer 基线 99.86%，CNN 基线 99.44%
- 在模拟数据上几乎完美

### 2. CNN vs Transformer：无显著差异
- 准确率基本持平
- CNN 模型 0.7MB，Transformer 34.8MB（50 倍）
- CPU 训练 Transformer 每 epoch 3-4 分钟，CNN 约 20 秒
- **推荐继续使用 CNN**

### 3. 数据增强没有帮助
- 漂移和噪声增强均未提升指标，反而略有下降
- 原因: 训练数据本身是模拟的，再加模拟扰动收益有限

### 4. 比例预测是当前瓶颈
- 所有实验 MAE 均在 0.18-0.23 之间
- 模型预测值集中在 ~0.5，无法区分极端比例
- strict 准确率 56-62%（受比例预测拖累）
- 可能原因: RT-TIC 一维特征对比例信息承载有限

---

## 五、模型文件位置

每个实验目录下包含:
```
outputs/<实验名>/
  ├── best_model.pth     # 训练好的模型权重
  ├── run_summary.json   # 实验参数和评估指标
  ├── X.npy              # 生成的训练数据
  ├── y_class1.npy       # 标签: 成分 1
  ├── y_class2.npy       # 标签: 成分 2
  ├── y_ratio.npy        # 标签: 比例
  ├── A_intensities.npy  # 9 种原始样本的插值强度
  └── A_names.npy        # 9 种样本名称
```

**当前推荐模型**: `outputs/cnn_r80_orderinv/best_model.pth`（CNN 基线，综合最优）

---

## 六、Web 预测工具

位于 `webapp/` 目录，详见 [webapp/使用说明.md](webapp/使用说明.md)。

- Flask 应用，本地运行
- 上传 .xlsx 色谱文件 → 自动识别成分和比例
- 支持 A 组和 B 组数据格式

---

## 七、代码文件说明

| 文件 | 用途 |
|------|------|
| `CNN.py` | CNN 训练脚本（主力，含数据生成、训练、评估） |
| `CNN_Transformer.py` | Transformer 训练脚本（基于 CNN.py，仅替换模型类） |
| `CNN+Transformer的自注意力机制.py` | 原始 Transformer 脚本（有多处 bug，未用于正式实验） |
| `CNN加偏移.py` | 早期偏移实验脚本（已弃用） |
| `webapp/app.py` | Web 应用主程序 |
| `webapp/inference.py` | 推理模块（模型加载 + 预测） |

---

## 八、数据说明

| 目录 | 内容 |
|------|------|
| `GCMS_单个样本数据/` | A 组 10 种单样本色谱（含新增的 NB21A2） |
| `单个色谱/单个色谱图-A/` | A 组完整库（181 种） |
| `单个色谱/单个色谱图-B/` | B 组完整库（87 种） |
| `组合后的叠加谱/` | 预计算的 45 对组合 × 9 比例 = 405 个叠加文件 |

**注意**: 所有叠加谱（包括训练数据和组合后的叠加谱）都是**数学模拟**（线性加权），不是真实混合后上机测量的数据。

---

## 九、下一步方向

1. **比例预测优化**: 换用分桶分类、换 loss 函数、或引入 m/z 特征
2. **真实数据验证**: 用真实混合样品验证模型泛化能力
3. **样本库扩展**: 从 9 种扩展到更多（B 组有 87 种可用）
4. **GPU 加速**: 已在 ai_env 中安装 CUDA PyTorch (2.6.0+cu124)，切换到独显模式后可用

---

## 十、环境信息

| 项目 | 值 |
|------|------|
| Python | 3.13 (conda base) / 3.10 (ai_env) |
| PyTorch (base) | 2.10.0+cpu |
| PyTorch (ai_env) | 2.6.0+cu124（需切换独显） |
| GPU | NVIDIA RTX 4060（办公模式下未启用） |
| 操作系统 | Windows 11 Pro for Workstations |
