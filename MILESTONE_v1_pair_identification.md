# GC-MS 混合样本识别 — 阶段性里程碑 v1

> 日期: 2026-03-18
> 状态: pair 识别基本完成，比例预测待突破，等待导师确认下一步方向

---

## 1. 当前最佳模型

| 项目 | 内容 |
|------|------|
| 脚本文件 | `CNN.py`（项目根目录） |
| Loss 版本 | **order-invariant loss**（对每个样本计算原序和交换序两种 loss，取 per-sample min） |
| 数据规模 | r=80（每对样本 80 个随机比例，共 7200 条合成混合谱） |
| 数据类型 | 合成线性混合（两条单样本 RT-TIC 按随机比例线性叠加） |
| 模型架构 | 1D CNN (3层卷积 + shared FC + 双分类头 + 比例回归头) |
| 最佳模型文件 | `outputs/cnn_r80_orderinv/best_model.pth` (726 KB) |
| 最终模型文件 | `outputs/cnn_r80_orderinv/overlay_cnn_fullA.pth` |
| 训练摘要 | `outputs/cnn_r80_orderinv/run_summary.json` |
| 损失曲线 | `outputs/cnn_r80_orderinv/loss_curve.png` |

---

## 2. 关键结果（单一 split, seed=42）

> **注意**: 以下结果基于合成数据，test set 同时用于 early stopping 模型选择，
> 不是独立测试结果，不能直接作为泛化性能的最终结论。

| 指标 | 数值 | 说明 |
|------|------|------|
| either-order acc_both | **99.4%** | 不考虑顺序，两个样本都识别正确 |
| strict acc_both | 60.1% | 考虑 head1/head2 顺序（在 order-inv loss 下意义有限） |
| acc_sample1 | 60.1% | head1 单独准确率 |
| acc_sample2 | 60.5% | head2 单独准确率 |
| MAE (best-order) | 0.176 | 取两种顺序中更小的比例误差 |
| MAE (strict) | 0.203 | 按标签顺序计算的比例误差 |
| best epoch | 34 | early stop 在 epoch 44 触发 |
| best test_loss | 0.163 | |

### 比例预测详细分析

| 真实比例区间 | 样本数 | 预测均值 | 真实均值 | MAE |
|-------------|--------|----------|----------|------|
| [0.1, 0.3) | 389 | 0.475 | 0.203 | 0.273 |
| [0.3, 0.5) | 393 | 0.475 | 0.400 | 0.116 |
| [0.5, 0.7) | 294 | 0.509 | 0.601 | 0.093 |
| [0.7, 0.9) | 291 | 0.513 | 0.800 | 0.287 |

**结论**: 模型预测值集中在 0.29~0.72 (std=0.088)，基本在输出"大约五五开"。
"谁是主要成分"的判断正确率仅 67%（随机猜为 50%）。

---

## 3. 复现说明

### 环境

```
OS: Windows 11 Pro for Workstations
GPU: NVIDIA RTX 4060 (8GB)
Python: miniconda3/envs/ai_env
依赖: torch, numpy, pandas, scipy, matplotlib, openpyxl
```

### 运行命令

```bash
# 训练（order-invariant loss 版本，r=80）
cd "f:/样本叠加测试"
NUM_RATIOS_PER_PAIR=80 EPOCHS=100 OUTPUT_DIR="f:/样本叠加测试/outputs/cnn_r80_orderinv" \
  /c/Users/Raelon/miniconda3/envs/ai_env/python.exe CNN.py

# 对照评测（同时评测 baseline 和 order-inv 两个 checkpoint）
/c/Users/Raelon/miniconda3/envs/ai_env/python.exe compare_eval.py
```

### 主要参数

| 参数 | 值 | 说明 |
|------|----|------|
| NUM_RATIOS_PER_PAIR | 80 | 每对样本生成的混合比例数 |
| EPOCHS | 100 | 最大训练轮数 |
| EARLY_STOPPING_PATIENCE | 10 | 早停耐心值 |
| SEED | 42 | 全局随机种子 |
| LAMBDA_REG | 5.0 | 回归 loss 权重 |
| learning_rate | 5e-4 | Adam 学习率 |
| batch_size | 32 | 批大小 |
| test_ratio | 0.2 | 测试集比例 |

### 数据

- 单样本数据目录: `GCMS_单个样本数据/`（9 个 Excel 文件）
- RT 范围: 4~50 min，插值到 1000 点
- 混合方式: 线性叠加 overlay = r1 * sample_A + (1-r1) * sample_B
- 生成数据缓存: `outputs/cnn_r80_orderinv/X.npy`, `y_class1.npy`, `y_class2.npy`, `y_ratio.npy`

### 输出文件说明

| 文件 | 内容 |
|------|------|
| `best_model.pth` | early stopping 选出的最佳 checkpoint |
| `overlay_cnn_fullA.pth` | 训练结束时的最终 checkpoint |
| `run_summary.json` | 完整训练参数和最终指标 |
| `loss_curve.png` | 训练/测试 loss 曲线 |
| `X.npy` / `y_*.npy` | 生成的训练数据缓存 |

---

## 4. 当前阶段结论

### 已解决

- **十选二 pair 识别**: 在合成线性混合数据上，either-order acc_both 达到 99.4%。
  模型能几乎完美地识别混合物由哪两种物质组成。
- **order-invariant loss 有效**: 相比固定顺序 loss，either-order 从 69.5% 提升到 99.4%，
  test_loss 从 1.328 降到 0.163。证明消除顺序歧义是关键改进。
- **标签 bug 已修复**: Order 2 的 y_ratio 标签从错误的 r2 修正为 r1。

### 未解决

- **比例预测**: ratio head 基本输出 ~0.5，没有真正学到比例信息。
  MAE ~0.18，"谁是主要成分"判断正确率仅 67%。
  **根本原因**: RT-TIC 一维特征中比例信息极其有限。
- **真实数据泛化**: 所有结果都基于合成线性混合，未在真实混合样品上验证。
  真实场景存在非线性效应、基线漂移、共洗脱等问题。
- **m/z 特征未接入**: 质谱的 m/z 维度（携带定量化学指纹信息）尚未使用。
- **独立评测**: test set 参与了 early stopping，不是真正独立的测试集。
  未做多种子实验或交叉验证。

### 当前不再继续做

- 不继续在 RT-TIC 上扩数据（r=80 已看到比例预测饱和）
- 不继续调 ratio head 结构（特征层缺信息，head 再改也没用）
- 暂不上 Transformer（当前 CNN 已经足够解决 pair 识别）
- 暂不做更多超参搜索（主要瓶颈不在超参数）

---

## 5. 实验历程摘要

| 实验 | 样本数 | either-order | strict | MAE | 关键变化 |
|------|--------|-------------|--------|-----|----------|
| baseline (固定比例) | 693 | - | 28.1% | 0.127 | 初始基线 |
| baseline+norm | 693 | - | 24.5% | 0.131 | z-score 归一化，效果不佳 |
| continuous r=20 | 1800 | - | 30.6% | 0.114 | 连续随机比例 |
| r=40 (bugfix前) | 3600 | - | 42.8% | 0.111 | 学习曲线 |
| r=40 (bugfix后) | 3600 | 61.0% | 40.4% | 0.188 | 修复标签 bug，加 either-order 指标 |
| r=60 (bugfix) | 5400 | 57.7% | 37.1% | 0.187 | 数据扩展 |
| r=80 (bugfix) | 7200 | 69.5% | 44.0% | 0.185 | 数据扩展 |
| **r=80 + order-inv loss** | **7200** | **99.4%** | **60.1%** | **0.176** | **当前最佳** |

---

## 6. 待确认问题清单（等导师确认）

### 必须确认

1. **业务目标优先级**: 最终更看重"识别哪两种物质"还是"精确预测比例"？
   - 如果主要是 pair 识别 → 当前已接近目标，重点转向真实数据验证
   - 如果比例预测同样重要 → 必须引入 m/z 特征

2. **有没有真实混合数据**: 哪怕 2~3 组已知成分和比例的真实混合样品 GC-MS 数据，
   就能验证合成数据训练的模型是否能迁移到真实场景

3. **m/z 数据是否可用**: 当前 9 个样本的 GC-MS 原始数据中是否包含 m/z 维度？
   如果有，以什么格式存储？（mzML / mzXML / 厂商格式 / 已提取的特征表？）

### 建议确认

4. **ratio 最终定义**: 比例应该表示什么？
   - "第一种物质的质量/体积占比"？
   - "较多成分的占比"（永远 >= 0.5）？
   - 还是直接输出两个比例值？

5. **物质种类扩展计划**: 后续会从 9 种扩展到多少种？
   这影响模型架构和评估方案的设计

6. **最终交付形态**: 是一个训练好的模型文件，还是一个可以输入谱图自动输出结果的工具？
