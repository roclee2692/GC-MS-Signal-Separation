# 项目部署说明（Windows / PowerShell）

## 1. 前置条件
- 已安装 Python 3.10 到 3.12（建议 3.10 或 3.11）
- 在项目根目录执行命令

优先建议：先复用你已有的 conda 环境（如 `ai_env`），仅安装缺失依赖；不必重复新建环境。

```powershell
conda activate ai_env
pip install torch xlrd
```

## 2. 创建并激活虚拟环境
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```
如果项目里已有 `.venv` 但报错（例如 `No module named encodings`），请先删除再重建：
```powershell
Remove-Item -Recurse -Force .\.venv
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

## 3. 安装依赖
```powershell
pip install -r requirements.txt
```

## 4. 数据目录约定
默认读取目录：`GCMS_单个样本数据`

默认输出目录：`outputs/cnn_baseline`

可通过环境变量覆盖：
```powershell
$env:DATASET_A_DIR = '你的数据目录'
$env:OUTPUT_DIR = '你的输出目录'
```

## 5. 启动训练与评估
方式一（推荐，一键运行）：
```powershell
.\run_cnn.ps1
```
说明：`run_cnn.ps1` 会优先使用可用的 `.venv`，若检测到 `.venv` 异常会自动回退到系统 `python`。

方式二（手动）：
```powershell
python .\CNN.py
```

## 6. 可调参数
在 PowerShell 中通过环境变量修改：
```powershell
$env:EPOCHS = '50'
$env:LAMBDA_REG = '1.0'
.\run_cnn.ps1
```

## 7. 主要输出文件
- `outputs/cnn_baseline/overlay_cnn_fullA.pth`：模型权重
- `outputs/cnn_baseline/loss_curve.png`：训练曲线
- `outputs/cnn_baseline/*.npy`：中间数据与标签

## 8. 常见问题
1. Excel 读取失败：确认安装了 `openpyxl` 和 `xlrd`。
2. 找不到样本文件：确认 `DATASET_A_DIR` 目录下有 `.xlsx` 或 `.xls`。
3. 显存不足：减小 `batch_size`（在 `CNN.py` 里调整）。
