# GC-MS 混合物识别工具

基于 1D CNN 的 GC-MS 二元混合物成分识别 Web 工具（MVP）。

## 定位

实验验证用工具，用于快速测试"上传色谱 → 识别混合成分"的完整流程。
当前模型在合成数据上的成分对识别准确率约 99%，比例预测仍在优化中。
**不是生产系统**，结果仅供实验参考。

## 启动

```bash
cd webapp
python app.py
# 浏览器打开 http://127.0.0.1:5000
```

如果使用 conda 环境：
```bash
/path/to/miniconda3/envs/ai_env/python.exe app.py
```

## 依赖

- Python 3.8+
- flask
- torch
- numpy, pandas, scipy, matplotlib, openpyxl

安装：
```bash
pip install flask torch numpy pandas scipy matplotlib openpyxl
```

## 输入格式

Excel 文件（.xlsx / .xls），需包含以下列：
- RT 列：列名含 `rt` 或 `保留时间(分钟)`
- 强度列：列名含 `归一化强度(10^8)` 或 `intensity`

与 `GCMS_单个样本数据/` 目录下的文件格式一致。

## 当前模型

| 项目 | 内容 |
|------|------|
| 模型 | CNN-OrderInv v1 |
| Loss | Order-invariant loss |
| 输入 | RT-TIC 1D（1000 点） |
| 样本库 | 9 类 |
| 模型文件 | `outputs/cnn_r80_orderinv/best_model.pth` |

## 环境变量（可选）

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `MODEL_PATH` | `../outputs/cnn_r80_orderinv/best_model.pth` | 模型文件路径 |
| `SAMPLE_DIR` | `../GCMS_单个样本数据` | 样本库目录 |

## 文件结构

```
webapp/
├── app.py           # Flask 路由
├── inference.py     # 模型加载 + 预处理 + 推理
├── templates/
│   ├── index.html   # 上传页
│   └── result.html  # 结果页
├── static/          # 生成的图表
└── README.md
```
