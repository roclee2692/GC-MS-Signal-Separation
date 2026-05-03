"""
GC-MS 混合物识别工具 — MVP
Flask 应用：上传 → 推理 → 结果展示
"""
import os
import uuid
import traceback
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 无 GUI 后端
import matplotlib.pyplot as plt

# 中文字体支持
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'PingFang SC', 'Noto Sans CJK SC', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
from flask import Flask, request, render_template, redirect, url_for

from inference import (
    GCMSPredictor,
    read_chromatogram_from_bytes,
    interpolate_to_grid,
    RT_MIN, RT_MAX, N_RT_POINTS,
)

# ==================== 配置 ====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)

MODEL_PATH = os.environ.get(
    'MODEL_PATH',
    os.path.join(PROJECT_DIR, 'outputs', 'cnn_B87_baseline', 'best_model.pth'),
)
SAMPLE_DIR = os.environ.get(
    'SAMPLE_DIR',
    os.path.join(PROJECT_DIR, '单个色谱', '单个色谱图-B'),
)
STATIC_DIR = os.path.join(BASE_DIR, 'static')
ALLOWED_EXTENSIONS = {'.xlsx', '.xls'}

# ==================== 初始化 ====================
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB 上传限制

# 启动时加载模型（只加载一次）
predictor = None
load_error = None
try:
    predictor = GCMSPredictor(MODEL_PATH, SAMPLE_DIR)
    print(f"模型加载成功: {MODEL_PATH}")
    print(f"样本库: {predictor.n_classes} 种样本")
except Exception as e:
    load_error = str(e)
    print(f"模型加载失败: {load_error}")


def allowed_file(filename):
    ext = os.path.splitext(filename)[1].lower()
    return ext in ALLOWED_EXTENSIONS


def generate_chart(rt_grid, intensity, result, chart_id):
    """生成色谱图 + 结果标注，保存为 PNG"""
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.fill_between(rt_grid, intensity, alpha=0.15, color='#667eea')
    ax.plot(rt_grid, intensity, color='#667eea', linewidth=0.9)
    ax.set_xlabel('保留时间 (min)', fontsize=10)
    ax.set_ylabel('强度', fontsize=10)
    ax.set_title('上传色谱图', fontsize=13, fontweight='bold')
    ax.set_xlim(RT_MIN, RT_MAX)

    # 在图上标注结果
    text = (
        f"成分 1: {result['component1']}\n"
        f"成分 2: {result['component2']}\n"
        f"比例: {result['ratio1']:.1%} : {result['ratio2']:.1%}"
    )
    ax.text(
        0.98, 0.95, text,
        transform=ax.transAxes, fontsize=9,
        verticalalignment='top', horizontalalignment='right',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='#f8fafc',
                  edgecolor='#667eea', alpha=0.95),
    )

    ax.grid(True, alpha=0.2, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout()
    chart_path = os.path.join(STATIC_DIR, f'{chart_id}.png')
    fig.savefig(chart_path, dpi=140, bbox_inches='tight')
    plt.close(fig)
    return f'{chart_id}.png'


# ==================== 路由 ====================

@app.route('/')
def index():
    return render_template('index.html', load_error=load_error)


@app.route('/predict', methods=['POST'])
def predict():
    # 检查模型是否可用
    if predictor is None:
        return render_template('result.html', error=f"模型未加载: {load_error}")

    # 检查文件
    if 'file' not in request.files:
        return render_template('result.html', error="没有选择文件")
    file = request.files['file']
    if file.filename == '':
        return render_template('result.html', error="没有选择文件")
    if not allowed_file(file.filename):
        return render_template('result.html',
                               error=f"不支持的文件格式: {file.filename}。请上传 .xlsx 或 .xls 文件。")

    try:
        # 读取并预处理
        file_bytes = file.read()
        rt_raw, int_raw = read_chromatogram_from_bytes(file_bytes, file.filename)
        rt_grid, intensity_grid = interpolate_to_grid(rt_raw, int_raw)

        # 推理
        result = predictor.predict(intensity_grid)

        # 生成图表
        chart_id = uuid.uuid4().hex[:8]
        chart_filename = generate_chart(rt_grid, intensity_grid, result, chart_id)

        return render_template(
            'result.html',
            result=result,
            filename=file.filename,
            chart_url=url_for('static', filename=chart_filename),
            n_points=len(rt_raw),
            rt_range=f"{rt_raw.min():.2f} ~ {rt_raw.max():.2f} min",
        )

    except ValueError as e:
        return render_template('result.html', error=f"文件解析失败: {e}")
    except Exception as e:
        traceback.print_exc()
        return render_template('result.html', error=f"推理过程出错: {e}")


# ==================== 启动 ====================
if __name__ == '__main__':
    os.makedirs(STATIC_DIR, exist_ok=True)
    app.run(host='127.0.0.1', port=5000, debug=True)
