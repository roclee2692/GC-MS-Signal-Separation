<div align="center">

# GC-MS Signal Deconvolution & Identification Engine
# GC-MS 色谱信号解耦与成分识别引擎

[**🇨🇳 中文版 (Chinese)**](#-中文版-chinese-version) | [**🇬🇧 English Version**](#-english-version)

</div>

---

## 🇨🇳 中文版 (Chinese Version)

**企业合作落地探索项目 · 工业级数据科学实践**

### 📖 项目背景
本项目源于一项真实的企业技术需求对接。合作企业需要对复杂的植物精油混合物（例如白兰花油、甜橙油、金银花油等共 **10 种基底原料**）进行成分拆解。由于气相色谱-质谱（GC-MS）数据经常出现信号重叠与干扰，依靠传统专家人工比对特征峰的分析方法效率极其低下。

**核心诉求**：构建一个基于深度学习的算法模型，直接输入 GC-MS 的一维色谱信号数据（RT-TIC 色谱图），实现端到端的“**混合物成分识别**”与“**核心原料混合配比预测**”。

### 💡 方案设计与大规模数据合成实验

由于工业界带有完美标签的真实混合数据难以低成本获取，本项目采用了算法工程中经典的“**仿真基线合成 (Synthetic Data Generation)**”路径，并在后续实验中**大规模扩充了数据量级**：

1. **构建超大规模合成物理数据集**：
早期实验基于简单的排列组合，但在最终阶段，我们提取了企业的 10 种基础原料色谱图，进行了全面组合（C(10,2) = 45 种原料级基础组合）。在此基础上，我们通过密集的**浓度比例遍历（数十种到上百种比例）**、**引入分析仪器的保留时间漂移（RT Drift）**以及**随机背景噪音（Gaussian Noise 数据增强）**，最终通过代码**自动化扩样生成了数万条高度复杂的模拟混合色谱图**来进行学习曲线（Learning Curve）测试。

2. **多模态与深度学习模型选型**：
- 将该工业问题正式定义为 **多标签分类 (Multi-label Classification) + 比例回归 (Regression)** 的联合任务。
- 独立设计并试验了基于 **1D-CNN** 和 **Transformer (Self-Attention)** 的信号处理架构。
- 针对混合物成分无序性导致的数学歧义，自研了 **Order-invariant Loss（顺序无关损失函数）**，打破了因排列而产生的信息干扰（修复了导致准确率卡脖子的 strict accuracy bug），让模型精准收敛。
- **实验结果**：在含有数万条样本的合成数据集上，CNN 基线模型达到了 **99.44%** 的成分盲盒识别准确率。尽管 Transformer 达到了 99.86%，但鉴于 CNN 模型体积极小 (仅 0.7MB) 与极快的训练推理速度（比Transformer快 5-10 倍），我们最终确立 **CNN** 为核心生产基线。

3. **端到端演示系统 (Demo App)**：
搭建了完整的 Web 前端应用界面服务。业务人员仅需直接拖拽仪器导出的 .xlsx 原始色谱文件，系统即会一键输出成分预估和调配比例。

### 🔬 工业界实验反思：Sim2Real 鸿沟 (Lessons Learned)

尽管算法在本地数万级模拟合成数据（Synthetic Data）上取得了近乎完美的准确度，但在交付前夕对**带有真实物理特性的盲测混合数据（Real Test Data）**进行验证时，我们遭遇了严峻的工程挑战：

- **Sim-to-Real 的相关性断层**：程序线性叠加与插值生成的合成色谱图，无法完美表征自然界复杂的物理化学干扰（如物理共流出效应、分子间相互作用导致的拖尾与偏移）。真实数据的分布与合成数据存在“域偏移（Domain Shift）”。
- **高质量落地数据的获取成本**：要进行基于真实域的数据微调（Fine-tuning），需要企业内部建立自动化流水线进行数万次的精密调配并过仪器进样，高昂的数据获取门槛与成本超出了落地预期。正如多次消融实验所暗示：短期的模型或硬件优化遭遇瓶颈后，继续盲目扩充仿真数据收益非常小，核心还是缺乏“高质量的、包含了真实 m/z 信息的混合数据”。

**项目总结**：
该落地流程最终暂时搁置。但这作为我的独立项目，带给我极大收获——它让我切身体会到了真实工业人工智能落地中的经典困局：**Data-centric AI 是破局的关键，高质量、贴合物理本质的数据构建其决定性意义远超单纯的算法设计**。这段从“需求沟通、数据合成、大规模实验”走向“业务验证碰壁、反思”的真实迭代经历，构成了我实践中最宝贵的认知财富。

### 🧩 技术栈
- **算法与深度学习**: PyTorch, 1D-CNN, Transformer, 定制 Loss Function (Order-invariant loss)
- **数据工程**: Pandas, NumPy, SciPy (泛型插值与大规模信号对齐/增强)
- **部署与产品化**: Python Flask, Javascript/HTML (前端构建与 Web 交互)

---

## 🇬🇧 English Version

**Enterprise Collaboration & Industrial Data Science Practice**

### 📖 Project Background
This project originated from a technical collaboration with an enterprise. The goal was to deconstruct complex plant essential oil mixtures (involving **10 discrete base raw materials**, such as White Orchid Oil, Sweet Orange Oil, Honeysuckle Oil, etc.). Since Gas Chromatography-Mass Spectrometry (GC-MS) data often suffers from signal overlap and interference, traditional analysis relying on manual peak comparison by experts is heavily inefficient.

**Core Objective**: Build an end-to-end deep learning model that directly takes 1D GC-MS signal data (RT-TIC chromatograms) as input to perform **mixture component identification** and **mixture ratio regression**.

### 💡 Methodology & Large-Scale Data Synthesis Experiment

Since massive labeled "real-world" mixture data is extremely expensive to acquire in the industry, this project adopted a classic **Synthetic Data Generation** pipeline, scaling it up massively during subsequent experimentation mapping:

1. **Constructing a Massive Synthetic Dataset**:
We extracted the individual GC-MS chromatograms of the 10 raw materials and combined them extensively (C(10,2) = 45 basic pairs). By iterating over **hundreds of mixture ratios**, and augmenting the data with **Retention Time (RT) drifts** and **Gaussian bottom-noise** to simulate real instrument conditions, we programmatically generated **tens of thousands of highly complex synthetic chromatogram samples** specifically to test the learning curves and model scaling limits.

2. **Model Selection (1D-CNN vs. Transformer)**:
- Formulated the industrial problem as a joint task of **Multi-label Classification and Regression**.
- Independently designed architectures based on **1D-CNN** and **Transformer (Self-Attention)**.
- To resolve the mathematical ambiguity caused by mixture permutations (which caused bugs in strict accuracy metrics during earlier phases), we proposed a custom **Order-invariant Loss**, which eliminated sequence interference and enabled robust convergence.
- **Results**: On the massive dataset, the baseline CNN model achieved a **99.44%** accuracy in blind component identification. Although the Transformer reached 99.86%, we selected the **CNN** as the core production baseline due to its tiny footprint (0.7MB) and blazing fast training/inference speed (5-10x faster than Transformer).

3. **End-to-End Demo System**:
Developed a full-stack Web application (Flask backend + UI Build). Business users can easily drag and drop raw .xlsx chromatogram files exported from instruments to instantly get model predictions for ingredients and mixing ratios.

### 🔬 Industrial Reflection: The Sim2Real Gap (Lessons Learned)

Although the algorithm reached near-perfect accuracy on tens of thousands of simulated records, we encountered a severe engineering hurdle when strictly validating against **Real Test Data** with physical chemical properties:

- **Sim-to-Real Domain Shift**: Linear combinations and programmatic interpolations failed to perfectly capture complex physical-chemical interferences in nature (e.g., co-elution effects, intermolecular interactions causing peak tailing and shifting). 
- **Prohibitive Data Acquisition Costs**: Fine-tuning the model in the real domain would require the enterprise to prepare tens of thousands of physically blended samples and run them through the GC-MS machine. The time and material costs far exceeded the project constraints. As confirmed by our ablation studies, blind generation of synthetic data yields diminishing returns; without high-quality genuine mixing data with proper m/z parameters, breaking through the performance wall is effectively impossible.

**Conclusion**:
The commercial deployment was subsequently put on hold. However, as an independent project, it provided profound insights into a classic AI industry dilemma: **Data-Centric AI is key; constructing high-quality datasets that align with physical reality is far more critical than fancy algorithm designs.** Going through the entire lifecycle—from requirements gathering and data synthesis to large-scale experimentation and hitting the real-world validation wall—has been an invaluable empirical lesson.

### 🧩 Tech Stack
- **Deep Learning**: PyTorch, 1D-CNN, Transformer, Custom Loss Algorithms
- **Data Engineering**: Pandas, NumPy, SciPy (Signal Processing & Massive Augmentation)
- **Deployment**: Python Flask, Javascript/HTML (Web UI)
