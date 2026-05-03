<div align="center">

# GC-MS Signal Deconvolution & Identification Engine
# GC-MS 色谱信号解耦与成分识别引擎

[**🇨🇳 中文版 (Chinese)**](#-中文版-chinese-version) | [**🇬🇧 English Version**](#-english-version)

</div>

---

## 🇨🇳 中文版 (Chinese Version)

**企业合作落地探索项目 · 工业级数据科学实践**

### 📖 项目背景
本项目源于一项真实的企业技术需求对接。合作企业需要对复杂的植物精油混合物进行成分拆解。由于气相色谱-质谱（GC-MS）数据经常出现信号重叠与干扰，依靠传统专家人工比对特征峰的分析方法效率极其低下。

**核心诉求**：构建一个基于深度学习的算法模型，直接输入 GC-MS 的一维色谱信号数据（RT-TIC 色谱图），实现端到端的“**混合物成分识别**”与“**核心原料混合配比预测**”。

### 💡 方案设计与大规模数据合成实验

由于工业界带有完美标签的真实混合数据难以低成本获取，本项目采用了算法工程中经典的“**仿真基线合成 (Synthetic Data Generation)**”路径，并在后期实验中**爆发式扩充了数据量级与搜索空间**：

1. **构建超过 7.4 万样本的超大规模合成物理数据集**：
- **从 10 种到 87 种**：最初阶段我们仅使用 **10 种** 基础原料进行验证，而在最终的大规模消融验证中，我们提取了多达 **87 种独立单体色谱图**。
- **全排列组合**：基于 87 种单体构建了所有两两配对组合（C(87,2) = 3741 种组合）。
- **比例与增强**：针对这 3741 种组合，我们系统性遍历了 10 种典型的浓度配比，叠加仪器的保留时间漂移（RT Drift）与随机高斯噪音（Gaussian Noise），最终通过代码**自动化生成了 74,820 条高度复杂的模拟混合色谱图大样本集**，以进行系统性的学习曲线（Learning Curve）压测。

2. **多模态与深度学习模型选型**：
- 将该工业问题正式定义为 **多标签分类 (Multi-label Classification) + 比例回归 (Regression)** 的联合学习任务。
- 独立设计并试验了基于 **1D-CNN** 和 **Transformer (Self-Attention)** 的光谱序列处理算法。
- 针对因混合组分排列引发的数学歧义，自研了 **Order-invariant Loss（顺序无关损失函数）**，打破了次序干扰的影响，使得模型精准收敛。
- **模型结论 (CNN vs Transformer)**：即便在 7.4 万大样本数据量下，CNN 基线模型依然表现非凡，成功将特征提取与盲识别跑通（成分盲盒识别在部分子集达 99.44%）。与此同时，尽管 Transformer 在理论上具有强大的拟合能力，但在此特定一维时序信号上的表现十分一般，没有带来任何实质提升，且训练和推断耗时极长。最终，权衡训练速度与模型体积（CNN仅不到 1MB 且快 5-10 倍），我们**果断弃用 Transformer，确立 CNN 为主力架构**。

3. **端到端演示系统 (Demo App)**：
搭建了完整的 Web 前端应用界面服务。业务人员仅需直接拖拽仪器导出的 .xlsx 原始色谱文件，系统即会一键输出成分预估和调配比例。

### 🔬 工业界实验反思：Sim2Real 鸿沟 (Lessons Learned)

尽管算法在本地 7 万级模拟合成数据（Synthetic Data）上取得了优异的表现，但在交付前夕对**带有真实物理特性的盲测混合数据（Real Test Data）**进行验证时，我们遭遇了严峻的工程挑战：

- **Sim-to-Real 的相关性断层**：因为程序线性叠加与插值生成的合成色谱图，无法完美表征自然界复杂的物理化学干扰（如物理共流出效应、分子间相互作用导致的拖尾与极性偏移）。真实数据的分布与合成模型数据存在不可忽视的“域偏移（Domain Shift）”。
- **高质量落地数据的获取沉没成本**：要跨越这层鸿沟进行基于真实域的数据微调（Fine-tuning），需要企业内部建立自动化流水线进行数以万次的精密调配、并实际通过机器测绘进样，高昂的数据获取门槛直接超出了落地预期。一如学习曲线与消融实验得出的确切证据：**当模型表现遭遇数据生成范式的物理瓶颈时，继续盲目增加粗糙的仿真组合数据（如扩大到 10 万以上）已经完全没有明显收益**。破局的唯一方式是补充高质量和带 m/z 信息的真实数据。

**项目总结**：
因为真实数据的匮乏，该落地合作最终遗憾搁置。但这成为我个人经历中一份极具价值的财富。它让我切身实地地体会到了真实工业人工智能落地中的经典箴言：**Data-centric AI，即高质量、贴近物理现实场景的数据，其决定性意义远远碾压单纯繁杂的算法架构堆砌**。这段涵盖“业务沟通、合成大规模数据、压栈实验”到最后“业务盲测碰壁并总结反思”的闭环经历，是我走向工业界路上至关重要的试错沉淀。

### 🧩 技术栈
- **算法与深度学习**: PyTorch, 1D-CNN, Transformer, 定制 Loss Function (Order-invariant loss)
- **数据工程**: Pandas, NumPy, SciPy (泛型特征级信号提取、增强、对齐与千万点运算)
- **部署与产品化**: Python Flask, Javascript/HTML (Web 交互)

---

## 🇬🇧 English Version

**Enterprise Collaboration & Industrial Data Science Practice**

### 📖 Project Background
This project originated from a technical collaboration with an enterprise. The goal was to deconstruct complex plant essential oil mixtures. Since Gas Chromatography-Mass Spectrometry (GC-MS) data often suffers from signal overlap and interference, traditional analysis relying on manual peak comparison by experts is heavily inefficient.

**Core Objective**: Build an end-to-end deep learning model that directly takes 1D GC-MS signal data (RT-TIC chromatograms) as input to perform **mixture component identification** and **mixture ratio regression**.

### 💡 Methodology & Large-Scale Data Synthesis Experiment

Since massive labeled "real-world" mixture data is extremely expensive to acquire in the industry, this project adopted a classic **Synthetic Data Generation** pipeline, scaling it up massively during subsequent experimentation:

1. **Constructing a Massive Synthetic Dataset of 74,000+ Samples**:
- **Scaling from 10 to 87 Inputs**: Early stage conceptualization started with just 10 ingredients, but for the final ablation benchmark, we pushed it to cover **87 independent raw single-component chromatograms**.
- **Complete Combinations**: We generated a complete matrix of pair-wise combinations resulting in **C(87,2) = 3741 unique material mix pairs**.
- **Scale and Augmentation**: Evaluating 10 specific mixing ratios for every combination, alongside augmenting with Retention Time (RT) drifts and Gaussian noise, the system programmatically generated **a staggering 74,820 deeply synthetic mixture chromatograms** to stretch the limits via deep learning curves.

2. **Model Selection (1D-CNN vs. Transformer)**:
- Formulated the industrial problem as a joint task of **Multi-label Classification and Regression**.
- Independently applied architectures based on **1D-CNN** and **Transformer (Self-Attention)**.
- To resolve the mathematical ambiguity caused by mixture permutations, we engineered a custom **Order-invariant Loss**, which eliminated sequence interference and enabled robust convergence.
- **Model Verdict (CNN vs. Transformer)**: The baseline CNN model handled feature extraction brilliantly and scaled efficiently across the 74K dataset. Conversely, although the Transformer model theoretically offers greater representation capacity, it performed poorly on this specific 1D sequential signal—bringing neither metric improvements nor reliability, while inflating training complexity exponentially. We decisively **abandoned the Transformer architecture** and established the lightweight (sub 1MB, 5x-10x faster) CNN as our primary workhorse.

3. **End-to-End Demo System**:
Developed a full-stack Web application (Flask backend + UI Build). Business users can easily drag and drop raw .xlsx chromatogram files exported from instruments to instantly output ingredients and mixing ratios predictions.

### 🔬 Industrial Reflection: The Sim2Real Gap (Lessons Learned)

Although the algorithm conquered the simulated 74K+ records seamlessly, we hit a severe engineering reality check when transitioning to validate on **blind Real Test Data**:

- **Sim-to-Real Domain Shift**: Linear physical combinations and interpolations essentially failed to mirror complex physical-chemical interference occurring in real labs, such as co-elutions or polarity shifting. Our synthetic dataset carried an unavoidable "Domain Shift".
- **Prohibitive Data Acquisition Costs**: Fine-tuning the model in the real domain would forcefully require setting up automation tracks to systematically run tens of thousands of liquid drops into GC machines—costs that completely derailed deployment estimations. The learning curve testing accurately revealed the stark warning: **Once a model plateaus physically on simulated data paradigms, inflating basic simulated pools further yields zero tangible gain.** Only high-grade, real-world data layered with full m/z spectra can break this ceiling.

**Conclusion**:
The corporate deployment ultimately stalled. However, this functioned as an incredible empirical lesson, grounding the core philosophy of **Data-Centric AI**: Having fundamentally sound, physics-aligned data trumps overly complex architectural flexing. The end-to-end journey traversing ideation, brute-forcing data scales, hitting physical hurdles, and rigorously analyzing failure boundaries stands as one of my most formative machine learning chapters.

### 🧩 Tech Stack
- **Deep Learning**: PyTorch, 1D-CNN, Transformer, Custom Order-invariant Loss Algorithm
- **Data Engineering**: Pandas, NumPy, SciPy (Signal Extractions, Large Array Mathematics)
- **Deployment**: Python Flask, Javascript/HTML (Web Application)
