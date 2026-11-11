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

由于工业界带有完美标签的真实混合数据难以低成本获取，本项目采用了算法工程中经典的“**仿真基线合成 (Synthetic Data Generation)**”路径，并在后期实验中**大规模爆发式扩充了数据量级与搜索空间**：

1. **构建覆盖 74,820 样本的超大规模合成数据**：
- **从 10 种暴增到 87 种**：最初阶段我们仅使用 **10 种** 基础原料，而在真正的大规模消融验证中，我们将基本盘直接拉高至对 **87 种独立单体色谱图**。
- **全排列组合**：算法遍历这 87 种单体构建了所有两两配对组合，产生了 **`C(87,2) = 3741` 种排列配置**。
- **比例与干扰增强**：针对这 3741 种组合，使用每组 10 种固定配比、每个配比生成 2 种顺序，共生成 **74,820 条模拟混合样本**。实验还对保留时间漂移（RT Drift）和高斯噪声增强进行了比较；74,820 是样本总量，并非全部为测试集。

2. **多模态与深度学习模型选型**：
- 将该工业问题正式定义为 **多标签分类 (Multi-label Classification) + 比例回归 (Regression)** 的联合学习任务。
- 独立设计搭建了基于 **1D-CNN** 和 **Transformer** 的序列信号特征处理算法。
- 针对因混合组分排列引发的数学歧义，自研了基于排列取 Min 的 **Order-invariant Loss（顺序无关损失函数）**，打破次序干扰影响。
- **模型对比**：在一组早期合成数据对照中，CNN 与 CNN+Transformer 的顺序无关成分组合识别率分别为 **99.44% 和 99.86%**，比例 MAE 分别为 0.203 和 0.202。该阶段使用单一随机划分，日志指出评估集参与了早停，不能视为独立测试。Transformer 训练更慢（日志记录 CPU 上约 5–10 倍），未在该对照中带来明显的比例预测改善，因此后续演示采用 CNN。这些结果不代表真实样本的识别准确率或生产部署效果。
- **87 类合成数据结果**：本地训练摘要记录，无增强 CNN 的顺序无关成分对识别率为 **88.96%**，强增强模型为 **29.00%**。两者均来自合成数据的一次划分，评估集也用于选择最佳模型，不能视为独立盲测。增强在划分之前施加，两次评估数据的扰动条件不同，不能把成绩差异直接归因于训练增强。

3. **端到端演示系统 (Demo UI)**：
搭建了完整的 Web 前端应用界面服务，支持文档中英文通过页面超链接实现一键平滑跳转。业务人员仅需拖拽 `.xlsx` 提取特征即可直出推理结果。

### 🔬 工业界反思：Sim2Real 鸿沟 (Lessons Learned)

合成数据实验取得了较高的组合识别率，但在后续真实混合样本测试中遇到了明显困难：

- **Sim-to-Real 的相关性鸿沟**：程序线性叠加与插值生成的极其规整的这七万余张合成信号图，无法表征自然界多分子共流出相互包裹导致的光谱拖尾与漂移。两批数据间存在直观的 “域偏移（Domain Shift）”。
- **真实样本验证的局限**：日志记录的 3 份真实混合样本、4 个模型共 12 次预测中，仅 1 次 Top-1 命中一个真实成分，没有完整识别出两个成分。更强的增强在这 3 份样本上也未改善完整识别。真实数据微调和引入 m/z 信息是值得验证的方向；现有实验不能证明所需样本数量，也不能证明继续增加合成数据或其他方法必然无效。

**总结反思**：
因为现实物理数据壁垒被迫归档停止，这构成了我走向人工智能一线极具痛点的案例背书，直观展示了在工业生产应用中，贯彻 **Data-centric AI**（拥抱真正来自业务物理环境的高质量数据），带来的作用远非枯燥敲击代码的闭门造车堆砌架构模型所能奢求的。

### 🧩 技术栈
- **算法架构**: PyTorch, 1D CNN, 定制次序损失算法损失模型
- **数据工程**: Pandas, NumPy, SciPy (超大型百万特征时序流向量化处理与合并操作)
- **部署产品化**: Python/Flask 及前端展示

---

## 🇬🇧 English Version

**Enterprise Collaboration & Industrial Data Science Practice**

### 📖 Project Background
The collaborating industrial enterprise needed to deconstruct the formulas of complex plant essential mixed oils. Because direct GC-MS signal outputs endure harsh peak-overlappings and interactions, traditional manual comparisons perform extremely poorly.
**Core Objective**: Build a deep learning pipeline absorbing 1-dimensional GC-MS (RT-TIC) chromatogram data directly, achieving End-to-End **ingredient composition identification** paired with **mixture proportion breakdown regression**.

### 💡 Massive Synthetic Scale-Up Experimentation

Since procuring physically authentic lab data with clean percentage labels operates at devastatingly high resource costs, we adopted an automated **Synthetic Generation Generation Pipeline**:

1. **Expanding Scale to Over 74,000 Giant Physical Samples**:
- **87 Total Monomer Foundations**: Moving far past the initially tiny 10 starting variables, our final ablation validations imported signals from exactly **87 completely individual monomer bases**.
- **Combinational Expansion Matrix**: We structured programmatic pairing across all samples generating the **`C(87,2) = 3741` full pairing configuration scope**.
- **Mixing ratios and augmentation**: The 3,741 pairs were combined at 10 fixed ratios in two orders, producing **74,820 synthetic samples in total**, not 74,820 test samples. Separate experiments compared retention-time drift and Gaussian-noise augmentation.

2. **Model Selection Battles (1D-CNN vs. Transformer)**:
- Engineered models using both sequence-efficient **1D-CNN** arrays against heavily attention-dependent **Transformer** setups.
- Constructed a tailored **Order-invariant Loss Function** defeating algebraic ambiguity introduced through raw ingredient arrays shuffling correctly.
- **Model comparison**: In an earlier synthetic-data comparison, CNN and CNN+Transformer achieved 99.44% and 99.86% order-invariant pair accuracy, respectively, with ratio MAE of 0.203 and 0.202. This stage used a single random split; the log notes that the evaluation set was also used for early stopping and was therefore not an independent test set. The Transformer trained more slowly (reported as roughly 5–10 times slower on CPU) without a meaningful improvement in ratio prediction in this comparison, so CNN was used for the subsequent demo. These figures do not establish real-sample accuracy or production deployment.
- **87-class synthetic results**: Local training summaries report **88.96%** order-invariant pair accuracy for the unaugmented CNN and **29.00%** for the stronger-augmentation run. Each uses a single synthetic-data split that also served for model selection, not an independent blind test. Augmentation was applied before splitting, so the evaluation inputs differ and the score difference cannot be attributed solely to training augmentation.

3. **End-to-End Demo System**:
Fully integrated responsive application dash featuring clickable anchoring bilingual language adaptations, accepting straightforward `.xlsx` Drag-And-Drop user interactions.

### 🔬 Unlocking Sim2Real Walls: Core Lessons Learned

Strong synthetic-data pair recognition did not carry over to the recorded real-mixture tests:

- **The Sim-to-Real Shift Gap**: 70,000 basic algorithm interpolations naturally missed capturing severe compound co-elution trailing distortions formed by real chemical fusions. The synthetics inherently hosted clear Domain Shift deviations.
- **Limits of real-sample validation**: Across 12 predictions from four models on three real mixtures, only one prediction identified one true ingredient at Top-1; none identified both. Stronger augmentation did not improve complete pair identification on these three samples. Fine-tuning with real mixtures and incorporating m/z information remain possible directions, but the experiments do not establish a required sample count or prove that more synthetic data or other approaches cannot help.

**Project Conclusion**:
Although formally halted by unachievable reality physical resource demands, personally encountering the brutal validation of **Data-Centric AI** limits heavily reframed my engineering methodology over strictly trying harder to carve fancier algorithms locally off paper.
