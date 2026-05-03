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

由于工业界带有完美标签的真实混合数据难以低成本获取，本项目采用了算法工程中经典的“**仿真基线合成 (Synthetic Data Generation)**”路径，并在后期实验中**大规模突变扩充了数据量级与搜索空间**：

1. **构建近 7.5 万样本的超大规模合成物理数据集**：
- **从 10 种到 87 种**：最初阶段我们仅使用 **10 种** 基础原料进行验证，而在最终的大规模消融验证中，我们覆盖单体扩大到了近 9 倍，极高密度地提取了多达 **87 种独立产品单体色谱图**。
- **全排列组合**：基于 87 种单体，通过算法构建了它们之间所有的两两配对，产生了 **`C(87,2) = 3741` 种排列组合**的庞大解空间。
- **十等分比例与干扰增强**：针对这 3741 种组合，我们系统性遍历了每组固定 10 种经典的浓度配比（从 1:9 到 9:1），并叠加极其恶劣的保留时间漂移（RT Drift）与随机底噪增强。最终自动化推流算出了 **74,820 条高度仿真的混合色谱图测试大集**（总数据量近 7.5万）。

2. **多模态与深度学习模型选型**：
- 将该工业问题正式定义为 **多标签分类 (Multi-label Classification) + 比例回归 (Regression)** 的联合学习任务。
- 独立搭建了基于 **1D-CNN** 和 **Transformer** 的序列信号特征处理算法。
- 针对因混合组分排列引发的数学歧义，自研基于排列取 Min 的 **Order-invariant Loss（顺序无关损失函数）**，打破了次序干扰的影响。
- **模型验证结论（CNN 胜出）**：即便在逼近 7.5 万的庞大数据量下，用 CNN 训练出的模型依然表现非凡，成功跑通了特征提取（成分盲识别率在子集达 99%以上）。与此同时我们发现，最初被寄予厚望的 Transformer 在这种一维特征色谱信号上完全“不好使”——它非但没有带来准确率的实质提升，反而伴随几十倍的内存开销与极慢的训练时长。**因此，我们弃用了 Transformer，全量落地使用 CNN 作为主力模型。**

3. **端到端演示系统 (Demo UI)**：
搭建了完整的预估前端可视化看板系统。支持中英文界面一键切换。业务人员仅需拖拽设备生成的 `.xlsx` 文件，即可直接在线推理。

### 🔬 工业界实战反思：Sim2Real 鸿沟 (Lessons Learned)

尽管由于规模扩容，算法在本地 7 万级模拟合成数据上依然横扫千军，但在最后对**真实的工厂盲测混合数据（Real Test Data）**进行验证时，我们遭遇了严峻的壁垒：

- **Sim-to-Real 的相关性断层**：因为程序线性叠加与插值生成的 7 万多张合成色谱图，无法完美反应自然界真实的物理共流出或分子间相互作用导致的严重极性脱尾偏移。模型在盲推中撞上了“域偏移（Domain Shift）”。
- **越发停滞的工程 ROI**：想跨过这层鸿沟的最后希望是依靠企业去提供数万份真实微调样本，但极高的高昂仪器耗时直接封死了路线。正如我们在“学习曲线”和“消融对照实验”末尾中痛苦意识到的：**“当模型在现阶段物理瓶颈（即缺乏真实m/z信息）上表现停滞时，仅仅盲目去扩建更多的组合伪仿真粗糙数据，短期内已经毫无任何明显收益了”**。

**总结沉淀**：
虽然项目因现实落地困难而搁置，但它比任何顺遂的实验室 Demo 都令我受益匪浅：**这切身体现了“Data-centric AI”在工业界的决定性意义。贴近物理客观环境的高质量数据，其价值永远碾压在算法上空洞的堆砌。**这也将是我带入未来所有数据开发工作中最宝贵的第一手复盘经验。

### 🧩 核心技术栈
- **核心算法设计**: PyTorch, 1D-CNN, Transformer, Custom Loss Algorithms
- **大规模数据工程**: Pandas, NumPy, SciPy (对数万条大体积数组进行矩阵特征提取推流算力)
- **部署接口**: Python Flask, Javascript/HTML

---

## 🇬🇧 English Version

**Enterprise AI Implementation Exploration · Industrial Data Science Practice**

### 📖 Project Background
The cooperative company needed to deconstruct the formulas of complex plant essential mixed oils. Because direct GC-MS signal outputs endure harsh peak-overlappings, manual comparison operations run at extremely inefficient capacity.
**Core Objective**: Build a deep learning pipeline absorbing 1-dimensional GC-MS (RT-TIC) data directly, achieving End-to-End **formula ingredient identification** and **proportion ratio regression**.

### 💡 Synthetic Scaled Big Data Experimentation

1. **Expanding the Data Limit to 74,820 Massive Synthetic Records**:
  - **A Jump from 10 to 87 Target Units**: Initially starting with 10 raw materials, we radically expanded vertically to extract **87 uniquely independent single-component targets**.
  - **The Pairwise Combination Matrix**: Formulated across 87 samples, we mapped completely crossing combinations totaling **`C(87,2) = 3741` fundamental pairs**.
  - **Ratios and Augmentations**: Multiplying these 3741 pairs by 10 systematic ratio slices each, accompanied by Retention Time Drifting and noise variations, the engine crashed out **74,820 synthesized chromatography test records**.

2. **Model Battle (CNN > Transformer)**:
  - Formulated as **Multi-label Classification + Regression**.
  - Deployed both lightweight **1D-CNN** architectures and deeply massive **Transformer** clusters independently.
  - Engineered an **Order-invariant Loss** function to clear mathematical ambiguities raised by formula order inversions.
  - **The CNN Definitively Won**: CNN efficiently dominated the 74K mega dataset running blindly past the 99% precision mark seamlessly. Conversely, the supposedly sophisticated Transformer model performed terribly under such 1D series signals. Offering drastically poorer optimization capability whilst bleeding processing power and draining train speeds to an unacceptable crawl, **we explicitly rejected Transformers to anchor purely on CNN.**

### 🔬 The Sim2Real Domain Void Wall 

Although effortlessly conquering the 74,000+ local proxy dataset, colliding with the **Real Test Data** hit a harsh reality wall:
- **Sim-to-Real Shift**: Code aggregations fail to encapsulate molecular co-eluting overlapping drifts naturally produced within real chemical machinery.
- **ROI Stagnation**: As our \"Ablation vs Learning Curve\" stats severely summarized, blindly generating more synthetic proxies post-bottleneck yields literally \"Zero Tangible Returns.\" Fine-tuning realistically needed actual real samples which was excessively costly for the client.

**Concluding Thought**:
While the enterprise deployment stalled directly due to insufficient true data collection capabilities, it provided something priceless. The principle of **Data-Centric AI** — realizing that acquiring physically exact information fundamentally outpaces blind algorithmic scale pushing.

