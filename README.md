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
- **比例与干扰增强**：针对这 3741 种组合，我们系统性遍历每组 10 种浓度配比固定值，叠加保留时间漂移（RT Drift）与底噪加强（Gaussian Noise）。通过代码**自动化生成了 74,820 条高度复杂的模拟混合色谱测试集（近7.5万大集规模）**，以进行系统性的学习曲线（Learning Curve）测试。

2. **多模态与深度学习模型选型**：
- 将该工业问题正式定义为 **多标签分类 (Multi-label Classification) + 比例回归 (Regression)** 的联合学习任务。
- 独立设计搭建了基于 **1D-CNN** 和 **Transformer** 的序列信号特征处理算法。
- 针对因混合组分排列引发的数学歧义，自研了基于排列取 Min 的 **Order-invariant Loss（顺序无关损失函数）**，打破次序干扰影响。
- **模型验证结论（确立 CNN 主力）**：即便在逼近 7.5 万的庞大数据量下，CNN 基线模型依然表现非凡，成功跑通特征提取并使成分识别率达标。与此同时，我们发现在其他领域性能出众的 Transformer，在这个项目中表现十分差劲——在一维特征色谱信号上它的表征甚至不如 CNN，准确率没有任何跨越突破，且伴随极其昂贵的内存开销与数倍延长的训练耗时。最终**果断弃用 Transformer，确立 CNN 为落地生产模型**。

3. **端到端演示系统 (Demo UI)**：
搭建了完整的 Web 前端应用界面服务，支持文档中英文通过页面超链接实现一键平滑跳转。业务人员仅需拖拽 `.xlsx` 提取特征即可直出推理结果。

### 🔬 工业界反思：Sim2Real 鸿沟 (Lessons Learned)

尽管算法最终在本地 7 万张粗略合成混合图上碾压全场，但在实际交付前夕对**带有真实工厂测绘环境特性盲测集（Real Test Data）**进行验证时，我们遭遇了壁垒：

- **Sim-to-Real 的相关性鸿沟**：程序线性叠加与插值生成的极其规整的这七万余张合成信号图，无法表征自然界多分子共流出相互包裹导致的光谱拖尾与漂移。两批数据间存在直观的 “域偏移（Domain Shift）”。
- **陷入停顿瓶颈圈的数据重构ROI**：要想填补这层微调，需要业务端自行搭建一条自动化微量试剂测绘流水线产出几万份真实的带有 m/z 质量通道的打靶数据，极高的数据与时间成本终结了部署计划。印证了我们在“对照实验”结尾总结到的一样：**模型已经卡死在数据生产范式的客观瓶颈下，单纯继续无节制地生成更多的这类物理规律粗糙的二维组合仿真图（如增加至十多万），在真实指标上实质将永远等于零收益！必须引入 m/z 物理通道。**

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
- **Ten Levels and Stacking Interference**: Evaluating 10 specific mixing scale variations each upon these 3741 matrix outputs, actively combined to simulate rough lab machinery Retention Time Drift and severe base-noises. The engine processed natively generating **74,820 synthesized GC-MS array records** pushing model learning curves.

2. **Model Selection Battles (1D-CNN vs. Transformer)**:
- Engineered models using both sequence-efficient **1D-CNN** arrays against heavily attention-dependent **Transformer** setups.
- Constructed a tailored **Order-invariant Loss Function** defeating algebraic ambiguity introduced through raw ingredient arrays shuffling correctly.
- **The Verdict (CNN Destroyed Transformer)**: Proving incredibly robust, CNN completely tamed the overwhelming 74,800+ dataset achieving near 99% blind accuracy metrics easily on isolated tracks. Transformer, on the opposite end, completely sank executing 1-dimensional raw GC-MS records providing ZERO true performance breakthrough compared to baseline architectures while simultaneously draining unacceptable magnitudes of server hardware speed/footprints. We effectively **terminated Transformers adopting sole implementation around lightweight CNNs**.

3. **End-to-End Demo System**:
Fully integrated responsive application dash featuring clickable anchoring bilingual language adaptations, accepting straightforward `.xlsx` Drag-And-Drop user interactions.

### 🔬 Unlocking Sim2Real Walls: Core Lessons Learned

Dominating simulated databases sadly stalled against pure absolute **Real Test Deployments** testing strict blind chemistry constraints:

- **The Sim-to-Real Shift Gap**: 70,000 basic algorithm interpolations naturally missed capturing severe compound co-elution trailing distortions formed by real chemical fusions. The synthetics inherently hosted clear Domain Shift deviations.
- **The ROI Deadlock Limit**: Jumping the fine-tune gap realistically required automated high-volume liquid physical test drops generating thousands of new genuine real-world recordings, financially breaking deployment schedules fundamentally. Mirroring perfectly what internal ablation testing forewarned: **Hitting objective algorithmic capacity limits using limited physics representations makes continuing massive mock-synthetic regenerations fundamentally mathematically pointless unless bringing real absolute m/z channel datasets.**

**Project Conclusion**:
Although formally halted by unachievable reality physical resource demands, personally encountering the brutal validation of **Data-Centric AI** limits heavily reframed my engineering methodology over strictly trying harder to carve fancier algorithms locally off paper.
