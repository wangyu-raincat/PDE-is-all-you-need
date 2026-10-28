# 跨现象 PDE 发现基准：来源与适用范围

本轮采用 Rudy、Brunton、Proctor、Kutz 的 PDE-FIND 作者仓库中的三份公开数值解：Burgers、双孤子 KdV 和 Kuramoto–Sivashinsky（KS）。它们分别覆盖非线性输运与耗散、色散波、四阶不稳定性与混沌动力学。目标是从时空场发现方程项和系数，不是只比较温度预测误差。

**这三份数据都是公开数值仿真基准，不是真实传感器测量，也不构成实物实验验证。** 每份文件只有一条既有轨迹；时间留出能检验该轨迹上的后续一致性，不能代替新初值、新设备或新物理系统上的独立验证。它们也不能证明任意 PDE 都可唯一识别。

## 固定来源与校验

- 作者仓库：[snagcliffs/PDE-FIND](https://github.com/snagcliffs/PDE-FIND)，固定提交 `86911349b9fac82996e35465a13d06205730cc05`。
- 原论文：[Data-driven discovery of partial differential equations](https://doi.org/10.1126/sciadv.1602614)，Science Advances，2017，DOI `10.1126/sciadv.1602614`。
- 固定版本[论文 PDF](https://github.com/snagcliffs/PDE-FIND/blob/86911349b9fac82996e35465a13d06205730cc05/sci_adv.pdf)与[补充材料](https://github.com/snagcliffs/PDE-FIND/blob/86911349b9fac82996e35465a13d06205730cc05/supplement.pdf)。补充材料 Table 1 给出名义时空范围；3.1.1、3.2、3.5 分别说明三个算例。
- 元数据核验时间：2026-10-28 07:54:17 UTC。

| 数据 | 原始文件 | 字节数 | MAT 字段与原始场形状 | 论文名义范围 |
|---|---|---:|---|---|
| Burgers | `burgers.mat` | 376124 | `x`, `t`, `usol`: 256 × 101 | x ∈ [-8, 8]；t ∈ [0, 10] |
| KdV | `kdv.mat` | 1535359 | `x`, `t`, `usol`: 512 × 201 | x ∈ [-30, 30]；t ∈ [0, 20] |
| KS | `kuramoto_sivishinky.mat` | 1589304 | `x`, `tt`, `uu`: 1024 × 251 | x ∈ [0, 100]；t ∈ [0, 100] |

文件名中的 `sivishinky` 是上游拼写，下载器应原样使用。三个场都按“空间 × 时间”存储；适配器转为“时间 × 空间”，并展开一维坐标。所有变量的 MAT 类型都是 double。原始 x/t 数组端点尚未在来源研究阶段读取；上表是论文的名义范围，不应据此声称保存了两个周期端点，也不应替代实际坐标校验。不将这些无量纲基准坐标擅自转换成米、秒或温度。

精确不可变下载 URL、字节数、Git blob SHA-1、SHA-256、字段名、方向与形状保存在 [`source_manifest.json`](../pde_discovery/data/generic_pde/source_manifest.json)。下载的原始字节已与固定提交的 Git blob 验证一致；只使用 `scipy.io.whosmat` 检查变量名、形状与类型，未读取场值或坐标值，未运行作者代码。

上段“未读取”指来源研究阶段。随后首次数据准备按 v1 协议解码 Burgers 时，遇到不满足虚部舍入阈值的原始 `usol`，因此在任何拟合、开发评分或最终评分前停止。重新核对作者源码后，v2 将读取规则修正为作者定义的观察场：Burgers notebook 的 `cells[2]` 使用 `u = real(data['usol'])`；KdV notebook 的 `cells[3]` 使用 `U = real(data['usol'])`，均没有按虚部大小决定是否取实部。这里按 notebook 全部 cells 的零起始索引计数。KS 的 `cells[2]` 直接使用 `u = data['uu']`，不作实部提取。这是来源格式约定，不代表已证明虚部微小或具有何种物理意义。

初次失败及修正保存在 [`GENERIC_PROTOCOL_v1_PREPARE_FAILURE.md`](GENERIC_PROTOCOL_v1_PREPARE_FAILURE.md)、[`GENERIC_INGESTION_AMENDMENT.json`](GENERIC_INGESTION_AMENDMENT.json) 和 `artifacts/generic-prepare-v1-failure.log`。修正发生在训练前，只调整来源读取约定；候选算法、选择规则和评分参考系数不变。后续报告应保留这次准备失败，不能将 v1 描述成原样成功执行。

三个 blob SHA-1 分别为：

- Burgers：`7060b54f07fef5dabaaeadeef20c09334c06cb0e`
- KdV：`3d9cd081abc7c3967fbb2408afc59f3d4266b716`
- KS：`bb2ed7e1d61a4f9bc1b879d63fd5d1ad4dbb7ed2`

## 参考方程的证据与隔离

评分参考方程单独位于 [`generic_reference_equations.json`](generic_reference_equations.json)。只允许在全部模型与选型决定冻结后，由评分阶段读取；来源 manifest 不含真实项系数。识别器必须使用同一个预先确定的候选库，不能按文件名选择方程项或系数。研究者事先知道这些经典基准，不能宣称盲测未知自然定律。

Burgers 存在需要保留的来源差异：作者 notebook 的引言和补充材料 Table 3 的“Correct PDE”将扩散项写为单位系数，而该表识别出的干净数据系数为 0.100。官方 PySINDy 的[固定版本特征说明](https://github.com/dynamicslab/pysindy/blob/c4421fcec275c8f4cc5c1e93bebb961b212067ae/examples/1_feature_overview/example.py)明确把同一文件定义为扩散系数 0.1。我们核对了 PySINDy 固定提交 `c4421fcec275c8f4cc5c1e93bebb961b212067ae` 下 `examples/data/` 三个 MAT 文件的 Git blob，均与作者原文件完全相同。因此评分参考采用已在发现之前确定的 0.1，并记录原文不一致，而非按本次拟合结果反推答案。

KdV 的参考来自作者补充材料 Table 2 和[双孤子 notebook](https://github.com/snagcliffs/PDE-FIND/blob/86911349b9fac82996e35465a13d06205730cc05/Examples/TwoSolitonKDV.ipynb)。作者说明它由不同振幅、不同中心的双孤子初值经谱方法产生。单一行波可能同时满足平流方程和非线性波方程；这是轨迹可辨识性限制，不能用低预测误差排除。KS 的参考来自 Table 6 和[作者 KS notebook](https://github.com/snagcliffs/PDE-FIND/blob/86911349b9fac82996e35465a13d06205730cc05/Examples/KuraSiva.ipynb)，包含四阶导数，不能沿用热方程的二阶扩散系数必须为正的约束。

## 边界条件与数值闭合的证据范围

原作者仓库没有这三份 MAT 的生成程序。补充材料明确说明 KdV 和 KS 使用谱方法生成，但没有为这三份文件逐一明确给出全部边界条件。

- 官方 PySINDy 对字节完全相同的 Burgers 文件，在上述固定版本特征说明中使用 `periodic=True`；[官方 PDE-FIND 示例](https://pysindy.readthedocs.io/en/stable/examples/10_PDEFIND_examples/example.html)对 KS 同样明确使用周期差分。这支持复现其周期数值处理，不等于证明具体物理装置的边界。
- 对 KdV，已检查的作者文本和官方示例未明确声明周期边界。“谱方法”本身不足以唯一确定边界。因此若当前引擎采用周期闭合，必须标记为**数值实现假设**，并仅用发现段检查网格及边缘接续的一致性；不能为提高最终留出分数而更改边界或裁剪规则。
- 周期离散中首尾相邻网格点通常不等值；不能仅凭两端值不完全相同就声称非周期。有效周期长度应由实际均匀网格间距乘空间点数计算，并检测是否重复保存了端点。

若训练段不满足冻结的周期适用规则，应报告不适用或受限结果，不得偷看后续留出段之后修复实验设定。

## 权利与分发

检查时 GitHub repository API 的 license 字段为 null，仓库树未发现 LICENSE。这里没有为原始 MAT 推定许可，也不把 PySINDy 仓库的代码许可自动套用到作者数据。项目交付包含来源说明、校验清单和下载器，不打包原始 MAT 或作者论文。下载器直接从固定作者提交获取数据，并核验字节数与哈希。

## 本轮能支持的结论

优先报告项集合是否恢复、系数误差、替代方程及其选型间隔，再报告冻结后的时间留出残差或数值推进误差。对于噪声和未见初值，只有真正执行的实验才能给出结论。当前三份来源本身不支持“真实世界已验证”“可求任意 PDE 的解析解”或“已接入华为平台”等表述。
