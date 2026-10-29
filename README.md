> 本实验分支已完成系数可靠性对照：最坏逐项系数误差24.61%→6.10%，但新数据上四组最坏平均预测误差均5.41%，门槛125/126，未合入。请先读 [实验报告](coefficient_study/REPORT.md) 和 [运行说明](COEFFICIENT_README.md)。回退分支为 `experiment/prefix-five-percent`；下文保留既有应用说明。

# 方程发现实验室 · PDE Discovery Lab

从时空观测数据 `u(x,t)` 中识别 PDE 微分项和系数的可运行科研原型。当前主线使用共同九项候选库与 L1 稀疏回归，在 Burgers、KdV、Kuramoto–Sivashinsky（KS）三个外部公开数值基准上验证发现能力：优先评价方程结构和系数，另行评价未来预测。

本项目包含 **数值策略控制器、L1 稀疏识别、验证和中文演示界面**，另提供热杆物理先验下的约束轨迹拟合方法作比较。当前没有调用大模型，也没有连接华为 AgentArts、Nexent、MindSpore 或昇腾；已提供可用于后续工具接入的 HTTP API 和 OpenAPI 描述。它不是任意 PDE 的发现器，也不寻找解析解。

最新冻结实验的**严格结构恢复为 0/3**：三个模型都找到全部参考项，但各自保留了多余项。不能按最终结果删去小系数，也不能把较低的预测误差算作方程恢复成功。这三个案例都是数值仿真，每个按同一原始轨迹的时间 60% / 20% / 20% 划分，不是真实测量或新初值独立实验。完整结论见 [通用发现报告](experiments/GENERIC_REPORT.md)。

## 快速运行

建议 Python 3.12；声明支持 Python 3.10 及以上。进入解压后的项目目录：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pde_discovery serve --port 8765
```

Windows 激活命令为 `.venv\Scripts\activate`。打开 **http://localhost:8765/discovery** 查看最新跨方程发现结果；根路径 **http://localhost:8765** 保留原七项合成/上传演示。默认仅绑定本机；无需 API Key，无需 GPU。也可用 `pip install .` 安装本项目，静态页面会随包安装。

主运行包包含代码、已有 Mendeley/ATMC CSV、报告、冻结模型和最新发现页所需派生图，可直接启动查看 `/discovery`。历史科学 PNG 另放补充包，按需解压到相同父目录并合并 `pde-discovery` 文件夹；此前保存 31 份 `.npz` 的历史数值轨迹补充包保持不变。独立热杆教学实验原始 TXT、新增 PDE 基准原始 MAT，以及这些来源的 prepared/观测 NPZ 均不分发，复现时从固定版本来源下载。查看已保存的最新发现结果不需要下载数据。具体归档范围见 `DELIVERY_NOTES.txt`。

已验证环境：Python 3.12.14、NumPy 2.3.5、SciPy 1.17.0、scikit-learn 1.8.0、FastAPI 0.141.1、Uvicorn 0.52.1。

复现实验优先使用 Python 3.12，并以 `python -m pip install -r requirements-repro.txt` 安装本次实际使用的固定版本。不同平台或数值库实现仍可能产生浮点末位差异。

## 当前主线：共同九项 L1 跨方程发现

打开 **http://localhost:8765/discovery** 查看三个完整案例、发现方程、逐项系数比较、结构指标、预测曲线和 JSON 导出。页面只读取保存结果，不重新训练或评分。

共同候选库是 `1, u, u², u³, u_x, u_xx, u_xxx, u_xxxx, u·u_x`，左侧固定为 `u_t`；任何右侧项都不是必选项。该引擎允许由四阶项稳定的负二阶系数，不沿用热传导分支“扩散必选且为正”的约束。它支持一维标量、常系数、均匀周期网格，不能处理任意 PDE。

| 公开数值基准 | 严格支持恢复 | 多余项 | 支持 recall | 系数向量相对 L2 误差 | 最终预测相对 L2 误差（次要） |
|---|---|---|---:|---:|---:|
| Burgers | 失败 | `u³, u_x` | 100% | 0.000372926% | 0.0000853051% |
| KdV | 失败 | `u_x` | 100% | 3.819691% | 0.649614% |
| KS | 失败 | `u³` | 100% | 0.001202108% | 0.003605936% |

系数误差包含额外项，依赖作者坐标和场的缩放。三个选中方程都能完成最终自由推进并优于持久基线，但完整候选池也都没有生成精确参考支持。本轮暴露的是候选生成与稀疏选择仍需改进，不能靠事后删小项改写为 3/3 成功。完整方程、全部六个候选和 KS 推进失败候选均在报告中保留。

训练采用统一空间投影、局部时间积分、LASSO 路径与最小二乘去偏；开发段按自由推进误差及固定复杂度规则选择模型。三个模型全部冻结后才对最终角色文件和参考方程评分。最终预测只输入该段首帧和坐标，不输入未来场或逐时边界。准备步骤会机械解码完整 MAT 后按预定行号分割，不宣称最终值从未进入准备程序内存。

来源固定为 [PDE-FIND](https://github.com/snagcliffs/PDE-FIND) 提交 `86911349b9fac82996e35465a13d06205730cc05`。三个数据都是公开数值解，不能与后面的真实热杆记录混合计算成功率。Burgers 原文的扩散系数书写存在不一致，评分参考依据字节相同的官方 PySINDy 文件交叉确认；KdV 周期闭合是明确声明的数值假设。首次 Burgers 导入曾因虚部阈值停止，v2 在任何拟合和评分前按作者 `real(usol)` 约定修正，原失败保留。

- [完整报告与冻结指标](experiments/GENERIC_REPORT.md)
- [固定实验协议](experiments/GENERIC_PROTOCOL.md)
- [来源、字段、许可与边界证据](experiments/GENERIC_SOURCE.md)
- [训练前来源格式修订](experiments/GENERIC_INGESTION_AMENDMENT.json)及[原失败协议](experiments/GENERIC_PROTOCOL_v1_PREPARE_FAILURE.md)

用独立输出目录复现，保留首次冻结结果：

```bash
# 下载固定提交的原始 MAT，核验字节数与哈希；不训练
python -m pde_discovery generic-benchmark --phase download
# 按固定规则解码、检查并生成训练/开发/最终三份角色文件；不训练
python -m pde_discovery generic-benchmark --phase prepare
# 只使用训练和开发段，冻结全部三个模型
python -m pde_discovery generic-benchmark --phase freeze --output artifacts/generic_pde_reproduction
# 核对全部冻结记录，再评分；不重新选模
python -m pde_discovery generic-benchmark --phase evaluate --output artifacts/generic_pde_reproduction
```

下载和准备使用 `pde_discovery/data/generic_pde/` 私有缓存；`--output` 指定的是冻结与结果目录，不改变数据缓存。完整流程也可用 `python -m pde_discovery generic-benchmark --phase all --output artifacts/generic_pde_reproduction`。页面默认读取 `artifacts/generic_pde` 首次归档。最终段已在本轮查看，后续重跑是复现；下一轮方法改进需要新的独立验证证据。

## 历史热杆实测分支

以下保留既有热杆工作及准确结果，均与当前九项发现分支分开。Mendeley 原始数据有 **9 组完整重复实验**：原 L1 方法有 4 组条件预测相对误差低于 5%；物理约束轨迹拟合在同一已查看记录上为 **9/9 低于 5%，7/9 误差下降**。这是回顾性比较，不是新盲验，也未确立唯一真实 PDE，见 [改进报告](experiments/REFINEMENT_REPORT.md)、[原 L1 实测报告](experiments/MENDELEY_REPORT.md)。

四根增材 304L 杆的 **12 条预定留出记录**条件预测误差为 **1.055%–4.279%，12/12 低于 5%**，全部优于两种简单基线。每根杆分别用自己的记录训练，仍需全程实测两端边界，不能称旧参数跨材料直接迁移或唯一物性认证。报告见 [新增记录验证](experiments/ADDITIVE_REPORT.md) 和 [可辨识性诊断](experiments/IDENTIFIABILITY_REPORT.md)。

另一个公开教学来源的铝、黄铜高电压留出记录，四内点 RMSE 分别为 **0.08670 K、0.04841 K**，相对原 L1 降低 **92.23%、94.43%**。两组均优于持久值和边界线性插值的总体 RMSE，但仍有多个近优结构，不能证明唯一 PDE，见 [独立来源报告](experiments/EXTERNAL_LAB_REPORT.md)。

## 历史独立数据集与高电压工况验证

打开 **http://localhost:8765/external** 查看独立来源的铝、黄铜热杆测试和新旧方法比较。来源为 [Heat-Conduction-Aerospace-Lab](https://github.com/TariqSawaqed1/Heat-Conduction-Aerospace-Lab) 固定提交 `aba5a8c3c1d52fb879710cdddcfb23d29b079f03`，作者标注为 ASEN 3802 教学实验。它与此前 Mendeley/ATMC 来源不同，未获得独立仪器校准认证。

每种材料固定用低电压记录前 60% 训练、后 40% 开发选模，另一整条高电压记录留出。只用 CH1–CH6 六个原始传感器，间距 0.0127 m；CH1/CH6 全程实测温度作边界，预测 CH2–CH5。两种方法、两组模型全部冻结后才解析高电压测试。训练首帧六点平均作为统一温度参考值，它不是实测环境温度。每种材料重新识别系数，不是旧系数直接迁移。

新方法使用原有四种物理先验支持及原数值引擎；对照使用原有 L1 路径与开发选模规则。主要指标是四内点 RMSE（K），另报告按训练参考中心化的相对 L2、持久值和同边界线性插值基线。不能直接把该相对误差与旧数据集的环境温升百分比作物性结论。来源、事先协议和完整结果分别见 [来源说明](experiments/EXTERNAL_LAB_SOURCE.md)、[固定协议](experiments/EXTERNAL_LAB_PROTOCOL.md)、[实验报告](experiments/EXTERNAL_LAB_REPORT.md)。

| 高电压留出记录 | 新模型 RMSE K | 原 L1 RMSE K | 同边界线性基线 RMSE K | 相对 L1 降低 | 新模型中心化相对 L2 |
|---|---:|---:|---:|---:|---:|
| 铝 28 V | 0.086700 | 1.116056 | 0.201097 | 92.23% | 1.7757% |
| 黄铜 29 V | 0.048409 | 0.868416 | 0.067512 | 94.43% | 0.5682% |

铝选择扩散加常数项，但有三个近优结构；黄铜选择纯扩散，但四个预设结构都近优。所有预定测试均报告。总体 RMSE 改善不代表每个传感器都更好，逐测点比较与参数歧义见报告。

```bash
# 只下载四份固定提交的数据，校验 Git blob SHA-1 并保存 SHA-256；不训练
python -m pde_discovery external-benchmark --phase download
# 用独立目录复现，保留首次计算归档
python -m pde_discovery external-benchmark --phase freeze --output artifacts/external_lab_reproduction
python -m pde_discovery external-benchmark --phase evaluate --output artifacts/external_lab_reproduction
```

`--phase all` 依次下载、冻结和评估；默认输出 `artifacts/external_lab`，界面读取该默认目录。两条高电压结果都是条件预测，不能据此证明通用跨装置泛化或唯一真实 PDE。首次评分之后的重跑只是复现。

## 新增留出记录验证与参数诊断

打开 **http://localhost:8765/validation** 查看本轮留出结果和旧训练数据的参数诊断。原始 CSV、清单、固定模型、完整分数及曲线随交付保存，运行不需要网络、GPU 或 API Key。

四组 `xn/xp/yn/yp` 是四根增材制造 304L 杆；作者文献支持每根杆五次测试。每组固定记录 1 训练、2 开发选模、3–5 留出，四组模型全部冻结后才读取任一留出数值。逐次风速、加热与重置条件未完全核实；同杆重复不能视为 12 个独立新装置。使用物理先验模型的配置与上一轮一致，不按新留出成绩调整。

| 杆组 | 三条留出相对 L2 范围 | 四内点 RMSE 范围 K | 预设模型 |
|---|---:|---:|---|
| xn | 2.966%–4.279% | 0.468–0.709 | 扩散 + 散热 + 常数 |
| xp | 1.055%–1.493% | 0.191–0.251 | 扩散 + 散热 + 常数 |
| yn | 2.216%–2.778% | 0.363–0.493 | 扩散 + 散热 + 常数 |
| yp | 1.912%–2.191% | 0.290–0.332 | 扩散 + 散热 + 常数 |

```bash
python -m pde_discovery additive-benchmark --phase freeze --output artifacts/additive_heat
python -m pde_discovery additive-benchmark --phase evaluate --output artifacts/additive_heat
python -m pde_discovery identifiability --output artifacts/identifiability
```

前两条也可合为 `python -m pde_discovery additive-benchmark --phase all`。复现原结果可另指定输出目录，保留归档；默认界面读取上面列出的默认目录。记录被本轮评价后已经查看，后续复现不产生新的盲验。详细来源与角色见 [ADDITIVE_PROTOCOL.md](experiments/ADDITIVE_PROTOCOL.md) 和 [元数据核查](experiments/UNUSED_DATA_METADATA.md)。

参数诊断只读取旧九组 repeat1：固定九点扩散系数网格，共 81 个剖面点；五块训练损失分别删除，共 45 次重拟合。删块时仍从原始初值沿完整边界和时间轴传播，不重新初始化。100ss 的离散近优扩散系数跨度约 3.97 倍，删块估计跨度约 4.82 倍，说明预测误差低仍可能伴随参数敏感性。**这些范围是固定容忍度下的描述性诊断，不是统计置信区间。**诊断没有反过来调整新增材模型。

## 原 L1 实测训练与结果

打开 **http://localhost:8765/real** 查看九组结果与逐组曲线。原始数据、固定清单、训练候选、冻结模型和评分随包保存；复现计算不需要联网、GPU 或 API Key。

```bash
python -m pde_discovery measured-benchmark --output artifacts/mendeley_heat
python -m pde_discovery measured-benchmark --case 50ss --output artifacts/mendeley_heat
```

| 材料 | 温度档位 °C | 最终条件预测误差 | 四内点 RMSE K | 保留项数 |
|---|---:|---:|---:|---:|
| 不锈钢 | 50 | 48.816% | 3.113 | 2 |
| 不锈钢 | 75 | 29.852% | 4.587 | 3 |
| 不锈钢 | 100 | 30.724% | 6.628 | 3 |
| 铝 | 50 | 1.247% | 0.203 | 5 |
| 铝 | 75 | 1.056% | 0.311 | 5 |
| 铝 | 100 | 1.905% | 0.752 | 5 |
| 铜 | 50 | 4.860% | 0.680 | 3 |
| 铜 | 75 | 58.151% | 14.247 | 1 |
| 铜 | 100 | 58.567% | 21.153 | 1 |

这些指标使用两个端点的完整实测温升作为边界输入，仅预测另外四个内点；首帧初始化不计分。它们不是未知未来边界下的自由预测，也不是“真实 PDE 发现成功率”。铝杆三组虽误差低，却保留全部五项；铜杆 50°C 的候选没有扩散项；不锈钢及其余铜杆未达到参考误差目标。详细候选与物理解释限制见 [诊断](experiments/MENDELEY_DIAGNOSTICS.md)。

本路径使用六测点非周期原网格、五项候选库及时间积分 L1；与下面的周期仿真上传接口分开。数据 DOI：[10.17632/sbcf36976g.1](https://data.mendeley.com/datasets/sbcf36976g/1)，CC BY 4.0，作者 Lauren Tomanek、Daniel Stutts。输入包由用户提供，本任务校验 ZIP CRC、本地文件哈希并核实作者代码中的几何与单位；没有独立比对远端每个文件的哈希。

先前 `al300test.csv` 样例与单记录时间留出结果保存在 `artifacts/real_heat`，可用 `python -m pde_discovery real-experiment` 复现；不与本次独立重复实验混合。

## 物理约束轨迹拟合改进

打开 **http://localhost:8765/refined** 查看新旧方法逐组比较，完整证据见 [改进报告](experiments/REFINEMENT_REPORT.md)。此分支使用 `theta_t = D theta_xx - h theta + c`，在拟合时约束 `D > 0`、`h >= 0`。扩散必选，散热与常数项各自可选，共四种预设结构。它是**有热传导先验的结构比较和参数识别**，不是 L1 优化，也不能称作从任意候选库发现真实定律；原 L1 实现与结果保持归档。

直接最小化原始四内点温度轨迹的预测误差，可避免将含噪空间差分直接作为回归特征。求解器对六测点线性半离散系统进行模态分解，在原始时间戳之间使用分段线性边界精确传播；不会增加空间观测点。物理限制、优化初值与界限均记录在冻结文件中。

```bash
python -m pde_discovery refined-benchmark --output artifacts/refined_heat
```

仍由每组 repeat1 拟合、repeat2 选模，先将九组模型全部冻结再评分 repeat3。**由于上一轮已经查看过 repeat3，本轮是回顾性方法比较，不是新的盲验。**改进代码不使用 repeat3 重新拟合或更换候选，但不能据此消除既往查看测试结果带来的方法开发影响。完整协议见 [REFINEMENT_PROTOCOL.md](experiments/REFINEMENT_PROTOCOL.md)。

所有预测均以全程实测两端边界和四个内点初始值为输入。低误差、少量项、正扩散或局部敏感性满秩，都不能单独证明已恢复唯一 PDE 或测得真实材料参数。新结果单独保存在 `artifacts/refined_heat`，不覆盖 `artifacts/mendeley_heat`。

| 工况 | 原 L1 误差 | 改进方法误差 | 原项数 → 改进项数 |
|---|---:|---:|---:|
| 50ss | 48.816% | 4.919% | 2 → 3 |
| 75ss | 29.852% | 4.633% | 3 → 3 |
| 100ss | 30.724% | 4.660% | 3 → 3 |
| 50al | 1.247% | 1.259% | 5 → 2 |
| 75al | 1.056% | 1.236% | 5 → 2 |
| 100al | 1.905% | 1.415% | 5 → 2 |
| 50cu | 4.860% | 4.445% | 3 → 2 |
| 75cu | 58.151% | 4.337% | 1 → 2 |
| 100cu | 58.567% | 4.285% | 1 → 2 |

50al 和 75al 的误差略升，换来较少的预设项；不能说全部工况的误差都改善。铝、铜六组的开发近优集合同时包含两项和三项结构，存在结构歧义。训练前后半参数稳定性等诊断详见逐组报告。

需要分阶段审计时，可先运行 `python -m pde_discovery refined-benchmark --phase freeze`，检查 `freeze_index.json`，再运行 `python -m pde_discovery refined-benchmark --phase evaluate`。第二阶段会先验证全部模型、代码、协议和数据清单哈希；改变代码或协议后必须重新冻结。

## 原七项分支：三分钟合成演示

以下演示位于根路径 `/`，使用原七项库、合成生成器和有限实验控制器。它不是 `/discovery` 新九项引擎，也不是三个外部数值基准的重新训练入口。

1. 选择“扩散方程”，保持无噪声，运行后恢复约 `u_t = 0.2 u_xx`，查看独立初值的预测热力图。
2. 选择“对流与扩散”，恢复约 `u_t = -0.7 u_x + 0.12 u_xx`。系数来自数据识别，发现模块不读取生成器的真值。
3. 选择“歧义与追加实验”，取消自动追加。首条单模态轨迹同时支持扩散和局部衰减，系统报告歧义、目标未完成。
4. 勾选自动追加，再运行。控制器比较 `1.5 sin(x)`、`sin(2x)`、`sin(3x)` 的候选模型分歧，调用独立隐藏系统生成选中实验的新观测，再用另一条保留轨迹验证。
5. 切换至 1% 噪声复现；最后导出 JSON，检查初轮候选、实验评分、最终方程和每一步记录。

上传模式支持独立数据；系统只提出追加实验建议，不会用自己猜出的方程生成数据来证明自己。

## 原七项分支：方法和目标完成条件

该旧分支的候选库固定为 `1, u, u², u³, u_x, u_xx, u·u_x`，左侧固定为 `u_t`，其系数固定为 1。PDE 参数在空间和时间上保持常数。以下 5% 目标完成规则不用于判定新九项基准的严格结构恢复。

1. 检查形状、有限值、均匀网格、周期边界和重复数据。
2. 空间保留前 6 个傅里叶模态，并进行时间局部多项式平滑。
3. 在短时间窗口上构建积分回归关系：`u(t₂)-u(t₁) = ∫ Θ(u)ξ dt`。避免直接对含噪观测计算时间导数。
4. 对候选矩阵各列做 RMS 缩放，遍历有限 LASSO 正则强度，用 L1 筛项。
5. 在保留支持集上重新做最小二乘，消除 L1 收缩偏差，并还原原尺度系数。
6. 根据训练拟合误差和复杂度保留候选；对高度相关的项保留近似等价解释。相关替换的容差依据候选列的几何关系，不读取真值。
7. 对歧义情况比较有限备选初值的模型预测差异，最多追加一次仿真实验，然后重新识别。
8. 从独立轨迹的初始场出发积分候选方程，预测后续所有时刻。最终留出数据不参与系数拟合、候选排序或歧义解除。

目标完成需要：训练阶段未检测到未解除的候选歧义、训练拟合检查通过、独立轨迹的相对 L2 预测误差不超过 **5%**。相对误差为 `||预测[1:] - 观测[1:]||₂ / ||观测[1:]||₂`，不把已给定的初始帧计入指标。

“完成”仅表示在有限候选库和测试工况内达标，不等于证明自然规律唯一。控制器最多执行 3 次发现调用；不能达标时明确返回原因。缺少验证数据返回 `waiting_for_data`，有等价解释返回 `ambiguous`，数据或预测不合格返回 `insufficient`。

## 原七项上传接口的数据格式

此格式用于根路径上传界面和 `/api/discover` 的原七项引擎；新九项基准目前通过上面的固定 MAT 下载适配器运行，不能把该上传接口描述为新九项发现入口。推荐上传一个对象，包含不同实验/初始条件的训练轨迹和最终验证轨迹：

```json
{
  "training": [
    {"x": [0.0, 0.1], "t": [0.0, 0.02], "u": [[0.0, 1.0], [0.01, 0.99]], "boundary": "periodic"}
  ],
  "validation": {
    "x": [0.0, 0.1], "t": [0.0, 0.02], "u": [[1.0, 0.0], [0.99, 0.01]], "boundary": "periodic"
  }
}
```

上面只展示结构，点数不足，不能直接用于运行。完整样例在 `examples/diffusion_bundle.json`，也可在界面下载。

- `u` 形状为 `[时间点数][空间点数]`；接口支持 32–192 个空间点、16–401 个时间点。
- `x`、`t` 严格递增且均匀采样；周期空间网格不重复右端点。所有轨迹用同一空间网格和一致单位。
- 所有训练轨迹必须来自相同方程和相同物理参数；不同初值可以不同。
- 最多 3 条训练轨迹和 1 条验证轨迹，总场值最多 300000，网页文件上限 5 MB。
- 只上传 `{x,t,u}` 也可以，但没有独立验证就不会宣布目标完成。
- 复制同一观测场、只改变时间原点等操作不能形成新验证证据。系统能检测完全重复场，但无法认证数据来源；不要用相邻时间窗口、轻微扰动副本冒充独立实验。
- 如需用验证数据排除候选，应把它明确加入训练/模型选择证据，并重新取得另一条最终留出轨迹。

## 复现实验与测试

```bash
python -m unittest discover -s tests -v
python -m pde_discovery demo --scenario ambiguity --output artifacts/demo.json
python -m pde_discovery demo --scenario ambiguity --no-auto-experiment --output artifacts/no_experiment.json
python -m pde_discovery benchmark --output artifacts
```

当前聚合 **128 项程序与科学验收通过**，日志在 `artifacts/generic-unit-tests.log`。涵盖原分支回归、通用九项引擎与独立数值解比较、跨方程支持、非法输入、源文件校验、三组全部冻结、参考真值与最终场隔离、篡改拒绝及失败结果保留。独立数值测试的 12 项已计入 128 项，不能重复相加。测试通过不改变外部基准严格结构恢复 0/3 的结论。

此前热杆独立来源阶段有 92 项验收通过，其中原有 73 项及新增 19 项结果仍保存在 `artifacts/external-validation-unit-tests.log`。这是历史阶段记录，不是当前聚合数量。

原七项分支的内置基准为 3 个场景 × 3 个噪声水平（0、0.1%、1%）× 2 个噪声种子（7、19），共 18 次运行，与新增三个外部数值基准分开。噪声标准差相对于干净轨迹的 RMS 定义；无噪声的两个种子运行是重复结果，不应当作独立泛化证据。

| 场景 | 达到本原型目标的运行数 | 6 次运行中最大留出预测相对误差 |
| --- | ---: | ---: |
| 扩散 | 6/6 | 约 1.080% |
| 对流扩散 | 6/6 | 约 1.083% |
| 单模态歧义，经追加实验 | 6/6 | 约 1.079% |

结果保存在 `artifacts/benchmark.json`，包含每次运行方程、系数、误差和耗时；每次实验另有完整 JSON。图表为 `artifacts/benchmark.png`。这些都是内置合成系统的结果，不是现实数据上的成功率。

可选浏览器检查需要额外安装 Playwright 和浏览器；先启动服务，再运行 `python tests/browser_smoke.py`。可通过 `CHROMIUM_PATH` 指定现有 Chromium。此检查覆盖桌面/移动端、正常识别、歧义、追加实验、上传错误、有效上传和下载导出。

实测页面另有 `python tests/browser_real_smoke.py`，检查九组切换、JSON 导出、移动端与失败报告不显示旧图；不重新训练真实数据。最终本地校验摘要见 `artifacts/final-verification.json`。

改进页面检查为 `python tests/browser_refined_smoke.py`，默认访问本机 8765 端口，可用 `PDE_BASE_URL` 更改地址。已检查工况切换、JSON 导出、回顾性标签、移动端、失败和缺报告状态；只发 GET 请求，不重复训练或评分。当前改进交付的校验摘要见 `artifacts/refinement-verification.json`。

历史新增留出与诊断页面检查为 `python tests/browser_validation_smoke.py`，同样支持 `PDE_BASE_URL`，只读取保存结果。已检查 12 行、记录切换、JSON 导出、移动端、失败/404 清图和禁用导出；该轮汇总见 `artifacts/expanded-validation-verification.json`。

独立来源页面检查为 `python tests/browser_external_smoke.py`，检查两条记录、切换、导出、歧义标识、移动端和失败/404 清图；只发 GET 请求。该轮汇总见 `artifacts/external-verification.json`。

最新九项发现页的只读浏览器检查入口为 `python tests/browser_generic_smoke.py`，针对 `/discovery` 的三个案例、结构与系数展示、导出、移动端及失败/缺报告状态；使用保存结果，不调用训练或最终评分。

## API 与华为平台接入边界

启动后访问 `/docs` 查看接口；OpenAPI 为 `/openapi.json`，静态副本在 `artifacts/openapi.json`。

| 方法 | 路径 | 功能 |
| --- | --- | --- |
| GET | `/api/health` | 运行状态与平台连接状态 |
| GET | `/api/generic-experiment?case=burgers` | 新九项引擎的冻结方程、结构/系数评分及预测结果 |
| GET | `/api/generic-experiment/benchmark` | 三个公开数值基准的完整结果，严格结构恢复 0/3 |
| GET | `/api/generic-experiment/plot?case=kdv` | 所选新基准的保存预测图 |
| POST | `/api/run` | 在原七项内置仿真实验室执行发现任务 |
| POST | `/api/discover` | 用原七项引擎识别上传轨迹，独立验证并给出建议 |
| GET | `/api/example?scenario=diffusion&kind=bundle` | 下载完整数据包 |
| GET / POST | `/api/real-experiment?case=50ss` | 读取 / 复现固定实测组 |
| GET | `/api/real-experiment/benchmark` | 九组实测汇总 |
| GET | `/api/real-experiment/plot?case=50ss` | 所选实测组验证图 |
| GET | `/api/refined-experiment?case=50ss` | 物理约束方法的回顾性比较报告 |
| GET | `/api/refined-experiment/benchmark` | 九组新旧误差与项数比较 |
| GET | `/api/refined-experiment/plot?case=50ss` | 所选组改进方法预测曲线 |
| GET | `/api/additive-experiment?case=xn3` | 所选增材杆留出记录报告 |
| GET | `/api/additive-experiment/benchmark` | 十二条预定留出记录汇总 |
| GET | `/api/additive-experiment/plot?case=xn3` | 对应留出记录预测图 |
| GET | `/api/identifiability/summary` | 旧九组训练参数诊断汇总 |
| GET | `/api/identifiability/plot` | 训练扩散系数剖面图 |
| GET | `/api/external-experiment?case=aluminum` | 独立来源高电压测试报告，新旧方法与基线 |
| GET | `/api/external-experiment/benchmark` | 两条高电压测试的完整比较 |
| GET | `/api/external-experiment/plot?case=brass` | 相应材料的条件预测曲线 |

`/api/run` 请求示例：

```json
{"scenario":"ambiguity","noise":0.01,"seed":7,"auto_experiment":true}
```

后续在所选华为赛题平台上，将数值服务封装为工具。平台层可负责理解目标、补齐缺失信息、选择下一步工具；数值层保留确定性的验证和停止规则。平台接入、鉴权、云部署、MindSpore/昇腾迁移及赛事适配尚未完成，不能将本原型描述为已在这些平台运行。当前服务适合本地演示，未配置公网鉴权或多租户隔离。

## 能力范围与研究限制

- 新九项发现和原七项合成/上传路径只支持一维周期标量场、常系数、各自指定候选库；实测热杆另有六测点非周期适配器。不支持任意几何、缺测或任意 PDE。
- 固定空间投影是平滑假设，可能滤掉真实高频结构。原内置基准均为平滑的低模态线性动力学；新增非线性外部基准未追加噪声，不能据此宣称普遍抗噪。
- 原合成生成器与线性候选预测均使用傅里叶演化。新九项引擎采用通用 ETDRK4 推进，并已用独立 Fourier/ODE 参考测试，但三份外部数据仍是数值模拟，不建立真实跨物理实验泛化。
- 已完成三个非线性方程的外部数值发现基准，全部参考项找到但严格支持恢复 0/3。低预测误差不能取代结构正确性，单轨迹时间留出不能代替新初值独立实验。
- 歧义检测主要覆盖高相关项与已生成候选，不构成一般的可辨识性证明。正确项不在库中，或多组机制产生相同观测时，不保证能识别真实规律。
- 该系统是基于已有稀疏辨识方法的科研原型。L1、SINDy、PDE-FIND 不是本项目发明；可进一步研究的贡献是实验选择、证据不足诊断和可审计验证流程。

参考：Rudy et al., *Data-driven discovery of partial differential equations*, Science Advances (2017), https://arxiv.org/abs/1609.06401。

## 文件结构

```text
pde_discovery/
  generic_engine.py # 当前九项L1发现、积分回归与通用ETDRK4推进
  generic_experiment.py # 固定源下载、角色分割、全组冻结、结构与预测评分
  data/generic_pde/source_manifest.json # 固定来源/字段/哈希；不含参考系数
  engine.py       # 原七项候选库、稀疏识别、候选 PDE 预测
  simulator.py    # 独立的隐藏方程观测生成器
  controller.py   # 有限预算的目标控制器、上传验证
  real_heat.py    # 六测点非周期时间积分 L1 与边界条件预测
  real_experiment.py # 训练、开发选模、冻结与最终评分
  real_benchmark.py # 九组固定实测重复实验
  real_heat_refined.py # 物理约束轨迹拟合与原时刻线性系统传播
  refined_experiment.py # 先冻结全部模型，再进行回顾性比较
  additive_experiment.py # 四增材杆、十二条预定留出记录
  heat_identifiability.py # 训练剖面与保留完整传播的删块诊断
  external_experiment.py # 独立教学热杆来源的固定训练和条件验证
  data/mendeley_heat/ # 原始 CSV、固定清单与来源信息
  data/additive_heat/ # 二十份增材CSV、清单与来源
  app.py          # FastAPI 与 OpenAPI
  __main__.py     # 服务、单次实验和基准 CLI
  static/discovery.html # 当前三个外部数值基准的只读发现页
  static/index.html  # 原七项合成/上传中文界面
tests/            # 科学验收与可选浏览器检查
examples/         # 完整示例数据
artifacts/        # 合成与实测分目录记录、图表和界面截图
experiments/      # 固定实验协议、实测报告与诊断
```
