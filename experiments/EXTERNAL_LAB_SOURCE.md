# 独立来源热杆测量：来源与适配审计

本轮选用 Tariq Sawaqed、Brady Comer、Josh Colgrove、Graham Dobbie 发布的 [Heat-Conduction-Aerospace-Lab](https://github.com/TariqSawaqed1/Heat-Conduction-Aerospace-Lab)。仓库介绍为 ASEN 3802 热传导实验：对铝、黄铜和钢杆加热，以八个热电偶记录升温至稳态的过程。这是作者提供的实验测量资料，不是本项目生成的仿真；尚无独立仪器校准审计或同行评议数据说明。来源与既有 Tomanek/Stutts Mendeley 热杆数据、ATMC 示例不同。

固定仓库提交为 `aba5a8c3c1d52fb879710cdddcfb23d29b079f03`。机器可读 URL、Git blob SHA-1 和文件字节数见 [source_manifest.json](../pde_discovery/data/external_lab/source_manifest.json)。本文件只记录来源；数据角色和温度参考值由本轮实验协议独立冻结。

## 测点与单位核验

- [README](https://github.com/TariqSawaqed1/Heat-Conduction-Aerospace-Lab/blob/aba5a8c3c1d52fb879710cdddcfb23d29b079f03/README.md) 说明时间单位为秒、温度为摄氏度，八个热电偶名义间距为 0.5 英寸，名义采样间隔为 10 秒。
- [作者几何计算](https://github.com/TariqSawaqed1/Heat-Conduction-Aerospace-Lab/blob/aba5a8c3c1d52fb879710cdddcfb23d29b079f03/Part%201/Lab2_Part1.m) 使用第一测点 1.375 英寸、后续逐点增加 0.5 英寸，并乘 0.0254 转为米。因此相邻测点距离为 **0.0127 m**。这些代码仅作为文字元数据阅读，未执行。
- README 对物理坐标原点的描述不足以替代装置测量记录。本项目仅采用相对测点坐标，不把 CH1、CH6 说成金属杆物理端点。
- [作者导入脚本](https://github.com/TariqSawaqed1/Heat-Conduction-Aerospace-Lab/blob/aba5a8c3c1d52fb879710cdddcfb23d29b079f03/Part%202/read_data.m) 确认一行表头、逗号分隔、九列数值，依次为时间与 CH1 至 CH8 的温度。
- 温度是**摄氏温度读数**，不是旧数据集中的环境温升。没有单独环境温度通道；项目计算的温度参考值不得称为实测环境温度。
- 作者图中使用 ±2°C 误差棒；这只是来源分析使用的假设，不等于本项目获得了热电偶校准证书。

## 文件与条件

全部文件取固定提交的 `Part 1/` 路径；`ASEN3802_HeatConduction_FA25/` 和 `cases/` 中相同文件是重复副本，不能作为额外试验。`Part 2/` 也有近乎相同的副本，不能增加样本量。

| 文件 | 作者材料标识 | 条件 | 原始字节数 | Git blob SHA-1 |
|---|---|---|---:|---|
| Aluminum_26V_250mA.txt | 7075-T651 铝 | 26 V、250 mA | 7329 | `968990e38e4fe0a1f1b981d117d8bf92a90f163c` |
| Aluminum_28V_269mA.txt | 7075-T651 铝 | 28 V、269 mA | 11491 | `51d15860ab7fd2ec0b8f28982b3f20901aaf282e` |
| Brass_26V_245mA.txt | C360 黄铜 | 26 V、245 mA | 21094 | `e6e74c879730e11fb50805e76b9516470e4a1e1d` |
| Brass_29V_273mA.txt | C360 黄铜 | 29 V、273 mA | 21267 | `beae9f168e3393837038e0de566f96779db49786` |
| Steel_21V_192mA.txt | 退火 T-303 钢 | 21 V、192 mA | 54351 | `d960828e25971b09192b290a4da1745a41cee401` |

Git blob SHA-1 对 `blob <字节数>\0<文件原始字节>` 求散列，不能与普通文件 SHA-1 混淆。下载后另存 SHA-256。电压、电流用于区别试验条件，不作为已知均匀体热源强度输入模型。

## 日期、授权与结论范围

仓库创建于 2026-08-04；导入脚本注释记载生成时间为 2025-03-07，文件夹名带 FA25。这些都不能单独确定原始数据采集日期，故不报告精确采集年份或日期。[科罗拉多大学官方课程目录](https://catalog.colorado.edu/courses-a-z/asen/) 能确认 ASEN 3802 包含热传导和实验模型验证，但没有逐份认证本仓库记录。

GitHub 仓库元数据的许可字段为空，目录中未发现 LICENSE。公开可读不意味着已获得再分发许可，因此交付包保留本项目的下载器、固定源地址、校验值和衍生实验结果，**不打包该仓库的原始温度文件或作者代码**。

适配选择应在读取最终高电压记录前固定：仅用原始 CH1 至 CH6，CH1/CH6 作观测子域边界，CH2 至 CH5 为待预测内部测点；CH7/CH8 不用于选型或补点。铝与黄铜有两种电压工况，可分开建模与留出验证。钢只有一个工况，预先排除于本轮跨工况测试。此选择由数据设计而非测试误差决定。

真实测量的公开来源并不保证模型适配成功。最终报告必须保留失败结果，比较同输入信息的基线，并说明条件预测需要整段实测边界。这轮能评估新来源、新工况下的预测表现，不能证明唯一真实连续 PDE 已恢复。
