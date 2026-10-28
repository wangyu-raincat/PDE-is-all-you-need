# 空间与时间弱形式的独立改进实验

目标仍是恢复 PDE 项和系数。保留 `pde_discovery/` 默认代码、上一轮 `l1_study/` 与 `challenge_study/`；本轮仅增加 `weak_study/`。改进把空间/时间导数移到已知测试函数上，避免直接对含噪观测求高阶导数。候选仍为原九项，没有强制保留任何真实项。

空间测试采用 Fourier 模式，包含零频以识别常量/反应项；非零模式的实虚部按一致权重展开。时间测试是紧支撑多项式 `(1-z²)^4`，边界为零，使用积分分部得到 `∫u_t φ = -∫u φ_t`。多项式非线性通过 2N 零填充计算；积分在原始观测点完成，不把插值当成新增测量。拟合复用原自适应 L1 路径和 OLS 去偏。

三个固定积分设置的支持集候选一起交给原开发集预测误差规则：最优 RMSE 的1.05倍加1e-12范围内，优先项数少，再比较误差和固定ID。全部候选、设置及失败均记录。不使用真实噪声方差、参考方程、解析导数或最终场选择设置。不采用跨 λ 惩罚目标比较或路径频次规则。

## 两种证据分开

- 旧90配置用于回顾性开发；探查记录在 `development/`，固定最终菜单的完整对照在 `results/retrospective/`。这些数据已经查看，不能声称为新留出。
- 新初值验证在 `results/first_run/`。三族全部为独立数值生成：Allen–Cahn、低黏度 Burgers、KS；每族3训练、2开发、3最终初值。方法与菜单在生成正式训练/开发前冻结，90模型全局冻结后才生成最终场。这里检验已知方程族的新初值泛化，没有新增真实仪器测量或未知方程族。

公开 Allen–Cahn 与本轮生成 Allen–Cahn 的初值分布、时间和网格不同，不能直接把两轮恢复数量当成可比成绩；只在同一轮、相同观测上对比三方法。噪声和粗采样条件共享底层来源。270次最终预测使用9个底层最终初值，不能当作270次独立实验。

## 预定验收

主要比较 `weak_adaptive` 与较强的旧 `adaptive`，同时报告 `plain`。严格项集合、漏项、多余项、系数向量和每个真实项误差都是主指标；预测为辅助。完整规则见 `protocol.json`。除跨族结构收益和不增加漏/余项外，每个真实系数在无噪及0.1%噪声下误差≤10%，1%噪声下≤25%。小扩散项不能被反应项掩盖。任何门槛失败则不合并；回归通过不能替代科学指标达标。

## 运行

使用 Python 3.12，安装根目录 `requirements-repro.txt`。提供匹配生成器源码的独立校准收据；数组通过确定性生成器重建，无需下载新公开数据。

```bash
python -m unittest discover -s tests -v
python -m weak_study.run prepare
python -m weak_study.run freeze --workers 3 --output weak_study/results/reproduction
python -m weak_study.run evaluate --output weak_study/results/reproduction
```

准备或结果已存在时拒绝覆盖；部分模型/最终场也保留为证据。复现已报告结果不是新的未见验证。若要比较进一步改动，另立协议、数据和结果目录，不能重用最终集来调整规则后仍称独立验证。

回顾性复现需要上一轮保留的完整数据目录；旧60条 plain/adaptive 成绩从原冻结结果直接读取，弱形式重拟合30条：

```bash
python -m scripts.run_weak_retrospective --previous /path/to/pde-discovery-l1-challenge --output weak_study/results/retrospective-reproduction
```

校准只比较独立校准初值的加密网格/减半步长，并有独立解析与小网格求解测试；不是每条正式轨迹的收敛证明或真实系统认证。方法适用范围仍限一维、周期、均匀采样、标量、自治常系数及既定九项库。
