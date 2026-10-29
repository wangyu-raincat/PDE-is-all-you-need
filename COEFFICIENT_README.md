# 系数精度独立对照实验

本分支 `experiment/coefficient-reliability` 从 `experiment/prefix-five-percent` 的 `7bacc8ce5a1628b76c6f95581cd37ba1c3ca050e` 分离。原版本及默认应用保留。目标是在完整恢复方程项、维持最坏条件平均预测误差≤5%的同时，改善小系数精度。验证范围是三个一维周期合成 PDE 方程族的新初值，不能当成新的实测或任意 PDE 泛化验证。

本轮结果：四组均30/30完整恢复、无多余/缺失项。融合候选最坏逐项系数误差24.6133%→6.1022%，相对改善75.2079%；仅改选模为6.0971%，单独噪声修正为24.6206%。四组最坏条件平均预测误差均5.4104%，29/30条件≤5%。因此效果门槛125/126，不合入；487项自动测试、独立审计及36次最差条件预测复算通过。新数据上的基线也是5.4104%；此前4.0708%属于旧初值组。完整结果见 [REPORT.md](coefficient_study/REPORT.md)。

## 方法与冻结规则

四组均使用完全相同的短历史条件预测和完整未来时长：

| 标识 | 系数特征 | 选模 |
|---|---|---|
| `prefix_baseline` | 原弱形式 | 原开发集前缀预测误差 |
| `stable_selection` | 原弱形式 | 预测合格后比较共同弱残差与整条训练轨迹留一敏感性 |
| `noise_corrected` | 非线性矩的观测噪声修正 | 原开发集前缀预测误差 |
| `noise_stable` | 非线性矩的观测噪声修正 | 共同弱残差与轨迹留一敏感性 |

融合组 `noise_stable` 在正式数据生成前就被指定为候选，不根据最终结果更换候选。选模先保留预测误差在最佳值1.05倍以内的方案，再保留最小项数，随后比较同一弱形式验证块上的残差，最后比较系数对训练轨迹留一的敏感性。全过程不读取真实系数、方程族标签或生成器噪声参数；真实系数只用于评分。存在诊断失败时明确记录并回退至原预测选择，不隐藏失败。

噪声修正从观测高频谱估计方差，对二次/三次矩作修正，并传播傅里叶插值的方差。它依赖高频物理信号较弱的假设，不消除全部自变量测量误差偏差。留一结果是敏感性诊断，不是置信区间。已有列归一化、联合轨迹拟合、重加权L1和去偏重拟合继续沿用。

完整采样历史11帧、粗采样6帧，末帧是预测起点。Allen–Cahn未来时长0.5、Burgers 1.5、KS 20，历史长度为未来时长的10%；历史不计入预测区间。坐标使用基准的无量纲单位。

每族3条训练、2条开发、3条最终轨迹。预先冻结全部120个模型，再生成9条最终数值轨迹；这些轨迹在采样/噪声/方法条件间复用。新初值与旧记录分开，旧AC两条件探针只算回顾性开发诊断。首次结果产生后，重复运行只算复现。

126项预设检查涵盖来源校准、完整项集、逐项系数阈值、系数向量回归、各族系数尾部回归、30条件预测均值≤5%、AC最坏逐项系数误差至少相对降低20%，以及全体最坏逐项系数误差不增加。任何检查失败都不合入默认版本；单条轨迹可以超过5%，会单独报告。详见 `coefficient_study/protocol.json`。

## 安装、演示与复现

验证环境为 Python 3.12。代码包与实验记录包解压到同一父目录，共享 `pde-discovery-coefficient-reliability/` 顶层目录。原始数值场可按种子重建。请从干净解压目录开始，避免覆盖首次记录：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-repro.txt
python -m unittest discover -s tests -v
python -m coefficient_study.run prepare
python -m coefficient_study.run freeze --workers 3 --output coefficient_study/results/reproduction
python -m coefficient_study.run evaluate --workers 3 --output coefficient_study/results/reproduction
python scripts/render_coefficient_study.py --output coefficient_study/results/reproduction
```

既有基线界面可直接启动：

```bash
python -m pde_discovery serve --host 127.0.0.1 --port 8765
```

打开 http://127.0.0.1:8765/discovery 。本实验独立接口为 `coefficient_study.engine.fit_observations(training, development)`；数据适配格式见 `coefficient_study/data.py`。预测使用 `prefix_study.forecast.forecast_with_diagnostics(terms, x, future_t, prefix_t, prefix_u)`，该接口不接收未来观测值。界面尚未自动切换到本实验方法。

最终报告和判定已保存到 `coefficient_study/REPORT.md`、`coefficient_study/DECISION.json`，逐模型收据和评分位于 `coefficient_study/results/first_run/`。`coefficient_study/DESIGN_REVIEW.md` 是性能评估前的独立设计审阅，不代表验证已通过。原基线可通过 Git 分支 `experiment/prefix-five-percent` 回退，切换前保存自己的修改。
