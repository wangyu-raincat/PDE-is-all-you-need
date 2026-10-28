# 短历史前缀条件预测实验

本分支 `experiment/prefix-five-percent` 从单帧实验 `0b32f67` 分离，保留原基线和所有历史实验。默认应用尚未改为本方法。本实验使用合成 PDE 数据，不构成真实测量或华为平台集成验证。

本轮结果：**最坏条件平均预测误差4.0708%，30/30条件达到5%目标；方程项完整恢复30/30，无多余/缺失项**。同一批数据单帧滤波对照最坏均值8.0642%。但AC最坏逐项系数误差22.3396%，对照原弱形式13.9417%，预设检查156/157，因此**不合入默认版本**。2/90次单轨迹误差超过5%，最大5.3547%。全量396项测试通过。

目标：30 个条件中，最差条件的三条最终轨迹平均相对 L2 预测误差（对干净数值参考）≤5%。同时检查完整方程项、额外/缺失项、逐项系数误差与对照回归。单条轨迹可以超过5%，会单独披露。低预测误差不能豁免方程识别失败。

## 输入和预测时长

用户已同意提供一段短历史。完整采样使用11帧，粗采样使用6帧；历史终点与预测起点是同一次观测。每帧经固定频谱阈值滤波，用候选 PDE 传播到预测起点，然后等权平均，形成预测初态。所有权重、阈值规则、候选项和选模规则均在新训练/开发数据前固定。

| 方程 | 历史区间 | 未来预测区间 | 完整未来时长 |
|---|---|---|---:|
| Allen–Cahn | 0–0.05 | 0.05–0.55 | 0.5 |
| 低黏度 Burgers | 0–0.15 | 0.15–1.65 | 1.5 |
| KS 不稳定增长基准 | 0–2 | 2–22 | 20 |

坐标均为基准的无量纲坐标。历史不计入预测时长，单帧对照从同一时刻起预测同一段未来。空间/时间采样、噪声等级、方程参数、初值分布均沿用困难基准；生成器独立于发现模型的预测器。训练 PDE 使用预测区间内的训练轨迹，不额外利用其历史前缀。

四臂：原弱形式 `weak_resolved`，单帧滤波及训练候选中位数 `denoised_median`，前缀弱形式 `prefix_weak`，前缀及训练候选中位数 `prefix_median`。中位数只在同一方程项集的不同训练系数向量间计算，不跨方程混合、不按路径频次投票。开发集预测用于选模；真实系数只参与最终评分。

## 安装和使用

验证环境 Python3.12.14，在代码目录执行：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-repro.txt
python -m unittest discover -s tests -v
python -m pde_discovery serve --host 127.0.0.1 --port 8765
```

浏览器打开 http://127.0.0.1:8765/discovery 查看既有基线演示。前缀方法提供独立 Python 接口，`terms` 必须来自已冻结模型；`future_t` 只有时间坐标，接口不接收未来观测值：

```python
from prefix_study.forecast import forecast_with_diagnostics
prediction, diagnostics = forecast_with_diagnostics(
    terms, x, future_t, prefix_t, prefix_u
)
```

`prefix_u` 为(历史帧数,空间点数)，末帧时间等于 `future_t[0]`。需要均匀周期空间网格；本规则使用2–32帧，正式基准固定11/6帧。

## 复现实验

代码包和实验记录包解压到同一父目录，两者共用 `pde-discovery-prefix-five/` 顶层文件夹。历史包单独包含 Git bundle。原始数值场可按冻结种子重建，不随交付包发布；历史实测/公开数据夹具沿用已有包。

结果已经生成后重复运行属于复现或回顾性分析，不是新留出实验。为避免覆盖首次记录，请从干净解压目录开始：

```bash
python -m prefix_study.run prepare
python -m prefix_study.run freeze --workers 3 --output prefix_study/results/reproduction
python -m prefix_study.run evaluate --workers 3 --output prefix_study/results/reproduction
python scripts/render_prefix_study.py --output prefix_study/results/reproduction
```

`prepare` 在方法冻结后只生成9条训练、6条开发轨迹；最终9条轨迹必须在120个模型全量冻结及摘要验证后才会生成。首次报告位于 `prefix_study/REPORT.md`、`prefix_study/DECISION.json`，原始逐配置记录位于 `prefix_study/results/first_run/`。包内不放运行态的 `data/prepared.json`，审计副本保存在结果 provenance 目录，复现会生成新收据。

## 边界

只测试三个已知一维周期、标量、自治、常系数数值方程族与九项候选库。频谱滤波有稀疏频谱假设，可能误删真实弱模态或宽带信号。前缀平均依赖候选 PDE，传播后的误差不能当成独立样本。9条最终初值在所有采样/噪声/方法间共享，360次预测不是360个独立物理实验。最终测试只有每族3条轨迹，不能给出广泛分布或最坏单轨迹保证。旧初值延长的开发探针明确为回顾性分析。

## 回退历史

`pde-prefix-history.zip` 解压得到 `git-history.bundle`，包含基线与七个实验分支。

```bash
git clone git-history.bundle restored-pde
cd restored-pde
git switch experiment/prefix-five-percent
```

回到本轮之前的单帧实验可切换 `experiment/prediction-five-percent`，原通用基线为 `baseline/frozen-generic`。切换前保存自己的修改；本轮未修改默认应用或合并到这些分支。
