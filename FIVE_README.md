# 单帧 5% 目标：历史开发版本

**当前状态：单帧回顾性最坏误差6.48%，未达5%，未做新的正式留出验证。协议保留draft，`prepare`会明确拒绝。用户已选择短历史前缀方案，后续入口见 `PREFIX_README.md`（在后续前缀分支）。本文件保留单帧方法和历史开发说明。**

本目录对应独立分支 `experiment/prediction-five-percent`，不要覆盖旧项目。目标是每个测试条件的三条最终轨迹平均相对 L2 误差都不超过5%，不是要求每条轨迹都低于5%。同时保留严格项恢复、逐项系数和相对已有方法的系数回归要求。最终是否达到目标、是否允许合入，见 `five_study/REPORT.md` 和 `five_study/DECISION.json`。

方法只从观测首帧估计噪声，进行固定 Fourier 硬阈值滤波，再用四倍内部计算网格推进。新方法不锁定旧方程结构；在训练候选中，为相同项集增加系数中位数候选，原候选全部保留，再用相同的滤波预测器进行开发选模。相同系数向量去重，不按 L1 路径出现频率加权。

## 下载与安装

主代码包可以单独运行。完整本轮结果单独放在 `pde-five-results.zip`，为便于阅读报告，请把它与主代码包解压到**同一个父目录**；两者内部均以 `pde-discovery-five-percent/` 开头，会形成同一个项目目录。`pde-five-history.zip` 是单独的回退历史，不包含在主代码包内。

进入 `pde-discovery-five-percent`。验证环境 Python 3.12.14：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-repro.txt
python -m unittest discover -s tests -v
python -m five_study.run prepare
python -m five_study.run freeze --workers 3 --output five_study/results/reproduction
python -m five_study.run evaluate --workers 3 --output five_study/results/reproduction
```

Windows 激活命令为 `.venv\Scripts\activate`。数值场由确定性生成器本地重建，无需下载新数据。校准收据与源码匹配；运行用准备收据和原始场不随包分发，审计副本位于结果目录的 `provenance/`。已有准备或结果拒绝覆盖，输出目录应尚不存在。复现已评分的同一数据不构成新的留出验证。

## 自有观测调用

```python
from five_study.engine import fit_observations
from five_study.forecast import forecast_with_diagnostics

# training / development 为事先独立划分的观测字典列表，
# 每项包括 x[Nx], t[Nt], u[Nt,Nx], boundary="periodic"。
results = fit_observations(training, development)
model = results["denoised_median"]["selected"]
if model is not None:
    print(model["equation"])
    prediction, initial_diagnostics = forecast_with_diagnostics(
        model["terms"], future_x, future_t, observed_initial)
```

方法入口不接受家族、参考方程、真实噪声尺度或最终未来场。开发数据只供选模；不能把最终数据放入开发列表。失败应保留 `selected=None`，不能强行输出方程。

滤波只有首帧输入，隐含“高频带多数为噪声、初始频谱较稀疏”的先验。真实宽带高频信号或弱物理模态可能被删除，因此不是任意场景的无损处理。输出首帧是过滤后的计算初值，不是新测量；方法不会增加观测信息。输入范围仍是一维均匀周期、自治常系数标量九项库，最多512个空间观测点。报告包含宽带反例测试及本批数值分布的适用限制。

原界面保持基线：`python -m pde_discovery serve --host 127.0.0.1 --port 8000`，浏览器打开 `http://127.0.0.1:8000`。新方法通过独立命令行和上述 Python 接口使用。

## 回退与历史

解压 `pde-five-history.zip` 得到 `git-history.bundle`，其中包含基线与六个实验分支：

```bash
git clone git-history.bundle restored-pde-five
cd restored-pde-five
git switch experiment/prediction-five-percent
```

没有未保存修改时，可切回 `experiment/fused-weak-pde`、`experiment/resolved-weak-pde`、`experiment/weak-pde-improvement`、`experiment/l1-challenge`、`experiment/l1-analytic-assessment` 或 `baseline/frozen-generic`。旧工作区和私有交付附件全部保留。

本轮旧数据诊断和回顾性比较位于 `five_study/development/`，新冻结验证位于 `five_study/results/first_run/`，两者不能混合报告。当前运行与分包方式以本文件为准，其他 README 说明各自历史版本。
