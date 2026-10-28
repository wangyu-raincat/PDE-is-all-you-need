# 结构锚定与系数融合实验：运行入口

本目录对应独立分支 `experiment/fused-weak-pde`，不要覆盖旧项目。新方法先用原弱形式确定方程项，再在相同项集的近优弱变体系数之间取等权中位数，用固定四倍内部网格检验开发误差并推进。失败时回退同结构最佳候选；不同结构不混合，不使用 L1 路径出现频率。

最终成绩与合入决定见 `fusion_study/REPORT.md`、`fusion_study/DECISION.json`。`fusion_study/development/retrospective/` 是已查看旧数据上的回顾性分析；`fusion_study/results/first_run/` 是规则冻结后的新初值验证，不能混为同一批证据。

## 安装与复现

解压进入 `pde-discovery-fused-weak`，验证环境为 Python 3.12.14：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-repro.txt
python -m unittest discover -s tests -v
python -m fusion_study.run prepare
python -m fusion_study.run freeze --workers 3 --output fusion_study/results/reproduction
python -m fusion_study.run evaluate --workers 3 --output fusion_study/results/reproduction
```

Windows 激活命令为 `.venv\Scripts\activate`。完整数值场由本地确定性生成器重建，无需下载新数据。包内校准收据匹配生成器。运行准备收据不打包，其审计副本位于首轮结果的 `provenance/`。已有准备或结果拒绝覆盖；首次运行使用尚不存在的输出目录。复现已评分数据不构成新的留出验证。

## 在自己的周期观测上使用

```python
from fusion_study.engine import fit_observations
from resolved_study.forecast import forecast

# training/development 是事先独立划分的观测列表；
# 每项包含 x[Nx], t[Nt], u[Nt,Nx], boundary="periodic"。
results = fit_observations(training, development)
model = results["fused_median"]["selected"]
if model is not None:
    print(model["equation"])
    prediction = forecast(model["terms"], future_x, future_t, observed_initial)
```

入口不接受家族标签、参考方程、真实噪声尺度或最终未来观测。开发数据用于多个选模阶段，其得分不是独立泛化验证。没有合格结构锚时返回失败；不能把已看到的最终观测放入开发列表。

当前范围是一维、均匀周期网格、自治常系数、标量九项库。固定四倍计算网格最多接收 512 个空间观测点，超预算拒绝。计算插值点不是新测量；三个弱变体共享训练数据，不是独立实验。

原界面仍运行旧基线：`python -m pde_discovery serve --host 127.0.0.1 --port 8000`，浏览器打开 `http://127.0.0.1:8000`。新融合方法通过上面的独立命令行及 Python 接口使用。

## 回退历史

主代码包不内置 Git bundle。原私有交付页单独提供 `pde-fusion-history.zip`，解压得到 `git-history.bundle`，包含基线和五个实验分支。

```bash
git clone git-history.bundle restored-pde-fusion
cd restored-pde-fusion
git switch experiment/fused-weak-pde
```

没有未保存修改时可切到 `experiment/resolved-weak-pde`、`experiment/weak-pde-improvement`、`experiment/l1-challenge`、`experiment/l1-analytic-assessment` 或 `baseline/frozen-generic`。既有工作区和交付文件保留。

历史 `RESOLVED_README.md`、`WEAK_README.md` 等只说明相应历史版本；当前运行及分包以本文件和当前交付页为准。
