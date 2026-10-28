# PDE 弱形式改进版：先读此文件

本目录是独立实验分支 `experiment/weak-pde-improvement`，请勿覆盖原项目。新增可运行的空间与时间弱形式识别器，保留原九项库、自适应 L1 与开发集选模。默认演示仍使用原基线，本轮没有合并或更改默认方法。

旧数据的回顾性完整项恢复从自适应17/30升到弱形式29/30；新初值验证从23/30升到28/30，多余项10→4、没有漏项。新验证的逐真实项系数误差均满足预设10%/10%/25%门槛，但粗采样KS五个条件的预测相对误差升至约88%–107%，完整门槛只通过97/102。因此保留改进候选和全部失败记录，**不合并**。不能把这理解成全面优于旧方法：旧公开Allen–Cahn在1%噪声下的小扩散系数仍可错约65%。

先阅读：

- `weak_study/REPORT.md`：方法、回顾性与新初值结果、限制。
- `weak_study/RESULT_REVIEW.md`：独立审阅及全部90行新配置主要指标。
- `weak_study/DECISION.json`：固定门槛与不合并决定。
- `weak_study/results/retrospective/`、`weak_study/results/first_run/`：冻结模型、JSON、CSV和科学图。
- `weak_study/development/DEVELOPMENT.md`：36次已查看旧数据上的开发试验，没有隐去失败。

全部为数值基准，没有新增真实测量。旧90配置对照中60条基线/自适应结果直接复用旧冻结记录，30条弱形式重新拟合；新90配置全部重新拟合。新验证含3方程族×2采样×5噪声条件×3方法，270次预测共享9个最终初值，不是270次独立物理实验。

## 安装与运行

解压后进入 `pde-discovery-weak-improvement`。已验证 Python 3.12.14；建议独立虚拟环境。

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-repro.txt
python -m unittest discover -s tests -v
```

Windows 激活命令为 `.venv\Scripts\activate`。232项测试覆盖历史版本以及本轮数学、生成器、数据隔离和验收逻辑，不等于232次科学验证。

本轮数据可在本地确定性生成，无需下载新数据；包内提供源码匹配的独立校准收据，不包含完整观测数组。复现命令：

```bash
python -m weak_study.run prepare
python -m weak_study.run freeze --workers 3 --output weak_study/results/reproduction
python -m weak_study.run evaluate --output weak_study/results/reproduction
```

请保留 `first_run` 原始记录。方法和菜单在正式训练/开发数据生成前冻结，所有90模型冻结后才生成最终初值；最终从带噪首帧预测，干净未来仅用于评分。已报告的数据再次用于改进时属于回顾性，不能冒称新留出。运行时 `data/prepared.json` 不打包，其审计副本保存在 `results/first_run/provenance/`。

原型演示仍可运行：

```bash
python -m pde_discovery serve --host 127.0.0.1 --port 8000
```

浏览器打开 `http://127.0.0.1:8000`。弱形式实验通过上述独立命令行运行，不静默替换原网页结果。

## 历史与回退

`EXPERIMENT_README.md`、`CHALLENGE_README.md` 是保留的前两轮说明。包内 `git-history.bundle` 含基线与三个实验分支。可在新的目录恢复：

```bash
git clone git-history.bundle restored-pde-weak
cd restored-pde-weak
git switch experiment/weak-pde-improvement
```

没有未保存修改时，可查看此前版本：

```bash
git switch experiment/l1-challenge
git switch experiment/l1-analytic-assessment
git switch baseline/frozen-generic
```

本轮方法与协议冻结提交 `540a0624fc171d36ddc7f31fcc1e3cc733ee207d`。原基线目录449个原文件、上一轮解析实验567个原文件、挑战轮693个原文件均保持不变；不需要对生产目录执行回退。所有历史包继续保留在原私有交付页。
