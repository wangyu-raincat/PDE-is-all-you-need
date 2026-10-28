# 挑战实验交付：90 个固定配置

本包是独立实验分支 `experiment/l1-challenge`，不覆盖原项目。一次自适应 L1 的严格项集合恢复为 17/30，普通 L1 为 0/30，五轮 IRL1 为 11/30；但自适应方法没有通过两项预定系数精度门槛，**不合并，继续使用原默认基线**。195 项程序测试通过。全部失败、漏项和微小多余项均保留。

先读 [挑战报告](challenge_study/REPORT.md)、[独立结果审阅](challenge_study/RESULT_REVIEW.md) 和 [决定记录](challenge_study/DECISION.json)。完整 90 行 CSV、冻结模型、比较 JSON、三张图在 `challenge_study/results/first_run/`。`EXPERIMENT_README.md` 与 `l1_study/` 是保留的上一轮 63 配置实验，不是本轮成绩；原网页仍是默认基线界面。

本轮为一个公开 Allen–Cahn 数值来源和两个独立生成数值方程族（低黏度 Burgers、KS）。完整/粗采样以及无噪、0.1%/1% 噪声交叉比较；没有新增真实仪器测量。90 配置共享来源和配对扰动，150 次预测只涉及 5 个底层最终来源，不能视为独立物理实验。150 次均返回有效结果，也不等于精度都达标；粗采样 KS 的一个条件误差约 51%。

## 安装、测试与原型演示

在解压得到的 `pde-discovery-l1-challenge` 目录运行。已验证环境为 Python 3.12.14；建议使用独立虚拟环境。

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-repro.txt
python -m unittest discover -s tests -v
python -m pde_discovery serve --host 127.0.0.1 --port 8000
```

演示页面为 `http://127.0.0.1:8000`。挑战实验通过下面的命令行复现，不静默替换原页面算法和分数。Windows 的虚拟环境激活命令是 `.venv\Scripts\activate`。

## 复现本轮

提供源码匹配的 `challenge_study/data/calibration.json`，独立生成器的分辨率/步长校准已通过。原始第三方 MAT、完整观测 NPZ 和生成场数组未打包；固定版本下载器与确定性生成器负责重建。`prepare` 下载 AC.mat，需要访问公开 GitHub raw 地址；不执行作者脚本。来源提交、原字节哈希、坐标及许可证说明均在 `challenge_study/source/`。

```bash
python -m challenge_study.run prepare
python -m challenge_study.run freeze --workers 3 --output challenge_study/results/reproduction
python -m challenge_study.run evaluate --output challenge_study/results/reproduction
```

顺序不能交换。准备阶段只生成训练和开发初值；90 个模型全部冻结后，才生成新数值族的最终场并评分。Allen–Cahn 压缩文件在准备阶段机械解码并分角色，最终角色不参与拟合或选模。最终预测从受扰动首帧开始，干净未来只在冻结后评分。全部预定结果和失败均保留。

包内 `first_run` 是已查看的原始记录；复现或后续利用这些数据比较新方法，应标为回顾性。不要覆盖 `first_run`。运行时准备收据不随包提供，以免缺少数组却误认为准备已完成；其审计副本位于 `challenge_study/results/first_run/provenance/prepared.json`。若准备已经存在，保留它并另选新的结果目录；不要为重跑删除原实验证据。

## 分支恢复与回退

`git-history.bundle` 包含三条分支：`baseline/frozen-generic`、`experiment/l1-analytic-assessment`、`experiment/l1-challenge`。挑战源码/协议冻结提交为 `a8409f8f673e1cc6fac5bf7fa061ae1415b49edf`，后续提交仅增加结果、报告和交付材料。

```bash
git clone git-history.bundle restored-pde-challenge
cd restored-pde-challenge
git switch experiment/l1-challenge
```

在这个新克隆、没有未保存修改时，查看原回退代码：

```bash
git switch baseline/frozen-generic
```

基线提交为 `b8f24a456ce63253994d05e446528c54ac88d32e`；上一轮结果提交为 `22be770aa998e55d7564b303c1a263a4fd7392b4`。原工作区生产目录未修改，无需执行回退操作。Git bundle 主要保留源码和结果记录；数据按上述步骤获取/生成。
