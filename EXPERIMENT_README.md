# L1 方法独立实验交付

本目录是独立实验副本 pde-discovery-l1-experiment，不要覆盖原 pde-discovery 文件夹。默认生产版本仍是原普通L1基线，本轮没有合入。

先读 l1_study/REPORT.md、l1_study/RESULT_REVIEW.md 和 l1_study/DECISION.json。全部63配置的冻结模型、结果JSON、CSV及两张科学图在 l1_study/results/first_run/。原README和原网页仍描述基线，最新实验不静默替换旧分数。

结论：旧公开数值数据的回顾性结构恢复从0/3升到加权方法3/3，多余项4→0；新解析点态估计导数对照从1/3升到3/3；原积分主路径在新解析套件中三种方法已均3/3。自适应一次加权与5轮IRL1全部21配对输出相同，多轨迹未增加结构恢复数。预设数值gate和165回归通过，但审查决定保留候选，不为没有额外收益的5轮迭代引入默认复杂度；这不等于把通过的gate改判失败。

## 运行

```bash
python -m pip install -r requirements-repro.txt
python -m l1_study.gaussian_check
python -m unittest discover -s tests -v
```

这三条不训练公开数据。完整首次实验复现见 l1_study/README.md：先用固定版本generic-benchmark下载/准备原公开数据，再study prepare，然后以新的output目录freeze和evaluate。本包不包含第三方MAT、第三方完整源码、任何第三方完整观测NPZ；新解析数组也由确定性生成器重建以减小包体。原准备收据的审计副本保存在 l1_study/results/first_run/provenance/prepared.json，而不是运行时data目录，以免阻止首次prepare。

当前包沿用基线主包的代码、数据清单、既有可再分发CSV和结果；旧科学图/历史NPZ补充包及完整基线仍在同一私有交付页。最新两张实验对照图在本包内。

## 分支与回退

基线提交：b8f24a456ce63253994d05e446528c54ac88d32e
实验算法/协议冻结提交：d4cec74d0383c7ef3c0f4169b7aaf3c75e4e3c2d

包内 git-history.bundle 包含两条本地分支与本轮记录；可在新目录恢复：

```bash
git clone git-history.bundle restored-pde-study
cd restored-pde-study
git switch experiment/l1-analytic-assessment
```

需要查看回退代码时，在没有未保存修改的该克隆中执行：

```bash
git switch baseline/frozen-generic
```

Git bundle 不包含被忽略的数据数组；按复现命令获取/生成。不要在有未保存工作的目录用reset --hard。原工作区的生产目录无需切换，因为本轮从未修改它。
