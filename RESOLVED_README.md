# 内部网格加密改进版：先读此文件

本目录对应独立分支 `experiment/resolved-weak-pde`，请勿覆盖原项目。新增固定4倍内部计算网格的预测器，并让开发选模与最终预测使用同一预测器。原时间积分识别、弱形式三变体、自适应L1路径和原默认演示代码均保留。

本轮新验证中，改进版完整项恢复29/30；上一版弱形式为30/30，使用同一加密预测器的原积分方法为22/30。相对上一版弱形式，最坏条件平均预测误差从98.85%降至8.54%，但一个AC高噪条件的小扩散系数仍错37.29%，另一个AC条件多出一个四阶项。因此预设门槛通过169/172，**不合并默认版本**。267项程序测试通过；改善与失败全部保留。

请先读 `resolved_study/REPORT.md`、`resolved_study/RESULT_REVIEW.md` 和 `resolved_study/DECISION.json` 查看完整成绩与是否通过门槛。`resolved_study/results/first_run/` 保存120配置的冻结模型、全部评分、CSV与科学图；`resolved_study/development/DIAGNOSIS.md` 及72次控制记录解释改进依据，明确属于旧数据回顾性诊断。

本轮四臂为：原积分/原预测、弱形式/原预测、原积分/加密预测、弱形式/加密预测。新数据保持上一轮三个数值方程族的物理参数、初值分布及采样规格，使用新的固定角色种子。每族3训练、2开发、3最终初值，合计120配置、360次预测；底层最终初值只有9个。没有新增真实测量，不能把计算网格插值点当成观测，也不把新初值称作新PDE家族。

## 安装与实验复现

解压进入 `pde-discovery-resolved-weak`。已验证 Python 3.12.14，建议新建虚拟环境。

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-repro.txt
python -m unittest discover -s tests -v
python -m resolved_study.run prepare
python -m resolved_study.run freeze --workers 3 --output resolved_study/results/reproduction
python -m resolved_study.run evaluate --output resolved_study/results/reproduction
```

Windows 的虚拟环境激活命令为 `.venv\Scripts\activate`。本轮完整观测数组由本地确定性生成器重建，无需新下载；提供匹配源码的独立校准收据。方法与协议在正式训练/开发前冻结，120个模型全部冻结后才首次生成最终场。最终从带噪首帧推进，干净未来只评分。

`first_run` 保留首轮证据，已有准备和结果拒绝覆盖；运行时准备收据不打包，其审计副本位于 `resolved_study/results/first_run/provenance/`。已评分数据再次用于改方法是回顾性，不能当作新的留出验证。完整预定门槛与失败处理见 `resolved_study/protocol.json`。

## 在自己的周期数据上调用

先安装依赖，在本目录用 Python 调用：

```python
from resolved_study.run import fit_observations
from resolved_study.forecast import forecast

# training / development 是事先独立划分的观测字典列表。
# 每项包含 x[Nx], t[Nt], u[Nt,Nx], boundary="periodic"。
result = fit_observations(training, development, "weak_resolved")
model = result["selected"]
if model is not None:
    print(model["equation"])
    prediction = forecast(model["terms"], future_x, future_t, observed_initial)
```

接口不接受参考系数、真实噪声尺度或未来观测场。`future_t` 是要预测的时间坐标，`observed_initial` 只是对应首帧；不能把最终观测拿来选模型。没有合格模型时 `selected` 为 `None`，应保留失败信息，不强行输出方程。适用范围仍是一维、均匀周期网格、自治常系数标量九项库；默认4倍计算网格最多接收512个观测空间点，超过预算明确拒绝。

原型演示仍可运行：

```bash
python -m pde_discovery serve --host 127.0.0.1 --port 8000
```

浏览器打开 `http://127.0.0.1:8000`。默认界面保持旧基线；本轮新方法通过独立命令行和上述Python接口使用。

## 回退历史单独下载

为控制单个附件大小，**主代码包不内置 Git bundle**。原私有交付页另有 `pde-resolved-history.zip`，解压得到 `git-history.bundle`；将它放在本目录或使用其实际路径。此历史包包含原基线及四个实验分支。

```bash
git clone git-history.bundle restored-pde-resolved
cd restored-pde-resolved
git switch experiment/resolved-weak-pde
```

没有未保存修改时，可切换到 `experiment/weak-pde-improvement`、`experiment/l1-challenge`、`experiment/l1-analytic-assessment` 或 `baseline/frozen-generic`。本轮方法/协议冻结提交为 `ea0fb061e93fd018546cbfa3501917441df82af1`。原工作区及旧交付均保留，不需对原生产目录执行回退。

`WEAK_README.md`、`CHALLENGE_README.md`、`EXPERIMENT_README.md` 是历史交付说明；当前运行与分包方式以本文件为准。
