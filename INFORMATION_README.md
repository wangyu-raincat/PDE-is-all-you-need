# 丰富初值开局后的 PDE 参数信息设计

独立分支为 `experiment/active-information-design`，回退基线为上一轮主动发现的
`40f0241d5a3f5ea9fd4347fa26432a9a597c210e`。本轮只修改实验选择策略，
识别器、候选项库、完整轨迹自助重选、数值预测器和旧结果均保留。
对照结论：**保留基线，不合并**。1%噪声主指标为0.5831%，优于固定方案的0.7860%，
但未超过随机方案的0.5015%；3%噪声仍有5/18个条件无法输出模型。
见 [结论与失败解释](information_study/RESULTS_REVIEW.md)、[实验报告](information_study/REPORT.md)、
[实际判定](information_study/results/first_run/assessment.json) 与
[预定判定规则](information_study/decision_protocol.json)。不自动合入默认版本。

## 这轮要回答什么

上轮固定丰富激励只需追加一次实验就达到80%成功标准，在共同初始预算5之后已达到最低查询数。
继续用这项指标寻求严格更省实验并不合理。新研究预先将**同预算系数精度**作为主指标：
总预算9、1%噪声下，18组参数各自的最坏真项相对误差再取均值。
随机实验在旧数据上的平均误差优于原结构分歧，因此保留随机强对照，不能只挑最有利的尾部比较。
原80%联合成功标准保持不变，作为成本次指标；3%噪声压力测试单独报告。

六策略共享同一识别器、初始观测和32个合法初值：

| 策略 | 第一次追加 | 随后3次追加 |
|---|---|---|
| fixed_rich | 固定丰富激励 | 其它固定丰富相位 |
| random | 随机 | 随机，不重复 |
| ensemble_variance | 候选预测方差 | 候选预测方差 |
| structural_disagreement | 原结构分歧 | 原结构分歧 |
| rich_then_variance | 与fixed_rich逐值相同 | 原集成方差，用于开局消融 |
| rich_then_information | 与fixed_rich逐值相同 | 当前模型下的相对参数A-opt设计代理 |

两种新策略与fixed_rich首个动作、噪声观测字节、预算6选中模型必须一致。
新增方法若只是因为开局更好，不能将提升归因于之后的信息评分。

信息策略使用当前选中模型预测各候选初值，计算新增弱特征对参数相对误差的局部设计价值。
所有动作共用从当前训练观察计算的尺度与正则化强度；没有参考系数、真实噪声或未查询观测输入。
模型预测使用33帧的时间近似，实际采集仍是111帧×64测点，识别及最终预测仍保留101帧、完整1.0时长。
参数信息矩阵只是设计代理：相关弱窗口、噪声特征及可能错误的选中结构使它不具备已校准的Fisher信息或后验保证。
33帧代理的弱时间窗为0.25，真实101帧对应0.24；这也是近似的一部分。
信息策略同时采用一个选中锚模型与更长设计预测时段，若改善，只能归因于整套选择策略，
不能将效应全部归给A-opt公式。A-opt优化的是局部相对系数平方误差代理，不保证正式的最坏真项指标下降。
没有可靠模型、秩不足或所有动作计算失败时，明确记录原因并回退固定丰富初值。

## 验证设计

新参数任务索引120–137，每类3组，共18组，分别加1%和3%噪声，形成36个配对条件。
噪声标准差分别为0.004和0.012，以固定初值RMS=0.4为参照。
六类仍为衰减、扩散、四阶耗散、对流扩散、反应扩散与Burgers。
其中反应扩散仅含线性反应项，六类中只有Burgers是非线性方程。
36个条件不能当作36个独立物理系统；本轮全部是合成数据。

初始3次训练和2次开发全部计费，每个合法开始的追加实验即使失败也扣预算。
216个回合、1080个预算槽位全部冻结后才生成54条干净最终参考；3240个预测槽位保留全部失败。
最终预测使用11帧历史前缀估计起点。已查看的旧任务只用于回顾性诊断，新开发任务不进入正式评价。
来源校准复用未修改数值provider的12项检查，不宣称连续参数域统一误差保证。

主要晋级门槛预先固定：1%噪声终端均值至少优于fixed-rich和开局消融10%，
不差于其余强对照，同时通过结构、系数尾部、3%压力与预测护栏。完整规则在decision_protocol.json。
每项失败都保留；全部通过也仅表示可以审阅，不自动合并。18参数的分层配对重采样只提供描述性区间，
每类仅3任务的限制仍在。自助包含频率继续作为算法稳定性描述，不是项存在概率。

## 运行

建议Python3.12，在源码根目录：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-repro.txt
python -m information_study.app --port 8768
```

打开 <http://localhost:8768> 查看保存结果；页面只读，无需API Key或GPU。
Windows激活命令为 `.venv\Scripts\activate`。旧回放仍可用 `python -m active_study.app --port 8767`。

在独立源码副本中复现，保留首次记录。开始时须有随源码分发的
`information_study/data/calibration.json` 和 `final_release.json`，
同时没有 `information_study/data/prepared.json` 与 `initial/*.npz`。
分发目录刻意不包含这些可再生的初始缓存；已有缓存的工作目录应另作副本，不删除原记录：

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python -m information_study.run prepare
python -m information_study.run freeze --output artifacts/information_reproduction --workers 3
python -m information_study.run evaluate --output artifacts/information_reproduction --workers 3
python scripts/audit_information_study.py --output artifacts/information_reproduction
python -m information_study.assess --output artifacts/information_reproduction
python scripts/render_information_study.py --output artifacts/information_reproduction
python -m information_study.app --output artifacts/information_reproduction --port 8768
```

记录的实验每个进程使用单个BLAS线程；PowerShell对应设置为
`$env:OPENBLAS_NUM_THREADS="1"; $env:OMP_NUM_THREADS="1"`。

已有初始缓存时跳过prepare；不覆盖首次或部分运行。固定最终数据发布后再次使用应称复现或回顾性分析，
不能重新宣称未查看验证。修改科学规则应另开实验与新最终任务；不能改哈希回执绕过冻结。
首次准备回执另存于 `information_study/results/first_run/prepared_snapshot.json`，
其哈希与首次freeze_index一致；该历史副本不作为新prepare输入。

开发联调与测试：

```bash
python -m information_study.run develop --output information_study/development/new_pilot --workers 3
python -m unittest discover -s tests -v
```

完整重跑涉及所有策略和预算，耗时取决于CPU。原始NPZ数值字段可由固定种子重新生成。
本轮不改变旧KS的5.4104%结果，不涉及测点或边界优化、真实设备、大模型或华为平台接入。
