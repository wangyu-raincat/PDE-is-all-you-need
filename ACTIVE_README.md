# 有预算的 PDE 主动发现与结构不确定性

这是 `experiment/active-pde-discovery` 独立实验，回退基线为
`experiment/coefficient-reliability` 的 `da846c189bfe0ced47a3515fad55d43333a4106b`。
旧应用和旧实验不被替换；本轮不自动合入。结果与决定见
[实验报告](active_study/REPORT.md) 和 [设计审阅](active_study/DESIGN_REVIEW.md)。

本轮把两个方向连成一个循环：由完整轨迹自助重采样重新选择结构，保留观测上难区分的候选；
用候选对合法新初值的预测分歧选一次追加实验，获取观测后重新识别，直到预算用完。
这是一维周期 PDE 的数值控制器，当前未接大模型、华为平台或实际设备。

## 运行与复现

在源码根目录使用 Python 3.12；CPU 即可，不需要 API Key：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-repro.txt
python -m active_study.app --port 8767
```

打开 <http://localhost:8767> 查看已保存实验，切换任务、策略与预算，检查候选方程、
自助频率、选择依据和实验账本。页面只读，不会训练、追加实验或重新评分。
旧应用仍用 `python -m pde_discovery serve --port 8765` 启动。

从不带私有字段缓存的全新源码目录复现实验：

```bash
python -m active_study.run prepare
python -m active_study.run freeze --output artifacts/active_reproduction --workers 3
python -m active_study.run evaluate --output artifacts/active_reproduction --workers 3
python scripts/audit_active_study.py --output artifacts/active_reproduction
python scripts/render_active_study.py --output artifacts/active_reproduction
python -m active_study.app --output artifacts/active_reproduction --port 8767
```

`prepare` 生成 120 次初始观测，包含所有策略共享的每条件 3 次训练和 2 次开发实验；
这 5 次全部计费。`freeze` 为四种策略分别执行至多 4 次选中的追加实验，保存全部候选、
动作评分与账本。所有 96 个策略回合冻结后，`evaluate` 才生成 36 条最终干净参考轨迹。
24 个条件来自 12 组参数各配无噪和 1% 噪声，不能当作 24 个独立物理系统。
每个条件/策略保留 5 个预算槽位及每槽位 3 次最终预测；失败不得从分母删除。

固定种子的最终字段现已属于可复现基准，再次运行应称复现或回顾性实验；不能据此反复调参后继续称盲验。
已有初始或部分结果时命令会停止以保存记录，使用新的输出目录复现回合，不要覆盖首次文件。
若修改科学源或协议，应创建新的开发/冻结实验，而非修改现有回执绕过核验。

仅运行开发联调（不会生成正式最终数据）：

```bash
python -m active_study.run develop --output active_study/development/new_pilot --workers 3
python -m unittest discover -s tests -v
```

## 实验范围与指标

六族为衰减、扩散、四阶耗散、对流扩散、反应扩散和 Burgers；采用共同九项候选库，
没有按任务强制真项。每次观测是 111 帧 × 64 个均匀周期测点，共 7,104 个标量。
初值从 32 个符合 RMS=0.4、幅值上限 1.2 的 Fourier 方案中选择。
本轮只优化初值，传感器网格和周期边界固定；同成本动作并不检验异质成本优化。
预算 5 到 9 次对应 35,520 到 63,936 个训练/开发测量标量，初始与前缀重叠帧只计一次。

四个策略共用同一个识别器与初始观测：固定丰富激励、随机、集成预测方差、结构分歧。
固定丰富激励会直接激发全部六个允许波数，是有竞争力的对照。
结构分歧先对各支持集等权，再比较结构间方差与结构内系数方差和估计噪声；
它是判别用评分，不是经过证明的信息增益。候选一致或不可推进时会保留原因并探索，
不会将候选一致视为已确立唯一真实 PDE。

主要指标是完整项恢复、多余/缺失项、逐真项系数误差；另报候选集是否含真结构、集合大小、
错误单集合以及正负项分开计算的 Brier 描述性诊断。辅助预测使用 11 帧历史估计起点，
保持之后完整 1.0 时间单位，不使用未来观测。离线达标定义预先固定为：精确结构，
最坏逐真项误差 ≤10%，三条最终轨迹平均相对 L2 ≤5%。
只有双方达到同一预定 80% 条件达标率时才比较实验节省；否则节省字段为 null。
真值判定的首次达标预算只用于离线分析，不用于在线停止。

## 限制与已有工作

- 本轮全部是受控合成数据，不是真实测量或新实物实验；不替代旧真实热杆记录。
- 每轮 B=8 次完整轨迹重采样提供算法稳定性频率，未校准为项存在概率，未提供后验或覆盖保证。
  重采样不会增加独立实验数量。观测上难区分的集合不等于数学上等价的 PDE。
- 训练谱过滤假设高频主要是白测量噪声，可能删除弱物理模式；候选库也可能遗漏真实物理。
- 校准通过仅覆盖声明的 12 个数值检查，不能推及整个连续参数域。
- E-SINDy 已包含集成稀疏识别与主动学习；这里的方差策略是受其启发的比较，不声称精确复现论文。
  本原型的贡献是预算、歧义、完整失败分母和冻结验证的集成与审计，不声称首次发明主动发现。
- 旧系数实验的 KS 最坏平均预测 5.4104% 仍保留。本轮新任务即使达到 5% 也不代表修复旧条件。

详细规则和文献身份保存在 [protocol.json](active_study/protocol.json) 与
[references.json](active_study/references.json)。
