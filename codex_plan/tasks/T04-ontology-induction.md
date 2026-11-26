# T04 低资源本体半自动构建算法 + 评测（算法创新 25% 的核心）

## 方法名（暂定）
**证据约束的主动本体归纳（Evidence-Constrained Active Ontology Induction, ECAOI）**
低资源 = 只有十几份资料、没有领域标注语料、没有大规模预训练领域模型。

## 算法组成
1. **种子本体**（人工 30 分钟可写完）：`competition_agent/ontology/seed.yaml`，约 15 类：Organization, Equipment, Sensor(热电偶), MeasurementPoint, Specimen(试样/杆), Material, BuildDirection, PorosityClass, ProcessCondition, Experiment(测试次), ObservationField, Unit, PhysicalModel, ModelParameter, ValidationResult, StandardClause；约 12 关系，带 domain/range 与基数约束。
2. **标准术语表**：`ontology/standards.yaml`，只存编号、名称、发布/实施日期、条款章节标题、我们自写的一句话释义，不复制标准正文。已核实（2026-11-26，国家标准馆等公开目录）：
   - GB/T 22588-2008《闪光法测量热扩散系数或导热系数》，2008-12-15 发布，2009-10-01 实施，现行。
   - GB/T 3651-2008《金属高温导热系数测量方法》，2008-06-09 发布，2008-12-01 实施，代替 GB/T 3651-1983；章节：范围、规范性引用文件、测量原理、试样要求、测量装置与仪器、测量程序、数据处理与测量结果、测量记录与报告、附录A 装置校准。
   - ASTM E1461-13(2022) Thermal Diffusivity by the Flash Method（其适用范围强调近乎完全致密、低孔隙、均匀各向同性材料——这一点可与 p/n 孔隙类别对齐，形成“方法适用性”推理）。
   - 本项目方法是“一维瞬态对流模型参数估计”，**不是**上述任一标准方法；对齐关系应是 `different_method_from` / `comparable_quantity`，不能写成“符合标准”。
3. **多抽取器候选生成**：
   - E1 结构映射：CSV/XLSX 表头、manifest、源码字面量 → 规则映射到本体。
   - E2 模式抽取：句式模板 + 单位/数值正则。
   - E3 LLM few-shot：给种子本体 + 3 个示例，要求输出三元组且**必须附逐字引文**，引文不逐字命中即丢弃。
4. **证据与约束校验**：引文定位、domain/range、单位量纲一致性、跨文档冲突检测（沿用现有冲突保留逻辑）。
5. **多信号置信度**：`score = f(抽取器一致票数, 跨文档佐证数, 约束满足, 标准术语对齐分, 抽取器先验精度)`，用 logistic 组合，权重在开发集上拟合（小样本，写清楚）。
6. **三档分流**：高置信自动进入草案版本；中置信进入**审核队列，按期望信息增益排序**（不确定度 × 该节点在决策问题证据路径上的中心度），让人用最少确认次数解锁最多决策问题；低置信丢弃但留痕。
7. **新概念/新关系归纳**：对未能映射的高频名词短语聚类，默认用字符 n-gram TF-IDF（scikit-learn 已有）；**不要引入 torch/sentence-transformers**（体积大、本机下载困难）。embedding 仅作为可选插件，缺失时自动降级。
8. **增量更新**：新资料到达只重算受影响的实体邻域；本体与图谱各自版本号、diff、可回滚；冻结后的案例不受影响。
9. **标准对齐**：概念 ↔ 标准术语映射（别名 + embedding 相似度 + 人工确认），输出 `alignment.json`，在决策报告里显示“本结论涉及的方法与 GB/T xxxx 第 x 条的差异”。

## 已知素材（来自 T03）
- 数据论文 PDF 第 3 页有构建方向 x/y、孔隙类别 p/n 的原文；“每杆五次测试”在该论文中**没有**原句，证据来自作者源码第 227/232/237/242 行与 AdditiveRodResults.xlsx 各表 B1:F1。这正好是跨文档佐证的天然样例，用作多信号置信度的示范。
- Wind Results.xlsx / Temperature Results.xlsx 含风速与温度档位结果，可为 ProcessCondition 类提供实例。

## 评测（必须真实执行，结果写进 T09）
- 金标准：用 `scripts/competition_annotate.py`（Codex 写的命令行/简易网页标注工具）由 2 名队员对资产包标注 80–150 条三元组，计算标注一致性（Cohen's κ）。
- 对照：B1 仅规则（E1+E2）、B2 仅 LLM（E3）、B3 全部候选全人工审核、Ours。
- 指标：P/R/F1；**每得到 1 条正确三元组所需人工确认次数**；引文有效率；新增 3 份资料时的增量更新耗时；在 5 个决策问题上“证据路径可完整解锁”所需的最少审核次数。
- 结果写 `competition_agent/verification/ontology_eval/REPORT.md`，失败项与小样本局限照实写。

## 接口
- MCP 新工具：`ontology_induce(case_id)`、`ontology_review_queue(case_id, limit)`、`standard_align(concept)`；人工审核仍只走本地操作员命令（沿用 `demo review-ontology`，扩展为批量）。

## 验收标准
- 一条命令跑完整评测并生成 REPORT.md 与 JSON 结果；固定随机种子可复现。
- 无 LLM 密钥时 E3 跳过，其他流程可运行；测试用打桩 LLM。
- 测试 `tests/test_competition_ontology_*.py`。
