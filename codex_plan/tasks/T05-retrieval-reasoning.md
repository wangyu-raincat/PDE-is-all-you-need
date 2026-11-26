# T05 “检索-推理”双驱动与决策溯源

## 目标
把赛题要求的“检索-推理双驱动协同执行流”和“跨文档多跳推理、决策链路可溯源”做成显式、可演示、可测试的结构。

## 步骤
1. **检索侧**
   - 向量检索：资产块（T03 blocks）用小型中文 embedding 建本地索引；同时提供导入 Nexent 知识库的文件清单与切片说明（平台内用 Nexent 原生知识库）。
   - 词面检索：沿用现有 BM25。
   - 图谱检索：扩展 `evidence_path`，支持按本体类型约束的多跳查询 `kg_query(pattern)`，例如 `Specimen -has_class-> PorosityClass` 且 `Specimen -tested_in-> Experiment -observed-> ObservationField`。
   - 融合：RRF 融合三路结果，返回统一 evidence 列表。
2. **推理侧**
   - 数值推理：`heat_identify / heat_compare`（T02）、`pde_identify`（合成对照）。
   - **补一个工具 `heat_repeatability(group)`**：对同杆 5 次测试记录各自独立拟合 D/h/c，返回均值、极差、标准差及逐条结果，作为“同杆复测离散度”。T02 的 heat_compare 没有区间，决策问题 1（xp 与 xn 有无可辨别差异）要靠它判断差异是否大于复测离散度。注意：编号 3–5 已被查看过，只能称“描述性复测离散度”，不能称盲测或置信区间；不改动已冻结的训练/开发/留出划分与评分。
   - 符号/规则推理：决策规则表 `competition_agent/decision/rules.yaml`（如“同组重复次数<3 → 不建议作为仿真输入”“存在未解决冲突 → 结论不可用于业务”“方法与标准不一致 → 提示需对标复测”）。
   - LLM 推理：只做组织与解释，所有数值引用工具返回值。
3. **决策对象与溯源**：新 MCP 工具
   - `decision_make(case_id, question)` → 返回 `{conclusion, confidence_level(定性), gaps[], next_experiments[], trace_id}`
   - `decision_trace_get(trace_id)` → 结论 ← 规则/模型 ← 图谱路径 ← 原文引文（含 PDF 页码 bbox / 表格单元格 / 源码行号）的完整树，可导出 JSON。
   trace 写入现有 SQLite 事件哈希链。
4. **执行流**：协调者按“检索→推理→（证据不足则再检索/提议实验）→决策→验证”循环，最多 N 轮；`case_handoff` 增加 `loop_reason`（evidence_gap / model_ambiguous / conflict）。

## 验收标准
- PLAN.md §2 的 5 个决策问题在本地 DeepSeek 运行器（扩展 `deepseek_acceptance.py`）中全部跑通，每个答案都能用 `decision_trace_get` 展开到原文。
- 问题 1 的证据路径至少跨 3 份不同模态资产（论文 PDF、源码、CSV/结果表）。
- 问题 4、5 按预期保留冲突/拒绝，不被模型绕过。
- 测试 `tests/test_competition_decision_*.py`；真实模型运行记录保存在 `verification/decision_v1/`（含请求数、工具调用数、token，照实写）。
