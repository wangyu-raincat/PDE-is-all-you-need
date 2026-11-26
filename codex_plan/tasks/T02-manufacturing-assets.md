# T02 制造场景资产包与真实热物性推理工具

## 目标
把仓库里已有的真实增材制造/金属热杆数据（Mendeley CC BY 4.0）变成智能体的“沉睡数据资产”，并提供一个面向真实数据的推理工具 `heat_identify`，让演示主线从合成 fixture 切换到真实制造场景。

## 输入（只读）
- `pde_discovery/data/additive_heat/`（xn/xp/yn/yp × 5 次，manifest、metadata_verification.md、preparation_record.json）
- `pde_discovery/data/mendeley_heat/`（ss/al/cu × 50/75/100 档 × 多次）
- `pde_discovery/data/mendeley_heat/author_oneDkhEstimator_V1.0.0_full_source.txt`（几何参数在第 223–250 行附近）
- `experiments/ADDITIVE_REPORT.md`、`experiments/MENDELEY_REPORT.md`、`experiments/UNUSED_DATA_METADATA.md`
- 引擎：`pde_discovery/real_heat*.py`、`pde_discovery/additive_experiment.py`（先读懂其公开函数，import 调用，不改）

## 步骤
1. **资产目录** `competition_agent/scenario_am/catalog.py`：扫描上述目录，为每个文件生成资产卡片：
   `asset_id, title, modality(csv/xlsx/pdf/code/image/markdown), domain_role(raw_measurement/method_code/paper/result_table/protocol/report), source_url, doi, license, sha256, byte_size, rows/cols(表格), linked_entities[]`。
   输出 `competition_agent/scenario_am/asset_catalog.json`，并给出“沉睡度”统计（已用于建模 / 从未使用 / 已用于最终评估不可再当盲测）。
2. **场景说明** `competition_agent/scenario_am/SCENARIO.md`：业务痛点（实验室多年测温数据散落、参数口口相传、工艺仿真缺可信材料输入、复测成本高）、角色（材料工程师/工艺工程师/质量）、5 个决策问题（见 PLAN.md §2）、数据许可与署名。
3. **推理工具** `competition_agent/scenario_am/heat_tools.py`：封装既有方法为
   - `heat_identify(asset_group, split='train')` → 用训练/开发记录拟合并返回模型项、系数、与已冻结结果的一致性，标注方法名“给定实测边界的四内点条件预测（物理先验 D+h+c）”。
   - `heat_holdout_summary(asset_group)` → 读取已有留出评分结果（不重跑最终评估、不改划分），返回逐记录误差及来源文件路径。
   - `heat_compare(group_a, group_b)` → 比较两组拟合系数及其 bootstrap 区间（若引擎支持；不支持则仅报告点估计并说明）。
   所有返回带 `evidence: [{asset_id, locator}]`。
4. 在 `mcp_server.py` 注册上述 3 个工具（新增工具，不动原 15 个），并更新 `bundle.py` 的角色工具绑定（pde_agent 获得 heat_*）。
5. 预处理：让 `competition_agent.preprocess` 能读取 catalog 中的 CSV 头部与 manifest 作为元数据证据（不把 CSV 正文当训练输入，数值走 heat_tools）。

## 验收标准
- catalog 覆盖全部原始 CSV 与说明文件，哈希与已有 manifest 一致（测试校验）。
- `heat_identify('xn')` 结果与 `experiments/ADDITIVE_REPORT.md` 中冻结模型一致（模型项相同，系数在浮点容差内）。
- `heat_holdout_summary` 返回的 12 条增材留出误差与报告表逐条一致。
- 新测试 `tests/test_competition_am_*.py`；原 15 工具测试全部通过。
