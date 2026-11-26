# T08 Nexent 平台实测与调试迭代（人主导，Codex 辅助）

## 人工
1. 启动 MCP（`--host 0.0.0.0`，仅内网），在 Nexent 注册 `pde_lab_tools`，确认工具数量与 T02/T04/T05 新增工具一致。
2. 上传 Skill ZIP，导入智能体 JSON，绑定模型，发布子智能体。
3. 知识库：按 T05 清单上传资产（PDF/XLSX/文本），记录切片与检索设置。
4. 逐条运行 5 个决策问题 + 4 个边界探针，**每条保存：对话截图、Nexent 调用链/日志截图、MCP 端 tool-trace.json**。
5. 每次失败记录到 `verification/nexent_live/ITERATIONS.md`：现象 → 原因 → 修改（提示词/工具/Skill）→ 复测结果。这一节就是提交材料要求的“调试迭代经验”。

## Codex
- 写 `scripts/competition_nexent_collect.py`：从 MCP 端日志与本地 trace 汇总每轮运行（问题、调用的工具序列、耗时、是否通过预期），生成 `verification/nexent_live/RUNS.md` 表格。
- 根据 ITERATIONS.md 中的问题修复工具/Skill（每次修复单独提交）。
- 导出最终在平台上运行通过的 agent JSON 到 `deliverables/`（平台导出物优先于我们生成的配置）。

## 验收标准
- 5 个决策问题在 Nexent 上真实运行有截图；结果与预期不符的照实记录，不删除。
