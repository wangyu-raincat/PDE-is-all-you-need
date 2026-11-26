# T06 Skill 分层与可复用工作流模板 + 跨行业迁移示例

## 目标
评分项要求“Skill 分层编排逻辑清晰”“沉淀可复用、可快速部署的 Skill 工作流模板，具备通用推广价值”。

## 步骤
1. **三层 Skill**（目录 `competition_agent/skills/`）：
   - L1 原子技能（每个对应一组 MCP 工具的正确用法与参数约束）：asset-parse、kg-query、ontology-review、heat-reason、decision-trace…
   - L2 角色技能：现有 6 个角色（精简重复的免责声明，公共约束抽到 `_shared/RULES.md` 并在各 SKILL.md 引用）。
   - L3 工作流模板：`workflow-asset-to-decision`（资产认知→本体→检索推理→决策→验证），参数化为行业配置。
2. **行业配置** `competition_agent/templates/industries/manufacturing_am.yaml`：种子本体路径、标准表、决策规则、推理工具映射、示例问题。模板引擎据此生成 Nexent 配置 + Skill ZIP（复用 bundle.py）。
3. **迁移示例**（证明通用性，P1，可裁剪）：`templates/industries/government_policy.yaml`——用 2–3 份公开政策文件（如高新技术企业认定相关公开办法，人工下载）走同一模板：资产识别→条款本体→“企业是否满足某条件”的可溯源判断。推理工具换成规则推理，不需要数值引擎。**只做最小可运行演示，文档里标“迁移示例”**。
4. 一键部署脚本 `python -m competition_agent.template deploy --industry manufacturing_am --model-name ... --mcp-url ...` 输出可导入 Nexent 的完整包。

## 验收标准
- 两个行业配置都能生成通过 v2 离线校验的接入包。
- 迁移示例在本地跑通 2 个问答并生成 decision trace。
- 文档 `competition_agent/templates/README.md`：新增一个行业需要改哪些文件（目标：只改 YAML + 种子本体 + 规则表）。
