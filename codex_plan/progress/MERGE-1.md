# MERGE-1：T01/T02 集成

已在 `competition/nexent-multi-agent` 完成 T02 的非快进合并。接入层 **103/103 通过**；v2 bundle 实测生成 **6 个智能体、6 个 Skill ZIP、18 个 MCP 工具**。关键证据单独提交，空分支已清理。

## 提交与冲突解决

- 合并前 HEAD：`23fa4c1`，规划/规则提交 `32cd686`；保险标签 `before-merge-1` 保留。
- 合并提交：`e0f4b09`（双父：`23fa4c1`、`ae74a9a`）。执行 `git merge --no-ff comp/T02-manufacturing-assets`。
- `bundle.py` 保留 T01 的 v2/legacy 选择、`schema.model_validate`、UTF-8 JSON 写入及 Skill 显式绑定导入顺序；合入 T02 制造提示与角色工具绑定，工具数/导入提示更新为 18。
- `service.py` import 取等价的 T02 写法；源码哈希目录元组保留 `competition_agent/scenario_am`。
- 6 份 SKILL.md 及对应 ZIP 内文本与 T02 一致。确认本次合并文件无冲突标记、无未解决索引项；没有纳入仅 CRLF/LF 差异的改动。
- `test_competition_bundle_v2.py` 的工具数改为 18。旧 AST 检查不能识别 T02 动态注册函数，改为无网络监听地读取实际 MCP 注册目录。legacy 原快照仍为 15 工具，测试现在核对旧格式及全部原工具不变，并明确要求仅 pde_agent 增加三个 heat_*；制造提示允许按任务更新。未删除或跳过失败测试。

## 验证

运行目录：仓库根目录；只使用现有 `competition_agent/build/t02-env/Scripts/python.exe`，未新建环境或下载依赖。

```powershell
competition_agent/build/t02-env/Scripts/python.exe -m unittest discover -s tests -p "test_competition*.py" -v
competition_agent/build/t02-env/Scripts/python.exe -m competition_agent.bundle --nexent-version v2
```

- 最终接入层：103 项，总通过 103、失败 0、错误 0、跳过 0；18.825 秒，结果 `OK`。源日志：`competition_agent/build/MERGE-1/integration-tests.log`。
- Bundle：nexent_version=v2；schema_reference_tag=v2.8.5；schema_reference_commit=`628b53d66c875094eb6fa1ed3b2dff6590bad9e5`；agent_count=6、skill_count=6、mcp_tool_count=18。
- 六个 ZIP：pde-coordinator、asset-agent、graph-agent、pde-agent、experiment-agent、validation-agent；每包根目录仅 SKILL.md，文本与 T02 一致。
- pde_agent 绑定：case_status、heat_identify、heat_holdout_summary、heat_compare、pde_identify。
- 导入步骤保留显式 Skill 绑定、绑定模型后发布与核对子版本；`nexent_live_verified=false`，本次仅离线格式与本地 MCP 验证。
- `git diff --check`、暂存区检查通过。未重跑科学实验或最终留出；按本次合并要求只进行接入层、打包与已有只读基线核验，未扩展运行历史全库回归。

## 证据独立提交

证据提交：`73d665b`；目录 `competition_agent/verification/am_v1/`。

- acceptance.json：从 `build/T02/acceptance.json` 原样复制，SHA-256 相同；是原 T02 数值验收，不冒充本次重新评分。
- test-summary.json：原 T02 93/93 与本次合并 103/103 的结果、命令、耗时和源日志哈希。
- readonly-hashes.json / readonly-verification.json：原基线与本次逐文件结果，**137/137 未变化，差异 0**。
- bundle-summary.json：6/6/18 输出、工具名、ZIP 名称与 Skill 一致性。
- evidence-provenance.json / README.md：来源和适用范围。

## 分支清理

在主仓库路径确认 `comp/T01-nexent-v2-compat` 位于 `e54a123`、无未合入当前分支的提交后，使用 `git branch -d` 删除；已执行 `git worktree prune`。

`comp/T01-nexent-v2-isolated`、`comp/T02-manufacturing-assets` 及其工作内容保留；保险标签保留。原有未跟踪的 cloud-delivery/ 未纳入提交。本次按请求完成本地提交，未推送远端。

## git log --oneline --graph -10

以下是证据提交完成后、提交本进度记录前的快照：

```text
* 73d665b T02: 归档制造资产验收与合并核验证据
*   e0f4b09 MERGE-1: 合并T02制造资产与Nexent v2接入
|\
| * ae74a9a T02: 增材制造资产目录与可溯源热推理工具
* |   23fa4c1 Merge T01: Nexent v2 兼容
|\ \
| * | 633f510 T01: 更新Nexent v2接入兼容并记录待完成验收
| |/
* / 32cd686 Add Codex plan and AGENTS rules
|/
* e54a123 Refine DeepSeek role contexts and pass live MCP acceptance
* a870c26 Add bounded live DeepSeek MCP acceptance and retain first-run failures
* 13b7991 Prepare and prioritize evidence before agent orchestration
* 593f2d4 Improve evidence grounding, structured handoffs and repeat-safe local workflows
```
