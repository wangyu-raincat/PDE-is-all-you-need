# AGENTS.md — Codex 工作约定（ICT 创新赛 · 赛题 3）

## 背景
本仓库原是 PDE 方程发现科研原型（`pde_discovery/` + 各 `*_study/`）。比赛作品在 `competition_agent/`：
基于华为 ModelEngine **Nexent** 的「增材制造材料研发 · 实验数据资产认知与可溯源决策」多智能体。
初赛截止 **2026-12-15 18:00（北京时间）**。总规划见 `codex_plan/PLAN.md`，任务卡见 `codex_plan/tasks/`。

## 每次开工
1. 先读 `codex_plan/PLAN.md` 和指定任务卡 `codex_plan/tasks/Txx-*.md`，只做该卡范围内的事。
2. 读 `codex_plan/progress/` 里已完成任务的记录，避免重复或冲突。
3. 在分支 `comp/Txx-<短名>` 上工作；不要改写 main 的历史。

## 目录边界
- 新代码只放 `competition_agent/`、`tests/test_competition_*.py`、`scripts/competition_*.py`、`codex_plan/`。
- **只读**：`pde_discovery/` 数值引擎、所有 `*_study/`、`artifacts/`、`experiments/`、`pde_discovery/data/` 原始数据。
  需要用引擎时 import 调用；确需修改须在 progress 里写明原因并保留原测试全部通过。
- 生成物放 `competition_agent/build/`（已 gitignore），交付物放 `competition_agent/deliverables/`。

## 环境与命令
- Windows + Python 3.12。**统一使用仓库根目录 `.venv`**；在人工修好 `.venv` 之前，使用已验证可跑通接入层 93 项测试的 `competition_agent/build/t02-env`（含 pywin32）。不要再新建其他隔离环境、不要反复联网重试下载。
  缺依赖时：停止相关步骤，在 progress 里写明缺哪个包，交给人工安装，然后继续能做的部分。
- 接入层测试：`.venv\Scripts\python -m unittest discover -s tests -p "test_competition*.py" -v` —— **必须全部通过**才算任务完成。
- 全量回归：仓库历史研究分支在 Windows 上有既有失败（路径分隔符、CRLF 导致的冻结哈希不一致等）。
  判定标准是**与 main 基线对比无新增失败**，不要求全绿；不要为此修改只读目录或历史 receipt。
- MCP：`python -m competition_agent.mcp_server --transport streamable-http --port 8769`
- 新依赖写入 `competition_agent/requirements.txt` 并固定版本；优先 CPU 可跑、轻量的库。

## 效率（15 天工期）
- 只做任务卡范围内的验证。不要做超出任务卡的上游源码逐文件哈希核对、多轮镜像下载重试等。
- 同时有多个 Codex 任务时，**开工前检查 `git status` 与其他 `comp/*` 分支**；要改的共享文件（`bundle.py`、`mcp_server.py`、`service.py`）若已被别的任务改动，在独立 worktree 工作，并在 progress 写明合并注意事项。

## 硬性约束（评审会看，违反等于丢分）
- **不编造**：数值、指标、问答截图、平台运行结果必须来自真实运行；未实测的写“待实测”。
- **来源可溯**：每条事实带 `source_id + 原文引文 + 位置`（文本行号/字符跨度、PDF 页码+bbox、表格 sheet/单元格）。
- **自制资料必须标注“自制/示例”**；不得伪装成企业真实数据。
- **标准文本**：只存标准编号、条款号、标题和我们自己写的短摘要，不复制标准原文。
- **密钥**：只从环境变量读（`DEEPSEEK_API_KEY` 等），不写进代码、配置、日志、测试快照。
- **数据许可**：Mendeley 数据集 CC BY 4.0 可再分发并需署名；其他来源无许可的原始数据不进入开源包。
- 训练/开发/留出划分沿用 `experiments/ADDITIVE_REPORT.md`、`MENDELEY_REPORT.md` 的既定划分，不按结果重新分组。

## 证据存放
`competition_agent/build/` 不入 Git。任务验收用到的关键证据（acceptance.json、测试日志摘要、对照表）须复制到 `competition_agent/verification/<任务短名>/` 并提交，T09 写文档要引用。

## 完成定义（每张任务卡）
1. 任务卡中的“验收标准”逐条满足；新增功能有 `tests/test_competition_*.py` 覆盖。
2. 接入层测试全部通过；涉及共享代码时跑全量回归。
3. 写 `codex_plan/progress/Txx.md`：做了什么、改了哪些文件、测试命令与结果（贴数字）、未完成项与风险。
4. 一个任务一个 PR/合并提交，提交信息 `Txx: <中文摘要>`。

## 文风
文档中文、代码标识英文。文档先讲业务价值与结果，限制说明集中放在“适用范围与限制”一节，不要每段都写免责声明。
