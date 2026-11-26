# T00 人工准备（不派给 Codex）

1. ~~确认报名~~（已完成）。分工建议：1 人负责 Nexent 部署与平台实测（T00/T08），1 人负责资料、标注与审核（T04 标注、核对 Codex 输出），1 人负责派发 Codex 与文档（T09）。
2. **部署 Nexent v2.0**（关键路径，越早越好）
   - 机器：建议 8 核 / 16 GiB / 100 GiB；Windows 需 Docker Desktop（WSL2 后端，给 WSL 分 ≥12 GiB 内存）。
   - `git clone https://github.com/ModelEngine-Group/nexent.git && cd nexent && bash deploy.sh docker`（在 WSL 或 Git Bash 中执行）。
   - 记录：Nexent 版本号/提交号、访问地址、部署耗时、遇到的报错 → 写入 `codex_plan/progress/T00.md`（T09 的“调试迭代经验”要用）。
3. **模型**：在 Nexent 里配置可用的大模型（DeepSeek 或其他 OpenAI 兼容接口）+ 一个 embedding 模型（知识库用），以及一个**视觉模型**（T03 图片理解用，如支持图像输入的 Qwen-VL/GLM-4V 类接口）。密钥只放环境变量/平台配置。
4. **手工导出样例**：在 Nexent UI 手工建一个带 1 个 MCP 工具和 1 个子智能体的测试智能体，导出 JSON，保存到 `competition_agent/verification/nexent_v2/manual_export_sample.json`（T01 以它为准）。
5. **资料下载**（Codex 环境可能无外网）：放到 `competition_agent/assets_raw/`
   - 数据论文 PDF：DOI 10.1016/j.dib.2021.107632（PMC8813594，开放获取，先确认许可为 CC BY）。
   - Mendeley 数据集中的 3 个 XLSX 结果表（原始 ZIP 已在本地，找出来复制即可）。
   - 相关国标/ISO 的**目录页信息**（编号、名称、条款标题），不需要全文，供 T04 标准对齐用。
