# 制造资产 v1 验收证据

原 T02 数值验收文件以字节不变方式归档为 acceptance.json；其环境版本和结果以文件内容为准。test-summary.json 分别记录原 T02 的 93/93 和合并后的 103/103 接入层结果、命令及源日志哈希。

readonly-hashes.json 为原始核验基线，readonly-verification.json 是本次合并后逐文件复核结果：137/137 未变化。bundle-summary.json 记录 v2.8.5 离线打包的 6 个智能体、6 个 Skill ZIP 和 18 个 MCP 工具，以及 Skill 与 T02 一致性检查。

evidence-provenance.json 记录原文件位置与哈希。本次未重新运行最终留出评分；这些证据不表示真实 Nexent 平台已部署或通过导入。历史全库失败及环境限制仍见 codex_plan/progress/T02.md，合并验证见 codex_plan/progress/MERGE-1.md。
