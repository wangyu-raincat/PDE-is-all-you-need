# T10 打包、开源准备与提交

1. 精简作品仓库（Codex）：新建 `deliverables/opensource/` 导出脚本，只包含 `competition_agent/`、所需的 `pde_discovery` 引擎模块、测试、CC BY 4.0 数据（附署名与 DOI）、LICENSE、README；排除无再分发许可的数据和大体积历史工件。在干净 venv 中按 README 从零安装跑通测试，记录结果。
2. 许可检查（Codex）：列出所有第三方数据/代码来源、许可、是否随包分发，写 `THIRD_PARTY.md`。
3. 密钥扫描（Codex）：`git grep` 常见密钥模式 + 检查 build/verification 中的日志，不得含密钥。
4. 提交（人）：12/15 中午前上传，留 6 小时缓冲；上传后下载回来检查文件可打开。
5. 入围后需开源到 GitCode 或 GitHub——提前准备好仓库但先设私有。
