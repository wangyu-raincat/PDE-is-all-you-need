# T01 Nexent v2.0 兼容性重核与接入包更新

## 目标
现有 `competition_agent/bundle.py` 按 Nexent 旧提交 d289f67 生成配置。Nexent 已发布 v2.0，需确认导入 schema、MCP 注册、Skill 上传格式仍然可用，并让接入包能一键生成 v2.0 可导入物。

## 步骤
1. 拉取 Nexent 最新 release/tag 源码（只读，放 `competition_agent/build/nexent_src/`，不提交）。对比旧提交与新版本中：
   `backend/contracts/agent.py`（或其新位置）、`sdk/nexent/core/agents/agent_model.py`、`backend/management/services/agent/transfer.py`、`backend/management/services/tool/discovery.py`、Skill 相关代码与 `doc/docs/zh/integration/integration-in/{mcp,skills}.md`。
2. 写 `competition_agent/verification/nexent_v2/SCHEMA_DIFF.md`：逐字段列出差异（新增/删除/改名/类型变化）、对我们配置的影响。
3. 若有 `manual_export_sample.json`（T00 第 4 步），以它为最高优先级参照。
4. 修改 `bundle.py`：新增 `--nexent-version {legacy,v2}`，默认 v2；保留 legacy 生成路径与原测试。更新离线 Pydantic 校验片段到 v2 版本并注明来源提交号与 MIT 许可。
5. MCP：确认 v2 支持的传输方式（streamable-http/SSE），在 README 写清楚容器访问宿主机地址 `host.docker.internal`。给 `mcp_server.py` 增加 `--host` 参数说明（已有则补文档），默认仍 127.0.0.1。
6. 更新 `config/bundle-status.json` 的 `schema_reference_commit`。

## 验收标准
- `python -m competition_agent.bundle --nexent-version v2 --model-name X --mcp-url http://host.docker.internal:8769/mcp` 生成配置 + 6 个 Skill ZIP，并通过 v2 离线 schema 校验。
- `tests/test_competition_bundle_v2.py` 覆盖：字段齐全、子智能体引用正确、工具 source/usage 对应 MCP 服务名、Skill ZIP 根目录只含 SKILL.md（若 v2 格式要求不同，以 v2 为准并在测试中体现）。
- SCHEMA_DIFF.md 写明“离线校验通过 ≠ 平台导入通过，平台导入在 T08 实测”。
