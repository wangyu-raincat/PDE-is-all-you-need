# Nexent v2 导入兼容性核查

本次核查固定官方最新稳定 release v2.8.5（2026-11-26 查询），提交 `628b53d66c875094eb6fa1ed3b2dff6590bad9e5`。`--nexent-version v2` 指这个固定目标，不表示保证所有 v2.x 兼容。旧参照为 `d289f67ef11558bbdb5d9a750dbb8fefcbce8cfc`；它已经包含 v2.6 字段，不能将它误称为 v1。未发现 T00 的 `manual_export_sample.json`；收到平台导出样本后应优先复核。

## 结果

四个导入模型的 42 个字段逐项比较：新增 0、删除 0、改名 0、类型或默认值变化 0。保留原配置字段即可通过两套离线 schema。实际变化是契约类从旧 `backend/contracts/agent.py` 移到本 release 的 `backend/consts/model.py`；`ToolConfig` 仍在 `sdk/nexent/core/agents/agent_model.py`。旧版与新版本单独保存校验片段，不能因为字段相同就省略版本来源。

- `managed_agents` 仍是整数 ID 列表；导入器先创建子节点，再将原 ID 映射到新 ID，关系使用子版本 1。
- 模型仍优先有效 `model_ids`，其次按 `model_names` 查显示名。我们只填写显示名，`X` 是离线验收占位符，平台必须使用已配置的真实显示名。
- MCP 导入流程依赖租户已登记工具；JSON 中的 `mcp_info` 不代替平台注册。工具仍是 `source=mcp`、`usage=pde_lab_tools`，`class_name` 为实际 MCP 工具名。导入器按 class_name+source 查工具；平台须检查同名工具没有连错服务。
- 旧与新 `transfer.py` 内部身份参数、数据库和常量导入路径变化，不影响生成 JSON。v2.8.5 使用 authorization Header；我们不调用这些后端内部函数。
- 普通 JSON 导入函数不创建 Skill 实例；`import_agent_with_skills_impl` 才在组合导入后逐一绑定。现有“JSON + 六个独立 Skill ZIP”路线可保留，但上传后必须手工为六个角色各绑定同名 Skill，再发布并检查子版本。不要将上传成功等同于角色已绑定。
- v2 Skill 文档接受根目录 SKILL.md，SDK 以 ZIP 魔数或扩展名识别。文档对“也可在子目录”与 FAQ 存在不一致；本包选双方共同支持的根目录，仅包含 SKILL.md，保留 YAML name/description。`skill/service.py::_save_skill_upload` 与 SDK 要求根目录 ZIP 显式提供 skill_name；上传时填写 ZIP 文件名去掉扩展名，不能假定仅从 Front Matter 取名。
- `tool/discovery.py::_create_mcp_transport` 对 `/sse` 使用 SSETransport，对 `/mcp` 使用 StreamableHttpTransport；其余后缀默认后者。两者支持，推荐 `/mcp`。文档侧仍以 HTTP/SSE 表述，以源码为具体判据。

## 逐字段对比

类型与默认值取自 AST；“必填”表示无默认表达式。下表同时列出未改变字段，避免遗漏字段被误判为删除。

| 字段 | legacy 类型与默认值 | v2.8.5 类型与默认值 | 差异 |
|---|---|---|---|
| ToolConfig.class_name | `('str', "Field(description='Tool class name')")` | `('str', "Field(description='Tool class name')")` | 不变 |
| ToolConfig.name | `('Optional[str]', "Field(description='Tool name')")` | `('Optional[str]', "Field(description='Tool name')")` | 不变 |
| ToolConfig.description | `('Optional[str]', "Field(description='Tool description', default=None)")` | `('Optional[str]', "Field(description='Tool description', default=None)")` | 不变 |
| ToolConfig.description_zh | `('Optional[str]', "Field(description='Chinese tool description', default=None)")` | `('Optional[str]', "Field(description='Chinese tool description', default=None)")` | 不变 |
| ToolConfig.inputs | `('Optional[str]', "Field(description='Tool inputs', default=None)")` | `('Optional[str]', "Field(description='Tool inputs', default=None)")` | 不变 |
| ToolConfig.output_type | `('Optional[str]', "Field(description='Tool output type', default=None)")` | `('Optional[str]', "Field(description='Tool output type', default=None)")` | 不变 |
| ToolConfig.params | `('Dict[str, Any]', "Field(description='Initialization parameters', default=None)")` | `('Dict[str, Any]', "Field(description='Initialization parameters', default=None)")` | 不变 |
| ToolConfig.source | `('str', "Field(description='Tool source, can be local or mcp', default='local')")` | `('str', "Field(description='Tool source, can be local or mcp', default='local')")` | 不变 |
| ToolConfig.usage | `('Optional[str]', "Field(description='MCP server name', default=None)")` | `('Optional[str]', "Field(description='MCP server name', default=None)")` | 不变 |
| ToolConfig.metadata | `('Optional[Dict[str, Any]]', "Field(description='Metadata', default=None)")` | `('Optional[Dict[str, Any]]', "Field(description='Metadata', default=None)")` | 不变 |
| ToolConfig.labels | `('Optional[List[str]]', "Field(description='Tool labels for filtering', default=None)")` | `('Optional[List[str]]', "Field(description='Tool labels for filtering', default=None)")` | 不变 |
| ExportAndImportAgentInfo.agent_id | `('int', '必填')` | `('int', '必填')` | 不变 |
| ExportAndImportAgentInfo.tenant_id | `('Optional[str]', 'None')` | `('Optional[str]', 'None')` | 不变 |
| ExportAndImportAgentInfo.name | `('str', '必填')` | `('str', '必填')` | 不变 |
| ExportAndImportAgentInfo.display_name | `('Optional[str]', 'None')` | `('Optional[str]', 'None')` | 不变 |
| ExportAndImportAgentInfo.description | `('str', '必填')` | `('str', '必填')` | 不变 |
| ExportAndImportAgentInfo.author | `('Optional[str]', 'None')` | `('Optional[str]', 'None')` | 不变 |
| ExportAndImportAgentInfo.max_steps | `('int', '必填')` | `('int', '必填')` | 不变 |
| ExportAndImportAgentInfo.requested_output_tokens | `('Optional[int]', 'Field(default=None, gt=0)')` | `('Optional[int]', 'Field(default=None, gt=0)')` | 不变 |
| ExportAndImportAgentInfo.is_main_agent | `('bool', 'True')` | `('bool', 'True')` | 不变 |
| ExportAndImportAgentInfo.provide_run_summary | `('bool', '必填')` | `('bool', '必填')` | 不变 |
| ExportAndImportAgentInfo.allow_chat_metadata | `('bool', 'False')` | `('bool', 'False')` | 不变 |
| ExportAndImportAgentInfo.enable_protocol_repair_retry | `('bool', 'False')` | `('bool', 'False')` | 不变 |
| ExportAndImportAgentInfo.verification_config | `('Optional[Dict[str, Any]]', 'None')` | `('Optional[Dict[str, Any]]', 'None')` | 不变 |
| ExportAndImportAgentInfo.context_policy | `('Optional[Dict[str, Any]]', 'None')` | `('Optional[Dict[str, Any]]', 'None')` | 不变 |
| ExportAndImportAgentInfo.prompt | `('Optional[str]', 'None')` | `('Optional[str]', 'None')` | 不变 |
| ExportAndImportAgentInfo.enabled | `('bool', '必填')` | `('bool', '必填')` | 不变 |
| ExportAndImportAgentInfo.tools | `('List[ToolConfig]', '必填')` | `('List[ToolConfig]', '必填')` | 不变 |
| ExportAndImportAgentInfo.managed_agents | `('List[int]', '必填')` | `('List[int]', '必填')` | 不变 |
| ExportAndImportAgentInfo.model_ids | `('Optional[List[int]]', 'None')` | `('Optional[List[int]]', 'None')` | 不变 |
| ExportAndImportAgentInfo.model_names | `('Optional[List[str]]', 'None')` | `('Optional[List[str]]', 'None')` | 不变 |
| ExportAndImportAgentInfo.business_logic_model_id | `('Optional[int]', 'None')` | `('Optional[int]', 'None')` | 不变 |
| ExportAndImportAgentInfo.business_logic_model_name | `('Optional[str]', 'None')` | `('Optional[str]', 'None')` | 不变 |
| ExportAndImportAgentInfo.skill_names | `('Optional[List[str]]', 'None')` | `('Optional[List[str]]', 'None')` | 不变 |
| ExportAndImportAgentInfo.model_params_override | `('Optional[Dict[str, Any]]', 'None')` | `('Optional[Dict[str, Any]]', 'None')` | 不变 |
| ExportAndImportAgentInfo.greeting_message | `('Optional[str]', 'None')` | `('Optional[str]', 'None')` | 不变 |
| ExportAndImportAgentInfo.example_questions | `('Optional[List[str]]', 'None')` | `('Optional[List[str]]', 'None')` | 不变 |
| MCPInfo.mcp_server_name | `('str', '必填')` | `('str', '必填')` | 不变 |
| MCPInfo.mcp_url | `('str', '必填')` | `('str', '必填')` | 不变 |
| ExportAndImportDataFormat.agent_id | `('int', '必填')` | `('int', '必填')` | 不变 |
| ExportAndImportDataFormat.agent_info | `('Dict[str, ExportAndImportAgentInfo]', '必填')` | `('Dict[str, ExportAndImportAgentInfo]', '必填')` | 不变 |
| ExportAndImportDataFormat.mcp_info | `('List[MCPInfo]', '必填')` | `('List[MCPInfo]', '必填')` | 不变 |

## 来源与校验范围

[官方 release](https://github.com/ModelEngine-Group/nexent/releases/tag/v2.8.5)；[v2 契约](https://github.com/ModelEngine-Group/nexent/blob/628b53d66c875094eb6fa1ed3b2dff6590bad9e5/backend/consts/model.py)；[导入实现](https://github.com/ModelEngine-Group/nexent/blob/628b53d66c875094eb6fa1ed3b2dff6590bad9e5/backend/management/services/agent/transfer.py)；[MCP 发现](https://github.com/ModelEngine-Group/nexent/blob/628b53d66c875094eb6fa1ed3b2dff6590bad9e5/backend/management/services/tool/discovery.py)；[Skill 格式](https://github.com/ModelEngine-Group/nexent/blob/628b53d66c875094eb6fa1ed3b2dff6590bad9e5/doc/docs/zh/integration/integration-in/skills.md)。逐文件 URL、SHA-256 和 Git blob 哈希见 source.json，MIT 许可见 NEXENT-LICENSE.txt。

schema_reference.py 的四个类逐字摘录自目标源码，局部补入导入语句与仅接受 None 的 context_policy helper。本生成器不输出自定义 context_policy；这里不模拟平台完整策略校验、数据库、认证或模型运行。

## 适用范围与限制

**离线校验通过 ≠ 平台导入通过，平台导入在 T08 实测。** 当前只验证生成物格式、引用、来源配置与本地 MCP；Nexent 部署、真实模型显示名、Skill 实例绑定、子版本发布和容器连通性都需 T08 留存真实证据。未新增 PDE 算法、未调用付费模型、未修改数值引擎。
