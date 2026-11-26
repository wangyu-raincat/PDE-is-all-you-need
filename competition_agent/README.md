# Nexent v2 接入包

本目录提供六角色配置、六个独立 Skill ZIP 和本地 MCP 服务。默认生成目标固定为 Nexent v2.8.5；旧配置结构保留为 legacy。字段对比和来源见 [SCHEMA_DIFF.md](verification/nexent_v2/SCHEMA_DIFF.md)。

## 生成

在仓库根目录、Python 3.12 环境运行：

```powershell
python -m competition_agent.bundle --nexent-version v2 --model-name X --mcp-url http://host.docker.internal:8769/mcp
python -m competition_agent.bundle --nexent-version legacy --output competition_agent/build/legacy
```

首条命令输出到 competition_agent/build：nexent-agents.json、bundle-status.json 和六个根目录仅含 SKILL.md 的 ZIP。X 只是离线验收示例，实际导入前换成平台模型显示名。不填写模型时生成空 model_names 并标记等待绑定。所有生成物默认使用 UTF-8。

## MCP 与容器连接

本机客户端默认：

```powershell
python -m competition_agent.mcp_server --transport streamable-http --port 8769
```

默认只监听 127.0.0.1，端点 http://127.0.0.1:8769/mcp。Windows Docker Desktop 中运行的 Nexent 连接宿主机时：

```powershell
python -m competition_agent.mcp_server --transport streamable-http --host 0.0.0.0 --port 8769
```

在 Nexent 注册服务名 pde_lab_tools，URL 使用 http://host.docker.internal:8769/mcp。容器内的 127.0.0.1 指容器自身；0.0.0.0 是监听地址，不是客户端 URL。只在可信网络使用，并将防火墙入站限制到需要的网络；本服务无身份认证。原生 Linux Docker 需另配 host-gateway 映射，不能假定该名称自动存在。

v2.8.5 源码支持 Streamable HTTP（/mcp）和 SSE（/sse）。如需 SSE，改用 `--transport sse`，并同步将包的 --mcp-url 改为 http://host.docker.internal:8769/sse。本地历史测试覆盖 stdio 与 Streamable HTTP；SSE 及 Nexent 容器端连通性仍待 T08 实测。

## 平台接入顺序

1. 配置真实模型，确认显示名。
2. 注册 MCP 服务 pde_lab_tools，刷新并检查 15 个工具的来源和服务名。
3. 上传六个独立 Skill ZIP，上传时显式填写与 ZIP 文件名（去掉 .zip）相同的 Skill 名称，例如 pde-coordinator。根目录 SKILL.md 的 ZIP 不提供目录名，不能仅依赖正文 name 自动命名。
4. 导入 nexent-agents.json，检查总控管理五个子智能体。
5. **逐角色绑定同名 Skill**。普通 JSON 导入不创建 Skill 实例，不能只上传后就认为绑定完成。
6. 检查六个角色模型与工具；绑定 Skill 后发布，确认总控引用了含 Skill 的子版本。
7. 在 T08 执行平台问答，保存真实调用链、截图及导出样本。

## 适用范围与限制

离线校验通过 ≠ 平台导入通过，平台导入在 T08 实测。v2 参数固定指本次审查的 v2.8.5，不保证全部 v2.x。T00 手工导出样本当前缺失；获得后优先与本生成物核对。历史总入口 NEXENT_README.md 保留旧交付记录，本文件说明 T01 新版本接入行为。
