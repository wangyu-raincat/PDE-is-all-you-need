# T08-prep：本机 Nexent 接入准备

## 改动

在主目录先切换到 competition/nexent-multi-agent，以 `git merge --no-ff comp/T05d-joint-fit` 合并，生成 `14e2b38`（双亲 ea5c6e9、4dd3610）。原协议提交 `7a5b8e1` 与结果提交 `4dd3610` 均保留，没有 squash。随后从主线创建 comp/T08-prep，没有新建 worktree。

1. Q1 运行时只核对 protocol.json 的 LF 规范化 SHA-256 与 REPORT.md 唯一记录值一致，不调用 git 或任何子进程。缺失、重复或不匹配的摘要拒绝通过。结果本身仍核对协议摘要和固定数值判定。未改协议、拟合结果或判定门槛，未重新进行联合拟合。
2. Git 提交字节及祖先链核验移到 scripts/competition_joint_fit_report.py 的归档阶段，在本次归档执行一次，结果写入 verification/joint_fit/archive_verification.json。服务运行不再依赖 Git 历史；历史“协议先提交”的证据仍由归档保留。
3. mcp_server.py 显式开启 TransportSecuritySettings.enable_dns_rebinding_protection，默认 Host 白名单为 127.0.0.1:*、localhost:*、[::1]:*、host.docker.internal:*。新增可重复 --allowed-host，仅追加且去重，不代替默认值。保留本地 Origin 白名单和 SDK 对非法 Host/Origin 的检查；没有关闭 DNS rebinding 防护，没有改变默认监听地址。
4. 纳入现有 verification/nexent_live/DEPLOY_LOG.md。提交前检查密钥、Bearer/JWT 模式、当前环境中的秘密值以及所有涉及凭据的文本行，未发现实际密钥或 token；日志中有关未配置密钥、历史部署失败和未提交 Git 的陈述保留为历史记录。

## 验证

环境：competition_agent/build/t02-env。关键记录在 verification/t08_prep/。

- 真实临时 HTTP 服务：test_competition_mcp.py 2/2 通过，8.985 秒。Host=host.docker.internal:8769 返回 200；untrusted.example:8769 和伪装子域 host.docker.internal.evil:8769 返回 421。两个 --allowed-host 分别生效，固定端口不匹配仍为 421，非法 Origin 返回 403。使用随机空闲端口，不连接或重启用户的 8769 服务。
- 协议回归：禁止 subprocess.Popen 的条件下可以完整读取联合拟合证据；协议内容篡改和报告缺失摘要均拒绝。
- 接入层测试：147/147 通过，30.568 秒。命令：`-m unittest discover -s tests -p "test_competition*.py" -v`。
- 六问本地 MCP + DeepSeek 验收：6/6 通过，6 次模型请求、25 次 MCP 调用；输入 4,038、输出 101，共 4,139 token。六问决策字段均与 v4 一致，Q1 数值证据也完全一致；每条 trace 已从 SQLite 回读核对。命令：`-m competition_agent.deepseek_acceptance --decision --output competition_agent/build/T08-prep/release --max-model-calls 6`。
- 完整回归：828 项，800 通过、1 失败、25 错误、2 跳过，128.841 秒。与主线已归档 decision_v4/tests.json 对比无新增或解决失败；未切换主线另跑基线。命令：`-m unittest discover -s tests -v`。

## 运行状态与后续操作

没有关闭现有 PowerShell 窗口，也没有停止或重启原 MCP 服务。测试只清理自己创建的临时服务进程。用户需要手动重启 MCP 服务，才能加载新白名单；本次没有将旧服务上的响应写成新配置已生效。

重启时沿用原命令即可使用默认 Docker Host 白名单；若需要追加受信任主机，可加入 `--allowed-host example.internal:*`，该参数可重复。没有在 Nexent 网页导入智能体或声称完成平台内多智能体验收。

工作期间另有 competition_agent/skills/pde-agent/SKILL.md 改动，以及已有 cloud-delivery/，均不修改、不暂存。

检查时原 PowerShell PID 49672 仍在运行，127.0.0.1:8769 仍由原服务 PID 62444 监听。
