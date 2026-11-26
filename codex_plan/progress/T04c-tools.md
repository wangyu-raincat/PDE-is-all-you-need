# T04c 本体增量归纳与标准对齐

在 `comp/T04-ontology` 上先以 `229d582` 单独提交 T04b 任务卡。随后实现 T04 第 7–9 步。案例可以从新资料提出概念、关系与标准对照建议，操作员可以查看证据、批量处理或回到先前版本。

## 实现与入口

- `competition_agent/ontology/tools.py` 从案例中已准备好且去重的资料提取候选。出现至少两次的未映射词组或关系按类型分开，以字符 1–3 gram 的 TF-IDF 余弦相似度构建连通分量（阈值 0.55）。队列保留词形、来源、原文和位置。标准目录中的精确术语首次出现即可送审。
- 图谱按归一化主体缓存来源行。新增资料或重复资料代表发生切换时，仅对受影响主体重新聚合；本体只更新受影响术语及其相连的簇。图谱与本体分别保存版本号和逐键旧值、新值。回滚创建新版本，保留审计历史。原样重放不产生新版本。
- `competition_agent/ontology/revision.py` 复用 T04b 的抽取器归属、证据校验和分流。`competition_agent/service.py` 将归纳状态纳入冻结摘要。归纳、审核和回滚均检查冻结状态。
- `competition_agent/mcp_server.py` 注册 `ontology_induce(case_id)`、`ontology_review_queue(case_id, limit)`、`standard_align(concept)`。`bundle.py` 与 `skills/graph-agent/SKILL.md` 把三者绑定给 graph_agent。
- 标准对齐使用项目方法规则、别名和字符相似度。可选相似度回调缺库或缺模型文件时使用字符相似度。方法比较标为 `different_method_from`，可比较的物理量标为 `comparable_quantity`。有依据时提供章节定位，例如 GB/T 3651-2008 附录 A；目录没有条款号时留空。
- `competition_agent/demo.py` 提供本地批量审核、版本导出和回滚。`competition_agent/reporting.py` 在案例报告展示涉及的标准、方法差异与审核状态。案例导出含 `alignment.json` 与 `versions.json`。

## 本地操作

在仓库根目录按实际案例指定 `$dir`、`$case` 与 `$out`。批量文件为 UTF-8 JSON 对象，将队列 ID 映射到 `accepted` 或 `rejected`。回滚的版本号按实际历史选择。

```powershell
$py = 'competition_agent/build/t02-env/Scripts/python.exe'
& $py -m competition_agent.demo review-ontology --directory $dir --case-id $case --batch decisions.json
& $py -m competition_agent.demo export-induction --directory $dir --case-id $case --output $out
& $py -m competition_agent.demo rollback-induction --directory $dir --case-id $case --kind ontology --version 1
```

## 验收

- 作者的 AdditiveRodResults.xlsx 经正常资料入口入库后，四个试样工作表 B1:F1 产生 20 条候选。首次图谱与本体版本均为 1；加入自制资料后均为 2，仅重算 xn。重复归纳的两份 diff 均为空。隔离案例的模拟操作员确认使本体到版本 3，回滚后在版本 4 恢复待审状态。
- 专项测试覆盖相近词聚类、未知关系、幂等重放、按主体更新、两个版本空间、回滚后再归纳、标准对齐和冻结写保护。接入层还通过真实 MCP stdio 会话调用三个新工具。
- `& $py -m unittest discover -s tests -p "test_competition*.py" -v`：128 项通过，0 跳过。
- `& $py -m competition_agent.bundle --nexent-version v2 --output competition_agent/build/T04c-tools/bundle`：6 个智能体、6 个独立 Skill ZIP、21 个 MCP 工具。
- `& $py -m unittest discover -s tests -v`：809 项，781 通过，1 失败，25 错误，2 跳过，耗时 142.023 秒。与已存主线基线逐项比较，新增失败/错误 0 项，消失 0 项。
- 证据见 `competition_agent/verification/ontology_tools/` 中的 `acceptance.json`、`alignment.json`、`tests.json` 和 `bundle-status.json`。

## 适用范围与限制

词组按字面模式提取，聚类阈值固定。词形相似并不保证同义，方法比较也不能作为符合标准的证明。已验证本地 MCP 与离线 Nexent v2 bundle，线上导入待实测。本次未引入 torch，未调用 E3，未拟合权重或运行人工评测；示例确认只验证软件流程。人工标注完成后再评估准确率、召回率和审核成本。
