# 质量改进对照记录

本轮提升本地工具、证据追溯和编排可靠性，未改动既有数值引擎。Nexent 与 LLM 协作尚未部署实测。原版可回退到提交 1045b6b。

## 定向行为检查

同一脚本针对旧版和新版运行14个人为构造用例，含2个正常控制：旧版2/14，新版14/14。选例针对已发现问题，部分使用空模型隔离证据逻辑；不是通用智能体准确率、独立科学基准或语义正确率。

| 检查 | 旧版 | 新版 |
|---|---|---|
| english_retrieval_control | 通过 | 通过 |
| chinese_question_retrieval | 未通过/未提供 | 通过 |
| late_english_match_is_in_quote | 未通过/未提供 | 通过 |
| late_chinese_match_is_in_quote | 未通过/未提供 | 通过 |
| ingest_replay_at_document_limit | 未通过/未提供 | 通过 |
| unrelated_fact_does_not_establish_readiness | 未通过/未提供 | 通过 |
| complete_metadata_control | 通过 | 通过 |
| negated_relation_is_not_verified | 未通过/未提供 | 通过 |
| metadata_must_match_simulator | 未通过/未提供 | 通过 |
| cross_document_path_has_both_sources | 未通过/未提供 | 通过 |
| changed_final_report_is_rejected | 未通过/未提供 | 通过 |
| changed_freeze_seal_rejected_before_final | 未通过/未提供 | 通过 |
| repeated_create_does_not_charge_twice | 未通过/未提供 | 通过 |
| status_read_during_reserved_write | 未通过/未提供 | 通过 |

## 全库与数值回归

731项全库测试通过，其中本地接入相关43项。实际stdio和Streamable HTTP会话涵盖14个工具目录与成功/错误路径。7组已查看的合成案例按原配置重演，结构、系数、预测误差、失败槽位和预算逐字段完全一致；Burgers 3%噪声依然没有模型。没有用这些最终结果调整任何数值参数。

每案MCP调用从15次增加至25次，共175次，新增开销用于显式交接和跨文档追溯；不声称节约调用、算力或实验数据。

## 可用性与边界

新增离线HTML报告，桌面1280与手机390像素宽度均无水平溢出，证据链接和引用展开已检查。截图来自本地工具报告，不能冒充Nexent平台截图。原始调用、计划、冻结记录、结果和HTML保存在matrix目录。

证据就绪只适用于固定合成元数据：三条要求完整一致、关系为原文整行显式断言，无冲突及未审核本体。自然语言关系、行业标准语义、原生知识库、多模态与真实企业收益尚未验证。未知或冲突保留为受阻状态；没有事实撤回和通用语义审核接口。

quality-before.json / quality-after.json 是逐项证据；numerical-comparison.json 是数值回归对照；acceptance.json 记录最终代码闭包哈希。
