# 五问决策证据报告

双击 `decision-report.html` 可离线查看五个决策问题。页面将原结论连接到决策规则、模型记录、带类型的图谱路径、冲突关系及原文定位；PDF 裁剪图片、表格单元格和源码行均内嵌，无 CDN。

输入固定为 `competition_agent/verification/decision_v2/trace-1.json` 至 `trace-5.json`。第 0 问不属于本次输入，合并后另补。报告不重新运行模型或实验，不改变原决策等级。

生成命令（仓库根目录）：

```powershell
& competition_agent/build/t02-env/Scripts/python.exe -X utf8 -m competition_agent.reporting
```

- `generation.json`：输入五条 trace 的文件摘要、报告摘要、预览数量、答复信息。
- `handoffs.json`：五条可用于 Nexent 答复的原结论、trace_id、报告路径和问题锚点。页面提供同样的复制按钮。
- 自定义输出：增加 `--output <完整HTML路径>`，对应答复路径同时更新。

## 适用范围与限制

浏览器冒烟尚未通过实测：当前浏览器连接不能创建页面，指定 Python 环境缺少 Playwright，未安装新依赖。390px 响应式样式已实现，但实际排版、点击展开及复制仍待浏览器验收。

Nexent 的运行器不在本次允许修改范围，未声称平台实际答复已自动附带报告路径。接入方需将 handoffs.json 的对应条目接入答复，平台内联调由 T08 验证。

资产按已有使用登记分组；未虚构“沉睡度”评分。T04 图表来自 v2 summary 的实际抽取数量和启发式分数分布；人工准确率、召回率、一致性与工时收益均待评测。

原文预览取生成时本地资料，归档引文始终保留。source_id 可能是解析器标识，不等于原始文件摘要。Q4 的 7.9 mm 冲突为自制验收样例。数据论文与测量数据署名见报告末尾，标准仅展示项目自写目录摘要。
