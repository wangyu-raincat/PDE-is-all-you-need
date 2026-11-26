# T03 验收证据

- acceptance.json：build/T03/acceptance.json 原样复制；作者源码 114 块已解析，6 个 PDF/XLSX/图片文件因缺依赖待验收。complete=false。
- test-summary.json：同一 t02-env 环境下任务前 b36d355 和本分支全量回归对照，含测试计数、失败 ID、异常原因差异、日志 SHA-256；接入层 113 通过/3 跳过，不能写成 116 全通过。
- asset-manifest.json：交付原始论文/表格及明确标注的自制图片、来源说明的文件哈希。CSV、引擎及历史结果未改。

日志保留在 build/T03（不提交）。真实资料下载与授权信息见 assets_raw/sources.json 和 README.md。模拟 OCR 测试仅验证下游定位、冲突和审核阻断，不等于 RapidOCR 实测。待人工安装依赖后的命令见 codex_plan/progress/T03.md。
