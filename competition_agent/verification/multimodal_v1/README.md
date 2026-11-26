# T03 最新真实文件验收

本次 7 个输入真实解析成功，完整接入层 **116/116、无跳过、无失败/错误**。原跳过的 PDF 夹具、XLSX 公式/缓存夹具、无密钥 OCR 均实际成功。

- acceptance.json：PDF 第 3 页方向/孔隙原文与 bbox；四组源码文件名和真实 XLSX B1:F1 补证五份编号记录；10 张表、1,867 个非空单元格摘要。原表公式为 0，另由夹具验证公式原文、缓存及缺缓存 null。
- graph_acceptance：真实 OCR 导入隔离 SQLite 图谱，已有来源事实 2 条，xn 冲突 1、xp 冲突 0；原事实未覆盖，OCR 自动入图被拒绝。记录包含源 ID、原文、bbox、事件链及缩略图哈希。新关系审批仅为测试驱动模拟，不是生产业务人工审批。
- test-summary.json 的 latest 是本次结果；test-excerpt.txt 明确列出三个原跳过测试成功。早期测试数据保留为历史。
- rerun-provenance.json：本次脚本、测试、完整日志与验收文件哈希。完整日志及解析原始输出位于 build/T03。
- installation.json、asset-manifest.json：保留既有安装与原始资料记录。

**来源限制**：数据论文未定位到“每杆五次测试”的明确原句，替代证据不能冒充 PDF 引文。因此 execution_checks_passed=true，但严格任务卡 complete=false。此项已调查并记录结论；其余三项补验收已通过。详见 codex_plan/progress/T03.md。
