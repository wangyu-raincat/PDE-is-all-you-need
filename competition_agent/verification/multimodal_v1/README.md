# T03 验收证据

- acceptance.json：安装后真实验收，7 个输入均 parsed；论文方向/孔隙在第 3 页，源码 114 块，XLSX 工作表 4/3/3，真实 OCR 检出 xn 冲突 1、xp 冲突 0。每杆五次测试未在 PDF 正文找到明确句子，且实际业务图谱审核流程尚待补验，complete=false。
- installation.json：用户授权安装记录、四个固定依赖与主要传递依赖的实际版本及路径。
- test-summary.json：保留安装前全量基线对照和接入层历史；最新 post_install 为 116/116、无跳过。原始日志在 build/T03。
- asset-manifest.json：原始交付物与自制样例哈希；原始文件未改。

安装后修复了 XLSX 缺少 worksheet dimension 元素的兼容问题；生成夹具覆盖此情况。来源与许可见 assets_raw，具体限制和命令见 codex_plan/progress/T03.md。
