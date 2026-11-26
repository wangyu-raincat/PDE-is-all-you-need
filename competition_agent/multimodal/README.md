# 多模态资产接入

T03 将本地文件转为统一资产和可定位证据；不执行作者代码、不计算 Excel 公式、不发送图片到外网。当前环境缺解析依赖，真实 PDF/XLSX/OCR 验收尚未完成，见 `codex_plan/progress/T03.md`。

## 使用

原 `asset_ingest(case_id, documents=[{"title":"...","kind":"text|csv","text":"..."}])` 保持兼容。
新增输入：`{"path":"competition_agent/assets_raw/dib-2021-107632.pdf","kind":"pdf"}`；kind 还支持 xlsx、image、code。
路径相对于启动服务的工作目录；推荐绝对路径。允许的输入根目录为 assets_raw、scenario_am、pde_discovery/data 和该 Lab runtime 下的 uploads；不开放任意本地文件读取。

统一结果是 `ParsedAsset` 字典：asset_id/source_id 为文件 SHA-256，modality、path、sha256、blocks。接入到 Lab 后，source_id 是现有文档摘要身份；`multimodal_assets` 同时保留文件 asset_id 与文档 source_id 的映射。

- PDF 文本带 1 起始页码、bbox（左上角原点，单位 point）、块内行号和字符跨度；图像保留哈希、候选图注、缩略图路径。pdfplumber 表格保留页码、bbox、表头与行记录。
- XLSX 保留 sheet、单元格范围、逐格坐标、公式字符串及已有缓存值。缺少缓存时返回 null 和 formula_cache_missing；不将公式原文假装为计算结果。
- 图片保留 OCR 文本、多边形 bbox、置信度和缩略图；全部需要审核。VLM 是可选项，本版禁用，未实现远端描述请求。PDF 内图片目前提供图注/位置/缩略图，不另做 OCR。
- 源码用 AST 抽取常量、列表和受限数值运算（加减乘除），保留分支作用域、原表达式、原文及行号。不会运行赋值；条件分支的常量不代表程序实际执行结果。作者增材参数形成 16 条带分组的候选事实，未知关系仍须本体现有人工审核流程。
- asset_context 按角色分页返回多模态摘要，图片返回描述/OCR 片段/缩略图路径，不返回 base64。原文投影与原始定位并存，graph_propose 将定位附到 citation。OCR 证据不能直接 graph_apply，冲突保留双方来源等待后续人工审核；本版没有新增 OCR 审核放行接口。

## 重建与验证

使用现有 `competition_agent/build/t02-env/Scripts/python.exe`：

```powershell
competition_agent/build/t02-env/Scripts/python.exe scripts/competition_render_cards.py
competition_agent/build/t02-env/Scripts/python.exe scripts/competition_multimodal_acceptance.py
competition_agent/build/t02-env/Scripts/python.exe -m unittest discover -s tests -p "test_competition*.py" -v
```

记录卡渲染脚本使用 Windows 自带 System.Drawing 和楷体；不依赖 Pillow、不下载字体。cards.json 的 expected_text 只用于夹具，不会交给解析器当成 OCR 结果。日期写“未记录”；20.8 C / 5 m/s 分别注明源码设定和猜测。xn 的方向 y 为唯一故意植入的事实冲突，不能引用为真实实验依据。

缺失依赖交人工安装，版本写在 competition_agent/requirements.txt：PyMuPDF 1.24.14、pdfplumber 0.11.4、openpyxl 3.1.5、rapidocr-onnxruntime 1.4.4。当前未实测这些版本的联合安装；RapidOCR 传递依赖包括 OpenCV/ONNX Runtime。无外网/无密钥时不调用 VLM；缺 OCR 包时返回清晰错误，不伪造识别结果。

解析 API 参考：[PyMuPDF](https://pymupdf.readthedocs.io/en/latest/recipes-images.html)、[openpyxl](https://openpyxl.readthedocs.io/en/stable/api/openpyxl.reader.excel.html)、[pdfplumber](https://github.com/jsvine/pdfplumber)。
