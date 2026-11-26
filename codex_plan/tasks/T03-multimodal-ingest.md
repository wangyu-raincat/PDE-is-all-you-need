# T03 多模态资产解析（文本 / 表格 / 图文）

## 目标
满足赛题“对文本、表格、图文等多元数据统一识别、理解与资产标识”。现有只支持纯文本和 CSV。

## 步骤
1. 新模块 `competition_agent/multimodal/`，统一输出 `ParsedAsset{asset_id, modality, blocks[]}`，block 类型：
   - `text`（页码、bbox、行号、字符跨度）
   - `table`（sheet/页码、单元格范围、表头、行记录）
   - `figure`（页码、bbox、图片哈希、图注、OCR 文字及 bbox、VLM 描述）
   - `code_literal`（文件、行号、变量名、字面值——用于作者源码中的几何参数）
2. 解析器（优先轻量、CPU）：
   - PDF：PyMuPDF 取文字与图片块位置；pdfplumber 取表格。
   - XLSX：openpyxl（保留公式原文与计算值）。
   - 图片：RapidOCR（onnxruntime）做文字+bbox；可选 VLM（OpenAI 兼容接口，环境变量配置）生成图片描述，**VLM 输出一律标记 `needs_review=true`**，不能直接成为已核验事实。
   - 源码：Python AST 提取字面量赋值（几何参数、坐标数组）。
3. 自制图文样例（演示“实验记录卡/工艺卡”这类企业常见扫描件）：用脚本渲染 2–3 张图片（含手写风格字段：试样号、构建方向、测试日期、环境温度、风速），**文件名和画面角落都标“自制示例”**；内容只用公开数据已有信息，可故意放 1 处与数据论文不一致的值，用于演示冲突保留。
4. 接入：`asset_ingest` 支持 `{"path": ..., "kind": "pdf|xlsx|image|code"}` 输入（保留原 title/kind/text 接口）；候选事实沿用现有引文机制，locator 扩展为多模态位置。
5. `asset_context` 按角色返回多模态摘要（图片返回描述+OCR 片段+缩略图路径，不返回整图 base64）。

## 验收标准
- 对数据论文 PDF：能抽出“构建方向 x/y、p/n 孔隙类别、每杆 5 次测试”等关键句，带页码与 bbox。
- 对作者源码：抽出 4 组增材杆的 D、L、密度与测点坐标，行号与 `metadata_verification.md` 记载一致。
- 对自制记录卡：OCR 得到字段并与图谱中已有事实比对，检出预设的 1 处冲突。
- 无外网/无 VLM 密钥时，流程降级为 OCR-only 且测试仍通过。
- 测试 `tests/test_competition_multimodal.py`（PDF 测试可用脚本生成的小 PDF 夹具，避免依赖下载文件）。
