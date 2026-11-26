# T04b 召回子集与标注工具修订

正式标注包现在有 36 个可独立阅读的原文片段，候选判断仍使用 v2 的同一份 100 条候选。原始资料范围及内容覆盖已逐项核验。

| 来源 | 旧版 | 新版 | 切分依据 |
|---|---:|---:|---|
| AdditiveRodResults.xlsx，xn 表 | 129 个非空单元格 | 24 行 | 每个非空行附 A1:F1 表头，并标出本行范围和所属模型。含表头行与模型标题行 |
| 作者源码，第 223–250 行 | 28 行 | 6 块 | 3 行初始化、4 个各 5 行的试样分支、5 行共用设置 |
| 数据论文，第 3 页 | 6 块 | 6 块 | 保留每个完整文本块、页码、原文和坐标 |
| **总计** | **163** | **36** | |

36 段低于原估计的 40–80 段。工作表第 11、18 行及末尾空行没有原始数据，原子集也没有为它们建立记录。回归测试将新版拆回旧版，逐项比对来源、位置、单元格值、源码行和 PDF 块内容。

## 标注与交付

- `scripts/competition_annotate.py` 的两种模式均显示来源、位置与原文。候选判断另显示待判三元组。PDF 合并断行词并整理空白；结果保留来源标识、定位字段与原始证据。穷举模式同时记录输入引文、原文匹配、规范化结果和原文字符区间（起始包含、末尾不含）。普通词内连字符和数值符号保留。
- 解压 `competition_agent/build/T04b/annotation_kit.zip` 后，标注员分别运行 run_A.bat、run_B.bat。练习包 `practice_kit.zip` 使用论文第 1 页的完整摘要块和关键词块，含两段穷举练习、五条自制判断题和参考答案。三个 BAT 均为 ASCII、CRLF，不切换代码页。
- 旧版正式子集保存在 `competition_agent/verification/ontology_induction/v2/gold-subset.json`，新版及核验摘要在相邻的 `v3/`。替换前的两个 ZIP 保存在 `competition_agent/build/T04b/before-subset-revision/`。
- 复现命令为 `competition_agent/build/t02-env/Scripts/python.exe -m scripts.competition_annotation_kit`。生成器读取冻结的 v2 子集和候选包，并从真实 PDF 取练习原文。

## 核验

使用 t02-env 执行 `-m unittest discover -s tests -p "test_competition*.py" -v`，131 项全部通过，0 跳过，耗时 23.559 秒。两份 ZIP 均实际解压，在临时目录运行 Windows 启动器和两种标注模式，验证退出续做、全部片段完成记录、判断结果和原始位置落盘。练习摘要输入 `convection` 成功定位到含换行连字符的原文。上述输入仅用于软件验收，没有写入正式标注目录或参与评测。包和测试日志的 SHA-256 已归档到 v3/acceptance.json。
