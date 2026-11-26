"""Rebuild the reviewer handoff from the frozen v2 evidence without rerunning extraction."""
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re
from zipfile import ZIP_DEFLATED, ZipFile

from competition_agent.ontology import ROOT
from competition_agent.multimodal import parse_asset

ARCH = ROOT / "competition_agent/verification/ontology_induction"
BUILD = ROOT / "competition_agent/build/T04b"


def encoded(value):
    return (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf8")


def revised_subset(old):
    """Regroup the fixed recall population; keep every nonempty cell and source line."""
    result = []
    sheets = defaultdict(list)
    code = []
    for record in old:
        locator = record["locator"]
        if "page" in locator:
            result.append(dict(record, locator=dict(locator, block=len(result) + 1)))
        elif "sheet" in locator:
            sheets[record["path"], locator["sheet"]].append(record)
        else:
            code.append(record)
    for (_, sheet), cells in sheets.items():
        rows = defaultdict(dict)
        for cell in cells:
            col, num = re.fullmatch(r"([A-Z]+)(\d+)", cell["locator"]["cell"]).groups()
            rows[int(num)][col] = cell["text"]
        columns = list(rows[1])
        header = " | ".join(rows[1][col] for col in columns)
        section = None
        for num, values in sorted(rows.items()):
            if set(values) == {columns[0]}:
                section = values[columns[0]]
            row = " | ".join(values.get(col, "") for col in columns)
            result.append(dict(path=cells[0]["path"], source_id=cells[0]["source_id"],
                               locator={"sheet": sheet, "cell_range": f"{columns[0]}{num}:{columns[-1]}{num}",
                                        "header_range": f"{columns[0]}1:{columns[-1]}1"},
                               section=section, text=header if num == 1 else header + "\n" + row))
    # The selected author source has a preamble, four specimen branches and common settings.
    for start, end in [(223, 225), (226, 230), (231, 235), (236, 240), (241, 245), (246, 250)]:
        block = [r for r in code if start <= r["locator"]["line"] <= end]
        if [r["locator"]["line"] for r in block] != list(range(start, end + 1)):
            raise ValueError(f"Incomplete author source block {start}-{end}")
        result.append(dict(path=block[0]["path"], source_id=block[0]["source_id"],
                           locator={"line": start, "end_line": end}, text="\n".join(r["text"] for r in block)))
    return result


def practice():
    pdf = ROOT / "competition_agent/assets_raw/dib-2021-107632.pdf"
    blocks = [b for b in parse_asset(pdf, "pdf")["blocks"] if b["type"] == "text" and b["locator"]["page"] == 1]
    def block(begin):
        b, = [b for b in blocks if b["text"].startswith(begin)]
        return dict(path=pdf.relative_to(ROOT).as_posix(), source_id=hashlib.sha256(pdf.read_bytes()).hexdigest(),
                    locator=b["locator"], text=b["text"])
    abstract, keywords = block("These data were acquired"), block("Keywords:")
    examples = [
        (abstract, ["Python program", "determines", "thermal conductivity"], "1", "摘要说明 Python 程序处理数据以求导热系数。"),
        (abstract, ["temperature acquisition system", "produces", "raw data"], "1", "摘要直接交代原始数据来源。"),
        (abstract, ["tested conductivity range", "has_upper_bound", "4000 W/mK"], "0", "摘要写的是 15–400 W/mK。"),
        (keywords, ["article", "has_keyword", "Parameter estimation"], "1", "关键词表中列出该词组。"),
        (keywords, ["analytical model", "uses_solver", "finite elements"], "?", "关键词没有给出求解器。")]
    items = [dict(id=f"practice-{i}", triple=triple, sources=[src]) for i, (src,triple,_,_) in enumerate(examples,1)]
    answers = "自制练习参考（完成后对照）\n\n原文 1：可录入原始数据来自温度采集系统。原文 2：可录入论文列出的关键词。\n\n"
    answers += "\n".join(f"{i}. {label}　{why}" for i, (_,_,label,why) in enumerate(examples,1))
    return [abstract, keywords], {"items": items}, answers


def launcher(subset, packet, directory, reviewer):
    commands = f'''@echo off
setlocal
cd /d "%~dp0"
if not exist competition_annotate.py (
  echo [ERROR] Extract the entire ZIP first.
  pause & exit /b 1
)
set "PY="
py -3 -c "import sys; assert sys.version_info >= (3,8)" >nul 2>&1 && set PY=py -3
if not defined PY python -c "import sys; assert sys.version_info >= (3,8)" >nul 2>&1 && set PY=python
if not defined PY if exist "%USERPROFILE%\\anaconda3\\python.exe" set PY="%USERPROFILE%\\anaconda3\\python.exe"
if not defined PY if exist "%USERPROFILE%\\miniconda3\\python.exe" set PY="%USERPROFILE%\\miniconda3\\python.exe"
if not defined PY (
  echo [ERROR] Install Python 3.8 or newer.
  pause & exit /b 1
)
%PY% competition_annotate.py missing {subset} {directory} {reviewer}
if errorlevel 3 goto failed
if errorlevel 2 goto saved
if errorlevel 1 goto failed
%PY% competition_annotate.py label {packet} {directory} {reviewer}
if errorlevel 1 goto failed
:saved
echo Saved. Run again to continue.
pause
exit /b 0
:failed
echo [ERROR] Take a screenshot of this window.
pause
exit /b 1
'''
    return commands.replace("\n", "\r\n").encode("ascii")


def write_zip(name, files):
    files["competition_annotate.py"] = (ROOT / "scripts/competition_annotate.py").read_bytes()
    target = BUILD / (name + ".zip")
    staging = target.with_suffix(".tmp")
    with ZipFile(staging, "w", ZIP_DEFLATED) as archive:
        for filename, content in files.items():
            archive.writestr(name + "/" + filename, content)
    staging.replace(target)


def main():
    BUILD.mkdir(parents=True, exist_ok=True)
    source = ARCH / "v2"
    revised = ARCH / "v3"
    revised.mkdir(exist_ok=True)
    old = json.loads((source / "gold-subset.json").read_bytes())
    subset = encoded(revised_subset(old))
    (revised / "gold-subset.json").write_bytes(subset)
    packet = (source / "packet.json").read_bytes()
    items = json.loads(packet)["items"]
    guide = f"""完整解压后，A 双击 run_A.bat，B 双击 run_B.bat。需要 Python 3.8 或更新版本。
两人独立标注。先穷举 {len(json.loads(subset))} 段原文中的事实，再判断 {len(items)} 条候选。
原文没有事实或已经录完时，在主语处按回车。输入 :q 可保存退出。下次启动继续。
表头和所属模型供理解本行，仅在首次出现时记录相同事实。引文可忽略空白与断行连字符，其他字符须与原文一致。
候选按 1（直接支持）、0（不支持或错误）、?（证据不足，需推测）判断。在备注中写修正或理由。q 保存退出。
完成后把 labels 目录交回负责人。请勿修改原文包或交换个人判断。
关系名参考（允许补充）：{', '.join(sorted({r['triple'][1] for r in items}))}
"""
    write_zip("annotation_kit", {"gold-subset.json": subset, "packet.json": packet, "README_说明.txt": guide,
                                 **{f"run_{r}.bat": launcher("gold-subset.json", "packet.json", "labels", "reviewer_"+r.lower())
                                    for r in ("A", "B")}})
    gold, samples, answers = practice()
    write_zip("practice_kit", {"gold_practice.json": encoded(gold), "packet_practice.json": encoded(samples),
                               "README_说明.txt": "完整解压，双击 run_practice.bat。先练习两段原文，再判断五条自制候选。完成后对照练习答案。结果保存在 labels_practice。",
                               "ANSWERS_练习答案.txt": answers,
                               "run_practice.bat": launcher("gold_practice.json", "packet_practice.json", "labels_practice", "practice")})


if __name__ == "__main__":
    main()
