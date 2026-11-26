"""Two independent annotators use the same immutable packet and separate journals."""
import argparse
import hashlib
import json
import random
import re
from pathlib import Path


# Only a hyphen between letters at an actual line break is discretionary.
WRAP = re.compile(r"(?<=[^\W\d_])[-\u00ad][^\S\r\n]*(?:\r\n|\r|\n)\s*(?=[^\W\d_])")
IGNORED = re.compile(WRAP.pattern + r"|\s+")


def quote_key(text):
    return IGNORED.sub("", text)


def match_quote(text, entered):
    key = quote_key(entered)
    if not key:
        return None
    exact = text.find(entered)
    if exact >= 0:
        return dict(quote=entered, quote_raw=entered, quote_normalized=key,
                    quote_span=[exact, exact + len(entered)])
    # Map a normalized match back to its exact, contiguous source span.
    offsets = []
    start = 0
    for gap in IGNORED.finditer(text):
        offsets.extend(range(start, gap.start()))
        start = gap.end()
    offsets.extend(range(start, len(text)))
    pos = quote_key(text).find(key)
    if pos < 0:
        return None
    first, last = offsets[pos], offsets[pos + len(key) - 1] + 1
    return dict(quote=entered, quote_raw=text[first:last], quote_normalized=key,
                quote_span=[first, last])


def show_source(source):
    loc = source["locator"]
    if "page" in loc:
        where = f"第 {loc['page']} 页"
        if "block" in loc:
            where += f"，文本块 {loc['block']}"
    elif "sheet" in loc:
        where = loc["sheet"] + "!" + (loc.get("cell_range") or loc.get("cell", ""))
    elif "line" in loc:
        end = loc.get("end_line", loc["line"])
        lines = str(loc["line"]) if end == loc["line"] else f"{loc['line']}–{end}"
        where = "第 " + lines + " 行"
        if "json_pointer" in loc:
            where += " · " + loc["json_pointer"]
    else:
        where = loc.get("id", "图像文字区域")
    original = source.get("text", source.get("quote", ""))
    display = " ".join(WRAP.sub("", original).split()) if "page" in loc else original
    if source.get("section"):
        where += " · " + source["section"]
    path = source.get("path") or loc.get("file") or "未记录文件名"
    print(f"来源文件　{path}\n位置　{where}\n原文\n{display}")


def prepare(snapshot, output, count, seed):
    if not 80 <= count <= 150:
        raise ValueError("choose 80..150 triples")
    run=json.loads(snapshot.read_text(encoding="utf8"))
    pool=[item for tier in ("draft","queue","discard") for item in run[tier]]
    if len(pool)<count: raise ValueError("not enough distinct candidates")
    rng=random.Random(seed)
    # Preserve rare relations by taking one candidate per relation in each round.
    groups={}
    for item in sorted(pool,key=lambda x:x["id"]):
        groups.setdefault(item["triple"][1],[]).append(item)
    for group in groups.values(): rng.shuffle(group)
    selected=[]
    while len(selected)<count:
        for group in groups.values():
            if group and len(selected)<count: selected.append(group.pop())
    rng.shuffle(selected)
    items=[{"id":r["id"],"triple":r["triple"],"sources":[{k:e[k] for k in
            ("path","source_id","quote","locator")} for e in r["evidence"]]} for r in selected]
    with output.open("x",encoding="utf8") as f:
        json.dump({"seed":seed,"items":items},f,ensure_ascii=False,indent=2)


def annotate(packet, directory, reviewer):
    if not reviewer.isidentifier(): raise ValueError("use a simple reviewer identifier")
    raw=packet.read_bytes()
    fingerprint=hashlib.sha256(raw).hexdigest()
    directory.mkdir(parents=True,exist_ok=True)
    journal=directory/(reviewer+".jsonl")
    done=set()
    if journal.exists():
        for line in journal.read_text(encoding="utf8").splitlines():
            record=json.loads(line)
            if record["packet_sha256"]!=fingerprint: raise ValueError("review packet changed")
            done.add(record["id"])
    for item in json.loads(raw)["items"]:
        if item["id"] in done: continue
        print("待判断事实　" + " → ".join(item["triple"]))
        for source in item["sources"]:
            show_source(source)
        answer=input("1 支持 / 0 不支持 / ? 不确定 / q 保存退出 > ").strip()
        while answer not in ("1","0","?","q"):
            answer=input("请输入 1、0、? 或 q > ").strip()
        if answer=="q": return
        note=input("备注或修正 > ")
        record=dict(id=item["id"],reviewer=reviewer,label=answer,note=note,packet_sha256=fingerprint,
                    triple=item["triple"],sources=item["sources"])
        with journal.open("a",encoding="utf8") as f:
            f.write(json.dumps(record,ensure_ascii=False)+"\n")


def missing(subset, directory, reviewer):
    """Annotate every source record independently, including zero-fact decisions."""
    if not reviewer.isidentifier(): raise ValueError("invalid reviewer identifier")
    raw=subset.read_bytes()
    digest=hashlib.sha256(raw).hexdigest()
    directory.mkdir(parents=True,exist_ok=True)
    journal=directory/(reviewer+".missing.jsonl")
    done=set()
    if journal.exists():
        for line in journal.read_text(encoding="utf8").splitlines():
            entry=json.loads(line)
            if entry["subset_sha256"]!=digest: raise ValueError("subset changed")
            if entry["action"]=="done": done.add(entry["index"])
    def save(action,index,source,**fields):
        record=dict(reviewer=reviewer,subset_sha256=digest,index=index,action=action,**source,**fields)
        with journal.open("a",encoding="utf8") as stream:
            stream.write(json.dumps(record,ensure_ascii=False)+"\n")
    for i,source in enumerate(json.loads(raw)):
        if i in done: continue
        show_source(source)
        while True:
            subject=input("主语（可补充任意遗漏事实），回车完成本片段，:q 退出 > ").strip()
            if subject==":q": return False
            if not subject:
                save("done",i,source);break
            predicate=input("关系 > ").strip()
            obj=input("宾语 > ").strip()
            quote=input("逐字引文（回车使用整个片段） > ") or source["text"]
            match = match_quote(source["text"], quote)
            if not predicate or not obj or match is None:
                print("关系、宾语及可定位引文均为必填");continue
            save("fact",i,source,triple=[subject,predicate,obj],**match)


def main():
    parser=argparse.ArgumentParser()
    cmds=parser.add_subparsers(dest="cmd",required=True)
    p=cmds.add_parser("prepare");p.add_argument("snapshot",type=Path);p.add_argument("packet",type=Path)
    p.add_argument("--count",type=int,default=100);p.add_argument("--seed",type=int,default=42)
    a=cmds.add_parser("label");a.add_argument("packet",type=Path);a.add_argument("directory",type=Path);a.add_argument("reviewer")
    m=cmds.add_parser("missing");m.add_argument("subset",type=Path);m.add_argument("directory",type=Path);m.add_argument("reviewer")
    args=parser.parse_args()
    if args.cmd=="missing" and missing(args.subset,args.directory,args.reviewer) is False:
        raise SystemExit(2)
    if args.cmd=="missing": return
    if args.cmd=="prepare": prepare(args.snapshot,args.packet,args.count,args.seed)
    else: annotate(args.packet,args.directory,args.reviewer)


if __name__=="__main__": main()
