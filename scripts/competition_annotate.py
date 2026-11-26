"""Two independent annotators use the same immutable packet and separate journals."""
import argparse
import hashlib
import json
import random
from pathlib import Path


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
        print(json.dumps({k:v for k,v in item.items() if k!="strata"},ensure_ascii=False,indent=2))
        answer=input("1 支持 / 0 不支持 / ? 不确定 / q 保存退出 > ").strip()
        while answer not in ("1","0","?","q"):
            answer=input("请输入 1、0、? 或 q > ").strip()
        if answer=="q": return
        note=input("备注或修正 > ")
        record=dict(id=item["id"],reviewer=reviewer,label=answer,note=note,packet_sha256=fingerprint)
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
        print(json.dumps(source,ensure_ascii=False,indent=2))
        while True:
            subject=input("主语（可补充任意遗漏事实），回车完成本片段，:q 退出 > ").strip()
            if subject==":q": return
            if not subject:
                save("done",i,source);break
            predicate=input("关系 > ").strip()
            obj=input("宾语 > ").strip()
            quote=input("逐字引文（回车使用整个片段） > ") or source["text"]
            if not predicate or not obj or not quote or quote not in source["text"]:
                print("关系、宾语及可定位引文均为必填");continue
            save("fact",i,source,triple=[subject,predicate,obj],quote=quote)


def main():
    parser=argparse.ArgumentParser()
    cmds=parser.add_subparsers(dest="cmd",required=True)
    p=cmds.add_parser("prepare");p.add_argument("snapshot",type=Path);p.add_argument("packet",type=Path)
    p.add_argument("--count",type=int,default=100);p.add_argument("--seed",type=int,default=42)
    a=cmds.add_parser("label");a.add_argument("packet",type=Path);a.add_argument("directory",type=Path);a.add_argument("reviewer")
    m=cmds.add_parser("missing");m.add_argument("subset",type=Path);m.add_argument("directory",type=Path);m.add_argument("reviewer")
    args=parser.parse_args()
    if args.cmd=="missing": return missing(args.subset,args.directory,args.reviewer)
    if args.cmd=="prepare": prepare(args.snapshot,args.packet,args.count,args.seed)
    else: annotate(args.packet,args.directory,args.reviewer)


if __name__=="__main__": main()
