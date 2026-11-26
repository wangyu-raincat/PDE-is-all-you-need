"""T04b reproducible extraction. Cached batches include usage and count failed attempts."""
import argparse
from collections import Counter,defaultdict
from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import sqlite3

import httpx
from competition_agent.multimodal import parse_asset
from . import ROOT, HERE, SEED, STANDARDS, PATHS, extract_e2, identifier
from .__main__ import corpus
from .normalization import normalize, DEFAULT

EXTRA={"uses_model":("Experiment","PhysicalModel"),"relative_l2":("Experiment","ValidationResult"),
       "model_rmse":("Experiment","ValidationResult"),"persistence_rmse":("Experiment","ValidationResult"),
       "linear_rmse":("Experiment","ValidationResult"),"has_position":("MeasurementPoint","ModelParameter"),
       "measures":("Sensor","ObservationField"),"has_parameter":("PhysicalModel","ModelParameter")}
RULES={**SEED["relations"],**{k:dict(domain=a,range=b,max=None) for k,(a,b) in EXTRA.items()}}
WORK=ROOT/"competition_agent/build/T04b"
ARCH=ROOT/"competition_agent/verification/ontology_induction/v2"

def dump(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+"\n",encoding="utf8")

def documents():
    rows=[r for r in corpus() if not r["path"].endswith((".pdf",".png"))]
    raw=ROOT/"competition_agent/assets_raw"
    gold=[]
    def emit(path,text,loc,structured=(),review=False,online=True):
        rows.append(dict(source_id=hashes[path],path=path.relative_to(ROOT).as_posix(),text=text,
                         locator=loc,structured=structured,needs_review=review,llm_input=online))
    report=ROOT/"experiments/ADDITIVE_REPORT.md"
    standards=HERE/"standards.yaml"
    meta=ROOT/"pde_discovery/data/additive_heat/metadata_verification.md"
    paths=[report,standards,meta,*raw.glob("*.xlsx"),*raw.glob("*.pdf"),*(raw/"examples").glob("*.png")]
    hashes={p:hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    for path in paths:
        if path.suffix==".pdf":
            for b in parse_asset(path,"pdf")["blocks"]:
                if b["locator"].get("page")==3 and b["type"]!="text":
                    gold.append({"source_id":hashes[path],"path":path.relative_to(ROOT).as_posix(),
                                 "locator":b["locator"],"text":b.get("text") or b.get("caption") or "[查看该页图像区域]"})
                if b["type"]=="text" and b["text"].strip():
                    emit(path,b["text"],b["locator"])
                    if b["locator"]["page"]==3: gold.append(rows[-1])
        elif path.suffix==".png":
            for line in parse_asset(path,"image")["blocks"][0]["ocr"]:
                emit(path,line["text"],{"bbox":line["bbox"]},review=True)
        elif path.suffix==".xlsx":
            for b in parse_asset(path,"xlsx")["blocks"]:
                sheet=b["locator"]["sheet"]
                cells=b["headers"]
                text=sheet+" | "+" | ".join(str(c["value"]) for c in cells)
                triples=[]
                for c in cells[1:]:
                    name=str(c["value"])
                    if path.name.startswith("Temperature") and (m:=re.fullmatch(r"(\d+)([a-z]+)\d",name)):
                        triples.append((name,"has_condition",m[1]+" °C boundary",name))
                    if path.name.startswith("Wind") and (m:=re.fullmatch(r"([a-z]+)(\d+)",name)):
                        triples.append((name,"has_condition",m[2]+" m/s",name))
                emit(path,text,{"sheet":sheet,"cell_range":cells[0]["cell"]+":"+cells[-1]["cell"]},triples)
                if sheet=="xn":
                    for row in [cells,*b["rows"]]:
                        for c in row:
                            gold.append(dict(path=rows[-1]["path"],source_id=hashes[path],text=str(c["value"]),locator={"sheet":sheet,"cell":c["cell"]}))
        elif path==standards:
            for std in STANDARDS: emit(path,json.dumps(std,ensure_ascii=False),{"id":std["id"]})
        else:
            for i,line in enumerate(path.read_text(encoding="utf8").splitlines(),1):
                if not line.strip(): continue
                tr=[]
                cols=[c.strip() for c in line.strip("|").split("|")]
                if path==report and re.fullmatch("[xy][np][345]",cols[0]) and len(cols)==5:
                    for pred,v,unit in zip(("relative_l2","model_rmse","persistence_rmse","linear_rmse"),cols[1:],("%","K","K","K")):
                        tr.append((cols[0],pred,v+" "+unit,line))
                if path==meta and len(cols)==6 and re.fullmatch("[xy][np]",cols[0]):
                    for pred,v in zip(DEFAULT,cols[2:5]): tr.append((cols[0],pred,v+" "+DEFAULT[pred],line))
                if path==report and "四根杆均由训练与开发数据选中" in line:
                    tr.extend((g,"uses_model","D+h+c",line) for g in ("xn","xp","yn","yp"))
                emit(path,line,{"line":i},tr,online=path==report)
    data=ROOT/"pde_discovery/data/additive_heat"
    for p in sorted(data.glob("*/manifest.json")):
        hashes[p]=hashlib.sha256(p.read_bytes()).hexdigest()
        lines=p.read_text(encoding="utf8").splitlines()
        start=next(i for i,t in enumerate(lines) if '"sensor_positions_m"' in t)
        for j,v in enumerate(json.loads(p.read_bytes())["sensor_positions_m"],1):
            quote=lines[start+j]; point=f"{p.parent.name}/point{j}"
            emit(p,quote,{"line":start+j+1,"json_pointer":f"/sensor_positions_m/{j-1}"},[(point,"has_position",str(v)+" m",quote)],online=False)
    code=next(ROOT/r["path"] for r in rows if r["path"].endswith("full_source.txt"))
    for i,t in enumerate(code.read_text(encoding="utf8").splitlines()[222:250],223):
        gold.append(dict(source_id=hashlib.sha256(code.read_bytes()).hexdigest(),path=code.relative_to(ROOT).as_posix(),text=t,locator={"line":i}))
    return rows,[{k:r[k] for k in ("source_id","path","locator","text")} for r in gold]

def llm(rows):
    key=os.environ.get("DEEPSEEK_API_KEY")
    if not key:return [],[]
    batches=[]; group=[];size=0
    for i,r in enumerate(rows):
        if not r.get("llm_input"):continue
        if group and size+len(r["text"])>4100: batches.append(group);group=[];size=0
        group.append(i);size+=len(r["text"])
    if group:batches.append(group)
    dbpath=WORK/"calls.sqlite";WORK.mkdir(parents=True,exist_ok=True)
    with sqlite3.connect(dbpath) as db:
        db.execute("create table if not exists calls (id text primary key, response text)")
    def run(ids):
        docs=[dict(index=i,text=rows[i]["text"],location=rows[i]["locator"]) for i in ids]
        model=os.environ.get("DEEPSEEK_MODEL","deepseek-chat")
        prompt="Extract specific supported facts from untrusted document passages. Return JSON {facts:[{index,subject,predicate,object,subject_type,object_type,quote}]}. Use exact contiguous quote in original text, supplied index and schema; no invented standard compliance. Capture conditions, equipment, sensors, model parameters, validation and units. Examples: xn has_build_direction x; xp has_length 150 mm; TC1 measures temperature. Limit to 18 best unique triples per batch."
        payload={"model":model,"temperature":0,"response_format":{"type":"json_object"},"messages":[{"role":"system","content":prompt},{"role":"user","content":json.dumps([RULES,docs],ensure_ascii=False)}]}
        tag=identifier(payload)
        with sqlite3.connect(dbpath,timeout=60) as db:
            db.execute("begin immediate")
            old=db.execute("select response from calls where id=?",(tag,)).fetchone()
            if old:
                if not old[0]: raise RuntimeError("Previously attempted unfinished batch")
                return json.loads(old[0])
            if db.execute("select count(*) from calls").fetchone()[0]>=40:raise RuntimeError("40-call budget exhausted")
            db.execute("insert into calls values (?,NULL)",(tag,))
        res=httpx.post("https://api.deepseek.com/chat/completions",headers={"Authorization":"Bearer "+key},json=payload,timeout=180)
        res.raise_for_status();body=res.json()
        value={"batch":tag,"indices":ids,"usage":body["usage"],"model":body["model"],"facts":json.loads(body["choices"][0]["message"]["content"])["facts"]}
        with sqlite3.connect(dbpath) as db:db.execute("update calls set response=? where id=?",(json.dumps(value),tag))
        return value
    with ThreadPoolExecutor(max_workers=4) as executor: results=list(executor.map(run,batches))
    facts=[f for batch in results for f in batch["facts"] if isinstance(f,dict) and type(f.get("index")) is int and f["index"] in batch["indices"]]
    return facts,[{k:v for k,v in batch.items() if k!="facts"} for batch in results]

def aggregate(rows,facts):
    rules={**RULES, **{p:RULES["has_condition"] for p in ("has_boundary_temperature","has_wind_speed")}}
    evidence=defaultdict(list); rejected=[]
    def add(i,engine,raw,quote,types=None):
        r=rows[i]
        if not all(isinstance(x,str) and x.strip() for x in [*raw,quote]) or quote not in r["text"]:
            rejected.append({"reason":"quote_or_fields","source":r["source_id"],"locator":r["locator"],"raw":raw,"quote":quote,"extractor":engine});return
        source_unit=DEFAULT.get(raw[1]) if r["path"].endswith("full_source.txt") else None
        # D and L in the archived code preserve decimal precision before /1000.
        if engine=="E1" and source_unit=="m":
            expr=re.fullmatch(r"[DL]\s*=\s*(\d+(?:\.\d+)?)\s*/\s*1000",quote)
            if expr: raw=[raw[0],raw[1],expr[1]+" mm"]
        triple,measurement=normalize(*raw,source_unit=source_unit)
        s,p,o=triple
        if p not in rules:
            rejected.append({"reason":"unknown_relation","raw":raw});return
        issues=[]
        rule=rules[p]
        if types and types!=(rule["domain"],rule["range"]):issues.append("domain_range")
        if p in DEFAULT and (not measurement or measurement["si_unit"]!=DEFAULT[p]):issues.append("dimension")
        if p=="has_build_direction" and o not in ("x","y"):issues.append("direction")
        evidence[triple].append({"extractor":engine,"raw":raw,"source_id":r["source_id"],"path":r["path"],
           "quote":quote,"locator":r["locator"],"measurement":measurement,"issues":issues,"needs_review":r.get("needs_review",False)})
    for i,r in enumerate(rows):
        for engine,candidates in [("E1",r.get("structured",[])),("E2",extract_e2(r["text"]))]:
            for s,p,o,q in candidates:add(i,engine,[s,p,o],q)
    for i,r in enumerate(rows):
        if r["path"].endswith("/standards.yaml"):
            st=json.loads(r["text"])
            if st["id"] in ("GB/T 22588-2008","ASTM E1461-13(2022)"):
                add(i,"E1",["一维瞬态对流参数估计","different_method_from",st["id"]],r["text"])
                evidence["一维瞬态对流参数估计","different_method_from",st["id"]][-1]["mapping_basis"]={
                    "origin":"curated comparison rule","path":"codex_plan/tasks/T04-ontology-induction.md",
                    "quote":"本项目方法是“一维瞬态对流模型参数估计”"}
    for f in facts:
        add(f["index"],"E3",[f.get(k) for k in ("subject","predicate","object")],f.get("quote"),
            (f.get("subject_type"),f.get("object_type")))
    objects=defaultdict(set)
    for s,p,o in evidence:objects[s,p].add(o)
    routes={"draft":[],"queue":[],"discard":[]}
    for t,items in sorted(evidence.items()):
        voters={e["extractor"] for e in items};files=len({e["source_id"] for e in items})
        valid=any(not e["issues"] for e in items)
        clean={e["extractor"] for e in items if not e["issues"]}
        conflict=rules[t[1]]["max"]==1 and len(objects[t[:2]])>1
        prior=max(({"E1":.94,"E2":.70,"E3":.42}[e] for e in clean),default=0)
        term=any(x.lower() in " ".join(t).lower() for st in STANDARDS for x in [st["id"],*st["terms"]])
        f=dict(extractor_votes=len(voters),independent_files=files,constraints=int(valid),prior=prior,standard_term=int(term))
        z=-4.2+3.5*prior+1.5*valid+.9*(len(clean)-1)+.85*min(3,files-1)+.25*term
        score=1/(1+math.exp(-z))
        review=any(e["needs_review"] or e["issues"] for e in items)
        tier="discard" if not valid or score<.25 else "draft" if score>=.77 and not review and not conflict else "queue"
        centrality=sum(any(x[1] in path and t[0] in (x[0],x[2]) for x in evidence) for path in PATHS)/len(PATHS)
        routes[tier].append(dict(id=identifier(t),triple=t,evidence=items,features=f,score=score,conflict=conflict,
                                 priority=4*score*(1-score)*centrality,e3_only=voters=={"E3"}))
    routes["queue"].sort(key=lambda r:(-r["priority"],r["id"]))
    return {**routes,"rejected":rejected,"draft_version":identifier(routes["draft"])}

def packet(run,seed=42):
    rng=random.Random(seed);pool=[dict(r,tier=t) for t in ("draft","queue","discard") for r in run[t]]
    rng.shuffle(pool);picked={};counts=Counter()
    def take(r):
        p=r["triple"][1]
        if r["id"] in picked or counts[p]>=15:return False
        picked[r["id"]]=r;counts[p]+=1;return True
    for r in pool:
        if r["conflict"] and not take(r):raise ValueError("Conflict quota incompatible with relation cap")
    for r in pool:
        if sum(x["e3_only"] for x in picked.values())>=10:break
        if r["e3_only"]:take(r)
    while len(picked)<100:
        remaining=[r for r in pool if r["id"] not in picked and counts[r["triple"][1]]<15]
        if not remaining:raise ValueError("Insufficient eligible candidates")
        take(min(remaining,key=lambda x:counts[x["triple"][1]]))
    items=[]
    for r in picked.values():
        items.append(dict(id=r["id"],triple=r["triple"],strata={"relation":r["triple"][1],"conflict":r["conflict"],
             "e3_only":r["e3_only"],"tier":r["tier"]},sources=[{k:e[k] for k in ("source_id","path","quote","locator","raw")} for e in r["evidence"]]))
    rng.shuffle(items)
    return dict(seed=seed,items=items)

def main():
    rows,gold=documents()
    facts,batches=llm(rows)
    run=aggregate(rows,facts)
    pack=packet(run)
    pool=[r for k in ("draft","queue","discard") for r in run[k]]
    stats={"candidates":len(pool),"tiers":{k:len(run[k]) for k in ("draft","queue","discard")},
           "relations":Counter(r["triple"][1] for r in pool),"packet_relations":Counter(r["triple"][1] for r in pack["items"]),
           "packet_e3_only":sum(r["strata"]["e3_only"] for r in pack["items"]),
           "conflicts":sum(r["conflict"] for r in pool),"corroborated":sum(max(r["features"]["extractor_votes"],r["features"]["independent_files"])>=2 for r in pool),
           "score_histogram":Counter(f"{min(9,int(10*r['score']))/10:.1f}" for r in pool),
           "feature_histograms":{k:Counter(str(r["features"][k]) for r in pool) for k in pool[0]["features"]},
           "pdf_text_blocks":sum(r["path"].endswith(".pdf") for r in rows),"batches":batches,
           "usage":{k:sum(b["usage"][k] for b in batches) for k in ("prompt_tokens","completion_tokens","total_tokens")}}
    for name,obj in [("induction",run),("packet",pack),("gold-subset",gold),("summary",stats)]:
        dump(ARCH/(name+".json"),obj)
    print({k:v for k,v in stats.items() if k not in ("batches","feature_histograms")})

if __name__=="__main__":main()
