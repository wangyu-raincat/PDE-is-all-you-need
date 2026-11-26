"""Build a reviewable induction snapshot from public manufacturing metadata."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

from competition_agent.multimodal import parse_asset
from . import ROOT, fit_weights, induce


def corpus():
    rows=[]
    data=ROOT/"pde_discovery/data/additive_heat"
    raw=ROOT/"competition_agent/assets_raw"
    def emit(path,source,text,locator,structured=(),**flags):
        rows.append(dict(path=path.relative_to(ROOT).as_posix(),source_id=source,text=text,
                         locator=locator,structured=structured,**flags))
    for path in sorted(data.glob("*/manifest.json")):
        content=path.read_bytes()
        lines=content.decode().splitlines()
        meta=json.loads(content)["experiment"]
        for key,relation in [("material","has_material"),("build_direction","has_build_direction"),("porosity_category","has_porosity_class")]:
            line=next(i for i,t in enumerate(lines) if t.strip().startswith('"'+key+'":'))
            emit(path,hashlib.sha256(content).hexdigest(),lines[line],{"line":line+1},
                 [(meta["specimen"],relation,meta[key],lines[line])])
    for path in sorted(data.glob("raw/*.csv")):
        with path.open(encoding="utf-8-sig") as stream: header=stream.readline().rstrip()
        emit(path,hashlib.sha256(path.read_bytes()).hexdigest(),header,{"line":1},
             [(path.stem,"records_field",v,v) for v in next(csv.reader([header]))])
    code=ROOT/"pde_discovery/data/mendeley_heat/author_oneDkhEstimator_V1.0.0_full_source.txt"
    asset=parse_asset(code,"code")
    for b in asset["blocks"]:
        groups=[g for g in ("xn","yn","xp","yp") if g.upper()+" == True" in b["scope"]]
        if not groups: continue
        group=groups[0];var=b["variable"];quote=b["text"]
        if var=="filenames": triples=[(group,"has_test",v,quote) for v in b["value"]]
        elif var in ("D","L","rho"):
            relation,unit={"D":("has_diameter","m"),"L":("has_length","m"),"rho":("has_density","kg/m3")}[var]
            triples=[(group,relation,format(b["value"],".12g")+" "+unit,quote)]
        else: continue
        emit(code,asset["source_id"],quote,dict(b["locator"],file=code.relative_to(ROOT).as_posix()),triples)
    path=raw/"AdditiveRodResults.xlsx"
    asset=parse_asset(path,"xlsx")
    for block in asset["blocks"]:
        group=block["locator"]["sheet"]
        for cell in block["headers"][1:]:
            value=cell["value"]
            emit(path,asset["source_id"],value,{"sheet":group,"cell":cell["cell"]},[(group,"has_test",value,value)])
    for path in sorted((raw/"examples").glob("*.png")):
        asset=parse_asset(path,"image")
        for line in asset["blocks"][0]["ocr"]:
            if "has_build_direction" in line["text"]:
                emit(path,asset["source_id"],line["text"],{"bbox":line["bbox"]},needs_review=True,llm_input=True)
    path=raw/"dib-2021-107632.pdf"
    asset=parse_asset(path,"pdf")
    for b in asset["blocks"]:
        if "build direction (x or y)" in b.get("text",""):
            emit(path,asset["source_id"],b["text"],b["locator"],llm_input=True)
    return rows


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--output",type=Path,default=ROOT/"competition_agent/build/T04/induction.json")
    ap.add_argument("--no-e3",action="store_true")
    ap.add_argument("--development",type=Path)
    args=ap.parse_args()
    weights=fit_weights(json.loads(args.development.read_text(encoding="utf8"))) if args.development else None
    result=induce(corpus(),use_e3=not args.no_e3,weights=weights)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf8")
    print(result["e3_status"],{k:len(result[k]) for k in ("draft","queue","discard","rejected")})


if __name__=="__main__": main()
