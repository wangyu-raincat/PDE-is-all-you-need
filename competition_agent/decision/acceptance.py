"""Live local MCP + DeepSeek acceptance for PLAN §2. The model can select cited evidence only."""
import json
from datetime import timedelta
from pathlib import Path
import sys

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from competition_agent.deepseek_acceptance import DeepSeek, Journal, ROOT, parse_json, tool_value, redact
from competition_agent.service import Lab
from . import QUESTIONS
from .retrieval import export_knowledge


def save(path,data):
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2)+"\n",encoding="utf8")


def validate(index, trace):
    reasoning = trace["reasoning"]
    expected = ["dispersion","transfer","standard","conflict","refuse"][index]
    if trace["rule"]["id"] != expected or not trace["rule"]["evidence"]["quote"]:
        raise ValueError("Decision did not follow the expected rule")
    if index == 0:
        fits = reasoning["numerical_tools"]
        assert all(len(v["fits"])==5 and v["summary"] for v in fits["heat_repeatability"].values())
        assert len(fits["historical_holdouts"])==6
        edges = [e for p in reasoning["graph_paths"] for e in p]
        assert {"has_porosity_class","has_diameter","has_test","records_field"} <= {e["triple"][1] for e in edges}
        assert any(len(p)>=2 for p in reasoning["graph_paths"])
        locations = json.dumps([v["locator"] for e in edges for v in e["evidence"]])
        assert all(k in locations for k in ('"bbox"','"sheet"','"cell"','"line"'))
    if index == 2:
        assert {m["relation"] for m in reasoning["standard_align"]}=={"comparable_quantity","different_method_from"}
    if index == 3:
        assert len({tuple(e["triple"]) for e in reasoning["conflicts"] if e.get("triple",[])[:2]==["xn","has_diameter"]})>=2
    if index == 4:
        assert not reasoning["numerical_tools"] and not reasoning["standard_align"]


def statements(trace):
    """Candidate explanation sentences come exclusively from verified tool fields."""
    why = trace["reasoning"]
    facts = {"rule": trace["rule"]["evidence"]["quote"]}
    if why["numerical_tools"].get("heat_repeatability"):
        for group,r in why["numerical_tools"]["heat_repeatability"].items():
            d = r["summary"]["D"]
            facts[group] = f"{group} 的平均 D 为 {d['mean']:.4g} m²/s，样本标准差为 {d['sample_std']:.4g} m²/s。"
    for group,count in why["numerical_tools"].get("documented_acquisitions",{}).items():
        facts[group] = f"{group} 有 {count} 条可定位测试记录。"
    for i,item in enumerate(why["standard_align"]):
        facts[f"standard{i}"] = f"{item['standard']}：{item['term']}，{item['relation']}（{item['status']}）。"
    for i,item in enumerate(why["conflicts"]):
        if item.get("triple"):
            facts[f"conflict{i}"] = "来源断言："+" → ".join(item["triple"])+"。"
    for i,item in enumerate(trace["decision"]["gaps"]):
        facts[f"gap{i}"] = "待补证据："+item+"。"
    for i,item in enumerate(trace["decision"]["next_experiments"]):
        facts[f"next{i}"] = item.rstrip("。")+"。"
    return facts


async def run(output,key,model,max_calls):
    output = Path(output).resolve()
    output.mkdir(parents=True,exist_ok=True)
    if any(output.iterdir()):
        raise ValueError("使用新的验收输出目录")
    journal = Journal(output/"model.jsonl",key)
    client = DeepSeek(key,journal,model,max_calls)
    runtime = output/"runtime"
    result = {"questions":[],"mcp_calls":[],"all_passed":False,"model":model}
    try:
        params = StdioServerParameters(command=sys.executable,args=["-m","competition_agent.mcp_server","--directory",str(runtime)],cwd=str(ROOT))
        async with stdio_client(params) as (r,w), ClientSession(r,w,read_timeout_seconds=timedelta(minutes=5)) as session:
            await session.initialize()
            async def call(name,**arguments):
                result["mcp_calls"].append(name)
                response = await session.call_tool(name,arguments)
                if response.isError:
                    raise ValueError(name+": "+response.content[0].text)
                return tool_value(response)
            # Only the case storage is shared with the synthetic harness. All reported heat results use real data.
            case = (await call("pde_case_create",request_id="manufacturing-decision-acceptance"))["case_id"]
            raw = ROOT/"competition_agent/assets_raw"
            docs = [(raw/"dib-2021-107632.pdf","pdf"),(raw/"AdditiveRodResults.xlsx","xlsx"),
                    (next((ROOT/"pde_discovery/data/mendeley_heat").glob("*full_source.txt")),"code")]
            await call("asset_ingest",case_id=case,documents=[{"path":str(path),"kind":kind} for path,kind in docs])
            await call("ontology_induce",case_id=case)
            await call("ontology_review_queue",case_id=case,limit=1)
            pattern = ["Specimen","has_test","Experiment","records_field","ObservationField"]
            paths = await call("kg_query",case_id=case,pattern=pattern,start="xp",limit=4)
            assert paths["paths"]
            search = await call("evidence_retrieve",case_id=case,query="xp1 TC1 热扩散 测点",pattern=pattern,start="xp",limit=12)
            assert set().union(*(e["ranks"] for e in search["evidence"]))=={"bm25","graph","vector"}
            with Lab(runtime).case(case,readonly=True) as state:
                export_knowledge(state,output/"knowledge")
            for i,question in enumerate(QUESTIONS):
                if i == 3:
                    await call("asset_ingest",case_id=case,documents=[{"title":"自制冲突样例（仅验收）","kind":"text","text":"xn diameter = 7.9 mm"}])
                answer = await call("decision_make",case_id=case,question=question)
                handoff = await call("case_handoff",case_id=case)
                assert handoff["next_tool"]=="decision_trace_get" and handoff["next_arguments"]["trace_id"]==answer["trace_id"]
                trace = await call("decision_trace_get",trace_id=answer["trace_id"])
                validate(i,trace)
                save(output/f"trace-{i+1}.json",trace)
                choices = statements(trace)
                msg = await client.complete("pde_coordinator",[
                    {"role":"system","content":"你为工程师编排已核验的证据解释。按问题相关性从给定字典选择一到四个不同的键并排序。只返回 JSON 对象 { \"evidence_order\": [键,...] }。"},
                    {"role":"user","content":json.dumps({"question":question,"decision":answer,"available_evidence":choices},ensure_ascii=False)}],[])
                order = parse_json(msg["content"])["evidence_order"]
                if not isinstance(order,list) or not 1<=len(order)<=4 or any(k not in choices for k in order) or len(set(order))!=len(order):
                    raise ValueError("模型引用了无效证据")
                result["questions"].append(dict(question=question,**answer,evidence_order=order,
                                                explanation=[choices[k] for k in order],trace_file=f"trace-{i+1}.json"))
        result["all_passed"]=True
    except Exception as exc:
        while isinstance(exc, BaseExceptionGroup):
            exc=exc.exceptions[0]
        result["error"]=redact(repr(exc),key)
        journal.add("run_error",detail=str(exc))
    finally:
        result.update(model_requests=client.calls,token_usage=client.usage,tool_call_count=len(result["mcp_calls"]))
        await client.close()
        save(output/"summary.json",result)
        print({k:v for k,v in result.items() if k not in ("questions","mcp_calls")},flush=True)
    return result
