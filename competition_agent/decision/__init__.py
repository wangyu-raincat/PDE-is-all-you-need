"""Manufacturing decisions derived from case evidence and fixed numerical tools."""
import hashlib
import json
from pathlib import Path
import re

from competition_agent.ontology.tools import induce_case, queue, standard_align
from competition_agent.scenario_am.catalog import ROOT, evidence
from competition_agent.scenario_am.heat_tools import heat_compare, heat_holdout_summary
from competition_agent.service import digest
from .retrieval import fused, graph_edges, typed_paths

RULE_FILE = Path(__file__).with_name("rules.yaml")
RULES = json.loads(RULE_FILE.read_bytes())
QUESTIONS = [
    "这批增材杆测温数据遵循什么传热规律？哪些结论可靠？",
    "xp（较多孔）和 xn 两根增材杆的热扩散特性有没有可辨别差异？依据是哪些记录、哪个模型？",
    "这批数据能否作为工艺仿真的材料输入？还缺什么实验？",
    "我们的瞬态对流法和国标闪光法测热扩散系数有什么区别，结论能不能对齐到标准？",
    "两份资料里同一根杆的直径不一致，还能用吗？",
    "把新关系写进正式本体 / 用最终验证结果再调模型。",
]


def intent(question):
    q = question.casefold()
    if any(term in q for term in ("正式本体","调参","调模型","重新选模","最终验证","final validation")):
        return "refuse"
    if any(term in q for term in ("冲突","不一致","矛盾")):
        return "conflict"
    if any(term in q for term in ("国标","闪光","标准","flash method")):
        return "standard"
    if any(term in q for term in ("传热规律", "传热方程", "测温数据遵循")):
        return "equation"
    if "xp" in q and "xn" in q:
        return "dispersion"
    if any(term in q for term in ("材料输入","工艺仿真","仿真输入","缺什么实验")):
        return "transfer"
    return "gap"


def rule_citation(key):
    return evidence(RULE_FILE, RULES[key])


def rod_paths(state, groups=("xp","xn")):
    paths = []
    for group in groups:
        for relation,target in (("has_porosity_class","PorosityClass"),("has_diameter","ModelParameter")):
            paths.extend(typed_paths(state,["Specimen",relation,target],group,4))
        paths.extend(typed_paths(state,["Specimen","has_test","Experiment","records_field","ObservationField"],
                                 group,5,"TC1"))
    return paths


def stainless_diagnostics():
    report = ROOT / "experiments/IDENTIFIABILITY_REPORT.md"
    groups = {}
    for line in report.read_text(encoding="utf8").splitlines():
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if cells[0] in ("50ss", "75ss", "100ss"):
            if len(cells) != 6 or cells[0] in groups:
                raise ValueError("不锈钢可识别性诊断表格式异常")
            groups[cells[0]] = dict(profile_ratio=float(cells[3]), deletion_ratio=float(cells[5]),
                                    evidence=evidence(report, line))
    if len(groups) != 3:
        raise ValueError("不锈钢诊断缺组")
    result = {"scope": "旧不锈钢温度组三条训练记录；有限采样和训练损失敏感性参照", "groups": groups}
    for key in ("profile_ratio", "deletion_ratio"):
        values = [g[key] for g in groups.values()]
        result[key] = {"min": min(values), "max": max(values)}
    return result


def author_diameters(paths):
    diameters = {}
    for path in paths:
        edge = path[0]
        group, relation, _ = edge["triple"]
        if relation != "has_diameter" or group not in ("xp", "xn"):
            continue
        for ev in edge["evidence"]:
            for location in ev["locator"].get("multimodal_locations", []):
                loc = location["locator"]
                if Path(loc.get("file", "")).name != "author_oneDkhEstimator_V1.0.0_full_source.txt":
                    continue
                measurement = ev.get("measurement") or {}
                if measurement.get("si_unit") == "m":
                    diameters[group] = {"mm": float(measurement["value"]) * 1000,
                                        "evidence": {"source_id": ev["source_id"], "locator": loc, "quote": ev["quote"]}}
    return diameters


def downgrade_triggers(a, b, diameter_a, diameter_b, observed_upper):
    delta_d = a["D"]["mean"] - b["D"]["mean"]
    delta_h = a["h"]["mean"] - b["h"]["mean"]
    ratio = max(a["D"]["mean"], b["D"]["mean"]) / min(a["D"]["mean"], b["D"]["mean"])
    return ratio, {"within_observed_sensitivity": ratio <= observed_upper,
                   "large_same_direction_changes": delta_d * delta_h > 0
                       and abs(delta_d) > max(a["D"]["range"], b["D"]["range"])
                       and abs(delta_h) > max(a["h"]["range"], b["h"]["range"]),
                   "different_diameters": diameter_a != diameter_b}


def comparison(paths):
    selection = heat_compare("xp","xn")
    repeats = selection["repeatability"]
    historical = []
    for group in repeats:
        for rec in heat_holdout_summary(group)["records"]:
            val = rec["validation"]
            path = ROOT / rec["source_path"]
            quote = next(line.strip() for line in path.read_text(encoding="utf8").splitlines()
                         if '"relative_l2":' in line and str(val["relative_l2"]) in line)
            historical.append(dict(test=rec["case"], relative_l2=val["relative_l2"], evidence=evidence(path,quote)))
    numeric = dict(heat_repeatability=repeats, historical_holdouts=historical,
                   heat_compare={"parameter_comparison":selection["parameter_comparison"],
                                 "selected_models":{group:selection[key]["selected"]["model_name"]
                                                    if selection[key]["selected"] else None
                                                    for group,key in (("xp","group_a"),("xn","group_b"))}})
    if not all(r["summary"] for r in repeats.values()) or not all(numeric["heat_compare"]["selected_models"].values()):
        return numeric, "部分记录未获得可比较的拟合结果。", False
    for group,key in (("xp","group_a"),("xn","group_b")):
        source = ROOT / selection[key]["frozen_consistency"]["source_path"]
        label = numeric["heat_compare"]["selected_models"][group]
        line = next(line.strip() for line in source.read_text(encoding="utf8").splitlines()
                    if line.strip().startswith(f'"model_name": "{label}"'))
        numeric["heat_compare"].setdefault("model_evidence",{})[group] = evidence(source,line)
    diag = stainless_diagnostics()
    diameters = author_diameters(paths)
    numeric.update(identifiability_evidence=diag, diameters=diameters)
    if set(diameters) != {"xp", "xn"}:
        return numeric, "缺少作者源码中可定位的两杆直径，暂缓材料比较。", False
    a, b = (repeats[g]["summary"] for g in ("xp", "xn"))
    upper = max(diag["profile_ratio"]["max"], diag["deletion_ratio"]["max"])
    ratio, triggers = downgrade_triggers(a, b, diameters["xp"]["mm"], diameters["xn"]["mm"], upper)
    delta = abs(a["D"]["mean"] - b["D"]["mean"])
    spread = max(a["D"]["range"], b["D"]["range"])
    numeric.update(mean_D_difference=delta, larger_within_rod_range=spread,
                   mean_D_ratio=ratio, downgrade_triggers=triggers)
    lines = []
    for group in ("xp", "xn"):
        stats = repeats[group]["summary"]
        corr = repeats[group]["D_h_correlation"]["r"]
        r_text = f"{corr:.3f}" if corr is not None else "未定义"
        lines.append(f"{group}（直径 {diameters[group]['mm']:.3f} mm）：五次拟合 D 均值 {stats['D']['mean']:.4g}、"
                     f"极差 {stats['D']['range']:.4g} m²/s；h 均值 {stats['h']['mean']:.4g}、"
                     f"极差 {stats['h']['range']:.4g} s⁻¹；描述性 D–h 相关 r={r_text}（n=5）。")
    profile, deletion = diag["profile_ratio"], diag["deletion_ratio"]
    lines.append(f"两杆平均 D 差 {delta:.4g} m²/s，倍数 {ratio:.3f}。旧不锈钢训练记录中，"
                 f"训练 RMSE 容许增加 5% 时 D 的近优剖面倍数为 {profile['min']:.3f}–{profile['max']:.3f}，"
                 f"五块删块倍数为 {deletion['min']:.3f}–{deletion['max']:.3f}。")
    if triggers["different_diameters"]:
        lines.append("直径不同可影响侧面散热衰减率 h，几何与孔隙类别混杂。")
    if a["D"]["mean"] > b["D"]["mean"]:
        lines.append("较多孔杆的拟合 D 较高，方向值得复核。")
    if any(triggers.values()):
        lines.append("模型参数存在差异，但不足以判定材料热扩散性质差异。")
    else:
        lines.append("仅报告两根杆的模型参数比较，材料性质仍需独立验证。")
    return numeric, "\n".join(lines), True


def decide(lab,case_id,question):
    if not isinstance(question,str) or not 1<=len(question.strip())<=600:
        raise ValueError("需要 1–600 字的问题")
    kind = intent(question)
    with lab.case(case_id,readonly=True) as state:
        lab.verify_seals(state)
        frozen = state["phase"] in ("frozen","validating","validated")
    if not frozen and kind!="refuse":
        induce_case(lab,case_id)
    pending = queue(lab,case_id,1)["total"]
    with lab.case(case_id) as state:
        lab.verify_seals(state)
        if not frozen and kind != "refuse":
            records = state.get("input_preparation", {}).get("records", {})
            ready = {key for key,row in records.items() if row["status"]=="ready" and not row["duplicate_of"]}
            for space in state.get("induction", {}).values():
                indexed = {k[7:] for k in space["values"] if k.startswith("source:")}
                if ready != indexed:
                    raise ValueError("资料集刚发生变更，请重新运行增量归纳及决策")
        groups = ("xn","xp","yn","yp") if kind in ("equation","transfer") else ("xp","xn")
        paths = rod_paths(state,groups) if kind!="refuse" else []
        conflicts = [e for e in graph_edges(state) if e.get("conflict")]
        conflicts.extend(lab.conflicts(state))
        conflicts.extend(i for i in state.get("input_preparation",{}).get("summary",{}).get("issues",[])
                         if i["priority"]=="P0")
        key = "conflict" if conflicts and kind!="refuse" else kind
        hits = fused(state,question,paths,6) if kind!="refuse" else []
        rounds = [{"stage":"retrieve","query":question,"blocks":[h["block_id"] for h in hits]}]
        mods = {h["modality"] for h in hits}
        if kind in ("transfer","dispersion") and not {"pdf","xlsx","code"} <= mods:
            query = "xp xn more porous specimen built along direction filenames diameter has_test"
            more = fused(state,query,paths,12)
            hits = list({h["block_id"]:h for h in hits+more}.values())
            rounds.append({"stage":"retrieve","loop_reason":"evidence_gap","query":query,"blocks":[h["block_id"] for h in more]})
        cited = {ev["source_id"] for path in paths for edge in path for ev in edge["evidence"]}
        expected = {"pdf": ROOT/"competition_agent/assets_raw/dib-2021-107632.pdf",
                    "xlsx": ROOT/"competition_agent/assets_raw/AdditiveRodResults.xlsx",
                    "code": ROOT/"pde_discovery/data/mendeley_heat/author_oneDkhEstimator_V1.0.0_full_source.txt"}
        modalities = set()
        if key in ("dispersion","equation"):
            for source,record in state.get("multimodal_assets",{}).items():
                asset = record["asset"]
                path = expected.get(asset["modality"])
                if source in cited and path and path.is_file() and asset["sha256"]==hashlib.sha256(path.read_bytes()).hexdigest():
                    modalities.add(asset["modality"])
        numeric,align,gaps,experiments = {},[],[],[]
        model_ambiguous = False
        confidence = "有限"
        if key == "refuse":
            confidence = "明确（操作权限）"
            conclusion = RULES[key]
            experiments = ["提交本地操作员审核；另立数据采集方案。"]
        elif key == "conflict":
            confidence = "暂缓判断"
            conclusion = RULES[key] if conflicts else "尚未定位到两份相互冲突的原文，暂缓使用该直径。"
            gaps = ["核查来源版本、单位和试样身份"]
            experiments = ["同一试样重新测径并保留原始记录"]
        elif key == "equation" and {"pdf","xlsx","code"} <= modalities:
            from .equation import equation_evidence, render_equation
            numeric = equation_evidence(stainless_diagnostics(), QUESTIONS[1])
            conclusion = render_equation(numeric)
            confidence = ("候选族内结构与条件预测可靠，单个系数受限" if all(
                r["supported"] for r in numeric["rods"].values()) else "结构与预测证据不足")
            gaps = ["独立材料系数验证"]
            experiments = ["参见 Q1 的独立试样闪光法复测方案"]
        elif key == "standard":
            method = ROOT/"competition_agent/scenario_am/heat_tools.py"
            numeric["method_evidence"] = evidence(method,"给定实测边界的四内点条件预测（物理先验 D+h+c）")
            align = standard_align("瞬态对流法")["matches"] + standard_align("热扩散率")["matches"]
            conclusion = "项目用六个测点的瞬态记录、实测边界与四个内点拟合有效参数；GB/T 22588-2008 采用闪光法测量热扩散系数或导热系数。"+RULES[key]
            gaps = ["标准方法的配对测量", "标准映射由本地操作员确认"]
            experiments = ["在相同试样与温度条件下做闪光法对标复测"]
        elif key == "dispersion" and {"pdf","xlsx","code"} <= modalities:
            numeric, conclusion, ok = comparison(paths)
            from competition_agent.scenario_am.joint_fit import read_joint_evidence
            joint = read_joint_evidence()
            numeric['heat_joint_fit'] = joint
            if not ok:
                model_ambiguous = True
                gaps.append("可比较的拟合或几何证据不足")
            elif any(numeric["downgrade_triggers"].values()):
                key = "identifiability"
                confidence = "证据不足（材料性质）"
                model_ambiguous = True
            if ok and joint.get('available') and joint['comparison']['distinguishable']:
                key = 'joint_identifiability'
                confidence = '有效参数可辨别（同装置同模型）'
                model_ambiguous = False
                comp = joint['comparison']
                values = joint['rods']
                conclusion = (f"预注册联合拟合：xp 的 D*={values['xp']['joint']['D']:.9g} m²/s，"
                    f"xn 的 D*={values['xn']['joint']['D']:.9g} m²/s；"
                    f"较大/较小比值 {comp['D_ratio']:.6f}，严格超过 LORO 倍数之积 {comp['loro_ratio_product']:.6f}。"
                    "两杆均满足联合剖面≤1.5、LORO≤1.3 的固定门槛。"
                    "同一装置、同一模型下两杆有效热扩散参数可辨别；仍不等于材料性质差异或材料证书，仍建议标准方法复核。")
            gaps.append("独立试样与逐次工况")
            experiments = ["按 GB/T 22588-2008 闪光法复测 xn、xp，每类至少 3 个独立试样；复测时记录风速与加热设定。"]
        elif key == "transfer" and paths:
            counts = {g:len({p[0]["triple"][2] for p in paths if p[0]["triple"][:2]==[g,"has_test"]}) for g in groups}
            scopes = []
            for group,count in counts.items():
                if count:
                    path = ROOT / "pde_discovery/data/additive_heat" / group / "manifest.json"
                    note = json.loads(path.read_bytes())["scope_note"]
                    scopes.append(evidence(path,note))
            numeric = {"documented_acquisitions":counts,"experimental_scope":scopes}
            if min(counts.values())<3: key="repeat_count"
            conclusion = RULES[key]
            gaps = ["逐次风速、加热与复位工况记录","独立材料试样","标准方法对照"]
            experiments = ["记录每次边界条件并复测","补充材料参数的标准方法对照"]
        else:
            key="gap"
            confidence="证据不足"
            conclusion=RULES[key]
            gaps=["需要可定位的试样、模型、测试记录及问题范围"]
            experiments=["补充相关资料后重新检索"]
        reason = ("conflict" if key=="conflict" else "model_ambiguous" if model_ambiguous
                  else "evidence_gap" if gaps else None)
        output = dict(conclusion=conclusion,confidence_level=confidence,gaps=gaps,next_experiments=experiments)
        trace = dict(case_id=case_id,question=question,decision=output,rule={"id":key,"evidence":rule_citation(key)},
                     reasoning={"numerical_tools":numeric,"standard_align":align,"graph_paths":paths,
                                "conflicts":conflicts,"pending_review_count":pending},
                     retrieval=[{**h,"quote":h["quote"][:1200]} for h in hits],
                     rounds=rounds,loop_reason=reason,graph_version=state.get("induction",{}).get("graph",{}).get("version",0))
        if kind=="dispersion" and numeric.get("heat_repeatability"):
            # Explicit joins: specimen -> named acquisition -> raw CSV -> fixed fit -> saved score.
            indexed = {item["test"]:item for item in numeric["historical_holdouts"]}
            links = []
            for path in paths:
                group,pred,test = path[0]["triple"]
                if pred=="has_test" and group in numeric["heat_repeatability"]:
                    row = next((r for r in numeric["heat_repeatability"][group]["fits"]
                                if test == group+str(r["repeat"])), None)
                    if row:
                        links.append(dict(group=group, test=test, edges=[e["triple"] for e in path],
                                          fit=row, historical_score=indexed.get(test)))
            trace["reasoning"]["acquisition_links"]=links
        trace_id = case_id+"."+digest(trace)
        if not any(e["operation"]=="decision_made" and e["details"]["trace_id"]==trace_id for e in state["events"]):
            lab.event(state,"pde_coordinator","decision_made",{"trace_id":trace_id,"trace":trace})
        return dict(output,trace_id=trace_id)


def get_trace(lab,trace_id):
    if not re.fullmatch(r"[0-9a-f]{32}\.[0-9a-f]{64}",trace_id):
        raise ValueError("Invalid trace ID")
    case_id,fingerprint = trace_id.split(".")
    with lab.case(case_id,readonly=True) as state:
        lab.verify_seals(state)
        for event in reversed(state["events"]):
            if event["operation"]=="decision_made" and event["details"]["trace_id"]==trace_id:
                trace = event["details"]["trace"]
                if digest(trace)!=fingerprint:raise ValueError("Decision trace changed")
                return dict(trace,event_sha256=event["sha256"],trace_id=trace_id)
    raise ValueError("Unknown trace ID")
