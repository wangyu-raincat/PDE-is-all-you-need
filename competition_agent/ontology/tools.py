"""Case-local ontology and graph drafts. Lab.case provides transaction isolation."""
from collections import defaultdict
from copy import deepcopy
from functools import lru_cache
import re

from sklearn.feature_extraction.text import TfidfVectorizer

from . import SEED, STANDARDS, identifier, extract_e2
from .normalization import DEFAULT
from .normalization import normalize
from .revision import RULES, aggregate
from competition_agent.multimodal import literal_candidates

FROZEN = {"frozen", "validating", "validated"}
KNOWN = {s.casefold() for s in (*SEED["classes"], *RULES)}
TERM = re.compile(r"[\u3400-\u9fff]{2,18}|[A-Za-z][A-Za-z_-]*(?: [A-Za-z][A-Za-z_-]*){0,2}")
ALIASES = {"闪光法": "flash method", "热扩散率": "热扩散系数", "thermal diffusivity": "热扩散系数",
           "thermal conductivity": "导热系数", "一维瞬态对流模型参数估计": "一维瞬态对流参数估计"}
PROJECT = {"一维瞬态对流参数估计", "瞬态对流法", "transient convection", "d+h+c"}


@lru_cache(maxsize=8192)
def similarity(left, right):
    if left == right:
        return 1.0
    vectors = TfidfVectorizer(analyzer="char", ngram_range=(1, 3)).fit_transform([left, right])
    return float(vectors[0].multiply(vectors[1]).sum())


def standard_align(concept: str) -> dict:
    """Return pending standard comparisons using the offline catalogue."""
    return match_standards(concept)


def match_standards(concept, embedding_similarity=None):
    """Find reviewable method/quantity comparisons with verified catalogue headings."""
    value = concept.strip().casefold()
    if not 1 <= len(value) <= 160:
        raise ValueError("Concept length must be 1..160")
    value = ALIASES.get(value, value)
    matches = []
    for std in STANDARDS:
        terms = [std["id"], std["name"], *std["terms"], *std["sections"]]
        semantic_terms = set()

        def score(term):
            label = term.casefold()
            label = ALIASES.get(label, label)
            lexical = similarity(value, label)
            if embedding_similarity is None:
                return lexical
            try:
                semantic = float(embedding_similarity(value, label))
            except (ImportError, FileNotFoundError):
                return lexical
            semantic_terms.add(term)
            return max(lexical, min(1.0, semantic))
        scores = {term: score(term) for term in terms}
        term = max(terms, key=scores.get)
        project = value in PROJECT
        if not project and scores[term] < .46:
            continue
        quantity = value in {"导热系数", "热扩散系数"} or term.casefold() in {"导热系数", "热扩散系数", "thermal diffusivity"}
        relation = "comparable_quantity" if quantity and not project else "different_method_from"
        matches.append({"concept": concept, "subject": concept if relation == "comparable_quantity" else "一维瞬态对流参数估计", "standard": std["id"], "title": std["name"], "relation": relation,
                        "term": term if not project else std["name"], "score": 1.0 if project else round(scores[term], 5),
                        "match_method": "curated_method_comparison" if project else ("alias_char_tfidf_embedding" if term in semantic_terms else "alias_char_tfidf"),
                        "section": next((sec for sec in std["sections"] if term in sec), None),
                        "clause_number": "附录A" if term in ("装置校准", "附录A 装置校准") else None,
                        "difference": "项目采用一维瞬态对流参数估计；标准方法为"+std["name"]+"。",
                        "source_id": identifier(std), "locator": {"file": "competition_agent/ontology/standards.yaml", "id": std["id"]},
                        "quote": std["name"], "status": "pending"})
    return {"concept": concept, "matches": sorted(matches, key=lambda m: -m["score"])}


def space():
    return {"version": 0, "values": {}, "history": []}


def commit(space, updates, reason):
    changes = {}
    for key, new in updates.items():
        old = space["values"].get(key)
        if old != new:
            changes[key] = {"old": old, "new": new}
    if changes:
        for key, delta in changes.items():
            if delta["new"] is None:
                space["values"].pop(key, None)
            else:
                space["values"][key] = delta["new"]
        space["version"] += 1
        space["history"].append({"version": space["version"], "reason": reason, "diff": changes})
    return sorted(changes)


def writable(state):
    if state["phase"] in FROZEN:
        raise ValueError("Evidence is frozen")


def rollback(lab, case_id, kind, version):
    if kind not in ("graph", "ontology"):
        raise ValueError("unknown version space")
    with lab.case(case_id) as state:
        writable(state)
        branch = state["induction"][kind]
        if not 0 <= version <= branch["version"]:
            raise ValueError("unknown version")
        updates = {}
        for change in reversed(branch["history"][version:]):
            for k, v in change["diff"].items():
                updates[k] = v["old"]
        changed = commit(branch, updates, "rollback to "+str(version))
        if changed:
            lab.event(state, "local_operator", "induction_rollback", {
                      "kind": kind, "target": version, "changed": changed})
        return {"version": branch["version"], "changed": changed}


def review(lab, case_id, decisions):
    with lab.case(case_id) as state:
        writable(state)
        draft = state.get("induction")
        if not draft:
            raise ValueError("Run ontology_induce before reviewing")
        if not isinstance(decisions, dict) or not 1 <= len(decisions) <= 100:
            raise ValueError("Need 1..100 decisions")
        branch = draft["ontology"]
        graph = draft["graph"]
        updates = {}
        graph_updates = {}
        for key, decision in decisions.items():
            if decision not in ("accepted", "rejected"):
                raise ValueError("invalid review decision")
            if key in branch["values"] and not key.startswith("source:"):
                updates[key] = dict(branch["values"][key], status=decision)
                continue
            entity = next((k for k, v in graph["values"].items() if k.startswith("entity:") and
                           any(c["id"] == key and c["tier"] == "queue" for c in v["candidates"])), None)
            if entity is None:
                raise ValueError("unknown review item")
            if entity not in graph_updates:
                graph_updates[entity] = deepcopy(graph["values"][entity])
            for candidate in graph_updates[entity]["candidates"]:
                if candidate["id"] == key:
                    candidate["status"] = decision
        oc = commit(branch, updates, "operator review")
        gc = commit(graph, graph_updates, "operator review")
        if oc or gc:
            lab.event(state, "local_operator", "induction_review", {"decisions": decisions})
        return {"ontology_version": branch["version"], "graph_version": graph["version"], "changed": oc+gc}


def source_rows(state, source):
    doc = state["documents"][source]
    record = state["input_preparation"]["records"][source]
    if record["status"] != "ready" or record["duplicate_of"]:
        return []
    mm = state.get("multimodal_assets", {}).get(source)
    rows = []
    if mm:
        for b in mm["asset"]["blocks"]:
            rows.append({"text": b["text"], "locator": b["locator"], "needs_review": b["needs_review"],
                        "skip_phrases": b["type"] in ("table", "code_literal"), "structured": []})
            if b["type"] == "table" and "sheet" in b["locator"]:
                sheet = b["locator"]["sheet"]
                for cell in b["headers"]:
                    if not isinstance(cell, dict):
                        continue
                    name = str(cell["value"])
                    if re.fullmatch("[xy][np]", sheet) and re.fullmatch(sheet+"[1-5]", name):
                        rows.append({"text": name, "locator": {"sheet": sheet, "cell": cell["cell"]},
                                     "structured": [(sheet, "has_test", name, name)]})
    else:
        rows = [{"text": line, "locator": {"line": i}, "structured": []}
                for i, line in enumerate(doc["text"].splitlines(), 1) if line.strip()]
    for c in [*record["candidates"], *literal_candidates({source: mm} if mm else {})]:
        f = c["fact"]
        pred = {"has_diameter_m": "has_diameter", "has_length_m": "has_length",
                "has_density_kg_m3": "has_density"}.get(f["predicate"], f["predicate"])
        obj = f["object"]+" "+DEFAULT[pred] if pred != f["predicate"] else f["object"]
        rows.append({"text": f["quote"], "locator": c["citation"], "structured": [(f["subject"], pred, obj, f["quote"])],
                     "needs_review": bool(mm and any(b["needs_review"] for b in mm["asset"]["blocks"]))})
    for r in rows:
        r.update(source_id=source, path=doc["title"])
    return rows


def induce_case(lab, case_id):
    with lab.case(case_id) as state:
        writable(state)
        draft = state.setdefault("induction", {"graph": space(), "ontology": space()})
        graph, ontology = draft["graph"], draft["ontology"]
        gu, ou = {}, {}
        # Canonical duplicate selection may change when a new source is ingested.
        records = state.get("input_preparation", {}).get("records", {})
        active = {s for s, r in records.items() if r["status"] == "ready" and not r["duplicate_of"]}

        def affected(branch):
            seen = {k[7:] for k in branch["values"] if k.startswith("source:")}
            return sorted(active-seen), sorted(seen-active)
        gnew, gone = affected(graph)
        onew, oout = affected(ontology)
        touched = set()
        for src in gone:
            for subject in graph["values"]["source:"+src]:
                touched.add(subject)
            gu["source:"+src] = None
        for src in oout:
            ou["source:"+src] = None
        bysubject = defaultdict(dict)
        for src in sorted(set(gnew+onew)):
            rows = source_rows(state, src)
            if src in gnew:
                groups = defaultdict(list)
                for row in rows:
                    for engine, fact in [*(("E1", f) for f in row["structured"]), *(("E2", f) for f in extract_e2(row["text"]))]:
                        s, p, o, q = fact
                        if p not in RULES:
                            continue
                        subject = normalize(s, p, o)[0][0]
                        groups[subject].append(
                            dict(row, structured=[fact], structured_engine=engine, preextracted=True))
                for subject, items in groups.items():
                    bysubject[subject][src] = items
                gu["source:"+src] = sorted(groups)
                touched.update(groups)
            if src in onew:
                phrases = {}
                for row in rows:
                    if row["structured"] or row.get("skip_phrases"):
                        continue
                    for match in TERM.finditer(row["text"]):
                        term = match.group().strip().casefold()
                        if len(term) < 2 or term in KNOWN:
                            continue
                        item = phrases.setdefault(
                            term, {"count": 0, "source_id": src, "quote": match.group(), "locator": row["locator"]})
                        item["count"] += 1
                    # Match catalogue terminology inside longer prose, retaining the exact surface form.
                    labels = {t for std in STANDARDS for t in (
                        *std["terms"], *std["sections"])} | set(ALIASES) | PROJECT
                    for label in labels:
                        for hit in re.finditer(re.escape(label), row["text"], re.I):
                            term = label.casefold()
                            if any(m.group().casefold() == term for m in TERM.finditer(row["text"])):
                                continue
                            item = phrases.setdefault(
                                term, {"count": 0, "source_id": src, "quote": hit.group(), "locator": row["locator"]})
                            item["count"] += 1
                relations = {}
                for row in rows:
                    if row["structured"] or row.get("skip_phrases"):
                        continue
                    for hit in re.finditer(r"(?m)^\s*\S+\s+([a-z]+(?:_[a-z]+)+)\s+\S+[^\n]*", row["text"]):
                        relations[hit[1], str(row["locator"])] = (hit[0].strip(), row["locator"])
                for c in records[src]["candidates"]:
                    term = c["fact"]["predicate"]
                    relations[term, str(c["citation"])] = (c["fact"]["quote"], c["citation"])
                for (term, _), (quote, locator) in relations.items():
                    if term not in RULES:
                        item = phrases.setdefault(
                            "relation:"+term, {"count": 0, "source_id": src, "quote": quote, "locator": locator})
                        item["count"] += 1
                ou["source:"+src] = phrases
        for subject in sorted(touched):
            key = "entity:"+subject
            sources = dict(graph["values"].get(key, {}).get("sources", {}))
            for src in gone:
                sources.pop(src, None)
            sources.update(bysubject[subject])
            if not sources:
                gu[key] = None
                continue
            run = aggregate([r for docs in sources.values() for r in docs], [])
            gu[key] = {"sources": sources, "candidates": [
                dict(c, tier=t) for t in ("draft", "queue", "discard") for c in run[t]]}
        if ou:
            exact_terms = {t.casefold() for std in STANDARDS for t in (
                std["id"], std["name"], *std["terms"], *std["sections"])} | set(ALIASES) | PROJECT
            docs = {k: v for k, v in {**ontology["values"], **
                                      ou}.items() if k.startswith("source:") and v is not None}
            terms = {t for src in onew for t in ou["source:"+src]}
            terms.update(t for src in oout for t in ontology["values"]["source:"+src])
            # Fixed per-pair TF-IDF keeps unrelated similarities stable across updates.
            for term in sorted(terms):
                key = "term:"+term
                ev = [v[term] for v in docs.values() if term in v]
                old = ontology["values"].get(key, {})
                # Standard comparisons are eligible on the first exact catalogue mention.
                frequent = sum(e["count"] for e in ev) >= 2
                matches = standard_align(term[:160])["matches"] if ev and (
                    frequent or term in exact_terms) and not term.startswith("relation:") else []
                for m in matches:
                    aid = "align:"+identifier([term, m["standard"]])
                    prior = ontology["values"].get(aid, {})
                    ou[aid] = dict(m, evidence=ev, status=prior.get("status", "pending")
                                   if prior.get("evidence") == ev else "pending")
                if not frequent:
                    if old:
                        ou[key] = None
                    continue
                ou[key] = {"term": term, "kind": "relation" if term.startswith("relation:") else "concept",
                           "cluster": old.get("cluster"), "evidence": ev, "status": old.get("status", "pending") if old.get("evidence") == ev else "pending"}
            pool = {k: v for k, v in {**ontology["values"], **ou}.items() if k.startswith("term:") and v}
            touched_clusters = {ontology["values"][k]["cluster"]
                                for k in ou if k.startswith("term:") and k in ontology["values"]}
            seeds = {k for k in pool if k in ou or pool[k]["cluster"] in touched_clusters}
            for key in sorted(seeds):
                if key not in seeds:
                    continue
                component = {key}
                stack = [key]
                while stack:
                    cur = pool[stack.pop()]
                    for k, v in pool.items():
                        if k in component or v["kind"] != cur["kind"]:
                            continue
                        if similarity(cur["term"].removeprefix("relation:"), v["term"].removeprefix("relation:")) >= .55:
                            component.add(k)
                            stack.append(k)
                cluster = identifier(sorted(component))[:16]
                for k in component:
                    if pool[k]["cluster"] != cluster:
                        ou[k] = dict(pool[k], cluster=cluster)
                seeds.difference_update(component)
            for k, v in ontology["values"].items():
                if k.startswith("align:") and v["concept"] in terms and k not in ou:
                    ou[k] = None
        gc = commit(graph, gu, "incremental induction")
        oc = commit(ontology, ou, "incremental induction")
        result = {"graph_version": graph["version"], "ontology_version": ontology["version"], "affected_entities": sorted(touched),
                  "graph_diff": gc, "ontology_diff": oc}
        if gc or oc:
            lab.event(state, "graph_agent", "ontology_induce", result)
        return result


def queue(lab, case_id, limit):
    if type(limit) != int or not 1 <= limit <= 100:
        raise ValueError("limit must be 1..100")
    with lab.case(case_id, readonly=True) as state:
        d = state.get("induction", {})
        items = [dict(v, id=k) for k, v in d.get("ontology", space())["values"].items()
                 if not k.startswith("source:") and v["status"] == "pending"]
        for k, v in d.get("graph", space())["values"].items():
            if k.startswith("entity:"):
                items.extend(c for c in v["candidates"] if c["tier"] ==
                             "queue" and c.get("status", "pending") == "pending")
        items.sort(key=lambda x: (-x.get("priority", 0), x["id"]))
        return {"total": len(items), "items": items[:limit]}


def alignments(state):
    return [v for k, v in state.get("induction", {}).get("ontology", space())["values"].items() if k.startswith("align:")]
