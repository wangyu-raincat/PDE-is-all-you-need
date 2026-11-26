"""Rank case evidence blocks; graph edges retain their T04 review provenance."""
from collections import defaultdict
from functools import lru_cache
import csv
import re
import hashlib
import json

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from competition_agent.evidence_quality import retrieve, tokens
from competition_agent.ontology.revision import RULES
from competition_agent.scenario_am.catalog import ROOT, evidence

ALIASES = {"孔隙": "porous porosity", "热扩散": "thermal diffusivity diffusion",
           "直径": "diameter", "工艺": "process condition", "风速": "wind speed",
           "闪光法": "flash method", "复测": "repeat test", "测点": "thermocouple"}


def expand(text):
    return text + " " + " ".join(v for k, v in ALIASES.items() if k in text)


def blocks(state):
    result = {}
    records = state.get("input_preparation", {}).get("records", {})
    for source, doc in state["documents"].items():
        if records[source]["status"] != "ready" or records[source]["duplicate_of"]:
            continue
        asset = state.get("multimodal_assets", {}).get(source, {}).get("asset")
        parts = asset["blocks"] if asset else [
            {"block_id": f"{source}:{i}", "text": line, "locator": {"line": i}}
            for i, line in enumerate(doc["text"].splitlines(), 1)]
        for part in parts:
            if part["text"].strip():
                result[part["block_id"]] = dict(source_id=source, block_id=part["block_id"],
                    title=doc["title"], modality=asset["modality"] if asset else doc["kind"],
                    quote=part["text"], locator=part["locator"], needs_review=part.get("needs_review",False))
    for test, ev in measurement_headers(state):
        result[ev["block_id"]] = dict(ev, title=test+".csv", modality="csv", needs_review=False)
    return result


@lru_cache(maxsize=4)
def vector_index(texts):
    """Small local Chinese-bigram/English LSA embedding, trained only on this case."""
    vec = TfidfVectorizer(tokenizer=tokens, token_pattern=None)
    try:
        matrix = vec.fit_transform(map(expand,texts))
    except ValueError:
        return None
    rank = min(64, min(matrix.shape) - 1)
    svd = TruncatedSVD(rank, random_state=42) if rank >= 2 else None
    vectors = normalize(svd.fit_transform(matrix)) if svd else matrix
    return vec, svd, vectors


def measurement_headers(state):
    seen = set()
    graph = state.get("induction", {}).get("graph", {}).get("values", {})
    for entity in graph.values():
        if not isinstance(entity, dict):
            continue
        for edge in entity.get("candidates", []):
            group, pred, test = edge["triple"]
            if (pred != "has_test" or edge["tier"] == "discard" or edge.get("status") == "rejected"
                    or test in seen or not re.fullmatch(r"[xy][np][1-5]", test) or test[:-1] != group):
                continue
            seen.add(test)
            path = ROOT / "pde_discovery/data/additive_heat/raw" / (test + ".csv")
            with path.open(encoding="utf8") as handle:
                header = handle.readline().rstrip("\r\n")
            ev = evidence(path, header)
            yield test, dict(ev, block_id=ev["source_id"]+":header")


def graph_edges(state):
    # Extend T04's has_test edges with measured columns of the named acquisition.
    for test, ev in measurement_headers(state):
        for field in next(csv.reader([ev["quote"]])):
            yield {"triple":[test,"records_field",field],"status":"dataset_header",
                   "conflict":False,"evidence":[ev]}
    for fact in state["facts"]:
        yield {"triple": [fact[k] for k in ("subject","predicate","object")],
               "status": "applied", "conflict": False, "evidence": [dict(fact,locator=fact.get("citation",{}))]}
    graph = state.get("induction", {}).get("graph", {}).get("values", {})
    for key, entity in graph.items():
        if key.startswith("entity:"):
            for item in entity["candidates"]:
                if item["tier"] != "discard" and item.get("status") != "rejected":
                    yield {**item, "status":item.get("status",item["tier"])}


def typed_paths(state, pattern, start="", limit=12, target=""):
    """Alternating type/predicate/type lists; traversal joins exact entity identifiers."""
    if not 1 <= limit <= 50 or len(pattern) not in (3,5,7,9):
        raise ValueError("Use one to four typed edges and limit 1..50")
    for a,p,b in zip(pattern[::2],pattern[1::2],pattern[2::2]):
        rule = RULES.get(p)
        if not rule or (rule["domain"],rule["range"]) != (a,b):
            raise ValueError("Invalid ontology signature: "+str(p))
    by_pred = defaultdict(list)
    for edge in graph_edges(state):
        by_pred[edge["triple"][1]].append(edge)
    found = []
    def walk(path, node, step):
        if len(found) >= limit:
            return
        if step == len(pattern):
            if not target or node == target:
                found.append(path)
            return
        for edge in by_pred[pattern[step]]:
            s,_,o = edge["triple"]
            if (not node or node == s) and edge not in path:
                walk(path+[edge],o,step+2)
                if len(found) >= limit:
                    break
    walk([],start,1)
    return found


def fused(state, question, paths, limit=12):
    catalog = blocks(state)
    if not catalog:
        return []
    docs = {k:{"title":v["title"], "text":v["quote"]} for k,v in catalog.items()}
    lexical = retrieve(docs,expand(question),len(docs)*2)["hits"]
    ids = list(docs)
    index = vector_index(tuple(doc["text"] for doc in docs.values()))
    dense = []
    if index:
        vec,svd,matrix = index
        query = vec.transform([expand(question)])
        scores = matrix @ (normalize(svd.transform(query)).T if svd else query.T)
        scores = np.asarray(scores.toarray() if hasattr(scores,"toarray") else scores).ravel()
        dense = [ids[i] for i in np.argsort(-scores,kind="stable") if scores[i] > 1e-8]
    graph = []
    for path in paths:
        for edge in path:
            for ev in edge["evidence"]:
                graph.extend(k for k,v in catalog.items() if v["source_id"]==ev["source_id"]
                             and ev["quote"] in v["quote"])
    rankings = defaultdict(dict)
    for channel, order in [("bm25",[r["source_id"] for r in lexical]),("vector",dense),("graph",graph)]:
        for rank,k in enumerate(dict.fromkeys(order),1):
            rankings[k][channel] = rank
    rows = [dict(catalog[k], ranks=ranks, rrf=sum(1/(60+n) for n in ranks.values()))
            for k,ranks in rankings.items()]
    return sorted(rows,key=lambda r:(-r["rrf"],r["block_id"]))[:limit]


def export_knowledge(state, directory):
    directory.mkdir(parents=True,exist_ok=True)
    manifest = []
    catalog = blocks(state)
    for block in catalog.values():
        filename = hashlib.sha256(block["block_id"].encode()).hexdigest()+".txt"
        (directory/filename).write_text(block["quote"],encoding="utf8")
        manifest.append({**{k:v for k,v in block.items() if k!="quote"}, "file":filename})
    (directory/"manifest.json").write_text(json.dumps({"chunking":"每个 UTF-8 文件一个来源块；按文件导入，保留 block_id 和 locator。超长块按平台长度上限分片并保留父块定位。",
        "files":manifest},ensure_ascii=False,indent=2),encoding="utf8")

    index = vector_index(tuple(row["quote"] for row in catalog.values())) if catalog else None
    if index:
        vec,svd,vectors = index
        dense = vectors if svd else vectors.toarray()
        np.savez_compressed(directory/"local-index.npz", embeddings=dense,
                            block_ids=list(catalog), idf=vec.idf_,
                            vocabulary=vec.get_feature_names_out().astype(str),
                            projection=svd.components_ if svd else np.empty((0,dense.shape[1])))
