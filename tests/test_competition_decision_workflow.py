import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from competition_agent.decision import decide, downgrade_triggers, get_trace, QUESTIONS, stainless_diagnostics
from competition_agent.decision.retrieval import export_knowledge, fused, typed_paths
from competition_agent.multimodal import project
from competition_agent.scenario_am.heat_tools import heat_repeatability
from competition_agent.service import Lab


class DecisionWorkflowTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.lab = Lab(self.root)
        self.case = self.lab.create_demo()["case_id"]

    def test_descriptive_five_fixed_fits(self):
        result = heat_repeatability("xn")
        self.assertEqual([r["repeat"] for r in result["fits"]],list(range(1,6)))
        self.assertEqual(result["scope"],"描述性复测离散度")
        for key,stats in result["summary"].items():
            values = [r["parameters"][key] for r in result["fits"]]
            self.assertAlmostEqual(stats["mean"],np.mean(values))
            self.assertAlmostEqual(stats["sample_std"],np.std(values,ddof=1))
            self.assertAlmostEqual(stats["range"],np.ptp(values))
        correlation = result["D_h_correlation"]
        self.assertEqual(correlation["n"], 5)
        paired = [[row["parameters"][key] for row in result["fits"]] for key in ("D","h")]
        self.assertAlmostEqual(correlation["r"], np.corrcoef(paired)[0,1])
        with self.assertRaises(ValueError):
            heat_repeatability("50ss")

    def test_downgrade_reasons_operate_independently(self):
        baseline = {"D": {"mean": 1, "range": 1}, "h": {"mean": 1, "range": 1}}
        def triggers(d, h, diameter):
            other = {"D": {"mean": d, "range": 1}, "h": {"mean": h, "range": 1}}
            _, flags = downgrade_triggers(other, baseline, diameter, 5, 2)
            return {key for key,value in flags.items() if value}
        self.assertEqual(triggers(2, 1, 5), {"within_observed_sensitivity"})
        self.assertEqual(triggers(3, 3, 5), {"large_same_direction_changes"})
        self.assertEqual(triggers(3, 1, 6), {"different_diameters"})
        self.assertFalse(triggers(3, 0.5, 5))
        report = stainless_diagnostics()
        self.assertEqual(report["profile_ratio"], {"min": 2.0, "max": 3.968})
        self.assertEqual(report["deletion_ratio"], {"min": 1.381, "max": 4.819})

    def test_typed_paths_rrf_and_export(self):
        inputs = [{"title":"自制表头","kind":"text","text":"probe measures temperature"},
                  {"title":"自制单位","kind":"text","text":"temperature uses_unit kelvin"}]
        sources = self.lab.ingest(self.case,inputs)["sources"]
        facts = []
        for doc,source in zip(inputs,sources):
            s,p,o = doc["text"].split()
            facts.append(dict(subject=s,predicate=p,object=o,quote=doc["text"],source_id=source["source_id"]))
        proposal = self.lab.graph_propose(self.case,facts)
        self.lab.graph_apply(self.case,proposal["proposal_id"])
        pattern = ["Sensor","measures","ObservationField","uses_unit","Unit"]
        with self.lab.case(self.case,readonly=True) as state:
            paths = typed_paths(state,pattern,"probe",target="kelvin")
            self.assertEqual(len(paths[0]),2)
            with self.assertRaises(ValueError):
                typed_paths(state,["Specimen","measures","ObservationField"])
            hits = fused(state,"temperature 单位",paths)
            self.assertEqual(set(hits[0]["ranks"]),{"bm25","vector","graph"})
            self.assertAlmostEqual(hits[0]["rrf"],sum(1/(60+n) for n in hits[0]["ranks"].values()))
            export_knowledge(state,self.root/"index")
            manifest=json.loads((self.root/"index/manifest.json").read_bytes())
            self.assertEqual(len(manifest["files"]),2)
            with np.load(self.root/"index/local-index.npz",allow_pickle=False) as index:
                self.assertEqual(len(index["embeddings"]),len(manifest["files"]))

    def test_conflict_refusal_missing_data_and_trace_integrity(self):
        answer = decide(self.lab,self.case,QUESTIONS[1])
        self.assertEqual(get_trace(self.lab,answer["trace_id"])["rule"]["id"],"gap")
        self.lab.ingest(self.case,[{"title":f"自制记录{i}","kind":"text","text":f"xn diameter = {i} mm"} for i in (6,7)])
        answer = decide(self.lab,self.case,QUESTIONS[4])
        trace = get_trace(self.lab,answer["trace_id"])
        self.assertEqual(trace["rule"]["id"],"conflict")
        self.assertEqual(len(trace["reasoning"]["conflicts"]),2)
        answer = decide(self.lab,self.case,QUESTIONS[5])
        trace = get_trace(self.lab,answer["trace_id"])
        self.assertEqual(trace["rule"]["id"],"refuse")
        self.assertFalse(trace["reasoning"]["numerical_tools"])
        with self.lab.case(self.case) as state:
            state["events"][-1]["details"]["trace"]["question"]="changed"
        with self.assertRaisesRegex(ValueError,"chain"):
            get_trace(self.lab,answer["trace_id"])

    def test_pdf_projection_preserves_raw_block_and_offsets(self):
        asset={"modality":"pdf","asset_id":"a","path":"paper.pdf","blocks":[
            {"text":"a\x01b","block_id":"b","locator":{"page":1},"needs_review":False}]}
        doc,spans=project(asset)
        self.assertEqual(doc["text"],"a b")
        self.assertEqual(spans[0]["end_char"],3)
        self.assertEqual(asset["blocks"][0]["text"],"a\x01b")
