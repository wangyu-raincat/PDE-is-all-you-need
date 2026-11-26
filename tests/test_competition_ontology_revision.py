import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from competition_agent.ontology.normalization import normalize
from competition_agent.ontology.revision import aggregate
from scripts.competition_annotate import missing

class RevisionTests(unittest.TestCase):
    def test_si_alias_precision(self):
        a,m=normalize("x 方向非较多孔杆","diameter","6.2060 mm")
        b,_=normalize("xn rod","has_diameter","0.006206","m")
        self.assertEqual(a,b)
        self.assertEqual(m["significant_digits"],5)
        self.assertNotEqual(normalize("xn","diameter","0.006206")[0],b)

    def test_headers_grouped_votes(self):
        rows=[dict(text="TC1",source_id=str(i),path="sample.csv",locator={"line":1},
                   structured=[("xn"+str(i),"records_field","TC1","TC1")]) for i in (1,2)]
        run=aggregate(rows,[])
        item=next(r for k in ("draft","queue") for r in run[k])
        self.assertEqual(item["triple"],("xn","records_field","TC1"))
        self.assertEqual(item["features"]["independent_files"],2)
        self.assertEqual({e["raw"][0] for e in item["evidence"]},{"xn1","xn2"})

    def test_exhaustive_annotation_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"subset.json"
            path.write_text(json.dumps([{"text":"test A","locator":{"page":3},"source_id":"s"}]))
            with patch("builtins.print"),patch("builtins.input",side_effect=["a","p","b","test A",""]):
                missing(path,path.parent,"reviewer")
            with patch("builtins.input",side_effect=AssertionError): missing(path,path.parent,"reviewer")
            entries=[json.loads(l) for l in (path.parent/"reviewer.missing.jsonl").read_text().splitlines()]
            self.assertEqual([e["action"] for e in entries],["fact","done"])
    def test_archived_packet_meets_sampling_constraints(self):
        from competition_agent.ontology.revision import ARCH
        from collections import Counter
        run=json.loads((ARCH/"induction.json").read_text(encoding="utf8"))
        items=json.loads((ARCH/"packet.json").read_text(encoding="utf8"))["items"]
        picked={item["id"] for item in items}
        self.assertEqual(len(items),100)
        self.assertEqual(len(picked),100)
        self.assertLessEqual(max(Counter(i["triple"][1] for i in items).values()),15)
        self.assertGreaterEqual(sum(i["strata"]["e3_only"] for i in items),10)
        pool=[r for tier in ("draft","queue","discard") for r in run[tier]]
        self.assertTrue(all(r["id"] in picked for r in pool if r["conflict"]))
