import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from competition_agent.ontology import fit_weights, induce
from scripts.competition_annotate import annotate, prepare


def passage(text,source="paper",structured=(),**flags):
    return dict(path=source,source_id=source,text=text,locator={"line":1},structured=structured,**flags)


class OntologyTests(unittest.TestCase):
    def test_corroboration_and_conflicts(self):
        q="xn has_build_direction x"
        docs=[passage(q,structured=[("xn","has_build_direction","x",q)]),passage(q,"table")]
        self.assertEqual(len(induce(docs,use_e3=False)["draft"]),1)
        run=induce(docs+[passage("xn has_build_direction y","card",needs_review=True)],use_e3=False)
        self.assertEqual(len(run["queue"]),2)
        self.assertTrue(all(r["conflict"] for r in run["queue"]))

    def test_quote_dimension_and_types(self):
        text="xn diameter=2 kg/m3"
        fake=Mock()
        def fact(quote,st):
            return dict(index=0,subject="xn",predicate="has_diameter",object="2 kg/m3",
                        subject_type=st,object_type="ModelParameter",quote=quote)
        fake.post.return_value.json.return_value={"choices":[{"message":{"content":json.dumps(
            {"facts":[fact("invented","Specimen"),fact(text,"Sensor")]})}}]}
        with patch.dict(os.environ,{"DEEPSEEK_API_KEY":"fixture"}):
            r=induce([passage(text,llm_input=True)],client=fake)
        self.assertEqual(r["rejected"][0]["reason"],"missing_literal_quote")
        issues=set(i for e in r["discard"][0]["evidence"] for i in e["issues"])
        self.assertEqual(issues,{"domain_range","dimension"})

    def test_missing_key_and_split_gate(self):
        with patch.dict(os.environ,{},clear=True):
            client=Mock()
            self.assertEqual(induce([passage("sample",llm_input=True)],client=client)["e3_status"],"skipped_missing_key")
            client.post.assert_not_called()
        with self.assertRaises(ValueError): fit_weights([{"split":"test"}])

    def test_independent_resume(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); snapshot=root/"run"; packet=root/"packet"
            snapshot.write_text(json.dumps({"draft":[],"discard":[],"queue":[
                {"id":str(i),"triple":[str(i),"records_field","x"],"evidence":[]} for i in range(80)]}))
            prepare(snapshot,packet,80,42)
            for reviewer in ["amy","bob","amy"]:
                with patch("builtins.print"),patch("builtins.input",side_effect=["1","fixture","q"]):
                    annotate(packet,root,reviewer)
            amy=[json.loads(x) for x in (root/"amy.jsonl").read_text().splitlines()]
            bob=json.loads((root/"bob.jsonl").read_text())
            self.assertEqual(amy[0]["id"],bob["id"])
            self.assertNotEqual(amy[1]["id"],amy[0]["id"])
