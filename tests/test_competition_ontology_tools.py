import tempfile
import unittest
from pathlib import Path

from competition_agent.ontology.tools import induce_case, queue, rollback, review, standard_align
from competition_agent.service import Lab


class CaseInductionTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.lab = Lab(Path(temp.name))
        self.case = self.lab.create_demo()["case_id"]

    def ingest(self, title, text):
        return self.lab.ingest(self.case, [{"title": title, "text": text, "kind": "text"}])

    def test_incremental_clusters_graph_versions_and_rollback(self):
        self.ingest("自制样例 A", "fiber probe\nfiber probe\nfibre probe\nfibre probe\nxn has_build_direction x")
        first = induce_case(self.lab, self.case)
        self.assertEqual(first["affected_entities"], ["xn"])
        candidates = queue(self.lab, self.case, 100)["items"]
        clusters = {c["term"]: c["cluster"] for c in candidates if c.get("kind") == "concept"}
        self.assertEqual(clusters["fiber probe"], clusters["fibre probe"])
        self.assertEqual(induce_case(self.lab, self.case)["graph_diff"], [])
        self.ingest("自制样例 B", "xp has_build_direction x")
        second = induce_case(self.lab, self.case)
        self.assertEqual(second["affected_entities"], ["xp"])
        old = rollback(self.lab, self.case, "graph", first["graph_version"])
        self.assertGreater(old["version"], second["graph_version"])
        with self.lab.case(self.case, readonly=True) as state:
            self.assertIn("entity:xn", state["induction"]["graph"]["values"])
            self.assertNotIn("entity:xp", state["induction"]["graph"]["values"])
            self.assertEqual(state["induction"]["ontology"]["version"], second["ontology_version"])
        self.assertEqual(induce_case(self.lab, self.case)["affected_entities"], ["xp"])

    def test_alignment_local_review_and_frozen_guards(self):
        self.ingest("自制样例", "对照标准的装置校准")
        first = induce_case(self.lab, self.case)
        match = next(c for c in queue(self.lab, self.case, 100)["items"] if c.get("clause_number") == "附录A")
        review(self.lab, self.case, {match["id"]: "accepted"})
        self.assertIn("accepted", [a["status"] for a in self.lab.evidence(self.case)["alignment"]])
        rollback(self.lab, self.case, "ontology", first["ontology_version"])
        self.assertIn(match["id"], [c["id"] for c in queue(self.lab, self.case, 100)["items"]])
        with self.lab.case(self.case) as state:
            state["phase"] = "frozen"
        for action in (lambda: induce_case(self.lab, self.case), lambda: rollback(self.lab, self.case, "graph", 0),
                       lambda: review(self.lab, self.case, {match["id"]: "accepted"})):
            with self.assertRaisesRegex(ValueError, "frozen"):
                action()

    def test_repeated_unknown_relation(self):
        self.ingest("自制关系样例", "probe1 senses_heat xn\nprobe2 senses_heat xp")
        induce_case(self.lab, self.case)
        relations = [item for item in queue(self.lab, self.case, 100)["items"]
                     if item.get("kind") == "relation"]
        self.assertIn("relation:senses_heat", [item["term"] for item in relations])

    def test_method_and_quantity_mappings(self):
        self.assertEqual({m["relation"] for m in standard_align(
            "瞬态对流法")["matches"]}, {"different_method_from"})
        self.assertEqual({m["relation"] for m in standard_align("热扩散率")["matches"]}, {"comparable_quantity"})
        self.assertEqual(standard_align("unrelated zebra")["matches"], [])
