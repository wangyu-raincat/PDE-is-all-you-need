"""Regression checks for split isolation and the published Q0 evidence tables."""
import json
import tempfile
import unittest
from unittest.mock import patch

from competition_agent.decision import QUESTIONS, decide
from competition_agent.decision.equation import l1_control
from competition_agent.scenario_am.heat_tools import ROOT, heat_identify
from competition_agent.service import Lab
from pde_discovery.real_experiment import load_measurements


class EquationDecisionTests(unittest.TestCase):
    def test_l1_only_opens_training_and_development(self):
        directory = ROOT / "pde_discovery/data/additive_heat/xn"
        manifest = json.loads((directory / "manifest.json").read_bytes())
        files = []

        def guarded(folder, record, schema):
            self.assertIn(record["repeat"], (1, 2))
            files.append(record["repeat"])
            return load_measurements(folder, record, schema)

        with patch("competition_agent.decision.equation.load_measurements", side_effect=guarded):
            result = l1_control(directory, manifest)
        self.assertEqual(files, [1, 2])
        self.assertEqual(result["library"], ["1", "theta", "theta^2", "theta_x", "theta_xx"])
        selected = result["selected"]
        self.assertIn([term["name"] for term in selected["terms"]], result["near_supports"])
        self.assertTrue(any(row["eligible"] and row["id"] == selected["candidate_id"] for row in result["candidates"]))

    def test_prior_table_reproduces_development_choice(self):
        result = heat_identify("yn")
        rows = result["candidate_table"]
        self.assertEqual(len(rows), 4)
        eligible = [r for r in rows if r["dev_rmse_K"] is not None]
        best = min(r["dev_rmse_K"] for r in eligible)
        near = [r for r in eligible if r["dev_rmse_K"] <= best * 1.05 + 1e-12]
        self.assertEqual(near, [r for r in rows if r["near_optimal"]])
        expected = min(near, key=lambda r: (r["support_size"], r["dev_rmse_K"], r["name"]))
        self.assertEqual([r for r in rows if r["selected"]], [expected])
        self.assertEqual(expected["name"], result["selected"]["model_name"])

    def test_missing_case_evidence_does_not_start_modeling(self):
        with tempfile.TemporaryDirectory() as directory:
            lab = Lab(directory)
            case = lab.create_demo()["case_id"]
            with patch("competition_agent.decision.equation.equation_evidence", side_effect=AssertionError):
                self.assertEqual(decide(lab, case, QUESTIONS[0])["confidence_level"], "证据不足")
