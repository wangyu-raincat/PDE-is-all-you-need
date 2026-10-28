"""Data isolation and failure-path checks using temporary manufactured records.

No public experimental temperature values are read by this test module.
"""

from copy import deepcopy
from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from fastapi import HTTPException

from pde_discovery import app as web
from pde_discovery import refined_experiment as experiment
from pde_discovery import real_heat_refined as refined
from pde_discovery.real_benchmark import CASES


def fitted_candidate(model_name="D+h"):
    equation_terms = [{"name": "theta_xx", "coefficient": 7e-5}]
    if "h" in model_name:
        equation_terms.append({"name": "theta", "coefficient": -0.06})
    if "c" in model_name:
        equation_terms.append({"name": "1", "coefficient": 0.01})
    return {
        "model_name": model_name,
        "candidate_id": model_name,
        "equation": "MANUFACTURED TEST CANDIDATE: NOT AN EXPERIMENTAL RESULT",
        "terms": equation_terms,
        "support_size": len(equation_terms),
        "fit_success": True,
        "train_score": {"rmse": 1e-7, "relative_error": 1e-7},
        "diagnostics": {"jacobian_rank": len(equation_terms),
                        "parameter_count": len(equation_terms),
                        "scaled_jacobian_condition_number": 2.0},
    }


class RefinedWorkflowTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="pde-refinement-fixtures-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.data_root = self.root / "data"
        self.baseline_root = self.root / "baseline_absent"
        self.t = np.linspace(0.0, 2.0, 41)
        self.manifests = {}
        modes = np.arange(1, 5)
        eigenvalues = -4.0 * np.sin(modes * np.pi / 10.0) ** 2 / 0.01**2
        vectors = np.sin(np.outer(np.arange(1, 5), modes) * np.pi / 5.0)
        for case in CASES:
            directory = self.data_root / case
            directory.mkdir(parents=True)
            entries = {}
            for role, amplitudes in [
                ("training", [4.0, 1.0, 0.7, 0.4]),
                ("development", [3.8, 0.8, 0.6, 0.3]),
                ("final", [3.5, 0.5, 0.4, 0.2]),
            ]:
                interior = (
                    np.exp(np.outer(self.t, 7e-5 * eigenvalues - 0.06)) * amplitudes
                ) @ vectors.T
                u = np.column_stack([np.zeros(len(self.t)), interior, np.zeros(len(self.t))])
                path = directory / f"{role}.csv"
                np.savetxt(path, np.column_stack([self.t, u]), delimiter=",")
                entries[role] = {"file": path.name,
                                 "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            manifest = {
                "experiment": {"title": f"MANUFACTURED WORKFLOW FIXTURE {case}"},
                "spacing_m": 0.01,
                "csv_format": {"delimiter": ",", "skiprows": 0},
                "split_kind": "independent_repeated_acquisitions",
                "splits": entries,
                "provenance_note": "Manufactured test fixture, not real measurements.",
            }
            self.manifests[case] = manifest
            (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        self.fit = patch.object(experiment, "fit_candidates", side_effect=lambda *_: [fitted_candidate()])
        self.half_fit = patch.object(experiment, "fit_candidate", side_effect=lambda *_, **__: fitted_candidate())
        self.plot_case = patch.object(experiment, "_plot_case")
        self.plot_summary = patch.object(experiment, "_plot_summary")
        for patcher in [self.fit, self.half_fit, self.plot_case, self.plot_summary]:
            patcher.start()
            self.addCleanup(patcher.stop)

    def freeze(self, output, cases=CASES):
        with redirect_stdout(io.StringIO()):
            return experiment.freeze_benchmark(
                output, data_root=self.data_root, cases=cases, baseline_root=self.baseline_root
            )

    def evaluate(self, output):
        with redirect_stdout(io.StringIO()):
            return experiment.evaluate_benchmark(
                output, data_root=self.data_root, baseline_root=self.baseline_root
            )

    def test_all_declared_models_are_frozen_before_any_final_numeric_read(self):
        output = self.root / "all_cases"
        original_loader = experiment.load_measurements
        events = []

        def checked_load(data_dir, entry, csv_format):
            events.append((Path(data_dir).name, entry["file"]))
            if entry["file"] == "final.csv":
                for case in CASES:
                    path = output / case / "frozen_model.json"
                    self.assertTrue(path.exists(), f"{case} must be frozen before any final read")
                    self.assertIsNotNone(json.loads(path.read_text())["selected"])
            return original_loader(data_dir, entry, csv_format)

        with patch.object(experiment, "load_measurements", side_effect=checked_load):
            self.freeze(output)
            self.assertEqual(len(events), len(CASES) * 2)
            self.assertFalse(any(name == "final.csv" for _, name in events))
            summary = self.evaluate(output)
        self.assertEqual(len(summary["records"]), len(CASES))
        self.assertEqual(sum(name == "final.csv" for _, name in events), len(CASES))
        self.assertTrue(all(r["validation_status"] == "evaluated" for r in summary["records"]))

    def test_development_selects_simpler_near_optimum_without_refitting_coefficients(self):
        output = self.root / "near_optimal"
        simple, larger = fitted_candidate("D"), fitted_candidate("D+h")
        simple["train_score"]["rmse"] = 100.0
        larger["train_score"]["rmse"] = 0.001
        training_seen = []
        development_seen = []
        original_loader = experiment.load_measurements

        def recorded_load(data_dir, entry, csv_format):
            result = original_loader(data_dir, entry, csv_format)
            if entry["file"] == "training.csv":
                training_seen.append(result)
            elif entry["file"] == "development.csv":
                development_seen.append(result)
            else:
                self.fail("freeze phase cannot load final values")
            return result

        def training_only(t, u, dx):
            self.assertIs(t, training_seen[0][0])
            self.assertIs(u, training_seen[0][1])
            return deepcopy([simple, larger])

        def development_only(equation, t, u, dx):
            self.assertIs(t, development_seen[0][0])
            self.assertIs(u, development_seen[0][1])
            return None, {"rmse": 1.04 if len(equation) == 1 else 1.0}

        with patch.object(experiment, "load_measurements", side_effect=recorded_load), \
             patch.object(experiment, "fit_candidates", side_effect=training_only), \
             patch.object(experiment, "_score", side_effect=development_only):
            self.freeze(output, cases=(CASES[0],))
        frozen = json.loads((output / CASES[0] / "frozen_model.json").read_text())
        self.assertEqual(frozen["selected"]["model_name"], "D")
        self.assertEqual(frozen["selected"]["terms"], simple["terms"])
        self.assertEqual(set(frozen["near_optimal_supports"]), {"D", "D+h"})

    def test_future_final_interiors_change_scores_but_not_frozen_model_or_prediction(self):
        original_loader = experiment.load_measurements
        frozen_models, reports, trajectories = [], [], []
        case = CASES[0]
        for perturb in [False, True]:
            output = self.root / ("altered_final" if perturb else "original_final")
            self.freeze(output, cases=(case,))
            frozen_models.append((output / case / "frozen_model.json").read_bytes())

            def altered_load(data_dir, entry, csv_format):
                t, u = original_loader(data_dir, entry, csv_format)
                if perturb and entry["file"] == "final.csv":
                    u = u.copy()
                    u[1:, 1:-1] += 50.0
                return t, u

            with patch.object(experiment, "load_measurements", side_effect=altered_load):
                self.evaluate(output)
            reports.append(json.loads((output / case / "report.json").read_text()))
            with np.load(output / case / "trajectories.npz") as arrays:
                trajectories.append(arrays["predicted_interior"].copy())
        self.assertEqual(frozen_models[0], frozen_models[1])
        self.assertEqual(reports[0]["discovery"], reports[1]["discovery"])
        np.testing.assert_array_equal(trajectories[0], trajectories[1])
        self.assertGreater(reports[1]["validation"]["rmse_K"], reports[0]["validation"]["rmse_K"] + 49.0)
        for report in reports:
            self.assertTrue(report["prior_seen_test"])
            self.assertFalse(report["new_blind_validation"])
            self.assertFalse(report["discovery"]["unique_physical_equation_established"])

    def test_last_frozen_file_tampering_is_detected_before_any_final_read(self):
        output = self.root / "tampered_model"
        self.freeze(output)
        path = output / CASES[-1] / "frozen_model.json"
        path.write_text(path.read_text() + "\n", encoding="utf-8")
        with patch.object(experiment, "load_measurements") as load:
            with self.assertRaises(ValueError):
                self.evaluate(output)
        load.assert_not_called()

    def test_failed_candidates_do_not_open_final_records(self):
        output = self.root / "failed_fit"
        failed = fitted_candidate()
        failed["fit_success"] = False
        failed["failure_reason"] = "manufactured optimizer failure"
        original_loader = experiment.load_measurements
        reads = []

        def checked_load(data_dir, entry, csv_format):
            reads.append(entry["file"])
            self.assertNotEqual(entry["file"], "final.csv")
            return original_loader(data_dir, entry, csv_format)

        with patch.object(experiment, "fit_candidates", return_value=[failed]), \
             patch.object(experiment, "load_measurements", side_effect=checked_load):
            self.freeze(output, cases=(CASES[0],))
            summary = self.evaluate(output)
        self.assertEqual(reads, ["training.csv", "development.csv"])
        report = json.loads((output / CASES[0] / "report.json").read_text())
        self.assertIsNone(summary["records"][0]["relative_l2"])
        self.assertFalse(report["plot_available"])
        self.assertFalse(report["trajectory_available"])

    def test_unexcited_data_rank_deficiency_cannot_select_a_physical_model(self):
        output = self.root / "unexcited"
        original_loader = experiment.load_measurements
        reads = []

        def no_excitation(data_dir, entry, csv_format):
            reads.append(entry["file"])
            self.assertNotEqual(entry["file"], "final.csv")
            t, u = original_loader(data_dir, entry, csv_format)
            return t, np.zeros_like(u)

        with patch.object(experiment, "fit_candidates", wraps=refined.fit_candidates), \
             patch.object(experiment, "load_measurements", side_effect=no_excitation):
            self.freeze(output, cases=(CASES[0],))
            summary = self.evaluate(output)
        self.assertEqual(reads, ["training.csv", "development.csv"])
        frozen = json.loads((output / CASES[0] / "frozen_model.json").read_text())
        self.assertIsNone(frozen["selected"])
        self.assertEqual(len(frozen["candidates"]), 4)
        for candidate in frozen["candidates"]:
            self.assertLess(candidate["diagnostics"]["jacobian_rank"],
                            candidate["diagnostics"]["parameter_count"])
            self.assertFalse(candidate["eligible"])
        self.assertIsNone(summary["records"][0]["relative_l2"])

    def test_damaged_old_baseline_does_not_hide_current_comparison_result(self):
        output = self.root / "old_report_damaged"
        case = CASES[0]
        old_directory = self.baseline_root / case
        old_directory.mkdir(parents=True)
        (old_directory / "report.json").write_text("not valid JSON", encoding="utf-8")
        self.freeze(output, cases=(case,))
        summary = self.evaluate(output)
        self.assertEqual(summary["records"][0]["validation_status"], "evaluated")
        self.assertIsNone(summary["records"][0]["baseline_relative_l2"])
        self.assertIsNone(summary["records"][0]["improved"])
        report = json.loads((output / case / "report.json").read_text())
        self.assertTrue(report["comparison"]["baseline_unavailable_reason"])

    def test_failed_final_integrity_is_recorded_without_refitting_or_reselection(self):
        output = self.root / "corrupt_final"
        case = CASES[0]
        self.freeze(output, cases=(case,))
        frozen_before = (output / case / "frozen_model.json").read_bytes()
        (output / case / "validation.png").write_bytes(b"obsolete previous-run plot")
        (output / case / "trajectories.npz").write_bytes(b"obsolete previous-run trajectory")
        with (self.data_root / case / "final.csv").open("a", encoding="utf-8") as stream:
            stream.write("corrupted input\n")
        with patch.object(experiment, "fit_candidates") as fit, \
             patch.object(experiment, "fit_candidate") as half_fit:
            self.evaluate(output)
        fit.assert_not_called()
        half_fit.assert_not_called()
        report = json.loads((output / case / "report.json").read_text())
        self.assertEqual(report["validation"]["status"], "failed")
        self.assertFalse(report["plot_available"])
        self.assertFalse(report["trajectory_available"])
        self.assertEqual(frozen_before, (output / case / "frozen_model.json").read_bytes())

    def test_old_plot_cannot_serve_failed_current_report_and_invalid_case_is_rejected(self):
        output = self.root / "stale_image"
        case = CASES[0]
        directory = output / case
        directory.mkdir(parents=True)
        (directory / "validation.png").write_bytes(b"obsolete image")
        with patch.object(web, "REFINED_OUTPUT", output):
            for report in [
                {"plot_available": False, "validation": {"status": "evaluated"}},
                {"plot_available": True, "validation": {"status": "failed"}},
                {"plot_available": False, "validation": {}},
            ]:
                with self.subTest(report=report):
                    (directory / "report.json").write_text(json.dumps(report), encoding="utf-8")
                    with self.assertRaises(HTTPException) as caught:
                        web.refined_experiment_plot(case)
                    self.assertEqual(caught.exception.status_code, 404)
            for function in [web.refined_experiment_plot, web.refined_experiment_report]:
                with self.assertRaises(HTTPException) as caught:
                    function("../invalid_case")
                self.assertEqual(caught.exception.status_code, 422)


if __name__ == "__main__":
    unittest.main()
