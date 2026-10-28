"""Audit the new additive-data workflow using only manufactured temporary CSVs.

All four models must be frozen before any of the twelve held-out records is
parsed. These tests do not read the real additive CSVs or workbook values.
"""

from contextlib import redirect_stdout
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from fastapi import HTTPException

from pde_discovery import additive_experiment as experiment
from pde_discovery import app as web


def manufactured_candidate():
    return {
        "candidate_id": "manufactured-D-h",
        "model_name": "D+h",
        "equation": "MANUFACTURED TEST FIXTURE ONLY",
        "terms": [{"name": "theta_xx", "coefficient": 7e-5},
                  {"name": "theta", "coefficient": -0.06}],
        "support_size": 2,
        "fit_success": True,
        "train_score": {"rmse": 0.0, "relative_error": 0.0},
        "diagnostics": {"jacobian_rank": 2, "parameter_count": 2,
                        "scaled_jacobian_condition_number": 2.0},
    }


class AdditiveWorkflowTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="pde-additive-test-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.data_root = self.root / "manufactured_data"
        self.output = self.root / "output"
        (self.data_root / "raw").mkdir(parents=True)
        self.t = np.linspace(0.0, 2.0, 41)
        self.manifests = {}
        modes = np.arange(1, 5)
        eigenvalues = -4.0 * np.sin(modes * np.pi / 10.0) ** 2 / 0.007**2
        vectors = np.sin(np.outer(np.arange(1, 5), modes) * np.pi / 5.0)
        for group_index, group in enumerate(experiment.GROUPS):
            directory = self.data_root / group
            directory.mkdir()
            entries = []
            for repeat in range(1, 6):
                amplitudes = np.array([4.0 + group_index * 0.2 + repeat * 0.1,
                                       0.8 + repeat * 0.03, 0.6, 0.3])
                interior = (
                    np.exp(np.outer(self.t, 7e-5 * eigenvalues - 0.06)) * amplitudes
                ) @ vectors.T
                u = np.column_stack([np.zeros(len(self.t)), interior, np.zeros(len(self.t))])
                path = self.data_root / "raw" / f"{group}{repeat}.csv"
                np.savetxt(path, np.column_stack([self.t, u]), delimiter=",")
                entries.append({"repeat": repeat, "file": f"../raw/{path.name}",
                                "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
            manifest = {
                "experiment": {"title": f"MANUFACTURED ADDITIVE FIXTURE {group}"},
                "spacing_m": 0.007,
                "csv_format": {"delimiter": ",", "skiprows": 0,
                               "time_column": 0, "temperature_columns": [1, 2, 3, 4, 5, 6]},
                "split_kind": "repeated_acquisitions_same_specimen",
                "splits": {"training": entries[0], "development": entries[1]},
                "holdouts": entries[2:],
                "provenance_note": "Manufactured temporary fixture; not real measurements.",
            }
            self.manifests[group] = manifest
            self.write_manifest(group)
        self.fit_patcher = patch.object(experiment, "fit_candidates", side_effect=lambda *_: [manufactured_candidate()])
        self.fit_mock = self.fit_patcher.start()
        self.addCleanup(self.fit_patcher.stop)
        self.half_patcher = patch.object(experiment, "fit_candidate", side_effect=lambda *_, **__: manufactured_candidate())
        self.half_patcher.start()
        self.addCleanup(self.half_patcher.stop)
        for attribute in ["_plot_case", "_plot_summary"]:
            patcher = patch.object(experiment, attribute)
            patcher.start()
            self.addCleanup(patcher.stop)

    def write_manifest(self, group):
        (self.data_root / group / "manifest.json").write_text(
            json.dumps(self.manifests[group]), encoding="utf-8"
        )

    def freeze(self, output=None):
        with redirect_stdout(io.StringIO()):
            return experiment.freeze_benchmark(
                output or self.output, data_root=self.data_root
            )

    def evaluate(self, output=None):
        with redirect_stdout(io.StringIO()):
            return experiment.evaluate_benchmark(
                output or self.output, data_root=self.data_root
            )

    def report_for(self, summary, group, repeat, output=None):
        record = next(r for r in summary["records"] if r["group"] == group and r["repeat"] == repeat)
        path = (output or self.output) / record["report"]
        return record, json.loads(path.read_text()), path.parent

    def test_all_four_models_freeze_before_any_of_twelve_holdout_reads(self):
        original_loader = experiment.load_measurements
        reads = []
        training = {}

        def checked_loader(data_dir, entry, csv_format):
            group = Path(data_dir).name
            reads.append((group, entry["repeat"]))
            if entry["repeat"] >= 3:
                for declared_group in experiment.GROUPS:
                    frozen_path = self.output / declared_group / "frozen_model.json"
                    self.assertTrue(frozen_path.exists())
                    self.assertIsNotNone(json.loads(frozen_path.read_text())["selected"])
                self.assertTrue((self.output / "freeze_index.json").exists())
            result = original_loader(data_dir, entry, csv_format)
            if entry["repeat"] == 1:
                training[group] = result
            return result

        with patch.object(experiment, "load_measurements", side_effect=checked_loader):
            self.freeze()
            self.assertEqual(len(reads), 8)
            self.assertTrue(all(repeat < 3 for _, repeat in reads))
            self.assertEqual(self.fit_mock.call_count, 4)
            for call, group in zip(self.fit_mock.call_args_list, experiment.GROUPS):
                self.assertIs(call.args[0], training[group][0])
                self.assertIs(call.args[1], training[group][1])
            summary = self.evaluate()
        self.assertEqual(len(summary["records"]), 12)
        self.assertEqual(sum(repeat >= 3 for _, repeat in reads), 12)
        self.assertEqual(len(set(reads)), 20)
        self.assertTrue(all(r["validation_status"] == "evaluated" for r in summary["records"]))
        self.assertTrue(summary["new_project_holdout"])
        self.assertFalse(summary["is_zero_shot_transfer"])
        self.assertEqual(summary["validation_scope"], "new_to_this_project_holdout_records")
        _, report, _ = self.report_for(summary, experiment.GROUPS[0], 3)
        self.assertFalse(report["discovery"]["unique_physical_equation_established"])
        self.assertIn("此前未用于建模", report["validation"]["kind"])
        self.assertIn("统计独立盲验", " ".join(report["limitations"]))

    def test_holdout_future_interiors_change_only_scores_not_models_or_forecasts(self):
        original_loader = experiment.load_measurements
        frozen_models, summaries, forecasts, reports = [], [], [], []
        group = experiment.GROUPS[0]
        for perturb in [False, True]:
            output = self.root / ("perturbed" if perturb else "original")
            self.freeze(output)
            frozen_models.append({g: (output / g / "frozen_model.json").read_bytes()
                                  for g in experiment.GROUPS})

            def altered_load(data_dir, entry, csv_format):
                t, u = original_loader(data_dir, entry, csv_format)
                if perturb and Path(data_dir).name == group and entry["repeat"] == 3:
                    u = u.copy()
                    u[1:, 1:5] += 100.0
                return t, u

            with patch.object(experiment, "load_measurements", side_effect=altered_load), \
                 patch.object(experiment, "fit_candidates") as refit:
                summary = self.evaluate(output)
            refit.assert_not_called()
            record, report, directory = self.report_for(summary, group, 3, output)
            summaries.append(record)
            reports.append(report)
            with np.load(directory / "trajectories.npz") as arrays:
                forecasts.append(arrays["predicted_interior"].copy())
        self.assertEqual(frozen_models[0], frozen_models[1])
        self.assertEqual(reports[0]["discovery"], reports[1]["discovery"])
        np.testing.assert_array_equal(forecasts[0], forecasts[1])
        self.assertGreater(summaries[1]["rmse_K"], summaries[0]["rmse_K"] + 99.0)

    def test_one_corrupt_holdout_preserves_other_eleven_results_without_refitting(self):
        self.freeze()
        frozen_before = {g: (self.output / g / "frozen_model.json").read_bytes()
                         for g in experiment.GROUPS}
        group = experiment.GROUPS[0]
        with (self.data_root / "raw" / f"{group}3.csv").open("a", encoding="utf-8") as stream:
            stream.write("corrupt after freeze\n")
        with patch.object(experiment, "fit_candidates") as refit, \
             patch.object(experiment, "fit_candidate") as half_refit:
            summary = self.evaluate()
        refit.assert_not_called()
        half_refit.assert_not_called()
        self.assertEqual(len(summary["records"]), 12)
        self.assertEqual(sum(r["validation_status"] == "failed" for r in summary["records"]), 1)
        self.assertEqual(sum(r["validation_status"] == "evaluated" for r in summary["records"]), 11)
        record, report, _ = self.report_for(summary, group, 3)
        self.assertIsNone(record["relative_l2"])
        self.assertFalse(report["plot_available"])
        self.assertFalse(report["trajectory_available"])
        for declared_group in experiment.GROUPS:
            self.assertEqual(frozen_before[declared_group],
                             (self.output / declared_group / "frozen_model.json").read_bytes())

    def test_last_frozen_model_tampering_blocks_all_holdout_numeric_reads(self):
        self.freeze()
        path = self.output / experiment.GROUPS[-1] / "frozen_model.json"
        path.write_text(path.read_text() + "\n", encoding="utf-8")
        with patch.object(experiment, "load_measurements") as load:
            with self.assertRaises(ValueError):
                self.evaluate()
        load.assert_not_called()

    def test_last_manifest_change_blocks_all_holdout_numeric_reads(self):
        self.freeze()
        group = experiment.GROUPS[-1]
        self.manifests[group]["spacing_m"] = 0.01
        self.write_manifest(group)
        with patch.object(experiment, "load_measurements") as load:
            with self.assertRaises(ValueError):
                self.evaluate()
        load.assert_not_called()

    def test_duplicate_path_hash_or_repeat_assignment_is_rejected_before_training(self):
        group = experiment.GROUPS[-1]
        original = deepcopy(self.manifests[group])
        invalid_manifests = []
        same_path = deepcopy(original)
        same_path["holdouts"][0]["file"] = f"../{group}/../raw/{group}1.csv"
        invalid_manifests.append(same_path)
        same_hash = deepcopy(original)
        same_hash["holdouts"][0]["sha256"] = original["splits"]["training"]["sha256"]
        invalid_manifests.append(same_hash)
        bad_repeat = deepcopy(original)
        bad_repeat["holdouts"][1]["repeat"] = 3
        invalid_manifests.append(bad_repeat)
        cropped = deepcopy(original)
        cropped["splits"]["training"]["rows"] = [0, 10]
        invalid_manifests.append(cropped)
        for manifest in invalid_manifests:
            with self.subTest(manifest=manifest):
                self.manifests[group] = manifest
                self.write_manifest(group)
                with patch.object(experiment, "load_measurements") as load, \
                     patch.object(experiment, "fit_candidates") as fit:
                    with self.assertRaises(ValueError):
                        self.freeze()
                load.assert_not_called()
                fit.assert_not_called()

    def test_cross_group_record_reuse_cannot_count_as_new_independent_data(self):
        first, last = experiment.GROUPS[0], experiment.GROUPS[-1]
        entry = deepcopy(self.manifests[first]["holdouts"][0])
        self.manifests[last]["holdouts"][0] = entry
        self.write_manifest(last)
        with patch.object(experiment, "load_measurements") as load, \
             patch.object(experiment, "fit_candidates") as fit:
            with self.assertRaises(ValueError):
                self.freeze()
        load.assert_not_called()
        fit.assert_not_called()

    def test_no_eligible_model_keeps_all_declared_holdout_rows_without_parsing_them(self):
        failed = manufactured_candidate()
        failed["fit_success"] = False
        with patch.object(experiment, "fit_candidates", return_value=[failed]):
            self.freeze()
        with patch.object(experiment, "load_measurements") as load:
            summary = self.evaluate()
        load.assert_not_called()
        self.assertEqual(len(summary["records"]), 12)
        self.assertTrue(all(r["relative_l2"] is None for r in summary["records"]))
        self.assertTrue(all(r["validation_status"] == "not_evaluated" for r in summary["records"]))

    def test_stale_plot_and_invalid_case_are_not_served_by_new_routes(self):
        directory = self.output / "xn" / "repeat3"
        directory.mkdir(parents=True)
        (directory / "validation.png").write_bytes(b"obsolete success image")
        with patch.object(web, "ADDITIVE_OUTPUT", self.output):
            for report in [
                {"plot_available": False, "validation": {"status": "evaluated"}},
                {"plot_available": True, "validation": {"status": "failed"}},
            ]:
                with self.subTest(report=report):
                    (directory / "report.json").write_text(json.dumps(report), encoding="utf-8")
                    with self.assertRaises(HTTPException) as caught:
                        web.additive_experiment_plot("xn3")
                    self.assertEqual(caught.exception.status_code, 404)
            for function in [web.additive_experiment_report, web.additive_experiment_plot]:
                with self.assertRaises(HTTPException) as caught:
                    function("../xn3")
                self.assertEqual(caught.exception.status_code, 422)
        diagnostics_output = self.root / "diagnostics"
        diagnostics_output.mkdir()
        (diagnostics_output / "diagnostics.png").write_bytes(b"obsolete diagnostics image")
        (diagnostics_output / "summary.json").write_text(json.dumps({"plot_available": False}), encoding="utf-8")
        with patch.object(web, "IDENTIFIABILITY_OUTPUT", diagnostics_output):
            with self.assertRaises(HTTPException) as caught:
                web.identifiability_plot()
            self.assertEqual(caught.exception.status_code, 404)


if __name__ == "__main__":
    unittest.main()
