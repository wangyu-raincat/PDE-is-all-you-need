"""Audit the real-experiment workflow with temporary manufactured fixtures.

These tests exercise provenance, freezing and data isolation. They never read
the public experimental holdout and cannot supply a real-data success claim.
"""

from copy import deepcopy
import hashlib
import inspect
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from fastapi import HTTPException

from pde_discovery import app as web
from pde_discovery import real_experiment as experiment


def candidate(terms, support_size):
    return {
        "terms": terms,
        "equation": "temporary manufactured test candidate",
        "fit_error": 0.001,
        "fit_relative_error": 0.001,
        "support_size": support_size,
        "diagnostics": {
            "lasso_all_support_runs_converged": True,
            "support_rank": support_size,
            "library": ["1", "theta", "theta^2", "theta_x", "theta_xx"],
        },
    }


class RealExperimentWorkflowAcceptanceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="pde-workflow-test-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.data_dir = self.root / "manufactured_fixtures"
        self.data_dir.mkdir()
        self.t = np.linspace(0.0, 1.6, 321)
        modes = np.arange(1, 5)
        eigenvalues = -4 * np.sin(modes * np.pi / 10) ** 2 / 0.2**2
        spatial_modes = np.sin(np.outer(np.arange(1, 5), modes) * np.pi / 5)
        splits = {}
        self.arrays = {}
        for role, amplitudes in [
            ("training", [2.0, 0.6, 0.35, 0.15]),
            ("development", [1.8, 0.3, 0.25, 0.2]),
            ("final", [1.5, 0.2, 0.5, 0.1]),
        ]:
            interior = (
                np.exp(np.outer(self.t, 0.08 * eigenvalues - 0.12)) * amplitudes
            ) @ spatial_modes.T
            u = np.column_stack([np.zeros(len(self.t)), interior, np.zeros(len(self.t))])
            path = self.data_dir / f"{role}.csv"
            np.savetxt(path, np.column_stack([self.t, u]), delimiter=",")
            splits[role] = {
                "file": path.name,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            self.arrays[role] = (self.t.copy(), u)
        self.manifest = {
            "experiment": "MANUFACTURED TEST FIXTURE: NOT REAL MEASUREMENTS",
            "spacing_m": 0.2,
            "csv_format": {"delimiter": ",", "skiprows": 0},
            "split_kind": "manufactured independent trajectories for workflow test",
            "splits": splits,
        }
        self.write_manifest()

    def write_manifest(self):
        (self.data_dir / "manifest.json").write_text(
            json.dumps(self.manifest), encoding="utf-8"
        )

    def use_temporal_split(self):
        """One temporary acquisition, split by rows; never the real CSV."""
        entry = self.manifest["splits"]["training"]
        self.manifest["split_kind"] = "chronological_single_acquisition"
        self.manifest["splits"] = {
            role: {**deepcopy(entry), "rows": rows}
            for role, rows in [
                ("training", [0, 192]),
                ("development", [192, 256]),
                ("final", [256, 321]),
            ]
        }
        self.write_manifest()

    def test_selection_has_no_final_input_and_separates_fitting_from_ranking(self):
        self.assertEqual(
            list(inspect.signature(experiment.select_model).parameters),
            ["t_train", "u_train", "t_dev", "u_dev", "dx"],
        )
        t_train, u_train = self.arrays["training"]
        t_dev, u_dev = self.arrays["development"]
        sparse = candidate([{"name": "theta_xx", "coefficient": 0.08}], 1)
        larger = candidate([
            {"name": "theta_xx", "coefficient": 0.08},
            {"name": "theta", "coefficient": -0.12},
        ], 2)
        fitted_inputs, ranked_inputs = [], []

        def fit_only_training(t, u, dx, window):
            fitted_inputs.append((t, u))
            self.assertIs(t, t_train)
            self.assertIs(u, u_train)
            return deepcopy([sparse, larger])

        def rank_only_development(terms, t, u, dx):
            ranked_inputs.append((t, u))
            self.assertIs(t, t_dev)
            self.assertIs(u, u_dev)
            # Predeclared 1.05 near-optimal rule should prefer one term.
            return None, {"rmse": 1.04 if len(terms) == 1 else 1.0}

        with patch.object(experiment, "build_candidates", side_effect=fit_only_training), \
             patch.object(experiment, "score_model", side_effect=rank_only_development):
            selected, candidates, supports = experiment.select_model(
                t_train, u_train, t_dev, u_dev, 0.2
            )
        self.assertEqual(len(fitted_inputs), len(experiment.WINDOW_FRACTIONS))
        self.assertEqual(len(ranked_inputs), 2 * len(experiment.WINDOW_FRACTIONS))
        self.assertEqual(selected["terms"], sparse["terms"])
        self.assertEqual(len(supports), 2)
        self.assertEqual(len(candidates), len(ranked_inputs))

    def test_future_final_interiors_change_only_scores_after_model_is_frozen(self):
        original_loader = experiment.load_measurements
        results, frozen_files, forecasts = [], [], []
        for perturb_future in [False, True]:
            output = self.root / ("perturbed" if perturb_future else "original")
            reads = []

            def checked_load(data_dir, entry, csv_format):
                reads.append(entry["file"])
                if entry["file"] == "final.csv":
                    frozen_path = output / "frozen_model.json"
                    self.assertTrue(frozen_path.exists(), "Freeze must precede final-data load")
                    self.assertIsNotNone(json.loads(frozen_path.read_text())["selected"])
                t, u = original_loader(data_dir, entry, csv_format)
                if perturb_future and entry["file"] == "final.csv":
                    u = u.copy()
                    u[1:, 1:5] += 50.0
                return t, u

            with patch.object(experiment, "load_measurements", side_effect=checked_load), \
                 patch.object(experiment, "_plot"):
                result = experiment.run_real_experiment(self.data_dir, output)
            self.assertEqual(reads, ["training.csv", "development.csv", "final.csv"])
            results.append(result)
            frozen_files.append((output / "frozen_model.json").read_bytes())
            with np.load(output / "trajectories.npz") as trajectories:
                forecasts.append(trajectories["predicted_interior"].copy())
        self.assertEqual(frozen_files[0], frozen_files[1])
        self.assertEqual(results[0]["discovery"], results[1]["discovery"])
        self.assertEqual(results[0]["frozen_model_sha256"], results[1]["frozen_model_sha256"])
        np.testing.assert_array_equal(forecasts[0], forecasts[1])
        self.assertGreater(results[1]["validation"]["rmse_K"], results[0]["validation"]["rmse_K"] + 49)

    def test_no_train_development_candidate_does_not_open_final_holdout(self):
        output = self.root / "no_model"
        original_loader = experiment.load_measurements
        reads = []

        def checked_load(data_dir, entry, csv_format):
            reads.append(entry["file"])
            self.assertNotEqual(entry["file"], "final.csv")
            return original_loader(data_dir, entry, csv_format)

        with patch.object(experiment, "load_measurements", side_effect=checked_load), \
             patch.object(experiment, "select_model", return_value=(None, [], [])), \
             patch.object(experiment, "_plot") as plot:
            report = experiment.run_real_experiment(self.data_dir, output)
        self.assertEqual(reads, ["training.csv", "development.csv"])
        self.assertEqual(report["discovery"]["status"], "insufficient")
        self.assertEqual(report["validation"], {})
        self.assertFalse(report["plot_available"])
        self.assertFalse(report["trajectory_available"])
        self.assertTrue((output / "frozen_model.json").exists())
        plot.assert_not_called()

    def test_reused_original_file_cannot_count_as_a_third_repeat(self):
        self.manifest["splits"]["final"] = deepcopy(self.manifest["splits"]["training"])
        self.write_manifest()
        with patch.object(experiment, "select_model") as select:
            with self.assertRaises(ValueError):
                experiment.run_real_experiment(self.data_dir, self.root / "reused")
        select.assert_not_called()

    def test_cropped_rows_cannot_be_reported_as_complete_repeated_acquisitions(self):
        self.manifest["splits"]["final"]["rows"] = [10, 100]
        self.write_manifest()
        with patch.object(experiment, "select_model") as select:
            with self.assertRaises(ValueError):
                experiment.run_real_experiment(self.data_dir, self.root / "cropped")
        select.assert_not_called()

    def test_final_solver_failure_is_saved_without_reselecting_the_frozen_model(self):
        output = self.root / "solver_failure"
        selected = candidate([{"name": "theta_xx", "coefficient": 0.08}], 1)
        expected = deepcopy(selected)
        with patch.object(experiment, "select_model", return_value=(selected, [selected], [["theta_xx"]])) as select, \
             patch.object(experiment, "score_model", side_effect=ValueError("manufactured final solver failure")) as score, \
             patch.object(experiment, "_plot") as plot:
            report = experiment.run_real_experiment(self.data_dir, output)
        select.assert_called_once()
        score.assert_called_once()
        plot.assert_not_called()
        self.assertEqual(report["validation"]["status"], "failed")
        self.assertFalse(report["plot_available"])
        self.assertFalse(report["trajectory_available"])
        self.assertIn("manufactured final solver failure", report["validation"]["failure_reason"])
        self.assertNotIn("rmse_K", report["validation"])
        frozen = json.loads((output / "frozen_model.json").read_text())
        self.assertEqual(frozen["selected"], expected)
        self.assertEqual(report["discovery"]["terms"], expected["terms"])
        self.assertEqual(
            report["frozen_model_sha256"],
            hashlib.sha256((output / "frozen_model.json").read_bytes()).hexdigest(),
        )
        self.assertEqual(json.loads((output / "report.json").read_text()), report)

    def test_zero_observed_signal_keeps_absolute_error_when_relative_error_is_undefined(self):
        output = self.root / "zero_signal"
        selected = candidate([{"name": "1", "coefficient": 0.1}], 1)
        original_loader = experiment.load_measurements

        def zero_final_signal(data_dir, entry, csv_format):
            t, u = original_loader(data_dir, entry, csv_format)
            if entry["file"] == "final.csv":
                u = np.zeros_like(u)
            return t, u

        with patch.object(experiment, "select_model", return_value=(selected, [selected], [["1"]])), \
             patch.object(experiment, "load_measurements", side_effect=zero_final_signal), \
             patch.object(experiment, "_plot"):
            report = experiment.run_real_experiment(self.data_dir, output)
        self.assertEqual(report["validation"]["status"], "evaluated")
        self.assertIsNone(report["validation"]["relative_l2"])
        self.assertGreater(report["validation"]["rmse_K"], 0.0)
        self.assertFalse(report["validation"]["below_descriptive_5pct"])
        self.assertIn("未定义", report["conclusion"])
        self.assertEqual(json.loads((output / "report.json").read_text()), report)

    def test_old_plot_is_not_served_as_a_failed_or_insufficient_current_result(self):
        output = self.root / "stale_artifacts"
        output.mkdir()
        (output / "validation.png").write_bytes(b"obsolete previous-run image")
        for validation in [{"status": "failed"}, {}]:
            with self.subTest(validation=validation):
                (output / "report.json").write_text(json.dumps({
                    "discovery": {"status": "insufficient" if not validation else "candidate"},
                    "validation": validation,
                    "plot_available": False,
                }), encoding="utf-8")
                with patch.object(web, "DEFAULT_OUTPUT", output):
                    with self.assertRaises(HTTPException) as caught:
                        web.real_experiment_plot()
                self.assertEqual(caught.exception.status_code, 404)

    def test_original_csv_corruption_is_detected_before_fitting(self):
        with (self.data_dir / "training.csv").open("a", encoding="utf-8") as stream:
            stream.write("0,0,0,0,0,0,0\n")
        with patch.object(experiment, "select_model") as select:
            with self.assertRaises(ValueError):
                experiment.run_real_experiment(self.data_dir, self.root / "corrupt")
        select.assert_not_called()

    def test_temporal_future_interiors_cannot_change_model_and_are_labelled_as_a_future_segment(self):
        self.use_temporal_split()
        original_loader = experiment.load_measurements
        results, frozen_files, forecasts = [], [], []
        for perturb_future in [False, True]:
            output = self.root / ("temporal_perturbed" if perturb_future else "temporal_original")
            rows_loaded = []

            def checked_load(data_dir, entry, csv_format):
                rows_loaded.append(entry["rows"])
                if entry["rows"] == [256, 321]:
                    self.assertTrue((output / "frozen_model.json").exists())
                t, u = original_loader(data_dir, entry, csv_format)
                if perturb_future and entry["rows"] == [256, 321]:
                    u = u.copy()
                    u[1:, 1:5] += 50.0
                return t, u

            with patch.object(experiment, "load_measurements", side_effect=checked_load), \
                 patch.object(experiment, "_plot"):
                report = experiment.run_real_experiment(self.data_dir, output)
            self.assertEqual(rows_loaded, [[0, 192], [192, 256], [256, 321]])
            self.assertIn("同一次实验的未来时段", report["validation"]["kind"])
            self.assertNotIn("独立重复实验", report["validation"]["kind"])
            self.assertIn("未来时段", report["conclusion"])
            self.assertNotIn("重复实验", report["conclusion"])
            self.assertFalse(report["discovery"]["unique_physical_equation_established"])
            results.append(report)
            frozen_files.append((output / "frozen_model.json").read_bytes())
            with np.load(output / "trajectories.npz") as trajectories:
                forecasts.append(trajectories["predicted_interior"].copy())
        self.assertEqual(frozen_files[0], frozen_files[1])
        self.assertEqual(results[0]["discovery"], results[1]["discovery"])
        np.testing.assert_array_equal(forecasts[0], forecasts[1])
        self.assertGreater(results[1]["validation"]["rmse_K"], results[0]["validation"]["rmse_K"] + 49)

    def test_temporal_overlaps_or_reversed_roles_are_rejected_before_model_fitting(self):
        for role, bad_interval in [
            ("development", [180, 256]),
            ("final", [255, 321]),
            ("final", [100, 150]),
        ]:
            with self.subTest(role=role, interval=bad_interval):
                self.use_temporal_split()
                self.manifest["splits"][role]["rows"] = bad_interval
                self.write_manifest()
                with patch.object(experiment, "select_model") as select, \
                     patch.object(experiment, "load_measurements") as load:
                    with self.assertRaises(ValueError):
                        experiment.run_real_experiment(self.data_dir, self.root / "overlap")
                select.assert_not_called()
                load.assert_not_called()

    def test_temporal_loading_does_not_parse_numeric_values_outside_the_role(self):
        self.use_temporal_split()
        path = self.data_dir / "training.csv"
        lines = path.read_text().splitlines(keepends=True)
        lines[257] = "invalid_future_numeric_observation\n"
        path.write_text("".join(lines), encoding="utf-8")
        for entry in self.manifest["splits"].values():
            entry["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        self.write_manifest()
        for role, count in [("training", 192), ("development", 64)]:
            t, u = experiment.load_measurements(
                self.data_dir, self.manifest["splits"][role], self.manifest["csv_format"]
            )
            self.assertEqual(len(t), count)
            self.assertEqual(u.shape, (count, 6))
        with self.assertRaises(ValueError):
            experiment.load_measurements(
                self.data_dir, self.manifest["splits"]["final"], self.manifest["csv_format"]
            )

    def test_failed_final_integrity_cannot_leave_a_previous_success_available(self):
        output = self.root / "failed_final_integrity"
        output.mkdir()
        (output / "validation.png").write_bytes(b"obsolete successful image")
        (output / "trajectories.npz").write_bytes(b"obsolete successful trajectories")
        (output / "report.json").write_text(json.dumps({
            "validation": {"status": "evaluated"},
            "plot_available": True,
            "trajectory_available": True,
        }), encoding="utf-8")
        with (self.data_dir / "final.csv").open("a", encoding="utf-8") as stream:
            stream.write("corrupted after manifest freeze\n")
        selected = candidate([{"name": "theta_xx", "coefficient": 0.08}], 1)
        with patch.object(experiment, "select_model", return_value=(selected, [selected], [["theta_xx"]])) as select, \
             patch.object(experiment, "score_model") as score:
            report = experiment.run_real_experiment(self.data_dir, output)
        select.assert_called_once()
        score.assert_not_called()
        self.assertEqual(report["validation"]["status"], "failed")
        self.assertIn("校验和", report["validation"]["failure_reason"])
        self.assertFalse(report["plot_available"])
        self.assertFalse(report["trajectory_available"])
        self.assertEqual(json.loads((output / "report.json").read_text()), report)
        self.assertEqual(json.loads((output / "frozen_model.json").read_text())["selected"], selected)
        with patch.object(web, "DEFAULT_OUTPUT", output):
            with self.assertRaises(HTTPException) as caught:
                web.real_experiment_plot()
        self.assertEqual(caught.exception.status_code, 404)

    def test_rank_deficiency_remains_ambiguous_on_success_and_final_solver_failure(self):
        selected = candidate([
            {"name": "theta_xx", "coefficient": 0.08},
            {"name": "theta", "coefficient": -0.12},
        ], 2)
        selected["diagnostics"]["support_rank"] = 1
        full_rank = candidate([{"name": "theta_xx", "coefficient": 0.08}], 1)

        def score_only_full_rank(terms, t, u, dx):
            self.assertEqual(terms, full_rank["terms"])
            return None, {"rmse": 1.0}

        with patch.object(experiment, "build_candidates", side_effect=lambda *_: deepcopy([selected, full_rank])), \
             patch.object(experiment, "score_model", side_effect=score_only_full_rank):
            chosen, candidates, _ = experiment.select_model(
                *self.arrays["training"], *self.arrays["development"], 0.2
            )
        self.assertEqual(chosen["terms"], full_rank["terms"])
        rank_deficient = [item for item in candidates if item["support_size"] == 2]
        self.assertTrue(rank_deficient)
        self.assertTrue(all(not item["eligible"] for item in rank_deficient))
        # Also defend the report boundary if an externally supplied selection
        # has a rank warning: final solver failure must not erase that warning.
        for fail_final in [False, True]:
            with self.subTest(fail_final=fail_final):
                output = self.root / ("rank_failure" if fail_final else "rank_success")
                score_effect = (ValueError("manufactured final solver failure")
                                if fail_final else experiment.score_model)
                with patch.object(experiment, "select_model", return_value=(deepcopy(selected), [deepcopy(selected)], [["theta_xx", "theta"]])), \
                     patch.object(experiment, "score_model", side_effect=score_effect), \
                     patch.object(experiment, "_plot"):
                    report = experiment.run_real_experiment(self.data_dir, output)
                self.assertEqual(report["discovery"]["status"], "ambiguous")
                self.assertEqual(report["discovery"]["diagnostics"]["support_rank"], 1)
                self.assertFalse(report["discovery"]["unique_physical_equation_established"])
                self.assertNotIn("开发数据支持多种近似候选结构", report["conclusion"])

    def test_mendeley_repeat_report_does_not_inherit_github_example_provenance(self):
        self.manifest["experiment"] = {
            "title": "MANUFACTURED FIXTURE FOR MENDELEY REPORT METADATA",
            "source_url": "https://data.mendeley.com/datasets/sbcf36976g/1",
            "doi": "10.17632/sbcf36976g.1",
            "license": "CC BY 4.0",
            "provenance_status": "published_experimental_dataset",
        }
        self.manifest["provenance_note"] = (
            "Mendeley published experimental dataset metadata fixture: "
            "DOI 10.17632/sbcf36976g.1; CC BY 4.0."
        )
        self.write_manifest()
        selected = candidate([{"name": "theta_xx", "coefficient": 0.08}], 1)
        with patch.object(experiment, "select_model", return_value=(selected, [selected], [["theta_xx"]])), \
             patch.object(experiment, "_plot"):
            report = experiment.run_real_experiment(self.data_dir, self.root / "mendeley_metadata")
        self.assertEqual(report["experiment"], self.manifest["experiment"])
        limitations = " ".join(report["limitations"])
        self.assertIn(self.manifest["provenance_note"], limitations)
        self.assertNotIn("CSV 是作者公开的采集示例", limitations)
        self.assertNotIn("不能等同已认证的论文原始数据", limitations)
        self.assertNotIn("60/20/20", limitations)
        self.assertIn("独立重复实验", report["validation"]["kind"])


if __name__ == "__main__":
    unittest.main()
