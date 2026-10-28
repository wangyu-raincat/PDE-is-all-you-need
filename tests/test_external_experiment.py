"""Audit independent-source validation on manufactured temporary records only.

These fixtures are not experimental evidence. No real high-voltage file is
read, and ordinary unit checks prohibit network requests.
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

from pde_discovery import external_experiment as experiment


def manufactured_candidate():
    return {
        "candidate_id": "manufactured-pure-diffusion",
        "model_name": "D",
        "equation": "MANUFACTURED TEST FIXTURE ONLY",
        "terms": [{"name": "theta_xx", "coefficient": 1e-6}],
        "support_size": 1,
        "fit_success": True,
        "train_score": {"rmse": 0.0, "relative_error": 0.0},
        "diagnostics": {
            "jacobian_rank": 1, "parameter_count": 1,
            "scaled_jacobian_condition_number": 1.0,
        },
    }


def git_blob_digest(data):
    return hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()


class ExternalWorkflowTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="pde-external-test-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.data_root = self.root / "manufactured_data"
        self.output = self.root / "output"
        (self.data_root / "raw").mkdir(parents=True)
        self.manifest = json.loads((experiment.DATA_ROOT / "source_manifest.json").read_text())
        self.raw_arrays = {}
        self.original_bytes = {}
        modes = np.arange(1, 5)
        eigenvalues = -4 * np.sin(modes * np.pi / 10) ** 2 / 0.0127**2
        vectors = np.sin(np.outer(np.arange(1, 5), modes) * np.pi / 5)
        for group_index, group in enumerate(experiment.GROUPS):
            for role_index, role in enumerate(("lower", "holdout")):
                filename = experiment.FILES[group][role]
                t = 10.0 * np.arange(51)
                amplitudes = np.array([4.0 + group_index + role_index, 0.8, 0.6, 0.3])
                interior = (np.exp(np.outer(t, 1e-6 * eigenvalues)) * amplitudes) @ vectors.T
                level = 20.0 + group_index * 3.0 + role_index * 8.0
                u = np.column_stack([np.zeros(len(t)), interior, np.zeros(len(t))]) + level
                # Ignored channels differ greatly and must never enter Tref or the model.
                raw = np.column_stack([t, u, np.full(len(t), 800.0), np.full(len(t), 900.0)])
                buffer = io.StringIO()
                np.savetxt(buffer, raw, delimiter=",", comments="", fmt="%.17g",
                           header="Time[s]," + ",".join(f"CH{i}[C]" for i in range(1, 9)))
                data = buffer.getvalue().encode()
                (self.data_root / "raw" / filename).write_bytes(data)
                entry = next(item for item in self.manifest["files"] if item["name"] == filename)
                entry.update(bytes=len(data), git_blob_sha1=git_blob_digest(data))
                self.raw_arrays[filename] = raw
                self.original_bytes[filename] = data
        self.write_manifest()
        protocol = self.root / "manufactured_protocol.md"
        protocol.write_text(experiment.PROTOCOL_PATH.read_text())
        self.protocol_path = protocol
        self.start_patch("PROTOCOL_PATH", protocol)
        self.no_network = patch.object(experiment.urllib.request, "urlopen",
                                       side_effect=AssertionError("Network forbidden in unit fixture"))
        self.network_mock = self.no_network.start()
        self.addCleanup(self.no_network.stop)
        experiment.download_data(self.data_root)
        self.fit_mock = self.start_patch("fit_candidates", side_effect=lambda *_: [manufactured_candidate()])
        self.l1_mock = self.start_patch("select_model", side_effect=lambda *_: (
            manufactured_candidate(), [manufactured_candidate()], [["theta_xx"]]))
        self.start_patch("_plot_case")
        self.start_patch("_plot_summary")

    def start_patch(self, name, *args, **kwargs):
        patcher = patch.object(experiment, name, *args, **kwargs)
        result = patcher.start()
        self.addCleanup(patcher.stop)
        return result

    def write_manifest(self):
        (self.data_root / "source_manifest.json").write_text(json.dumps(self.manifest))

    def freeze(self, output=None):
        with redirect_stdout(io.StringIO()):
            return experiment.freeze_benchmark(output or self.output, data_root=self.data_root)

    def evaluate(self, output=None):
        with redirect_stdout(io.StringIO()):
            return experiment.evaluate_benchmark(output or self.output, data_root=self.data_root)

    def frozen(self, group, output=None):
        return json.loads(((output or self.output) / group / "frozen_model.json").read_text())

    def report(self, group, output=None):
        return json.loads(((output or self.output) / group / "report.json").read_text())

    def test_both_groups_and_both_methods_frozen_before_any_holdout_numeric_read(self):
        original_loader = experiment.load_measurements
        reads = []
        final_names = {experiment.FILES[g]["holdout"] for g in experiment.GROUPS}

        def checked_loader(data_root, filename):
            reads.append(filename)
            if filename in final_names:
                self.assertTrue((self.output / "freeze_index.json").exists())
                for group in experiment.GROUPS:
                    frozen = self.frozen(group)
                    for method in ("physical", "l1"):
                        self.assertIsNotNone(frozen["models"][method]["selected"])
            return original_loader(data_root, filename)

        with patch.object(experiment, "load_measurements", side_effect=checked_loader):
            self.freeze()
            self.assertEqual(reads, [experiment.FILES[g]["lower"] for g in experiment.GROUPS])
            self.assertEqual(self.fit_mock.call_count, 2)
            self.assertEqual(self.l1_mock.call_count, 2)
            self.evaluate()
        self.assertEqual(reads[-2:], [experiment.FILES[g]["holdout"] for g in experiment.GROUPS])
        self.network_mock.assert_not_called()

    def test_split_is_disjoint_and_reference_uses_only_first_training_six_sensors(self):
        self.freeze()
        for index, group in enumerate(experiment.GROUPS):
            raw = self.raw_arrays[experiment.FILES[group]["lower"]]
            cutoff = int(np.floor(0.6 * len(raw)))
            reference = raw[0, 1:7].mean()
            physical_args = self.fit_mock.call_args_list[index].args
            l1_args = self.l1_mock.call_args_list[index].args
            np.testing.assert_array_equal(physical_args[0], raw[:cutoff, 0])
            np.testing.assert_allclose(physical_args[1], raw[:cutoff, 1:7] - reference)
            np.testing.assert_array_equal(l1_args[0], raw[:cutoff, 0])
            np.testing.assert_array_equal(l1_args[2], raw[cutoff:, 0])
            np.testing.assert_allclose(l1_args[1], raw[:cutoff, 1:7] - reference)
            np.testing.assert_allclose(l1_args[3], raw[cutoff:, 1:7] - reference)
            self.assertLess(l1_args[0][-1], l1_args[2][0])
            self.assertEqual(physical_args[2], 0.0127)

    def test_final_future_interiors_change_only_scores_not_fits_or_forward_predictions(self):
        self.freeze()
        frozen_bytes = {g: (self.output / g / "frozen_model.json").read_bytes()
                        for g in experiment.GROUPS}
        group = experiment.GROUPS[0]
        filename = experiment.FILES[group]["holdout"]
        original_loader = experiment.load_measurements
        original_predictions = {}
        self.evaluate()
        with np.load(self.output / group / "trajectories.npz") as arrays:
            for name in ("predicted_physical", "predicted_l1", "persistence", "linear_boundary"):
                original_predictions[name] = arrays[name].copy()
            original_observations = arrays["observed_centered"].copy()
        original_report = self.report(group)

        def changed_loader(data_root, name):
            t, u = original_loader(data_root, name)
            if name == filename:
                u = u.copy()
                u[1:, 1:5] += 100.0
            return t, u

        with patch.object(experiment, "load_measurements", side_effect=changed_loader), \
             patch.object(experiment, "fit_candidates") as physical_fit, \
             patch.object(experiment, "select_model") as l1_fit:
            self.evaluate()
        physical_fit.assert_not_called()
        l1_fit.assert_not_called()
        with np.load(self.output / group / "trajectories.npz") as arrays:
            for name, original in original_predictions.items():
                np.testing.assert_array_equal(arrays[name], original)
            np.testing.assert_array_equal(arrays["observed_centered"][0], original_observations[0])
            np.testing.assert_allclose(arrays["observed_centered"][1:, 1:5],
                                       original_observations[1:, 1:5] + 100.0)
        for declared_group in experiment.GROUPS:
            self.assertEqual(frozen_bytes[declared_group],
                             (self.output / declared_group / "frozen_model.json").read_bytes())
        self.assertNotEqual(original_report["models"]["physical"]["validation"],
                            self.report(group)["models"]["physical"]["validation"])

    def test_test_temperature_reference_is_training_reference_not_test_initial_mean(self):
        self.freeze()
        self.evaluate()
        for group in experiment.GROUPS:
            lower = self.raw_arrays[experiment.FILES[group]["lower"]]
            final = self.raw_arrays[experiment.FILES[group]["holdout"]]
            reference = lower[0, 1:7].mean()
            self.assertNotAlmostEqual(reference, final[0, 1:7].mean())
            with np.load(self.output / group / "trajectories.npz") as arrays:
                np.testing.assert_allclose(arrays["observed_centered"], final[:, 1:7] - reference)

    def test_tampered_last_frozen_model_blocks_all_final_numeric_reads(self):
        self.freeze()
        path = self.output / experiment.GROUPS[-1] / "frozen_model.json"
        path.write_text(path.read_text() + "\n")
        with patch.object(experiment, "load_measurements") as load:
            with self.assertRaises(ValueError):
                self.evaluate()
        load.assert_not_called()

    def test_protocol_source_or_manifest_changes_block_final_numeric_reads(self):
        for change in ("protocol", "code", "manifest"):
            with self.subTest(change=change):
                output = self.root / change
                self.freeze(output)
                with patch.object(experiment, "load_measurements") as load:
                    if change == "protocol":
                        before = self.protocol_path.read_text()
                        self.protocol_path.write_text(before + "\nCHANGED\n")
                        try:
                            with self.assertRaises(ValueError):
                                self.evaluate(output)
                        finally:
                            self.protocol_path.write_text(before)
                    elif change == "code":
                        original = experiment._source_hashes()
                        altered = {**original, next(iter(original)): "0" * 64}
                        with patch.object(experiment, "_source_hashes", return_value=altered):
                            with self.assertRaises(ValueError):
                                self.evaluate(output)
                    else:
                        path = self.data_root / "source_manifest.json"
                        before = path.read_text()
                        path.write_text(before + "\n")
                        try:
                            with self.assertRaises(ValueError):
                                self.evaluate(output)
                        finally:
                            path.write_text(before)
                load.assert_not_called()

    def test_existing_raw_corruption_is_not_silently_replaced(self):
        filename = experiment.FILES[experiment.GROUPS[0]]["lower"]
        path = self.data_root / "raw" / filename
        invalid = path.read_bytes() + b"corrupt\n"
        path.write_bytes(invalid)
        with self.assertRaises(ValueError):
            experiment.download_data(self.data_root)
        self.assertEqual(path.read_bytes(), invalid)
        self.network_mock.assert_not_called()

    def test_duplicate_manifest_file_name_is_rejected(self):
        self.manifest["files"].append(deepcopy(self.manifest["files"][0]))
        self.write_manifest()
        with self.assertRaises(ValueError):
            experiment.download_data(self.data_root)
        self.network_mock.assert_not_called()

    def test_invalid_raw_columns_or_time_are_rejected(self):
        filename = experiment.FILES[experiment.GROUPS[0]]["lower"]
        original = self.raw_arrays[filename]
        for label, raw in (("missing_sensor", original[:, :-1]),
                           ("duplicate_time", original.copy()),
                           ("nonfinite", original.copy())):
            with self.subTest(label=label):
                if label == "duplicate_time":
                    raw[1, 0] = raw[0, 0]
                if label == "nonfinite":
                    raw[5, 3] = np.nan
                buffer = io.StringIO()
                np.savetxt(buffer, raw, delimiter=",", comments="", fmt="%.17g",
                           header="Time[s]," + ",".join(f"CH{i}[C]" for i in range(1, 9)))
                data = buffer.getvalue().encode()
                (self.data_root / "raw" / filename).write_bytes(data)
                entry = next(item for item in self.manifest["files"] if item["name"] == filename)
                entry.update(bytes=len(data), git_blob_sha1=git_blob_digest(data))
                self.write_manifest()
                experiment.download_data(self.data_root)
                with self.assertRaises(ValueError):
                    experiment.load_measurements(self.data_root, filename)

    def test_missing_source_file_fails_closed_before_any_numeric_load(self):
        filename = experiment.FILES[experiment.GROUPS[-1]]["holdout"]
        (self.data_root / "raw" / filename).unlink()
        with patch.object(experiment, "load_measurements") as load:
            with self.assertRaises(FileNotFoundError):
                self.freeze()
        load.assert_not_called()
        for group in experiment.GROUPS:
            report = self.report(group)
            self.assertEqual(report["validation"]["status"], "failed")
            self.assertFalse(report["plot_available"])
        self.network_mock.assert_not_called()

    def test_download_rejects_bad_blob_without_writing_a_partial_file(self):
        filename = experiment.FILES[experiment.GROUPS[0]]["lower"]
        path = self.data_root / "raw" / filename
        path.unlink()
        entry = next(item for item in self.manifest["files"] if item["name"] == filename)
        wrong = b"x" * entry["bytes"]

        class Response(io.BytesIO):
            def __init__(self, data):
                super().__init__(data)
                self.read_sizes = []

            def read(self, size=-1):
                self.read_sizes.append(size)
                return super().read(size)

        response = Response(wrong)
        with patch.object(experiment.urllib.request, "urlopen", return_value=response) as download:
            with self.assertRaises(ValueError):
                experiment.download_data(self.data_root)
        download.assert_called_once()
        self.assertGreater(download.call_args.kwargs["timeout"], 0)
        self.assertLessEqual(download.call_args.kwargs["timeout"], 60)
        self.assertEqual(response.read_sizes, [entry["bytes"] + 1])
        self.assertEqual(download.call_args.args[0].full_url, entry["url"])
        self.assertFalse(path.exists())

    def test_download_timeout_retains_other_files_and_no_partial_target(self):
        filename = experiment.FILES[experiment.GROUPS[0]]["holdout"]
        path = self.data_root / "raw" / filename
        path.unlink()
        with patch.object(experiment.urllib.request, "urlopen", side_effect=TimeoutError("fixture timeout")):
            with self.assertRaises(TimeoutError):
                experiment.download_data(self.data_root)
        self.assertFalse(path.exists())
        for name, data in self.original_bytes.items():
            if name != filename:
                self.assertEqual((self.data_root / "raw" / name).read_bytes(), data)

    def test_one_final_parse_failure_retains_both_cases_and_scores_the_other(self):
        self.freeze()
        self.evaluate()
        failed_group = experiment.GROUPS[0]
        stale_plot = self.output / failed_group / "validation.png"
        stale_plot.write_bytes(b"old scientific plot fixture")
        original_loader = experiment.load_measurements

        def fail_one(data_root, filename):
            if filename == experiment.FILES[failed_group]["holdout"]:
                raise ValueError("manufactured malformed final record")
            return original_loader(data_root, filename)

        with patch.object(experiment, "load_measurements", side_effect=fail_one), \
             patch.object(experiment, "fit_candidates") as physical_fit, \
             patch.object(experiment, "select_model") as l1_fit:
            summary = self.evaluate()
        physical_fit.assert_not_called()
        l1_fit.assert_not_called()
        self.assertEqual(len(summary["records"]), 2)
        self.assertEqual([r["validation_status"] for r in summary["records"]], ["failed", "evaluated"])
        failed = self.report(failed_group)
        self.assertFalse(failed["plot_available"])
        self.assertFalse(failed["trajectory_available"])
        self.assertTrue(stale_plot.exists())  # status, not file presence, governs display
        self.assertTrue(all(failed["models"][m]["validation"]["status"] == "failed"
                            for m in ("physical", "l1")))
        successful = self.report(experiment.GROUPS[-1])
        self.assertTrue(all(successful["models"][m]["validation"]["status"] == "evaluated"
                            for m in ("physical", "l1")))

    def test_failed_training_method_is_retained_without_disabling_other_method(self):
        with patch.object(experiment, "select_model", side_effect=[
                ValueError("manufactured L1 selection failure"),
                (manufactured_candidate(), [manufactured_candidate()], [["theta_xx"]])]):
            self.freeze()
        failed = self.frozen(experiment.GROUPS[0])
        self.assertIsNone(failed["models"]["l1"]["selected"])
        self.assertIsNotNone(failed["models"]["physical"]["selected"])
        with patch.object(experiment, "select_model") as retry:
            summary = self.evaluate()
        retry.assert_not_called()
        self.assertEqual(len(summary["records"]), 2)
        report = self.report(experiment.GROUPS[0])
        self.assertEqual(report["models"]["physical"]["validation"]["status"], "evaluated")
        self.assertEqual(report["models"]["l1"]["validation"]["status"], "insufficient")
        self.assertIsNone(report["comparison"]["new_vs_l1_rmse_improvement_fraction"])

    def test_global_integrity_failure_invalidates_previous_successful_reports(self):
        self.freeze()
        self.evaluate()
        filename = experiment.FILES[experiment.GROUPS[-1]]["holdout"]
        path = self.data_root / "raw" / filename
        path.write_bytes(path.read_bytes() + b"changed\n")
        with patch.object(experiment, "load_measurements") as load:
            with self.assertRaises(ValueError):
                self.evaluate()
        load.assert_not_called()
        summary = json.loads((self.output / "benchmark.json").read_text())
        self.assertEqual(len(summary["records"]), 2)
        self.assertFalse(summary["plot_available"])
        for group in experiment.GROUPS:
            report = self.report(group)
            self.assertFalse(report["plot_available"])
            self.assertFalse(report["trajectory_available"])
            self.assertEqual(report["validation"]["status"], "failed")

