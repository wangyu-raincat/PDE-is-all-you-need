"""Audit PDE-discovery isolation using manufactured MAT files only.

No public benchmark arrays or external reference equations are read by these
tests. The fixtures below are numerical controls, not scientific evidence.
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
from scipy.io import savemat

from pde_discovery import generic_experiment as experiment


def digest(content):
    return hashlib.sha1(b"blob " + str(len(content)).encode() + b"\0" + content).hexdigest()


def candidate():
    return {
        "candidate_id": "MANUFACTURED-DIFFUSION",
        "equation": "MANUFACTURED FIXTURE: u_t = 0.1 u_xx",
        "terms": [{"name": "u_xx", "coefficient": 0.1}],
        "support_size": 1, "fit_success": True,
        "fit_error": 0.0, "fit_relative_error": 0.0,
        "diagnostics": {"support_rank": 1, "parameter_count": 1,
                        "lasso_all_support_runs_converged": True},
    }


def analytic_forecast(terms, x, t, u0):
    """Exact evolution for our single-mode diffusion fixture only."""
    coefficient = next(term["coefficient"] for term in terms if term["name"] == "u_xx")
    return np.exp(-coefficient * (np.asarray(t) - t[0]))[:, None] * np.asarray(u0)[None, :]


class GenericWorkflowTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="pde-generic-fixture-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.data_root = self.root / "data"
        self.output = self.root / "output"
        (self.data_root / "raw").mkdir(parents=True)
        self.protocol = self.root / "manufactured_protocol.md"
        self.protocol.write_text("MANUFACTURED ISOLATION PROTOCOL ONLY\n")
        self.reference = self.root / "manufactured_reference.json"
        self.reference.write_text(json.dumps({"references": {
            group: {"terms": [{"name": "u_xx", "coefficient": 0.1}]}
            for group in experiment.GROUPS
        }}))
        self.start_patch("PROTOCOL_PATH", self.protocol)
        self.start_patch("REFERENCE_PATH", self.reference)
        self.manifest = {
            "repo": "snagcliffs/PDE-FIND",
            "commit": "86911349b9fac82996e35465a13d06205730cc05",
            "source_kind": "MANUFACTURED UNIT FIXTURES ONLY", "license": None,
            "groups": {},
        }
        self.arrays = {}
        names = {"burgers": ("burgers.mat", "t", "usol"),
                 "kdv": ("kdv.mat", "t", "usol"),
                 "ks": ("kuramoto_sivishinky.mat", "tt", "uu")}
        for index, group in enumerate(experiment.GROUPS):
            filename, time_key, field_key = names[group]
            x = np.arange(64) * (2 * np.pi / 64)
            t = np.linspace(0, 2, 101)
            u = (index + 1.0) * np.exp(-0.1 * t[:, None]) * np.cos(x)[None, :]
            self.arrays[group] = {"x": x, "t": t, "u": u}
            self.manifest["groups"][group] = {
                "filename": filename,
                "url": f"https://raw.githubusercontent.com/snagcliffs/PDE-FIND/{self.manifest['commit']}/Datasets/{filename}",
                "mat_keys": {"x": "x", "t": time_key, "u": field_key},
                "orientation": "space_time", "field_shape": [64, 101],
                "boundary": "periodic", "data_kind": "manufactured unit fixture",
            }
            self.write_mat(group)
        self.write_manifest()
        self.network = self.start_patch("urllib.request.urlopen", side_effect=AssertionError("Network forbidden in unit tests"))
        self.fit = self.start_patch("fit_candidates", side_effect=lambda *_: [candidate()])
        self.forecast = self.start_patch("forecast", side_effect=analytic_forecast)
        self.start_patch("_plot_case")
        self.start_patch("_plot_summary")
        with redirect_stdout(io.StringIO()):
            experiment.download_data(self.data_root)
            experiment.prepare_data(self.data_root)

    def start_patch(self, name, *args, **kwargs):
        target = experiment
        parts = name.split(".")
        for part in parts[:-1]:
            target = getattr(target, part)
        patcher = patch.object(target, parts[-1], *args, **kwargs)
        result = patcher.start()
        self.addCleanup(patcher.stop)
        return result

    def write_mat(self, group):
        entry = self.manifest["groups"][group]
        arrays = self.arrays[group]
        keys = entry["mat_keys"]
        buffer = io.BytesIO()
        savemat(buffer, {keys["x"]: arrays["x"][:, None],
                        keys["t"]: arrays["t"][:, None],
                        keys["u"]: arrays["u"].T}, do_compression=True)
        data = buffer.getvalue()
        (self.data_root / "raw" / entry["filename"]).write_bytes(data)
        entry.update(bytes=len(data), git_blob_sha1=digest(data),
                     sha256=hashlib.sha256(data).hexdigest())

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

    def report(self, group):
        return json.loads((self.output / group / "report.json").read_text())

    def test_preparation_keeps_original_grid_and_disjoint_chronological_rows(self):
        self.fit.assert_not_called()
        self.forecast.assert_not_called()
        for group in experiment.GROUPS:
            original = self.arrays[group]
            for role, first, last in (("training", 0, 60), ("development", 60, 80), ("final", 80, 101)):
                part = experiment.load_partition(self.data_root, group, role)
                np.testing.assert_array_equal(part["x"], original["x"])
                np.testing.assert_array_equal(part["t"], original["t"][first:last])
                np.testing.assert_array_equal(part["u"], original["u"][first:last])
            training = experiment.load_partition(self.data_root, group, "training")
            development = experiment.load_partition(self.data_root, group, "development")
            final = experiment.load_partition(self.data_root, group, "final")
            self.assertLess(training["t"][-1], development["t"][0])
            self.assertLess(development["t"][-1], final["t"][0])

    def test_every_group_frozen_before_final_or_reference_is_loaded(self):
        original_loader, original_reference = experiment.load_partition, experiment.load_reference
        reads = []

        def all_frozen():
            self.assertTrue((self.output / "freeze_index.json").is_file())
            for group in experiment.GROUPS:
                self.assertIsNotNone(self.frozen(group)["selected"])

        def loader(data_root, group, role):
            reads.append((group, role))
            if role == "final":
                all_frozen()
            return original_loader(data_root, group, role)

        def reference(*args, **kwargs):
            all_frozen()
            return original_reference(*args, **kwargs)

        with patch.object(experiment, "load_partition", side_effect=loader), \
             patch.object(experiment, "load_reference", side_effect=reference) as ref:
            self.freeze()
            self.assertFalse(any(role == "final" for _, role in reads))
            ref.assert_not_called()
            self.assertEqual(self.fit.call_count, 3)
            self.evaluate()
            self.assertEqual([g for g, role in reads if role == "final"], list(experiment.GROUPS))
        self.network.assert_not_called()

    def test_reference_coefficients_do_not_change_candidates_or_selection(self):
        self.freeze()
        first = {g: deepcopy(self.frozen(g)["selected"]) for g in experiment.GROUPS}
        self.reference.write_text(json.dumps({"references": {
            g: {"terms": [{"name": "u^3", "coefficient": 500.0}]}
            for g in experiment.GROUPS
        }}))
        second_output = self.root / "second_output"
        with patch.object(experiment, "load_reference", side_effect=AssertionError("Reference cannot drive fitting")):
            self.freeze(second_output)
        for group in experiment.GROUPS:
            self.assertEqual(first[group], self.frozen(group, second_output)["selected"])

    def test_last_frozen_model_tamper_prevents_all_final_and_reference_reads(self):
        self.freeze()
        path = self.output / experiment.GROUPS[-1] / "frozen_model.json"
        path.write_text(path.read_text() + "\n")
        with patch.object(experiment, "load_partition") as load, \
             patch.object(experiment, "load_reference") as reference:
            with self.assertRaises(ValueError):
                self.evaluate()
        load.assert_not_called()
        reference.assert_not_called()

    def test_corrupted_prepared_final_prevents_all_numeric_scoring(self):
        self.freeze()
        path = self.data_root / "prepared" / experiment.GROUPS[-1] / "final.npz"
        path.write_bytes(path.read_bytes() + b"TAMPER")
        with patch.object(experiment, "load_partition") as load, \
             patch.object(experiment, "load_reference") as reference:
            with self.assertRaises(ValueError):
                self.evaluate()
        load.assert_not_called()
        reference.assert_not_called()

    def test_source_or_protocol_change_prevents_scoring(self):
        self.freeze()
        for label in ("source", "protocol"):
            with self.subTest(label=label):
                path = self.data_root / "source_manifest.json" if label == "source" else self.protocol
                before = path.read_bytes()
                path.write_bytes(before + b"\n")
                try:
                    with patch.object(experiment, "load_partition") as load, \
                         patch.object(experiment, "load_reference") as reference:
                        with self.assertRaises(ValueError):
                            self.evaluate()
                    load.assert_not_called()
                    reference.assert_not_called()
                finally:
                    path.write_bytes(before)

    def test_existing_bad_raw_data_is_not_overwritten_or_downloaded(self):
        entry = self.manifest["groups"][experiment.GROUPS[0]]
        path = self.data_root / "raw" / entry["filename"]
        invalid = path.read_bytes() + b"CORRUPT"
        path.write_bytes(invalid)
        with self.assertRaises(ValueError):
            experiment.download_data(self.data_root)
        self.assertEqual(path.read_bytes(), invalid)
        self.network.assert_not_called()

    def test_final_future_field_changes_only_scores_not_selection_or_prediction(self):
        self.freeze()
        frozen_bytes = {group: (self.output / group / "frozen_model.json").read_bytes()
                        for group in experiment.GROUPS}
        self.evaluate()
        group = experiment.GROUPS[0]
        old_report = self.report(group)
        with np.load(self.output / group / "trajectories.npz", allow_pickle=False) as arrays:
            predicted = arrays["prediction"].copy()
        original_loader = experiment.load_partition

        def changed_final(data_root, selected_group, role):
            data = original_loader(data_root, selected_group, role)
            if selected_group == group and role == "final":
                data["u"] = data["u"].copy()
                data["u"][1:] += 50.0
            return data

        self.forecast.reset_mock()
        with patch.object(experiment, "load_partition", side_effect=changed_final), \
             patch.object(experiment, "fit_candidates", side_effect=AssertionError("Final cannot fit")) as fit:
            self.evaluate()
        fit.assert_not_called()
        self.assertEqual(self.forecast.call_count, 3)
        for call in self.forecast.call_args_list:
            # Candidate terms, x, t, initial 1D field. No future field argument.
            self.assertEqual(len(call.args), 4)
            self.assertEqual(np.asarray(call.args[3]).ndim, 1)
        with np.load(self.output / group / "trajectories.npz", allow_pickle=False) as arrays:
            np.testing.assert_array_equal(arrays["prediction"], predicted)
        report = self.report(group)
        self.assertGreater(report["validation"]["rmse"], old_report["validation"]["rmse"] + 40)
        self.assertEqual(report["equation_recovery"], old_report["equation_recovery"])
        for selected_group in experiment.GROUPS:
            self.assertEqual((self.output / selected_group / "frozen_model.json").read_bytes(),
                             frozen_bytes[selected_group])

    def test_one_failed_training_group_is_retained_and_other_groups_are_scored(self):
        self.fit.side_effect = [[candidate()], ValueError("Manufactured unavailable candidate"), [candidate()]]
        self.freeze()
        summary = self.evaluate()
        self.assertEqual([record["case"] for record in summary["records"]], list(experiment.GROUPS))
        self.assertEqual(summary["records"][1]["validation"]["status"], "insufficient")
        self.assertEqual(summary["records"][1]["equation_recovery"]["status"], "insufficient")
        for index in (0, 2):
            self.assertEqual(summary["records"][index]["validation"]["status"], "evaluated")

    def test_one_final_load_failure_retains_all_groups_and_no_reselection(self):
        self.freeze()
        original_loader = experiment.load_partition

        def loader(data_root, group, role):
            if group == experiment.GROUPS[1] and role == "final":
                raise ValueError("Manufactured final parse failure")
            return original_loader(data_root, group, role)

        with patch.object(experiment, "load_partition", side_effect=loader), \
             patch.object(experiment, "fit_candidates", side_effect=AssertionError("No test rescue")) as fit:
            summary = self.evaluate()
        fit.assert_not_called()
        self.assertEqual(len(summary["records"]), 3)
        self.assertEqual(summary["records"][1]["validation"]["status"], "failed")
        self.assertFalse(summary["records"][1]["plot_available"])
        for index in (0, 2):
            self.assertEqual(summary["records"][index]["validation"]["status"], "evaluated")

    def test_integrity_failure_invalidates_previous_successful_reports(self):
        self.freeze()
        self.evaluate()
        path = self.output / experiment.GROUPS[-1] / "frozen_model.json"
        path.write_text(path.read_text() + "\n")
        with self.assertRaises(ValueError):
            self.evaluate()
        for group in experiment.GROUPS:
            report = self.report(group)
            self.assertEqual(report["validation"]["status"], "failed")
            self.assertFalse(report["plot_available"])

    def test_minimum_sixteen_frames_applies_to_each_partition(self):
        group = experiment.GROUPS[0]
        self.arrays[group] = {key: value[:60] if key in ("t", "u") else value
                              for key, value in self.arrays[group].items()}
        self.manifest["groups"][group]["field_shape"] = [64, 60]
        self.write_mat(group)
        self.write_manifest()
        experiment.download_data(self.data_root)
        with self.assertRaises(ValueError):
            experiment.prepare_data(self.data_root)

    def test_download_rejects_bad_blob_without_partial_file(self):
        entry = self.manifest["groups"][experiment.GROUPS[0]]
        path = self.data_root / "raw" / entry["filename"]
        path.unlink()
        response = io.BytesIO(b"x" * entry["bytes"])
        with patch.object(experiment.urllib.request, "urlopen", return_value=response) as fetch:
            with self.assertRaises(ValueError):
                experiment.download_data(self.data_root)
        self.assertFalse(path.exists())
        fetch.assert_called_once()
        self.assertGreater(fetch.call_args.kwargs["timeout"], 0)
        self.assertLessEqual(fetch.call_args.kwargs["timeout"], 60)

    def test_code_or_reference_hash_change_prevents_final_reads(self):
        self.freeze()
        with patch.object(experiment, "_source_hashes", return_value={"changed": "0" * 64}), \
             patch.object(experiment, "load_partition") as load, \
             patch.object(experiment, "load_reference") as reference:
            with self.assertRaises(ValueError):
                self.evaluate()
        load.assert_not_called()
        reference.assert_not_called()
        self.reference.write_text(self.reference.read_text() + "\n")
        with patch.object(experiment, "load_partition") as load, \
             patch.object(experiment, "load_reference") as reference:
            with self.assertRaises(ValueError):
                self.evaluate()
        load.assert_not_called()
        reference.assert_not_called()

    def test_invalid_time_or_nonfinite_measurements_are_rejected_during_prepare(self):
        group = experiment.GROUPS[0]
        for mode in ("duplicate_time", "nonfinite"):
            with self.subTest(mode=mode):
                original = {key: value.copy() for key, value in self.arrays[group].items()}
                try:
                    if mode == "duplicate_time":
                        self.arrays[group]["t"][1] = self.arrays[group]["t"][0]
                    else:
                        self.arrays[group]["u"][5, 2] = np.nan
                    self.write_mat(group)
                    self.write_manifest()
                    experiment.download_data(self.data_root)
                    with self.assertRaises(ValueError):
                        experiment.prepare_data(self.data_root)
                finally:
                    self.arrays[group] = original

    def test_documented_real_component_is_extracted_and_undeclared_complex_is_rejected(self):
        # This follows an explicit source field definition, not a claim that a
        # stored imaginary component is negligible numerical roundoff.
        for group, permission in (("burgers", True), ("ks", False)):
            original = self.arrays[group]["u"].copy()
            entry = self.manifest["groups"][group]
            for imaginary in (1e-12, 0.25):
                with self.subTest(group=group, imaginary=imaginary):
                    self.arrays[group]["u"] = original.astype(complex) + 1j * imaginary
                    entry["take_real_part"] = permission
                    self.write_mat(group)
                    self.write_manifest()
                    experiment.download_data(self.data_root)
                    if permission:
                        prepared = experiment.prepare_data(self.data_root)
                        conversion = prepared["records"][group]["real_component_conversion"]
                        self.assertTrue(conversion["applied"])
                        self.assertAlmostEqual(conversion["maximum_absolute_imaginary_component"], imaginary)
                        self.assertAlmostEqual(conversion["imaginary_to_real_l2_ratio"],
                                               imaginary * np.sqrt(original.size) / np.linalg.norm(original))
                        self.assertTrue(conversion["imaginary_component_is_not_asserted_to_be_roundoff"])
                        training = experiment.load_partition(self.data_root, group, "training")
                        np.testing.assert_array_equal(training["u"], original[:60])
                    else:
                        with self.assertRaises(ValueError):
                            experiment.prepare_data(self.data_root)
            self.arrays[group]["u"] = original
            self.write_mat(group)
            self.write_manifest()
        # Source field real-part permission must never project complex axes.
        for axis in ("x", "t"):
            with self.subTest(complex_axis=axis):
                original_axis = self.arrays["burgers"][axis].copy()
                self.arrays["burgers"][axis] = original_axis.astype(complex) + 1e-12j
                self.write_mat("burgers")
                self.write_manifest()
                experiment.download_data(self.data_root)
                with self.assertRaises(ValueError):
                    experiment.prepare_data(self.data_root)
                self.arrays["burgers"][axis] = original_axis


class GenericMetricTests(unittest.TestCase):
    def test_support_and_signed_coefficient_errors_include_missing_and_extra_terms(self):
        reference = [{"name": "u_xx", "coefficient": 1.0},
                     {"name": "u*u_x", "coefficient": -2.0}]
        found = [{"name": "u_xx", "coefficient": 0.5},
                 {"name": "u", "coefficient": 1.0}]
        score = experiment.reference_metrics(found, reference)
        self.assertFalse(score["exact_support"])
        self.assertEqual(score["precision"], 0.5)
        self.assertEqual(score["recall"], 0.5)
        self.assertEqual(score["f1"], 0.5)
        self.assertEqual(score["missing_terms"], ["u*u_x"])
        self.assertEqual(score["spurious_terms"], ["u"])
        self.assertAlmostEqual(score["coefficient_relative_l2"], np.sqrt(5.25 / 5.0))
        sign_reversed = experiment.reference_metrics([
            {"name": "u_xx", "coefficient": -1.0},
            {"name": "u*u_x", "coefficient": 2.0}], reference)
        self.assertTrue(sign_reversed["exact_support"])
        self.assertEqual(sign_reversed["f1"], 1.0)
        self.assertAlmostEqual(sign_reversed["coefficient_relative_l2"], 2.0)

    def test_empty_discovery_is_not_a_success(self):
        score = experiment.reference_metrics([], [{"name": "u_xx", "coefficient": 0.1}])
        self.assertFalse(score["exact_support"])
        self.assertEqual(score["f1"], 0.0)
        self.assertEqual(score["coefficient_relative_l2"], 1.0)

    def test_duplicate_reference_names_are_rejected_even_when_first_is_zero(self):
        with self.assertRaises(ValueError):
            experiment.reference_metrics([], [
                {"name": "u", "coefficient": 0.0},
                {"name": "u", "coefficient": 1.0},
            ])

    def test_forecast_metric_excludes_initial_and_handles_zero_denominator(self):
        observed = np.ones((3, 4))
        predicted = observed.copy()
        predicted[0] = 1000.0
        score = experiment.forecast_metrics(predicted, observed)
        self.assertEqual(score["rmse"], 0.0)
        self.assertEqual(score["relative_l2"], 0.0)
        self.assertEqual(score["scored_frames"], 2)
        self.assertEqual(experiment.forecast_metrics(np.zeros((3, 4)), np.zeros((3, 4)))["relative_l2"], 0.0)
        self.assertIsNone(experiment.forecast_metrics(np.ones((3, 4)), np.zeros((3, 4)))["relative_l2"])
        with self.assertRaises(ValueError):
            experiment.forecast_metrics(np.empty((3, 0)), np.empty((3, 0)))


if __name__ == "__main__":
    unittest.main()
