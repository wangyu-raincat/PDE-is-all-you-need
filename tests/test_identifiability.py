"""Manufactured checks for training-only parameter-identifiability diagnostics.

No real measurement files are read. Profile loss and delete-block sensitivity
are diagnostics, not confidence intervals or new model selection.
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

from pde_discovery import heat_identifiability as diagnostics


def manufactured_trajectory():
    increments = 0.06 * (1.0 + 0.2 * np.sin(np.arange(80)))
    t = np.r_[0.0, np.cumsum(increments)] + 11.0
    dx, D, h = 0.01, 7e-5, 0.06
    modes = np.arange(1, 5)
    eigenvalues = -4 * np.sin(modes * np.pi / 10.0) ** 2 / dx**2
    vectors = np.sin(np.outer(np.arange(1, 5), modes) * np.pi / 5.0)
    interior = (
        np.exp(np.outer(t - t[0], D * eigenvalues - h))
        * np.array([4.0, 1.3, 0.8, 0.4])
    ) @ vectors.T
    u = np.column_stack([np.zeros(len(t)), interior, np.zeros(len(t))])
    return t, u, dx, {"D": D, "h": h, "c": 0.0}


class IdentifiabilityDiagnosticTests(unittest.TestCase):
    def test_masked_fit_keeps_original_time_initial_state_and_full_boundaries(self):
        t, u, dx, reference = manufactured_trajectory()
        mask = np.ones(len(t), dtype=bool)
        mask[:20] = False
        mask[35:48] = False
        original_prepare = diagnostics._prepare
        prepared_calls = []

        def inspect_original_inputs(time, boundary, initial, spacing):
            prepared_calls.append(1)
            np.testing.assert_array_equal(time, t)
            np.testing.assert_array_equal(boundary, u[:, [0, 5]])
            np.testing.assert_array_equal(initial, u[0, 1:5])
            self.assertEqual(spacing, dx)
            return original_prepare(time, boundary, initial, spacing)

        with patch.object(diagnostics, "_prepare", side_effect=inspect_original_inputs):
            fitted = diagnostics.fit_masked(t, u, dx, "D+h", mask)
        self.assertTrue(prepared_calls)
        self.assertTrue(fitted["fit_success"])
        self.assertAlmostEqual(fitted["parameters"]["D"], reference["D"], delta=reference["D"] * 0.01)
        self.assertAlmostEqual(fitted["parameters"]["h"], reference["h"], delta=reference["h"] * 0.01)

    def test_excluded_future_interiors_do_not_change_masked_fit(self):
        t, u, dx, _ = manufactured_trajectory()
        mask = np.ones(len(t), dtype=bool)
        mask[1:22] = False
        mask[40:51] = False
        altered = u.copy()
        excluded_future = (~mask) & (np.arange(len(t)) > 0)
        altered[excluded_future, 1:5] += 1e5
        before = u.copy()
        original = diagnostics.fit_masked(t, u, dx, "D+h", mask)
        changed = diagnostics.fit_masked(t, altered, dx, "D+h", mask)
        self.assertTrue(original["fit_success"])
        self.assertTrue(changed["fit_success"])
        self.assertEqual(original["parameters"], changed["parameters"])
        np.testing.assert_array_equal(u, before)

    def test_fixed_diffusion_profile_has_minimum_at_manufactured_truth(self):
        t, u, dx, reference = manufactured_trajectory()
        profile = diagnostics.profile_diffusion(t, u, dx, "D+h", reference)
        records = profile["points"]
        self.assertEqual(len(records), len(diagnostics.PROFILE_MULTIPLIERS))
        scored = [r for r in records if r["fit_success"]]
        best = min(scored, key=lambda r: r["train_rmse_K"])
        self.assertAlmostEqual(best["fixed_D_m2_s"], reference["D"], delta=1e-12)
        self.assertLess(best["train_rmse_K"], 1e-6)
        self.assertGreater(scored[0]["train_rmse_K"], best["train_rmse_K"] + 1e-3)
        self.assertGreater(scored[-1]["train_rmse_K"], best["train_rmse_K"] + 1e-3)
        self.assertFalse(profile["is_confidence_interval"])

    def test_five_block_deletions_only_mask_losses_and_cover_each_future_row_once(self):
        t, u, dx, reference = manufactured_trajectory()
        calls = []
        original_fit = diagnostics.fit_masked

        def inspect_mask(time, observed, spacing, model_name, observation_mask=None, **kwargs):
            np.testing.assert_array_equal(time, t)
            np.testing.assert_array_equal(observed, u)
            self.assertEqual(spacing, dx)
            if observation_mask is not None:
                calls.append(np.asarray(observation_mask, bool).copy())
            return original_fit(time, observed, spacing, model_name, observation_mask, **kwargs)

        with patch.object(diagnostics, "fit_masked", side_effect=inspect_mask):
            result = diagnostics.block_delete_diagnostics(t, u, dx, "D+h", reference)
        deleted = [~mask[1:] for mask in calls if (~mask[1:]).any()]
        self.assertEqual(len(deleted), diagnostics.BLOCK_COUNT)
        np.testing.assert_array_equal(np.sum(deleted, axis=0), np.ones(len(t) - 1))
        for block in deleted:
            indices = np.flatnonzero(block)
            self.assertTrue(len(indices))
            np.testing.assert_array_equal(indices, np.arange(indices[0], indices[-1] + 1))
        self.assertFalse(result["is_confidence_interval"])
        self.assertFalse(result["used_for_model_selection"])

    def test_zero_excitation_is_insufficient_not_a_unique_equation(self):
        t = np.linspace(0.0, 4.0, 41)
        reference = {"D": 7e-5, "h": 0.06, "c": 0.0}
        report = diagnostics.diagnose(t, np.zeros((len(t), 6)), 0.01, "D+h", reference)
        self.assertEqual(report["status"], "insufficient")
        self.assertFalse(report["used_for_model_selection"])
        self.assertFalse(report["is_confidence_interval"])

    def test_insufficient_loss_rows_in_one_block_preserve_other_block_diagnostics(self):
        t, u, dx, reference = manufactured_trajectory()
        selection = np.r_[np.arange(7), len(t) - 1]
        # Six non-initial observations are concentrated in the first time
        # block. Deleting them leaves too little loss data for that fit only.
        result = diagnostics.block_delete_diagnostics(t[selection], u[selection], dx, "D+h", reference)
        self.assertEqual(len(result["blocks"]), diagnostics.BLOCK_COUNT)
        self.assertFalse(result["blocks"][0]["fit_success"])
        self.assertTrue(result["blocks"][0]["failure_reason"])
        self.assertTrue(result["blocks"][-1]["fit_success"])
        for block in result["blocks"][1:-1]:
            self.assertEqual(block["deleted_rows"], 0)
            self.assertFalse(block["fit_success"])
            self.assertTrue(block["failure_reason"])
        self.assertEqual(result["successful_fits"], 1)
        self.assertEqual(result["status"], "inconclusive")

    def test_invalid_masks_and_nonfinite_measurements_are_rejected(self):
        t, u, dx, reference = manufactured_trajectory()
        for mask in [np.zeros(len(t), dtype=bool), np.ones(len(t) - 1, dtype=bool),
                     np.ones((len(t), 4), dtype=bool)]:
            with self.subTest(mask_shape=mask.shape):
                with self.assertRaises(ValueError):
                    diagnostics.fit_masked(t, u, dx, "D+h", mask)
        bad = u.copy()
        bad[3, 2] = np.nan
        with self.assertRaises(ValueError):
            diagnostics.diagnose(t, bad, dx, "D+h", reference)
        before = deepcopy(reference)
        with self.assertRaises(ValueError):
            diagnostics.fit_masked(t, u, dx, "D+h", fixed_D=-1e-5)
        self.assertEqual(reference, before)


class IdentifiabilityRunnerTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="pde-identifiability-fixture-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.data_root, self.model_root = self.root / "data", self.root / "model"
        self.case = "50ss"
        data_dir, model_dir = self.data_root / self.case, self.model_root / self.case
        data_dir.mkdir(parents=True)
        model_dir.mkdir(parents=True)
        self.t, self.u, dx, self.parameters = manufactured_trajectory()
        train = data_dir / "training.csv"
        np.savetxt(train, np.column_stack([self.t, self.u]), delimiter=",")
        entries = {"training": {"file": train.name,
                                "sha256": hashlib.sha256(train.read_bytes()).hexdigest()}}
        for role in ["development", "final"]:
            path = data_dir / f"{role}.csv"
            path.write_text("These forbidden future numeric values must never be parsed.\n", encoding="utf-8")
            entries[role] = {"file": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        manifest_path = data_dir / "manifest.json"
        manifest_path.write_text(json.dumps({
            "spacing_m": dx, "csv_format": {"delimiter": ",", "skiprows": 0},
            "splits": entries,
        }), encoding="utf-8")
        self.model_path = model_dir / "frozen_model.json"
        self.model_path.write_text(json.dumps({
            "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            "selected": {"model_name": "D+h", "fit_success": True, "parameters": self.parameters},
        }), encoding="utf-8")
        (self.model_root / "freeze_index.json").write_text(json.dumps({
            "records": [{"case": self.case,
                         "frozen_model_sha256": hashlib.sha256(self.model_path.read_bytes()).hexdigest()}],
        }), encoding="utf-8")
        self.fake_diagnostic = {
            "model_name": "D+h", "status": "diagnostic_complete",
            "reference": {"parameters": self.parameters, "train_rmse_K": 0.0,
                          "jacobian_rank": 2, "parameter_count": 2},
            "profile": {"near_optimal_D_range_m2_s": [self.parameters["D"]] * 2,
                        "successful_points": 9, "near_optimal_touches_grid_edge": False,
                        "near_optimal_touches_parameter_bound": False},
            "block_deletion": {"parameter_ranges": {"D": {"min": self.parameters["D"],
                                                          "max": self.parameters["D"]}},
                               "successful_fits": 5},
        }

    def run_diagnostics(self):
        with patch.object(diagnostics, "CASES", (self.case,)), \
             patch.object(diagnostics, "_plot"), redirect_stdout(io.StringIO()):
            return diagnostics.run_diagnostics(self.root / "output", self.data_root, self.model_root)

    def test_runner_reads_only_original_training_and_preserves_frozen_model(self):
        original_loader = diagnostics.load_measurements
        before = self.model_path.read_bytes()
        reads = []

        def only_training(data_dir, entry, csv_format):
            reads.append(entry["file"])
            self.assertEqual(entry["file"], "training.csv")
            return original_loader(data_dir, entry, csv_format)

        def inspect_diagnostic_input(t, u, dx, model_name, parameters):
            np.testing.assert_allclose(t, self.t)
            np.testing.assert_allclose(u, self.u)
            self.assertEqual(model_name, "D+h")
            self.assertEqual(parameters, self.parameters)
            return deepcopy(self.fake_diagnostic)

        with patch.object(diagnostics, "load_measurements", side_effect=only_training), \
             patch.object(diagnostics, "diagnose", side_effect=inspect_diagnostic_input):
            summary = self.run_diagnostics()
        self.assertEqual(reads, ["training.csv"])
        self.assertEqual(summary["numerical_data_roles_loaded"], ["training"])
        self.assertFalse(summary["used_for_model_selection"])
        self.assertEqual(before, self.model_path.read_bytes())

    def test_modified_frozen_reference_is_rejected_before_training_load(self):
        self.model_path.write_text(self.model_path.read_text() + "\n", encoding="utf-8")
        with patch.object(diagnostics, "load_measurements") as load, \
             patch.object(diagnostics, "diagnose") as diagnose:
            summary = self.run_diagnostics()
        load.assert_not_called()
        diagnose.assert_not_called()
        self.assertEqual(summary["records"][0]["status"], "inconclusive")
        self.assertTrue(summary["records"][0]["failure_reason"])


if __name__ == "__main__":
    unittest.main()
