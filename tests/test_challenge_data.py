"""Manufactured data only: no public Allen--Cahn field is read here."""
from contextlib import contextmanager
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from scipy.io import savemat

from challenge_study import data


def manufactured(frames=40, points=32, offset=0.):
    x = np.linspace(0., 2*np.pi, points, endpoint=False)
    t = np.linspace(0., .4, frames)
    u = offset + .2 + np.cos(x)[None, :] * np.exp(-t[:, None])
    return {"x": x, "t": t, "u": u, "boundary": "periodic",
            "reference_coefficients": {"fake": 123}, "clean_future": "not an observation"}


def metadata_for(content):
    return {**data.FILE, "bytes": len(content),
            "sha256": hashlib.sha256(content).hexdigest(),
            "git_blob_sha1": hashlib.sha1(b"blob " + str(len(content)).encode() + b"\0" + content).hexdigest()}


@contextmanager
def manufactured_mat(root, *, complex_field=False, shifted_x=False, broken_time=False):
    path = Path(root)/"raw"/"AC.mat"
    path.parent.mkdir(parents=True)
    x = -1. + np.arange(512)/256.
    t = np.arange(201)*.005
    field = x[:, None]*.2 + t[None, :] + .7
    if complex_field:
        field = field + .25j*np.ones_like(field)
    if shifted_x:
        x = x + .001
    if broken_time:
        t[11] += .001
    savemat(path, {"x": x[None], "tt": t[None], "uu": field}, do_compression=True)
    with patch.object(data, "FILE", metadata_for(path.read_bytes())):
        yield field


class ChallengeDataTests(unittest.TestCase):
    def test_fixed_ac_orientation_and_disjoint_rows_are_preserved(self):
        with tempfile.TemporaryDirectory() as tmp, manufactured_mat(tmp) as field:
            record = data.prepare_ac(tmp)
            self.assertTrue(record["complete_mat_field_mechanically_decoded"])
            self.assertTrue(record["final_numeric_values_entered_preparation_memory"])
            self.assertFalse(record["training_or_selection_performed"])
            self.assertFalse(record["spatial_endpoint_removed"])
            self.assertEqual(record["records"]["training"]["frames"], 120)
            self.assertEqual(record["records"]["development"]["frames"], 40)
            self.assertEqual(record["records"]["final"]["frames"], 41)
            for role, (start, stop) in data.AC_ROLES.items():
                observed = data.read_clean(tmp, "allen_cahn", role)
                np.testing.assert_array_equal(observed["u"], field[:, start:stop].T)
                np.testing.assert_array_equal(observed["t"], np.arange(201)[start:stop]*.005)
                coarse = data.corrupt(observed, "allen_cahn", role, 0, "coarse", 0., 0, 1.)
                self.assertGreaterEqual(len(coarse["t"]), 16)
                self.assertEqual(len(coarse["x"]), 256)
            with self.assertRaises(ValueError):
                data.prepare_ac(tmp)

    def test_author_real_component_rule_accepts_and_audits_non_roundoff_imaginary_part(self):
        with tempfile.TemporaryDirectory() as tmp, manufactured_mat(tmp, complex_field=True) as field:
            record = data.prepare_ac(tmp)
            self.assertTrue(record["imaginary_component_present"])
            self.assertEqual(record["imaginary_max_abs"], .25)
            self.assertGreater(record["imaginary_relative_l2"], .1)
            np.testing.assert_array_equal(data.read_clean(tmp, "allen_cahn", "training")["u"],
                                          field.real[:, :120].T)

    def test_wrong_coordinates_fail_before_any_role_files(self):
        for kwargs in ({"shifted_x": True}, {"broken_time": True}):
            with self.subTest(kwargs=kwargs), tempfile.TemporaryDirectory() as tmp, manufactured_mat(tmp, **kwargs):
                with self.assertRaises(ValueError):
                    data.prepare_ac(tmp)
                self.assertFalse(list(Path(tmp).glob("clean/**/*.npz")))

    def test_raw_hash_mismatch_rejects_before_mat_decode(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"raw"/"AC.mat"
            path.parent.mkdir()
            path.write_bytes(b"not the immutable source")
            with patch.object(data, "loadmat", side_effect=AssertionError("Must not decode")):
                with self.assertRaises(ValueError):
                    data.prepare_ac(tmp)

    def test_mock_fixed_download_is_bounded_verified_and_idempotent(self):
        content = b"manufactured immutable file bytes"
        with tempfile.TemporaryDirectory() as tmp, patch.object(data, "FILE", metadata_for(content)), \
             patch.object(data, "CACHED_AC", Path(tmp)/"missing-cache"), \
             patch.object(data.urllib.request, "urlopen", return_value=io.BytesIO(content)) as request:
            first = data.download_ac(tmp)
            self.assertFalse(first["numeric_arrays_decoded"])
            self.assertFalse(first["author_code_executed"])
            self.assertEqual(request.call_args.kwargs["timeout"], 30)
            second = data.download_ac(tmp)
            self.assertEqual(first, second)
            self.assertEqual(request.call_count, 1)
            self.assertEqual((Path(tmp)/"raw"/"AC.mat").read_bytes(), content)

    def test_bad_cached_or_oversized_download_never_writes_unverified_destination(self):
        content = b"valid manufactured bytes"
        with tempfile.TemporaryDirectory() as tmp, patch.object(data, "FILE", metadata_for(content)), \
             patch.object(data, "CACHED_AC", Path(tmp)/"bad-cache"):
            data.CACHED_AC.write_bytes(b"corrupt")
            with patch.object(data.urllib.request, "urlopen", side_effect=AssertionError("Must not retry bad cache")):
                with self.assertRaises(ValueError):
                    data.download_ac(tmp)
            self.assertFalse((Path(tmp)/"raw"/"AC.mat").exists())
        with tempfile.TemporaryDirectory() as tmp, patch.object(data, "FILE", metadata_for(content)), \
             patch.object(data, "CACHED_AC", Path(tmp)/"missing-cache"), \
             patch.object(data.urllib.request, "urlopen", return_value=io.BytesIO(content+b"extra")):
            with self.assertRaises(ValueError):
                data.download_ac(tmp)
            self.assertFalse((Path(tmp)/"raw"/"AC.mat").exists())

    def test_coarse_noise_is_exactly_nested_and_metadata_does_not_escape(self):
        clean = manufactured()
        before = clean["u"].copy()
        args = (clean, "low_viscosity_burgers", "training", 0)
        full = data.corrupt(*args, "full", .01, 17, .7)
        coarse = data.corrupt(*args, "coarse", .01, 17, .7)
        self.assertEqual(set(full), {"x", "t", "u", "boundary"})
        np.testing.assert_array_equal(coarse["u"], full["u"][::2, ::2])
        np.testing.assert_array_equal(coarse["t"], full["t"][::2])
        np.testing.assert_array_equal(coarse["x"], full["x"][::2])
        np.testing.assert_array_equal(clean["u"], before)
        self.assertFalse(np.array_equal(full["u"][0], clean["u"][0]))

    def test_seed_sequence_matches_formula_and_roles_trajectories_are_distinct(self):
        clean = manufactured()
        expected_rng = np.random.default_rng(np.random.SeedSequence([55000, 2, 0, 1, 29]))
        expected = clean["u"] + expected_rng.normal(0., .01*.9, clean["u"].shape)
        value = data.corrupt(clean, "ks_unstable", "training", 1, "full", .01, 29, .9)
        np.testing.assert_array_equal(value["u"], expected)
        again = data.corrupt(clean, "ks_unstable", "training", 1, "full", .01, 29, .9)
        np.testing.assert_array_equal(again["u"], expected)
        other_role = data.corrupt(clean, "ks_unstable", "development", 1, "full", .01, 29, .9)
        other_initial = data.corrupt(clean, "ks_unstable", "training", 0, "full", .01, 29, .9)
        other_seed = data.corrupt(clean, "ks_unstable", "training", 1, "full", .01, 17, .9)
        for other in (other_role, other_initial, other_seed):
            self.assertFalse(np.array_equal(value["u"], other["u"]))

    def test_zero_noise_copies_and_training_scale_equalizes_trajectory_weight(self):
        first, second = manufactured(), manufactured(frames=80, offset=.4)
        scale = data.training_scale([first, second])
        expected = np.sqrt((np.mean(first["u"]**2)+np.mean(second["u"]**2))/2)
        self.assertAlmostEqual(scale, expected, places=15)
        clean = data.corrupt(first, "allen_cahn", "final", 0, "full", 0., 0, scale)
        np.testing.assert_array_equal(clean["u"], first["u"])
        clean["u"][0, 0] += 3.
        self.assertNotEqual(clean["u"][0, 0], first["u"][0, 0])

    def test_clean_role_storage_strips_oracle_fields_and_prevents_overwrite(self):
        clean = manufactured()
        with tempfile.TemporaryDirectory() as tmp:
            record = data.write_clean(tmp, "ks_unstable", "training", 2, clean)
            with np.load(Path(tmp)/record["file"], allow_pickle=False) as z:
                self.assertEqual(set(z.files), {"x", "t", "u"})
            value = data.read_clean(tmp, "ks_unstable", "training", 2)
            self.assertEqual(set(value), {"x", "t", "u", "boundary"})
            with self.assertRaises(ValueError):
                data.write_clean(tmp, "ks_unstable", "training", 2, clean)

    def test_undeclared_noise_identity_and_invalid_fields_are_rejected(self):
        clean = manufactured()
        args = dict(clean=clean, family="ks_unstable", role="training", index=0,
                    sampling="full", noise_level=.01, noise_seed=17, training_rms=.8)
        bad = [{"family": "other"}, {"role": "calibration"}, {"index": True}, {"index": 3},
               {"sampling": "adaptive"}, {"noise_level": .02}, {"noise_level": float("nan")},
               {"noise_seed": 2}, {"noise_seed": True}, {"noise_level": 0.},
               {"training_rms": float("nan")}, {"training_rms": -.1}]
        for update in bad:
            with self.subTest(update=update), self.assertRaises(ValueError):
                data.corrupt(**{**args, **update})
        for wrong in ({**clean, "u": clean["u"].astype(complex)},
                      {**clean, "boundary": "dirichlet"},
                      {**clean, "u": np.full_like(clean["u"], np.nan)}):
            with self.assertRaises(ValueError):
                data.corrupt(**{**args, "clean": wrong})
        with self.assertRaises(ValueError):
            data.training_scale([])


if __name__ == "__main__":
    unittest.main()
