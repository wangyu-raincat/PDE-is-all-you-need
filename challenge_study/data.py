"""Immutable Allen--Cahn ingestion and prespecified observation corruption.

This module contains no reference PDE or model-selection logic. The source MAT
must be mechanically decoded in full before its fixed temporal roles can be
written; that operation is explicitly disclosed rather than called unread data.
Only x/t/u observation dictionaries leave the corruption interface.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import urllib.request

import numpy as np
from scipy.io import loadmat

DATA = Path(__file__).resolve().parent / "data"
CACHED_AC = Path("/workspace/pde-challenge-source-audit/source/main/Data/AC.mat")
FILE = {
    "repo": "maziarraissi/PINNs",
    "commit": "932f50a2d8ef4e80d1456bbae6887a73ff5166ef",
    "url": "https://raw.githubusercontent.com/maziarraissi/PINNs/932f50a2d8ef4e80d1456bbae6887a73ff5166ef/main/Data/AC.mat",
    "bytes": 559196,
    "sha256": "266e97eedd9ef543d7f6d370fa5f06388761aa7da2b97e476562dcf1d644a175",
    "git_blob_sha1": "cd18945f3896e0571c4e2ed7c8a6099762056cb7",
}
FAMILY_IDS = {"allen_cahn": 0, "low_viscosity_burgers": 1, "ks_unstable": 2}
ROLE_IDS = {"training": 0, "development": 1, "final": 2}
ROLE_COUNTS = {"training": 3, "development": 2, "final": 2}
AC_ROLES = {"training": (0, 120), "development": (120, 160), "final": (160, 201)}
NOISE_LEVELS = (0.0, 0.001, 0.01)
NOISE_SEEDS = (17, 29)
PROTOCOL = {
    "source": FILE,
    "ac_shape_space_time": [512, 201],
    "ac_real_component": "Author explicitly uses real(uu).T; no magnitude-based imaginary threshold.",
    "ac_roles": {key: list(value) for key, value in AC_ROLES.items()},
    "ac_full_matrix_mechanical_decode": True,
    "source_data_kind": "public numerical simulation; not measured data",
    "samplings": {"full": 1, "coarse": 2},
    "noise_levels": list(NOISE_LEVELS),
    "noise_seeds": list(NOISE_SEEDS),
    "clean_noise_seed": 0,
    "training_scale": "sqrt(mean(per-trajectory mean(u**2))) on full-grid clean training roles only",
    "fixed_noise_standard_deviation": "noise_level * training_rms, same scale for all roles",
    "seed_sequence": "[55000, family_id, role_id, trajectory_index, noise_seed]",
    "family_ids": FAMILY_IDS,
    "role_ids": ROLE_IDS,
    "sampling_order": "generate independent full-role noise, then slice time and space within the role",
    "coarse_is_nested_observation": True,
    "observation_interface": "x,t,u,boundary only; no clean future, exact derivatives, sigma or reference labels",
    "clean_future_use": "frozen-final scoring only",
}


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write_json(path, record):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def _verify_bytes(content):
    digest = hashlib.sha256(content).hexdigest()
    blob = hashlib.sha1(b"blob " + str(len(content)).encode("ascii") + b"\0" + content).hexdigest()
    if (len(content) != FILE["bytes"] or digest != FILE["sha256"]
            or blob != FILE["git_blob_sha1"]):
        raise ValueError("Allen--Cahn source bytes do not match the immutable manifest.")
    return {"bytes": len(content), "sha256": digest, "git_blob_sha1": blob}


def download_ac(data_root=DATA):
    """Cache verified original bytes without decoding any MAT variable."""
    root = Path(data_root)
    path = root / "raw" / "AC.mat"
    if path.exists():
        content = path.read_bytes()
        method = "existing verified destination"
    elif CACHED_AC.exists():
        content = CACHED_AC.read_bytes()
        method = "existing audited immutable cache"
    else:
        request = urllib.request.Request(FILE["url"], headers={"User-Agent": "PDE-challenge-validation/1.0"})
        with urllib.request.urlopen(request, timeout=30) as response:
            content = response.read(FILE["bytes"] + 1)
        method = "fixed-commit HTTPS download"
    digest = _verify_bytes(content)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            stream.write(content)
    record = {**FILE, **digest, "method": method, "path": "raw/AC.mat",
              "numeric_arrays_decoded": False, "author_code_executed": False}
    receipt = root / "ac_download_receipt.json"
    if receipt.exists():
        old = json.loads(receipt.read_text())
        # Acquisition method may differ on an idempotent cached invocation.
        if any(old.get(key) != record[key] for key in (*FILE, "numeric_arrays_decoded", "author_code_executed", "path")):
            raise ValueError("Existing Allen--Cahn receipt is inconsistent with immutable source.")
        return old
    _write_json(receipt, record)
    return record


def _identity(family, role, index):
    if family not in FAMILY_IDS or role not in ROLE_IDS:
        raise ValueError("Unknown clean-data family or role.")
    if isinstance(index, (bool, np.bool_)) or not isinstance(index, (int, np.integer)):
        raise ValueError("Trajectory index must be an integer.")
    count = 1 if family == "allen_cahn" else ROLE_COUNTS[role]
    if not 0 <= int(index) < count:
        raise ValueError("Trajectory index lies outside the fixed role allocation.")


def _validated(clean):
    if not isinstance(clean, dict) or not all(k in clean for k in ("x", "t", "u")):
        raise ValueError("Need a clean or observed x/t/u dictionary.")
    raw = [np.asarray(clean[key]) for key in ("x", "t", "u")]
    if any(np.iscomplexobj(value) for value in raw):
        raise ValueError("Observation coordinates and fields must be real-valued.")
    x, t, u = [np.asarray(value, float) for value in raw]
    if (x.ndim != 1 or t.ndim != 1 or len(x) < 16 or len(t) < 16
            or u.shape != (len(t), len(x)) or u.size > 3_000_000
            or not all(np.isfinite(v).all() for v in (x, t, u))
            or clean.get("boundary", "periodic") != "periodic"):
        raise ValueError("Need finite matching periodic fields with at least 16 points on each axis.")
    for values in (x, t):
        delta = np.diff(values)
        if np.any(delta <= 0) or not np.allclose(delta, delta[0], rtol=1e-6, atol=1e-12):
            raise ValueError("Clean-data coordinates must be increasing and uniform.")
    return x, t, u


def write_clean(root, family, role, index, clean):
    """Write one named clean role without coefficients or oracle features."""
    _identity(family, role, index)
    x, t, u = _validated(clean)
    root = Path(root)
    path = root / "clean" / family / role / f"{index}.npz"
    if path.exists():
        raise ValueError("Clean role exists; preserve it instead of silently overwriting.")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        np.savez_compressed(stream, x=x, t=t, u=u)
    return {"file": str(path.relative_to(root)), "sha256": _sha(path),
            "space_points": len(x), "frames": len(t),
            "time_range": [float(t[0]), float(t[-1])], "family": family,
            "role": role, "trajectory_index": int(index)}


def read_clean(root, family, role, index=0):
    _identity(family, role, index)
    path = Path(root) / "clean" / family / role / f"{index}.npz"
    with np.load(path, allow_pickle=False) as stored:
        clean = {key: stored[key].copy() for key in ("x", "t", "u")}
    _validated(clean)
    return {**clean, "boundary": "periodic"}


def prepare_ac(data_root=DATA):
    """Decode the entire fixed MAT mechanically and write disjoint time roles."""
    root = Path(data_root)
    record_path = root / "ac_preparation.json"
    if record_path.exists():
        raise ValueError("Allen--Cahn roles already prepared; do not overwrite first preparation.")
    raw_path = root / "raw" / "AC.mat"
    source_digest = _verify_bytes(raw_path.read_bytes())
    parsed = loadmat(raw_path, variable_names=["x", "tt", "uu"])
    if not all(key in parsed for key in ("x", "tt", "uu")):
        raise ValueError("Author MAT keys x, tt, uu are required.")
    x = np.asarray(parsed["x"]).reshape(-1)
    t = np.asarray(parsed["tt"]).reshape(-1)
    field = np.asarray(parsed["uu"])
    if (np.iscomplexobj(x) or np.iscomplexobj(t) or field.shape != (512, 201)
            or not all(np.isfinite(a).all() for a in (x, t, field))):
        raise ValueError("Allen--Cahn metadata must match fixed real coordinates and finite 512x201 field.")
    expected_x = -1.0 + np.arange(512)/256.0
    expected_t = np.arange(201)*0.005
    if (x.shape != (512,) or t.shape != (201,)
            or not np.allclose(x, expected_x, rtol=0., atol=1e-12)
            or not np.allclose(t, expected_t, rtol=0., atol=1e-12)):
        raise ValueError("Allen--Cahn coordinates differ from audited nonduplicated periodic grid.")
    imaginary_max = float(np.max(np.abs(field.imag))) if np.iscomplexobj(field) else 0.0
    common = max(1.0, float(np.max(np.abs(field.real))), imaginary_max)
    real_norm = float(np.linalg.norm(field.real/common))
    imaginary_norm = float(np.linalg.norm(field.imag/common)) if np.iscomplexobj(field) else 0.0
    u = np.asarray(field.real.T, float)
    _validated({"x": x, "t": t, "u": u})
    records = {}
    for role, (start, stop) in AC_ROLES.items():
        records[role] = write_clean(root, "allen_cahn", role, 0,
            {"x": x, "t": t[start:stop], "u": u[start:stop]})
    result = {"source": source_digest, "repo": FILE["repo"], "commit": FILE["commit"],
              "records": records, "orientation": "uu[space,time].real.T",
              "imaginary_component_present": bool(np.iscomplexobj(field)),
              "imaginary_max_abs": imaginary_max,
              "imaginary_relative_l2": imaginary_norm/real_norm if real_norm else None,
              "imaginary_component_treatment": "author-declared real field, not a fitted roundoff tolerance",
              "complete_mat_field_mechanically_decoded": True,
              "final_numeric_values_entered_preparation_memory": True,
              "training_or_selection_performed": False, "author_code_executed": False,
              "role_row_bounds": {key: list(value) for key, value in AC_ROLES.items()},
              "spatial_endpoint_removed": False, "interpolation_performed": False}
    _write_json(record_path, result)
    return result


def training_scale(clean_training):
    """Benchmark corruption scale only; never exposed as a fitting input."""
    if not isinstance(clean_training, (list, tuple)) or not clean_training:
        raise ValueError("Need at least one clean training trajectory.")
    arrays = [_validated(item)[2] for item in clean_training]
    maximum = max(float(np.max(np.abs(a))) for a in arrays)
    if maximum == 0.0:
        return 0.0
    return float(maximum*np.sqrt(np.mean([np.mean((a/maximum)**2) for a in arrays])))


def corrupt(clean, family, role, index, sampling, noise_level, noise_seed, training_rms):
    """Construct only the observed field; neither clean future nor sigma escapes.

    Sampling is omitted from the deterministic seed so coarse observations are
    exact ::2 slices of their matching full-grid noisy observations. Roles and
    trajectory indices have distinct independent random streams.
    """
    _identity(family, role, index)
    x, t, u = _validated(clean)
    if sampling not in ("full", "coarse"):
        raise ValueError("Sampling must be full or coarse.")
    if isinstance(noise_level, (bool, np.bool_)) or noise_level not in NOISE_LEVELS:
        raise ValueError("Noise level must be one of the fixed benchmark levels.")
    if isinstance(noise_seed, (bool, np.bool_)) or not isinstance(noise_seed, (int, np.integer)):
        raise ValueError("Noise seed must be a fixed integer.")
    if (noise_level == 0 and noise_seed != 0) or (noise_level != 0 and noise_seed not in NOISE_SEEDS):
        raise ValueError("Use seed zero for clean data and a declared independent seed for noise.")
    if (isinstance(training_rms, (bool, np.bool_)) or np.ndim(training_rms) != 0
            or not np.isfinite(training_rms) or training_rms < 0):
        raise ValueError("Training RMS must be a finite nonnegative scalar.")
    sigma = float(noise_level)*float(training_rms)
    if not np.isfinite(sigma):
        raise ValueError("Noise standard deviation must be finite.")
    noisy = u.copy()
    if noise_level:
        sequence = np.random.SeedSequence([55000, FAMILY_IDS[family], ROLE_IDS[role], int(index), int(noise_seed)])
        noisy += np.random.default_rng(sequence).normal(0.0, sigma, size=u.shape)
    stride = 1 if sampling == "full" else 2
    observed = {"x": x[::stride].copy(), "t": t[::stride].copy(),
                "u": noisy[::stride, ::stride].copy(), "boundary": "periodic"}
    _validated(observed)
    return observed
