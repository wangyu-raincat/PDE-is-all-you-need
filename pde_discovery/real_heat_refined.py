"""Physics-constrained trajectory calibration on the original six sensors.

This is a prespecified four-model comparison, NOT L1 equation discovery.
Training observes all six sensors, but every forward rollout accepts only the
two boundary traces and the initial four interior values. No test data is read.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares
from scipy.signal import lfilter

from .real_heat import evaluate


MODEL_NAMES = ("D", "D+h", "D+c", "D+h+c")
D_BOUNDS = (1e-8, 1e-3)
H_BOUNDS = (0.0, 1.0)
C_BOUNDS = (-1.0, 1.0)
H_SCALE = 0.01
C_SCALE = 0.1
STARTS = ((1e-6, 0.001, 0.0), (1e-5, 0.01, 0.0), (1e-4, 0.1, 0.0))
MAX_NFEV = 250
FIT_TOLERANCE = 1e-9
DT_GROUP_RTOL = 1e-10
FIT_PROTOCOL = {
    "method": "prespecified physical supports with training-trajectory nonlinear least squares",
    "is_l1": False,
    "equation_family": "theta_t = D theta_xx - h theta + c",
    "model_names": list(MODEL_NAMES),
    "physical_prior": "D > 0; h >= 0; no advection or quadratic reaction; source c optional",
    "parameter_bounds": {"D_m2_s": list(D_BOUNDS), "h_per_s": list(H_BOUNDS),
                         "c_K_s": list(C_BOUNDS)},
    "initial_values_D_h_c": [list(s) for s in STARTS],
    "optimizer_coordinates": "log(D), h/0.01, c/0.1, with absent parameters omitted",
    "optimizer": "scipy.optimize.least_squares, trf, linear loss, 2-point Jacobian",
    "max_nfev_per_start": MAX_NFEV,
    "ftol": FIT_TOLERANCE, "xtol": FIT_TOLERANCE, "gtol": FIT_TOLERANCE,
    "training_residual": "four interior rollout errors excluding initial frame, divided by sqrt(count)",
    "start_selection": "lowest training RMSE among successful starts; no development or final input",
    "failed_start_handling": "record exception or failure and continue remaining fixed starts",
    "failed_support_handling": "retain an explicit failed candidate and continue other fixed supports",
    "propagator": "four-point sine eigenmodes with piecewise-linear measured boundary forcing",
    "time_step_group_relative_tolerance": DT_GROUP_RTOL,
    "time_step_group_representative": "mean original dt within each numerically equal contiguous block",
    "spatial_observation_count": 6, "interior_state_count": 4,
    "spatial_upsampling": False, "data_smoothing": "none",
    "parameter_boundary_hit_tolerance": "1e-5 of the optimizer-coordinate bound range",
    "scope": "coarse-grid effective parameters; not a proof of a unique continuous PDE",
}

_INDEX = np.arange(1, 5, dtype=float)
_Q = np.sqrt(2.0 / 5.0) * np.sin(np.pi * _INDEX[:, None] * _INDEX[None, :] / 5.0)
_LAPLACIAN_EIGENVALUES = -4.0 * np.sin(np.pi * _INDEX / 10.0)**2


def _prepare(t, boundaries, u0, dx):
    t, boundaries, u0 = np.asarray(t, float), np.asarray(boundaries, float), np.asarray(u0, float)
    dx = float(dx)
    if (t.ndim != 1 or not 2 <= len(t) <= 200000 or not np.isfinite(t).all()
            or np.any(np.diff(t) <= 0) or not np.isfinite(dx) or dx <= 0):
        raise ValueError("需有限严格递增时间坐标与正的传感器间距。")
    if (boundaries.shape != (len(t), 2) or u0.shape != (4,)
            or not np.isfinite(boundaries).all() or not np.isfinite(u0).all()):
        raise ValueError("预测仅接受 N×2 边界与4个初始内点；不得传入未来内点。")
    dt = np.diff(t)
    changes = np.flatnonzero(~np.isclose(dt[1:], dt[:-1], rtol=DT_GROUP_RTOL, atol=0.0)) + 1
    starts = np.r_[0, changes]
    ends = np.r_[changes, len(dt)]
    blocks = [(int(a), int(b), float(np.mean(dt[a:b]))) for a, b in zip(starts, ends)]
    boundary_modes = boundaries[:, :1] * _Q[0] + boundaries[:, 1:] * _Q[-1]
    return len(t), dx, boundary_modes, u0 @ _Q, blocks


def _advance(prepared, D: float, h: float, c: float):
    count, dx, boundary_modes, initial_modes, blocks = prepared
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        rate = D / dx**2
        eigenvalues = rate * _LAPLACIAN_EIGENVALUES - h
        forcing = rate * boundary_modes + c * _Q.sum(axis=0)
    if not np.isfinite(eigenvalues).all() or not np.isfinite(forcing).all():
        raise ValueError("系数或测点间距导致空间算子溢出。")
    modes = np.empty((count, 4))
    modes[0] = initial_modes
    for a, b, delta in blocks:
        with np.errstate(over="ignore", under="ignore"):
            z = eigenvalues * delta
            decay = np.exp(z)
        k0, k1 = np.empty(4), np.empty(4)
        small = np.abs(z) < 1e-4
        zs = z[small]
        k0[small] = delta * (1 + zs/2 + zs**2/6 + zs**3/24 + zs**4/120)
        k1[small] = delta * (0.5 + zs/6 + zs**2/24 + zs**3/120 + zs**4/720)
        large = ~small
        k0[large] = np.expm1(z[large]) / eigenvalues[large]
        k1[large] = (k0[large] / delta - 1) / eigenvalues[large]
        inputs = k0 * forcing[a:b] + k1 * (forcing[a+1:b+1] - forcing[a:b])
        for mode in range(4):
            values, _ = lfilter([1.0], [1.0, -decay[mode]], inputs[:, mode],
                                zi=[decay[mode] * modes[a, mode]])
            modes[a+1:b+1, mode] = values
    result = modes @ _Q.T
    if not np.isfinite(result).all():
        raise ValueError("预测输出溢出。")
    return result


def _parameters_from_terms(terms):
    coefficients = {"1": 0.0, "theta": 0.0, "theta_xx": 0.0}
    for term in terms:
        if not isinstance(term, dict) or term.get("name") not in coefficients:
            raise ValueError("改进模型仅允许 1、theta、theta_xx 三种项。")
        value = float(term["coefficient"])
        if not np.isfinite(value):
            raise ValueError("模型系数必须有限。")
        coefficients[term["name"]] += value
    D, h, c = coefficients["theta_xx"], -coefficients["theta"], coefficients["1"]
    if D <= 0 or h < 0:
        raise ValueError("物理先验要求 D > 0、h >= 0。")
    return D, h, c


def predict(terms, t, boundaries, u0, dx) -> np.ndarray:
    """Return N×4 conditional predictions; never accepts future interior data."""
    D, h, c = _parameters_from_terms(terms)
    return _advance(_prepare(t, boundaries, u0, dx), D, h, c)


def _terms(D, h, c, model_name):
    result = []
    if "c" in model_name:
        result.append({"name": "1", "coefficient": float(c)})
    if "h" in model_name:
        result.append({"name": "theta", "coefficient": float(-h)})
    result.append({"name": "theta_xx", "coefficient": float(D)})
    return result


def _equation(terms):
    parts = []
    for term in terms:
        value, name = term["coefficient"], term["name"]
        part = f"{abs(value):.9g}" + ("" if name == "1" else f" {name}")
        parts.append(("- " if value < 0 else "+ ") + part)
    return "theta_t = " + " ".join(parts).removeprefix("+ ")


def _failed_candidate(model_name, reason, runs=None):
    return {
        "candidate_id": "trajectory-" + model_name.replace("+", "-"),
        "model_name": model_name, "equation": "拟合失败", "terms": [],
        "support_size": 1 + int("h" in model_name) + int("c" in model_name),
        "fit_success": False, "failure_reason": str(reason),
        "train_score": None, "parameters": None,
        "diagnostics": {"method": FIT_PROTOCOL["method"], "is_l1": False,
                        "failure_reason": str(reason), "starts": runs or [],
                        "future_interior_is_prediction_input": False},
    }


def fit_candidate(t, u, dx, model_name: str) -> dict:
    """Calibrate one declared support on one training trajectory only."""
    if model_name not in MODEL_NAMES:
        raise ValueError(f"model_name 必须属于 {MODEL_NAMES}。")
    u, t = np.asarray(u, float), np.asarray(t, float)
    if (t.ndim != 1 or u.shape != (len(t), 6) or len(t) < 8
            or not np.isfinite(u).all()):
        raise ValueError("拟合需要至少8帧的原始 N×6 温升轨迹。")
    prepared = _prepare(t, u[:, [0, 5]], u[0, 1:5], dx)
    target = u[1:, 1:5]
    normalization = np.sqrt(target.size)
    names = ["D"] + (["h"] if "h" in model_name else []) + (["c"] if "c" in model_name else [])
    lower = [np.log(D_BOUNDS[0])]
    upper = [np.log(D_BOUNDS[1])]
    if "h" in model_name:
        lower.append(H_BOUNDS[0]/H_SCALE); upper.append(H_BOUNDS[1]/H_SCALE)
    if "c" in model_name:
        lower.append(C_BOUNDS[0]/C_SCALE); upper.append(C_BOUNDS[1]/C_SCALE)
    lower, upper = np.array(lower), np.array(upper)

    def decode(vector):
        D, h, c = float(np.exp(vector[0])), 0.0, 0.0
        offset = 1
        if "h" in model_name:
            h = float(vector[offset] * H_SCALE); offset += 1
        if "c" in model_name:
            c = float(vector[offset] * C_SCALE)
        return D, h, c

    def residual(vector):
        return (_advance(prepared, *decode(vector))[1:] - target).ravel() / normalization

    runs, results = [], []
    for start_index, (start_D, start_h, start_c) in enumerate(STARTS):
        initial = [np.log(start_D)]
        if "h" in model_name:
            initial.append(start_h/H_SCALE)
        if "c" in model_name:
            initial.append(start_c/C_SCALE)
        try:
            result = least_squares(residual, initial, bounds=(lower, upper), method="trf",
                                   jac="2-point", loss="linear", max_nfev=MAX_NFEV,
                                   ftol=FIT_TOLERANCE, xtol=FIT_TOLERANCE, gtol=FIT_TOLERANCE)
            rmse = float(np.linalg.norm(result.fun))
            finite = bool(np.isfinite(rmse) and np.isfinite(result.x).all())
            fitted = decode(result.x) if finite else None
            runs.append({"start_index": start_index, "initial_D_h_c": [start_D, start_h, start_c],
                         "success": bool(result.success and finite), "status": int(result.status),
                         "message": str(result.message), "nfev": int(result.nfev),
                         "njev": int(result.njev or 0), "training_rmse_K": rmse if finite else None,
                         "optimality": float(result.optimality) if np.isfinite(result.optimality) else None,
                         "fitted_parameters": dict(zip(("D", "h", "c"), fitted)) if fitted else None})
            results.append(result if finite else None)
        except (ValueError, RuntimeError, FloatingPointError, OverflowError, np.linalg.LinAlgError) as exc:
            runs.append({"start_index": start_index, "initial_D_h_c": [start_D, start_h, start_c],
                         "success": False, "status": None, "message": str(exc),
                         "failure_reason": str(exc), "training_rmse_K": None,
                         "fitted_parameters": None, "nfev": None, "njev": None, "optimality": None})
            results.append(None)
    successful = [i for i, run in enumerate(runs) if run["success"]]
    pool = successful or [i for i, result in enumerate(results) if result is not None]
    if not pool:
        return _failed_candidate(model_name, "所有固定初值均失败，未获得可评分参数。", runs)
    selected = min(pool, key=lambda i: (runs[i]["training_rmse_K"], i))
    best = results[selected]
    D, h, c = decode(best.x)
    terms = _terms(D, h, c, model_name)
    prediction = _advance(prepared, D, h, c)
    singular = np.linalg.svd(best.jac, compute_uv=False)
    rank = int(np.sum(singular > max(float(singular[0]) * 1e-8, 1e-12)))
    condition = float(singular[0] / max(float(singular[-1]), 1e-30))
    near_lower = np.abs(best.x-lower) <= 1e-5 * (upper-lower)
    near_upper = np.abs(best.x-upper) <= 1e-5 * (upper-lower)
    hits = [{"parameter": name, "bound": "lower" if lo else "upper"}
            for name, lo, hi in zip(names, near_lower, near_upper) if lo or hi]
    return {
        "candidate_id": "trajectory-" + model_name.replace("+", "-"),
        "model_name": model_name, "equation": _equation(terms), "terms": terms,
        "support_size": len(terms), "fit_success": bool(successful),
        "failure_reason": None if successful else "所有固定初值均未满足优化停止条件。",
        "train_score": evaluate(prediction, u[:, 1:5]),
        "parameters": {"D": D, "h": h, "c": c},
        "diagnostics": {
            "method": FIT_PROTOCOL["method"], "is_l1": False,
            "starts": runs, "selected_start_index": selected,
            "parameter_bounds_hit": hits, "jacobian_rank": rank,
            "parameter_count": len(names),
            "scaled_jacobian_condition_number": condition if condition < 1e16 else None,
            "training_rows": len(t), "scored_training_rows": len(t)-1,
            "spatial_observation_count": 6, "predicted_interior_count": 4,
            "dt_blocks": len(prepared[-1]),
            "identifiability_note": "local numerical rank is not proof of unique physical mechanism",
            "future_interior_is_prediction_input": False,
        },
    }


def fit_candidates(t, u, dx) -> list[dict]:
    """Return all four prespecified supports, never selecting on dev/test."""
    u, t = np.asarray(u, float), np.asarray(t, float)
    if (t.ndim != 1 or u.shape != (len(t), 6) or len(t) < 8
            or not np.isfinite(u).all()):
        raise ValueError("拟合需要至少8帧的原始 N×6 温升轨迹。")
    _prepare(t, u[:, [0, 5]], u[0, 1:5], dx)
    candidates = []
    for name in MODEL_NAMES:
        try:
            candidates.append(fit_candidate(t, u, dx, name))
        except (ValueError, RuntimeError, FloatingPointError, OverflowError, np.linalg.LinAlgError) as exc:
            candidates.append(_failed_candidate(name, exc))
    return candidates
