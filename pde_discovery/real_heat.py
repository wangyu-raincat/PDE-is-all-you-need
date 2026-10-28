"""Sparse, coarse-grid heat-rod identification from six measured sensors.

This module is independent of the periodic synthetic demonstration.  It uses
only the original six spatial locations: four interior evolution equations,
with the two measured end temperatures treated as exogenous boundary inputs.
The identified coefficients describe this finite-difference observation model;
they are not automatically continuous material-property measurements.

No train/test split, future-data smoothing, candidate selection on test data,
or spatial upsampling happens here.  The caller owns a documented, chronological
split and must freeze its selected candidate before evaluating the final test.
"""
from __future__ import annotations

import warnings

import numpy as np
from scipy.integrate import cumulative_trapezoid, solve_ivp
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Lasso


TERM_NAMES = ["1", "theta", "theta^2", "theta_x", "theta_xx"]
ALPHA_PATH = (1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4,
              1e-3, 3e-3, 1e-2, 3e-2, 1e-1)
MAX_RHS_EVALUATIONS = 40000


def _time(t) -> np.ndarray:
    t = np.asarray(t, dtype=float)
    if (t.ndim != 1 or not 3 <= len(t) <= 200000
            or not np.isfinite(t).all() or np.any(np.diff(t) <= 0)):
        raise ValueError("时间必须为至少3个有限、严格递增的观测时刻。")
    return t


def _spacing(dx) -> float:
    dx = float(dx)
    if not np.isfinite(dx) or dx <= 0:
        raise ValueError("传感器间距 dx 必须是有限正数。")
    return dx


def _condition(matrix: np.ndarray) -> tuple[float | None, int]:
    singular = np.linalg.svd(matrix, compute_uv=False)
    if not len(singular) or singular[0] == 0:
        return None, 0
    rank = int(np.sum(singular > singular[0] * 1e-10))
    condition = float(singular[0] / max(singular[-1], 1e-30))
    return (condition if condition < 1e14 else None), rank


def _equation(terms: list[dict]) -> str:
    pieces = []
    for term in terms:
        coefficient = float(term["coefficient"])
        part = f"{abs(coefficient):.8g}"
        if term["name"] != "1":
            part += " " + term["name"]
        pieces.append(("- " if coefficient < 0 else "+ ") + part)
    right = " ".join(pieces) or "0"
    if right.startswith("+ "):
        right = right[2:]
    return "theta_t = " + right


def build_candidates(t, u, dx, window_seconds: float) -> list[dict]:
    """Produce a fixed LASSO path, followed by support-wise least-squares refits.

    Parameters
    ----------
    t : (N,) measured time coordinates, in seconds; irregular sampling allowed.
    u : (N, 6) measured excess temperatures, in sensor order.
    dx : known uniform spatial separation, normally in metres.
    window_seconds : positive integration duration; windows end at an original
        observation at or after this duration. No interpolated measurements are
        created. Integration uses trapezoids over the original timestamps.

    Returns a list of candidates generated entirely from this training input.
    Ordering is deterministic by complexity then training residual; it is not
    a claim that the first candidate is the best predictive model.
    """
    t, dx = _time(t), _spacing(dx)
    u = np.asarray(u, dtype=float)
    if u.shape != (len(t), 6) or not np.isfinite(u).all():
        raise ValueError("训练温度必须为有限 N×6 数组；不接受插值增加的空间点。")
    if len(t) < 8:
        raise ValueError("积分识别至少需要8个观测时刻。")
    window_seconds = float(window_seconds)
    if (not np.isfinite(window_seconds) or window_seconds <= 0
            or window_seconds >= t[-1] - t[0]):
        raise ValueError("积分窗口必须大于0且小于训练轨迹时长。")

    theta = u[:, 1:-1]
    theta_x = (u[:, 2:] - u[:, :-2]) / (2 * dx)
    theta_xx = (u[:, 2:] - 2 * theta + u[:, :-2]) / dx**2
    library = np.stack([np.ones_like(theta), theta, theta**2,
                        theta_x, theta_xx], axis=-1)
    if not np.isfinite(library).all():
        raise ValueError("候选项计算溢出；请检查单位并对问题做一致的缩放。")
    integral = cumulative_trapezoid(library, x=t - t[0], axis=0, initial=0)

    # Five overlapping windows per requested duration limit redundant rows;
    # overlapping windows are not counted as independent experiments.
    nominal_points = window_seconds / float(np.median(np.diff(t)))
    stride = max(1, int(round(nominal_points / 5)))
    starts = np.arange(0, len(t) - 1, stride, dtype=int)
    ends = np.searchsorted(t, t[starts] + window_seconds, side="left")
    valid = (ends < len(t)) & (ends >= starts + 2)
    starts, ends = starts[valid], ends[valid]
    if len(starts) < 3:
        raise ValueError("积分窗口过大或采样过少，无法构成足够的原始观测窗口。")
    duration = t[ends] - t[starts]
    target = ((theta[ends] - theta[starts]) / duration[:, None]).reshape(-1)
    features = ((integral[ends] - integral[starts]) /
                duration[:, None, None]).reshape(-1, len(TERM_NAMES))
    scales = np.sqrt(np.mean(features**2, axis=0))
    target_scale = float(np.sqrt(np.mean(target**2)))
    if not np.isfinite(target_scale) or target_scale < 1e-12:
        raise ValueError("轨迹未包含可测的温度演化，无法识别动力学。")
    active = scales > max(1e-15, float(scales.max()) * 1e-12)
    scales = np.where(active, scales, 1.0)
    X, y = features / scales, target / target_scale
    full_condition, full_rank = _condition(X[:, active])
    norms = np.linalg.norm(X, axis=0)
    similarity = X.T @ X / np.maximum(np.outer(norms, norms), 1e-30)
    correlated = [{"terms": [TERM_NAMES[i], TERM_NAMES[j]],
                   "correlation": float(similarity[i, j])}
                  for i in range(len(TERM_NAMES))
                  for j in range(i + 1, len(TERM_NAMES))
                  if active[i] and active[j] and abs(similarity[i, j]) > 0.98]

    paths: dict[tuple, list[float]] = {}
    optimization_runs: dict[float, dict] = {}
    for alpha in ALPHA_PATH:
        model = Lasso(alpha=alpha, fit_intercept=False,
                      max_iter=50000, tol=1e-9)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            model.fit(X, y)
        optimization_runs[float(alpha)] = {
            "alpha": float(alpha), "iterations": int(model.n_iter_),
            "dual_gap": float(model.dual_gap_),
            "converged": not any(issubclass(w.category, ConvergenceWarning) for w in caught),
        }
        support = tuple(int(i) for i in np.flatnonzero(
            (np.abs(model.coef_) > 1e-7) & active))
        if support:
            paths.setdefault(support, []).append(float(alpha))

    candidates = []
    for support, alphas in paths.items():
        columns = np.asarray(support, dtype=int)
        normalized = np.linalg.lstsq(X[:, columns], y, rcond=1e-10)[0]
        coefficients = normalized * target_scale / scales[columns]
        terms = [{"name": TERM_NAMES[int(i)], "coefficient": float(c)}
                 for i, c in zip(columns, coefficients)]
        residual = target - features[:, columns] @ coefficients
        fit = float(np.linalg.norm(residual) / np.linalg.norm(target))
        condition, rank = _condition(X[:, columns])
        candidate = {
            "equation": _equation(terms), "terms": terms,
            "fit_error": fit, "fit_relative_error": fit,
            "fit_rmse": float(np.sqrt(np.mean(residual**2))),
            "condition_number": condition, "support_size": len(support),
            "alpha_values": alphas,
            "diagnostics": {
                "library": list(TERM_NAMES), "full_library_condition_number": full_condition,
                "full_library_rank": full_rank, "support_rank": rank,
                "correlated_columns": correlated,
                "regression_rows": len(target), "integration_windows": len(starts),
                "window_seconds_requested": window_seconds,
                "window_seconds_min": float(duration.min()),
                "window_seconds_max": float(duration.max()),
                "overlapping_windows": True, "spatial_observation_count": 6,
                "evolution_equation_count": 4, "spatial_upsampling": False,
                "smoothing": "none", "time_integration": "original-grid trapezoids",
                "lasso_runs": [optimization_runs[alpha] for alpha in alphas],
                "lasso_all_support_runs_converged": all(optimization_runs[a]["converged"] for a in alphas),
                "scope": "6点原测量网格的有效半离散系数，不自动等同连续材料物性",
            },
        }
        candidates.append(candidate)
    candidates.sort(key=lambda c: (c["support_size"], c["fit_error"], c["equation"]))
    return candidates


def predict(terms: list[dict], t, boundaries, u0, dx) -> np.ndarray:
    """Simulate four interior states using only initial states and boundary inputs.

    ``boundaries`` must have exactly two measured columns, left and right.
    Their interpolation supplies the non-periodic boundary forcing; it is not
    a spatially upsampled measurement. Future interior observations are neither
    accepted nor used. This is a boundary-conditioned forecast, not a forecast
    of unknown future boundary temperatures.
    """
    t, dx = _time(t), _spacing(dx)
    boundaries, u0 = np.asarray(boundaries, float), np.asarray(u0, float)
    if (boundaries.shape != (len(t), 2) or u0.shape != (4,)
            or not np.isfinite(boundaries).all() or not np.isfinite(u0).all()):
        raise ValueError("预测只接受 N×2 边界输入与4个初始内点温度。")
    coefficients = {name: 0.0 for name in TERM_NAMES}
    for term in terms:
        if not isinstance(term, dict) or term.get("name") not in coefficients:
            raise ValueError("候选含不支持的项。")
        c = float(term["coefficient"])
        if not np.isfinite(c):
            raise ValueError("候选系数必须有限。")
        coefficients[term["name"]] += c
    if coefficients["theta_xx"] < 0:
        raise ValueError("负扩散系数候选不用于正向热传导验证。")

    reaction = coefficients["theta"]
    quadratic = coefficients["theta^2"]
    transport = coefficients["theta_x"]
    diffusivity = coefficients["theta_xx"]
    laplacian = (np.diag(np.full(4, -2.0)) + np.diag(np.ones(3), 1)
                 + np.diag(np.ones(3), -1)) / dx**2
    gradient = (np.diag(np.ones(3), 1) - np.diag(np.ones(3), -1)) / (2 * dx)
    linear = diffusivity * laplacian + transport * gradient + reaction * np.eye(4)
    if not np.isfinite(linear).all():
        raise ValueError("空间算子溢出；请检查间距和系数单位。")
    origin = float(t[0])
    local_t = t - origin
    magnitude = max(1.0, float(np.max(np.abs(boundaries))), float(np.max(np.abs(u0))))
    divergence_limit = magnitude * 1000
    count = 0

    def rhs(time, state):
        nonlocal count
        count += 1
        if count > MAX_RHS_EVALUATIONS:
            raise ValueError("热杆候选积分超过40000次求值预算。")
        if not np.isfinite(state).all() or np.max(np.abs(state)) > divergence_limit:
            raise ValueError("热杆候选模型积分发散。")
        left = np.interp(time, local_t, boundaries[:, 0])
        right = np.interp(time, local_t, boundaries[:, 1])
        result = linear @ state + quadratic * state**2 + coefficients["1"]
        result[0] += (diffusivity / dx**2 - transport / (2 * dx)) * left
        result[-1] += (diffusivity / dx**2 + transport / (2 * dx)) * right
        return result

    def jacobian(_time, state):
        return linear + np.diag(2 * quadratic * state)

    span = float(local_t[-1])
    max_step = min(span / 50, 10 * float(np.median(np.diff(local_t))))
    result = solve_ivp(rhs, (0.0, span), u0, t_eval=local_t,
                       method="BDF", jac=jacobian, rtol=1e-7, atol=1e-9,
                       max_step=max_step)
    if (not result.success or result.y.shape != (4, len(t))
            or not np.isfinite(result.y).all()
            or np.max(np.abs(result.y)) > divergence_limit):
        raise ValueError("热杆候选积分未收敛或输出发散。")
    return result.y.T


def evaluate(predicted, observed, *, exclude_initial: bool = True) -> dict:
    """Report errors only; this function neither fits nor ranks candidates."""
    predicted, observed = np.asarray(predicted, float), np.asarray(observed, float)
    if (predicted.ndim != 2 or predicted.shape != observed.shape
            or predicted.shape[1] != 4 or len(predicted) < 2
            or not np.isfinite(predicted).all() or not np.isfinite(observed).all()):
        raise ValueError("评估数组必须为相同形状、有限的 N×4 内点温度。")
    start = 1 if exclude_initial else 0
    residual = predicted[start:] - observed[start:]
    denominator = float(np.linalg.norm(observed[start:]))
    error = float(np.linalg.norm(residual))
    relative = error / denominator if denominator > 1e-12 else (0.0 if error < 1e-12 else None)
    return {
        "relative_error": relative, "rmse": float(np.sqrt(np.mean(residual**2))),
        "mae": float(np.mean(np.abs(residual))),
        "max_absolute_error": float(np.max(np.abs(residual))),
        "per_sensor_rmse": np.sqrt(np.mean(residual**2, axis=0)).tolist(),
        "scored_rows": len(residual), "excluded_initial_row": bool(exclude_initial),
    }
