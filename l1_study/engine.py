"""Isolated equal-trajectory L1 assessment with fixed, auditable rules.

The production engine is imported but never changed. Neither labels nor truth
coefficients are accepted by the fitting or selection interfaces. Exact
derivative arrays must be supplied independently from an analytic solution;
this module never constructs a target from a proposed PDE's right hand side.
"""
from __future__ import annotations

from copy import deepcopy
import warnings

import numpy as np
from scipy.integrate import cumulative_simpson
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Lasso

from pde_discovery import generic_engine as baseline
from pde_discovery.generic_experiment import forecast_metrics

TERM_NAMES = tuple(baseline.TERM_NAMES)
ALPHA_PATH = tuple(baseline.ALPHA_PATH)
METHODS = ("plain", "adaptive", "irl1")
RIDGE_ALPHA = 1e-6
WEIGHT_EPSILON = 1e-3
WEIGHT_BOUNDS = (1e-2, 1e3)
IRL1_SOLVES = 5
SUPPORT_THRESHOLD = 1e-6
DEBIAS_ZERO_THRESHOLD = 1e-8
NEAR_FACTOR = 1.05
PROTOCOL = {
    "version": 1,
    "methods": list(METHODS),
    "terms": list(TERM_NAMES),
    "mandatory_terms": [],
    "alpha_path": list(ALPHA_PATH),
    "trajectory_weighting": "rows multiplied by sqrt(total_rows/(trajectory_count*own_rows)) before global RMS",
    "normalization": "one pooled training feature RMS per column and one pooled training target RMS",
    "ridge_pilot": "solve(X.T@X/n + alpha*I, X.T@y/n) in normalized coordinates",
    "ridge_alpha": RIDGE_ALPHA,
    "adaptive_weights": "clip(1/(abs(ridge_pilot)+epsilon), lower, upper)",
    "weight_epsilon": WEIGHT_EPSILON,
    "weight_bounds": list(WEIGHT_BOUNDS),
    "adaptive_solves_per_alpha": 1,
    "irl1_solves_per_alpha": IRL1_SOLVES,
    "irl1_initialization": "independent same ridge pilot for each alpha; no warm starts across alpha",
    "irl1_update": "same inverse-absolute-coefficient weights after each solve",
    "weighted_objective": "||y-X*beta||^2/(2*n) + alpha*sum(weight*abs(beta))",
    "lasso": {"max_iter": 100000, "tol": 1e-10, "fit_intercept": False},
    "normalized_support_threshold": SUPPORT_THRESHOLD,
    "normalized_debiased_zero_threshold": DEBIAS_ZERO_THRESHOLD,
    "debias": "OLS on original normalized candidate columns; rcond=1e-10",
    "cross_alpha_objective_selection": False,
    "path_frequency_selection": False,
    "exact_derivatives": "analytic features and analytic u_t supplied independently, never reconstructed from a fitted PDE",
    "pointwise_numerical": "baseline projected spatial library; np.gradient of projected u with edge_order=2; remove two frames at each end",
    "pointwise_exact": "supplied exact feature and u_t arrays; remove same two frames at each end",
    "integrated": "unchanged baseline Fourier features and Simpson window construction",
    "development_selection": "sqrt(mean(trajectory_MSE)); within 1.05*best+1e-12 choose fewest terms, error, stable ID",
    "forecast": "unchanged baseline forecast; each development trajectory initialized only from its own first frame",
    "reference_used_for_selection": False,
}


def validate_block(block):
    """Return finite float X,y arrays with exactly the shared nine columns."""
    if not isinstance(block, dict) or "X" not in block or "y" not in block:
        raise ValueError("Each trajectory block must contain X and y.")
    features, target = np.asarray(block["X"]), np.asarray(block["y"])
    if np.iscomplexobj(features) or np.iscomplexobj(target):
        raise ValueError("Regression blocks must be real-valued.")
    features, target = np.asarray(features, float), np.asarray(target, float)
    if (features.ndim != 2 or features.shape[1] != len(TERM_NAMES)
            or target.ndim != 1 or features.shape[0] != len(target)
            or len(target) < 2 or len(target) > 3_000_000
            or not np.isfinite(features).all() or not np.isfinite(target).all()):
        raise ValueError("Need finite matching X[n,9] and y[n] with 2..3000000 rows.")
    return features, target


def integrated_block(data):
    """Build one unchanged baseline integral-regression trajectory block."""
    x, t, raw = baseline._validate(data)
    u, library, projection = baseline._features(x, raw)
    integral = cumulative_simpson(library, x=t, axis=0, initial=0)
    width = max(3, min(12, len(t)//10))
    starts = np.arange(4, len(t)-width-4, 2, dtype=int)
    ends = starts + width
    if len(starts) < 2:
        raise ValueError("Too few complete original-grid integration windows.")
    durations = (t[ends]-t[starts])[:, None]
    result = {
        "X": ((integral[ends]-integral[starts])/durations[:, :, None]).reshape(-1, len(TERM_NAMES)),
        "y": ((u[ends]-u[starts])/durations).ravel(),
        "metadata": {"derivative_mode": "integrated", "projection_relative_l2": projection,
                     "window_intervals": width, "integration_windows": len(starts),
                     "training_frames": len(t), "spatial_points": len(x),
                     "overlapping_windows": True},
    }
    validate_block(result)
    return result


def pointwise_block(data, exact=False):
    """Build matched pointwise rows with numerical or analytic derivatives.

    Shapes for the exact fields are (Nt,Nx,9) and (Nt,Nx), matching u's time
    axis first. Exact arrays are an explicitly labeled diagnostic upper bound,
    not an option available for an arbitrary measured data upload.
    """
    x, t, raw = baseline._validate(data)
    if exact:
        if not all(key in data for key in ("exact_features", "exact_ut")):
            raise ValueError("Exact mode requires independent exact_features and exact_ut.")
        library, target = np.asarray(data["exact_features"]), np.asarray(data["exact_ut"])
        if np.iscomplexobj(library) or np.iscomplexobj(target):
            raise ValueError("Exact derivative arrays must be real-valued.")
        library, target = np.asarray(library, float), np.asarray(target, float)
        if (library.shape != (*raw.shape, len(TERM_NAMES)) or target.shape != raw.shape
                or not np.isfinite(library).all() or not np.isfinite(target).all()):
            raise ValueError("Exact arrays must be finite (Nt,Nx,9) features and (Nt,Nx) u_t.")
        mode, projection = "exact_pointwise", None
    else:
        u, library, projection = baseline._features(x, raw)
        target = np.gradient(u, t, axis=0, edge_order=2)
        mode = "numerical_pointwise"
    result = {"X": library[2:-2].reshape(-1, len(TERM_NAMES)), "y": target[2:-2].ravel(),
              "metadata": {"derivative_mode": mode, "projection_relative_l2": projection,
                           "training_frames": len(t), "used_frames": len(t)-4,
                           "spatial_points": len(x), "trimmed_temporal_edges": 2,
                           "exact_derivative_diagnostic": bool(exact)}}
    validate_block(result)
    return result


def pool_blocks(blocks):
    """Equalize trajectory influence before one global training normalization."""
    if not isinstance(blocks, (list, tuple)) or not blocks or len(blocks) > 32:
        raise ValueError("Supply 1..32 independent training trajectory blocks.")
    arrays = [validate_block(block) for block in blocks]
    total = sum(len(y) for _, y in arrays)
    if total > 3_000_000:
        raise ValueError("Pooled regression exceeds three million rows.")
    factors = [float(np.sqrt(total/(len(arrays)*len(y)))) for _, y in arrays]
    features = np.vstack([X*factor for (X, _), factor in zip(arrays, factors)])
    target = np.concatenate([y*factor for (_, y), factor in zip(arrays, factors)])
    if not np.isfinite(features).all() or not np.isfinite(target).all():
        raise ValueError("Trajectory weighting overflowed.")
    return features, target, {"trajectory_count": len(arrays),
                              "rows_per_trajectory": [len(y) for _, y in arrays],
                              "row_weights": factors}


def _weights(coefficients):
    return np.clip(1/(np.abs(coefficients)+WEIGHT_EPSILON), *WEIGHT_BOUNDS)


def _solve(X, y, alpha, weights, iteration):
    model = Lasso(alpha=alpha, fit_intercept=False, max_iter=100000, tol=1e-10)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(X/weights, y)
    coefficients = model.coef_/weights
    converged = bool(np.isfinite(coefficients).all() and
                     not any(issubclass(w.category, ConvergenceWarning) for w in caught))
    return coefficients, {"iteration": iteration, "alpha": float(alpha),
                          "iterations": int(model.n_iter_), "dual_gap": float(model.dual_gap_),
                          "converged": converged, "weights": weights.tolist()}


def fit_blocks(blocks, method="plain"):
    """Return fixed-path candidates; no development data or truth enters here."""
    if method not in METHODS:
        raise ValueError(f"Unknown L1 method: {method}")
    features, target, pooled = pool_blocks(blocks)
    scales = np.sqrt(np.mean(features**2, axis=0))
    target_scale = float(np.sqrt(np.mean(target**2)))
    if not np.isfinite(scales).all() or not np.isfinite(target_scale):
        raise ValueError("Training normalization overflowed.")
    if target_scale <= 1e-12:
        raise ValueError("No measurable temporal excitation; dynamics cannot be identified.")
    active = scales > max(1e-15, float(scales.max())*1e-12)
    safe_scales = np.where(active, scales, 1.0)
    X, y = features/safe_scales, target/target_scale
    full_condition, full_rank = baseline._condition(X[:, active])
    norms = np.linalg.norm(X, axis=0)
    correlations = X.T@X/np.maximum(np.outer(norms, norms), 1e-30)
    pairs = [{"terms": [TERM_NAMES[i], TERM_NAMES[j]], "correlation": float(correlations[i, j])}
             for i in range(len(TERM_NAMES)) for j in range(i+1, len(TERM_NAMES))
             if active[i] and active[j] and abs(correlations[i, j]) > .995]
    pilot = None if method == "plain" else np.linalg.solve(
        X.T@X/len(y)+RIDGE_ALPHA*np.eye(len(TERM_NAMES)), X.T@y/len(y))
    if pilot is not None and not np.isfinite(pilot).all():
        raise ValueError("Ridge pilot is not finite.")
    paths, runs = {}, {}
    for alpha in ALPHA_PATH:
        weights = np.ones(len(TERM_NAMES)) if method == "plain" else _weights(pilot)
        iterations = IRL1_SOLVES if method == "irl1" else 1
        history = []
        for iteration in range(1, iterations+1):
            coefficients, record = _solve(X, y, alpha, weights, iteration)
            history.append(record)
            weights = _weights(coefficients)
        runs[alpha] = {"alpha": float(alpha), "converged": all(r["converged"] for r in history),
                       "reweighting_solves": iterations, "history": history}
        support = tuple(int(i) for i in np.flatnonzero((np.abs(coefficients) > SUPPORT_THRESHOLD)&active))
        if support:
            paths.setdefault(support, []).append(alpha)
    candidates = []
    for support, alphas in sorted(paths.items()):
        indices = np.asarray(support, int)
        normalized = np.linalg.lstsq(X[:, indices], y, rcond=1e-10)[0]
        keep = np.abs(normalized) > DEBIAS_ZERO_THRESHOLD
        indices, normalized = indices[keep], normalized[keep]
        if not len(indices):
            continue
        condition, rank = baseline._condition(X[:, indices])
        coefficients = normalized*target_scale/safe_scales[indices]
        terms = [{"name": TERM_NAMES[i], "coefficient": float(c)} for i, c in zip(indices, coefficients)]
        residual = target-features[:, indices]@coefficients
        converged = all(runs[alpha]["converged"] for alpha in alphas)
        success = bool(converged and rank == len(indices) and np.isfinite(coefficients).all())
        prefix = "lasso" if method == "plain" else method
        candidates.append({
            "candidate_id": prefix+"-"+"-".join(str(i) for i in indices)+"-"+f"{alphas[0]:g}",
            "method": method, "equation": baseline._equation(terms), "terms": terms,
            "support_size": len(indices), "fit_success": success,
            "fit_relative_error": float(np.linalg.norm(residual)/np.linalg.norm(target)),
            "fit_rmse": float(np.sqrt(np.mean(residual**2))), "alpha_values": list(alphas),
            "failure_reason": None if success else "Unconverged LASSO path or rank-deficient selected support.",
            "diagnostics": {**pooled, "support_rank": rank, "parameter_count": len(indices),
                            "condition_number": condition, "full_library_rank": full_rank,
                            "full_library_condition_number": full_condition,
                            "active_library_count": int(active.sum()),
                            "lasso_all_support_runs_converged": converged,
                            "lasso_runs": [runs[a] for a in alphas],
                            "all_alpha_runs": [runs[a] for a in ALPHA_PATH],
                            "correlated_columns": pairs, "regression_rows": len(target),
                            "reference_equation_used": False,
                            "feature_rms": scales.tolist(), "target_rms": target_scale,
                            "ridge_pilot": None if pilot is None else pilot.tolist(),
                            "block_diagnostics": [deepcopy(b.get("metadata", {})) for b in blocks]},
        })
    # OLS's fixed zero rule may collapse two originally different supports.
    # Keep the deterministic first result and retain duplicate-run evidence.
    unique = {}
    for candidate in candidates:
        support = tuple(t["name"] for t in candidate["terms"])
        if support not in unique:
            unique[support] = candidate
        else:
            previous = unique[support]
            previous["diagnostics"].setdefault("collapsed_candidates", []).append(candidate)
    return sorted(unique.values(), key=lambda c: (c["support_size"], c["fit_relative_error"], c["candidate_id"]))


def select_candidates(candidates, developments):
    """Choose by held-out development trajectories, never reference equations."""
    if not isinstance(developments, (list, tuple)) or not developments:
        raise ValueError("At least one development trajectory is required.")
    arrays = [baseline._validate(data) for data in developments]
    scored = deepcopy(candidates)
    for candidate in scored:
        candidate["eligible"] = False
        if not candidate.get("fit_success"):
            candidate["rejection_reason"] = "Training convergence, finite coefficients or rank check failed."
            continue
        try:
            individual = []
            for x, t, u in arrays:
                prediction = baseline.forecast(candidate["terms"], x, t, u[0])
                individual.append(forecast_metrics(prediction, u))
            error = float(np.sqrt(np.mean([m["rmse"]**2 for m in individual])))
            if not np.isfinite(error):
                raise ValueError("Development aggregate is not finite.")
            candidate["development"] = {"rmse": error, "per_trajectory": individual,
                                         "aggregation": "sqrt(mean(trajectory_MSE))",
                                         "trajectory_count": len(individual)}
            candidate["eligible"] = True
        except (ValueError, RuntimeError, FloatingPointError, np.linalg.LinAlgError) as exc:
            candidate["rejection_reason"] = str(exc)
    eligible = [c for c in scored if c["eligible"]]
    if not eligible:
        return {"selected": None, "candidates": scored, "near_optimal_supports": [],
                "failure_reason": "No candidate passed training and development checks."}
    best = min(c["development"]["rmse"] for c in eligible)
    pool = [c for c in eligible if c["development"]["rmse"] <= NEAR_FACTOR*best+1e-12]
    selected = min(pool, key=lambda c: (c["support_size"], c["development"]["rmse"], c["candidate_id"]))
    near = sorted({tuple(sorted(t["name"] for t in c["terms"])) for c in pool})
    return {"selected": selected, "candidates": scored,
            "near_optimal_supports": [list(support) for support in near], "failure_reason": None}
