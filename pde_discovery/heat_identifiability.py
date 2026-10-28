"""Training-only diagnostics for an already selected physical heat-rod model.

Profile fitting and deletion of loss blocks are descriptive sensitivity checks,
not confidence intervals, equation discovery, or a replacement model selection.
The numerical propagator and its frozen physical priors are reused unchanged.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

from .real_benchmark import CASES, DATA_ROOT
from .real_experiment import _sha, _write_json, load_measurements
from .real_heat_refined import (
    C_BOUNDS, C_SCALE, D_BOUNDS, FIT_PROTOCOL, FIT_TOLERANCE, H_BOUNDS,
    H_SCALE, MAX_NFEV, MODEL_NAMES, STARTS, _advance, _equation, _prepare, _terms,
)

PROFILE_MULTIPLIERS = (0.25, 0.4, 0.63, 0.8, 1.0, 1.25, 1.6, 2.5, 4.0)
BLOCK_COUNT = 5
NEAR_OPTIMAL_FACTOR = 1.05
SENSITIVITY_STEP = 1e-5
RANK_RTOL = 1e-8
RANK_ATOL = 1e-12
BOUND_HIT_TOLERANCE = 1e-5
OUTPUT_ROOT = Path("artifacts/identifiability")
MODEL_ROOT = Path("artifacts/refined_heat")

IDENTIFIABILITY_PROTOCOL = {
    "kind": "training_only_identifiability_diagnostics",
    "data_role": "only complete original repeat1 training records",
    "model_role": "already frozen selected support and coefficients; never replaced",
    "is_l1": False, "used_for_model_selection": False,
    "is_confidence_interval": False, "unique_physical_equation_established": False,
    "profile_multipliers": list(PROFILE_MULTIPLIERS),
    "profile_reference": "existing frozen full-training D, not a newly chosen optimum",
    "profile_bounds": list(D_BOUNDS),
    "profile_clipping": "clip to original D bounds, deduplicate identical clipped points, retain multiplier mappings",
    "profile_refit": "D fixed; refit only h/c present in the already selected support",
    "near_optimal_factor": NEAR_OPTIMAL_FACTOR,
    "near_optimal_rule": "successful discrete profile RMSE <= minimum successful profile RMSE * 1.05; no interpolation or CI",
    "loss_block_count": BLOCK_COUNT,
    "loss_block_definition": "equal elapsed-time intervals on [original t0, original t_end]; first four right-open, last closed; initial row never scored",
    "block_refit": "same selected support, all its parameters; delete only a block of interior observation residuals",
    "propagation": "always original initial interior state, full original time axis and full measured boundaries",
    "reference_sensitivity": "forward finite difference in old scaled optimizer coordinates; backward at upper bound",
    "reference_sensitivity_step": SENSITIVITY_STEP,
    "rank_relative_threshold": RANK_RTOL, "rank_absolute_threshold": RANK_ATOL,
    "bound_hit_coordinate_range_fraction": BOUND_HIT_TOLERANCE,
    "fitting": json.loads(json.dumps(FIT_PROTOCOL)),
    "profile_start_rule": "old three fixed starts projected onto the free h/c coordinates; no sequential warm starts",
    "failure_rule": "record each failed start; retain failed diagnostic points; an incomplete profile or deletion series is inconclusive",
    "empty_block_rule": "a deletion interval containing no future observation is insufficient and is not refitted",
    "frozen_reference_integrity": "verify each model JSON against its existing freeze_index SHA256 and each training manifest against the frozen model",
    "reporting": "report all declared cases, profile grid edges, parameter bounds, and block parameter ranges",
    "limitations": [
        "Descriptive loss sensitivity is not statistical uncertainty or confidence coverage.",
        "A locally full-rank Jacobian does not establish a unique physical law.",
        "Parameters remain effective six-sensor discretization coefficients under a physical prior.",
        "No development, old repeat3, or new held-out measurements enter this diagnostic.",
    ],
}


def _validate(t, u, dx, model_name):
    if model_name not in MODEL_NAMES:
        raise ValueError(f"model_name 必须属于 {MODEL_NAMES}。")
    t, u = np.asarray(t, float), np.asarray(u, float)
    if t.ndim != 1 or len(t) < 8 or u.shape != (len(t), 6) or not np.isfinite(u).all():
        raise ValueError("诊断需要至少8帧原始 N×6 有限温升观测。")
    prepared = _prepare(t, u[:, [0, 5]], u[0, 1:5], dx)
    return t, u, prepared


def _coordinates(model_name, fixed_D=None):
    names = ([] if fixed_D is not None else ["D"])
    names += (["h"] if "h" in model_name else []) + (["c"] if "c" in model_name else [])
    bounds = {"D": np.log(D_BOUNDS), "h": np.asarray(H_BOUNDS)/H_SCALE,
              "c": np.asarray(C_BOUNDS)/C_SCALE}
    lower = np.array([bounds[name][0] for name in names])
    upper = np.array([bounds[name][1] for name in names])

    def encode(parameters):
        values = {"D": np.log(parameters["D"]), "h": parameters["h"]/H_SCALE,
                  "c": parameters["c"]/C_SCALE}
        return np.array([values[name] for name in names])

    def decode(vector):
        parameters = {"D": float(fixed_D) if fixed_D is not None else 0.0, "h": 0.0, "c": 0.0}
        for name, value in zip(names, vector):
            parameters[name] = float(np.exp(value) if name == "D" else value * (H_SCALE if name == "h" else C_SCALE))
        return parameters

    return names, lower, upper, encode, decode


def _rank(jacobian):
    count = jacobian.shape[1]
    if not count:
        return {"jacobian_rank": 0, "parameter_count": 0, "scaled_jacobian_condition_number": None}
    singular = np.linalg.svd(jacobian, compute_uv=False)
    rank = int(np.sum(singular > max(float(singular[0]) * RANK_RTOL, RANK_ATOL)))
    condition = float(singular[0] / singular[-1]) if singular[-1] > 0 else float("inf")
    return {"jacobian_rank": rank, "parameter_count": count,
            "scaled_jacobian_condition_number": condition if np.isfinite(condition) else None}


def _bounds_hit(names, vector, lower, upper):
    hits = []
    for name, value, lo, hi in zip(names, vector, lower, upper):
        if abs(value-lo) <= BOUND_HIT_TOLERANCE * (hi-lo):
            hits.append({"parameter": name, "bound": "lower"})
        elif abs(value-hi) <= BOUND_HIT_TOLERANCE * (hi-lo):
            hits.append({"parameter": name, "bound": "upper"})
    return hits


def fit_masked(t, u, dx, model_name, observation_mask=None, *, fixed_D=None):
    """Fit a selected support; the mask changes loss only, never rollout inputs.

    ``observation_mask`` is an N-vector of booleans. The initial frame is always
    excluded. All boundary samples, including those in masked rows, remain known
    external forcing. Future interior samples appear only in scored residuals.
    """
    t, u, prepared = _validate(t, u, dx, model_name)
    if fixed_D is not None and (not np.isfinite(fixed_D) or not D_BOUNDS[0] <= fixed_D <= D_BOUNDS[1]):
        raise ValueError("固定 D 必须位于原冻结参数界限内。")
    if observation_mask is None:
        mask = np.ones(len(t), dtype=bool)
    else:
        mask = np.asarray(observation_mask)
        if mask.shape != (len(t),) or mask.dtype != np.dtype(bool):
            raise ValueError("observation_mask 必须是与时间等长的布尔向量。")
        mask = mask.copy()
    mask[0] = False
    if mask.sum() < 2:
        raise ValueError("至少保留两个非初始时刻用于拟合。")
    names, lower, upper, encode, decode = _coordinates(model_name, fixed_D)
    target, normalization = u[mask, 1:5], np.sqrt(4*int(mask.sum()))

    def residual(vector):
        parameters = decode(vector)
        trajectory = _advance(prepared, parameters["D"], parameters["h"], parameters["c"])
        return (trajectory[mask]-target).ravel() / normalization

    runs, solutions = [], []
    starts = STARTS if names else (STARTS[0],)
    for start_index, start in enumerate(starts):
        initial = encode(dict(zip(("D", "h", "c"), start)))
        try:
            if names:
                result = least_squares(residual, initial, bounds=(lower, upper), method="trf", jac="2-point",
                                       loss="linear", max_nfev=MAX_NFEV, ftol=FIT_TOLERANCE,
                                       xtol=FIT_TOLERANCE, gtol=FIT_TOLERANCE)
                vector, error = result.x, float(np.linalg.norm(result.fun))
                success = bool(result.success and np.isfinite(error) and np.isfinite(vector).all())
                jacobian, status, message = result.jac, int(result.status), str(result.message)
                nfev, njev = int(result.nfev), int(result.njev or 0)
            else:
                vector, error = initial, float(np.linalg.norm(residual(initial)))
                success, jacobian, status, message = bool(np.isfinite(error)), np.zeros((target.size, 0)), 0, "No free parameter: direct rollout."
                nfev, njev = 1, 0
            parameters = decode(vector)
            runs.append({"start_index": start_index, "initial_D_h_c": [float(fixed_D) if fixed_D is not None else start[0], start[1], start[2]],
                         "success": success, "status": status, "message": message,
                         "nfev": nfev, "njev": njev, "retained_loss_rmse_K": error if np.isfinite(error) else None,
                         "fitted_parameters": parameters if np.isfinite(list(parameters.values())).all() else None})
            solutions.append((vector, jacobian) if success else None)
        except (ValueError, RuntimeError, FloatingPointError, OverflowError, np.linalg.LinAlgError) as exc:
            runs.append({"start_index": start_index, "success": False, "failure_reason": str(exc),
                         "retained_loss_rmse_K": None, "fitted_parameters": None})
            solutions.append(None)
    successful = [i for i, solution in enumerate(solutions) if solution is not None]
    base = {"model_name": model_name, "fixed_D_m2_s": float(fixed_D) if fixed_D is not None else None,
            "fit_success": bool(successful), "starts": runs, "loss_rows": int(mask.sum()),
            "rollout_rows": len(t), "rollout_time_start_s": float(t[0]), "rollout_time_end_s": float(t[-1]),
            "original_initial_interior": u[0, 1:5].tolist(), "full_boundaries_retained": True,
            "initial_condition_reset": False, "future_interior_is_prediction_input": False}
    if not successful:
        return {**base, "status": "inconclusive", "failure_reason": "No fixed start converged.",
                "parameters": None, "terms": [], "train_rmse_K": None, "retained_loss_rmse_K": None}
    selected = min(successful, key=lambda i: (runs[i]["retained_loss_rmse_K"], i))
    vector, jacobian = solutions[selected]
    parameters = decode(vector)
    prediction = _advance(prepared, parameters["D"], parameters["h"], parameters["c"])
    rank = _rank(jacobian)
    excluded = ~mask
    excluded[0] = False
    terms = _terms(parameters["D"], parameters["h"], parameters["c"], model_name)
    return {**base, **rank, "selected_start_index": selected, "failure_reason": None,
            "status": "insufficient" if rank["jacobian_rank"] < rank["parameter_count"] else "fit_complete",
            "parameters": parameters, "terms": terms, "equation": _equation(terms),
            "parameter_bounds_hit": _bounds_hit(names, vector, lower, upper),
            "retained_loss_rmse_K": runs[selected]["retained_loss_rmse_K"],
            "train_rmse_K": float(np.sqrt(np.mean((prediction[1:]-u[1:, 1:5])**2))),
            "deleted_loss_rmse_K": float(np.sqrt(np.mean((prediction[excluded]-u[excluded, 1:5])**2))) if excluded.any() else None}


def _reference(t, u, dx, model_name, parameters):
    t, u, prepared = _validate(t, u, dx, model_name)
    parameters = {name: float(parameters.get(name, 0.0)) for name in ("D", "h", "c")}
    if (not np.isfinite(list(parameters.values())).all()
            or not D_BOUNDS[0] <= parameters["D"] <= D_BOUNDS[1]
            or not H_BOUNDS[0] <= parameters["h"] <= H_BOUNDS[1]
            or not C_BOUNDS[0] <= parameters["c"] <= C_BOUNDS[1]
            or ("h" not in model_name and parameters["h"] != 0)
            or ("c" not in model_name and parameters["c"] != 0)):
        raise ValueError("参考参数必须符合原支持与原参数界限。")
    names, lower, upper, encode, decode = _coordinates(model_name)
    vector = encode(parameters)

    def residual(value):
        p = decode(value)
        return (_advance(prepared, p["D"], p["h"], p["c"])[1:] - u[1:, 1:5]).ravel() / np.sqrt(4*(len(t)-1))

    baseline = residual(vector)
    jacobian = np.empty((len(baseline), len(names)))
    for column in range(len(names)):
        step = SENSITIVITY_STEP if vector[column]+SENSITIVITY_STEP <= upper[column] else -SENSITIVITY_STEP
        shifted = vector.copy()
        shifted[column] += step
        jacobian[:, column] = (residual(shifted)-baseline)/step
    return {"parameters": parameters, "model_name": model_name,
            "train_rmse_K": float(np.linalg.norm(baseline)), **_rank(jacobian),
            "parameter_bounds_hit": _bounds_hit(names, vector, lower, upper)}


def _failed_diagnostic_fit(t, u, model_name, observation_mask, fixed_D, reason, status="inconclusive"):
    mask = np.ones(len(t), dtype=bool) if observation_mask is None else np.asarray(observation_mask, bool).copy()
    mask[0] = False
    return {"model_name": model_name, "fixed_D_m2_s": float(fixed_D) if fixed_D is not None else None,
            "fit_success": False, "status": status, "failure_reason": str(reason),
            "parameters": None, "terms": [], "starts": [], "train_rmse_K": None,
            "retained_loss_rmse_K": None, "loss_rows": int(mask.sum()), "rollout_rows": len(t),
            "rollout_time_start_s": float(t[0]), "rollout_time_end_s": float(t[-1]),
            "original_initial_interior": np.asarray(u)[0, 1:5].tolist(),
            "full_boundaries_retained": True, "initial_condition_reset": False,
            "future_interior_is_prediction_input": False}


def _safe_diagnostic_fit(t, u, dx, model_name, observation_mask=None, *, fixed_D=None):
    """Retain a per-point failure after public inputs have been validated."""
    try:
        return fit_masked(t, u, dx, model_name, observation_mask, fixed_D=fixed_D)
    except (ValueError, RuntimeError, FloatingPointError, OverflowError, np.linalg.LinAlgError) as exc:
        return _failed_diagnostic_fit(t, u, model_name, observation_mask, fixed_D, exc)


def profile_diffusion(t, u, dx, model_name, reference_parameters):
    """Refit nuisance coefficients over a fixed discrete diffusion profile."""
    reference = _reference(t, u, dx, model_name, reference_parameters)
    D = reference["parameters"]["D"]
    requested = [{"multiplier": factor, "requested_D_m2_s": D*factor,
                  "D_m2_s": float(np.clip(D*factor, *D_BOUNDS)),
                  "clipped_to_parameter_bound": bool(not D_BOUNDS[0] <= D*factor <= D_BOUNDS[1])}
                 for factor in PROFILE_MULTIPLIERS]
    grid = sorted({point["D_m2_s"] for point in requested})
    points = []
    for value in grid:
        result = _safe_diagnostic_fit(t, u, dx, model_name, fixed_D=value)
        points.append({**result, "D_m2_s": value, "D_over_reference": value/D,
                       "requested_multipliers": [r["multiplier"] for r in requested if r["D_m2_s"] == value]})
    successful = [point for point in points if point["fit_success"]]
    best = min((point["train_rmse_K"] for point in successful), default=None)
    near = [point for point in successful if point["train_rmse_K"] <= best*NEAR_OPTIMAL_FACTOR] if best is not None else []
    for point in points:
        point["near_optimal"] = point in near
        point["rmse_ratio_to_profile_minimum"] = point["train_rmse_K"]/best if point["fit_success"] and best and best > 0 else None
    return {"reference": reference, "requested_grid": requested, "points": points,
            "status": "inconclusive" if len(successful) < len(points) else ("insufficient" if reference["jacobian_rank"] < reference["parameter_count"] else "diagnostic_complete"),
            "all_points_successful": len(successful) == len(points),
            "successful_points": len(successful), "minimum_train_rmse_K": best,
            "near_optimal_factor": NEAR_OPTIMAL_FACTOR,
            "near_optimal_D_values_m2_s": [point["D_m2_s"] for point in near],
            "near_optimal_D_range_m2_s": [near[0]["D_m2_s"], near[-1]["D_m2_s"]] if near else None,
            "near_optimal_touches_grid_edge": bool(near and (near[0]["D_m2_s"] == grid[0] or near[-1]["D_m2_s"] == grid[-1])),
            "near_optimal_touches_parameter_bound": any(point["D_m2_s"] in D_BOUNDS for point in near),
            "is_confidence_interval": False, "used_for_model_selection": False,
            "range_note": "Only sampled near-optimal values; unsampled values are not classified, and an edge hit leaves the range unresolved."}


def block_delete_diagnostics(t, u, dx, model_name, reference_parameters):
    """Delete each of five time blocks from loss, retaining full trajectories."""
    t, u, _ = _validate(t, u, dx, model_name)
    reference = _reference(t, u, dx, model_name, reference_parameters)
    edges = np.linspace(t[0], t[-1], BLOCK_COUNT+1)
    block_ids = np.searchsorted(edges[1:-1], t, side="right")
    records = []
    for block in range(BLOCK_COUNT):
        deleted = block_ids == block
        deleted[0] = False
        retained = ~deleted
        retained[0] = False
        indices = np.flatnonzero(deleted)
        fit = (_safe_diagnostic_fit(t, u, dx, model_name, retained) if len(indices) else
               _failed_diagnostic_fit(t, u, model_name, retained, None,
                                      "Deleted interval contains no observed future rows.", "insufficient"))
        records.append({**fit, "block_index": block,
                        "deleted_interval_s": [float(edges[block]), float(edges[block+1])],
                        "interval_right_closed": block == BLOCK_COUNT-1,
                        "deleted_rows": int(deleted.sum()),
                        "deleted_row_indices": indices.tolist(),
                        "deleted_first_row": int(indices[0]) if len(indices) else None,
                        "deleted_last_row": int(indices[-1]) if len(indices) else None,
                        "deleted_observed_time_range_s": [float(t[indices[0]]), float(t[indices[-1]])] if len(indices) else None})
    successful = [record for record in records if record["fit_success"]]
    ranges = {}
    names = ["D"] + (["h"] if "h" in model_name else []) + (["c"] if "c" in model_name else [])
    for name in names:
        values = [record["parameters"][name] for record in successful]
        full = reference["parameters"][name]
        ranges[name] = {"reference": full, "min": min(values) if values else None,
                        "max": max(values) if values else None,
                        "absolute_range": max(values)-min(values) if values else None,
                        "range_relative_to_reference": (max(values)-min(values))/abs(full) if values and abs(full) > 1e-12 else None}
    return {"reference": reference, "blocks": records, "block_edges_s": edges.tolist(),
            "successful_fits": len(successful), "parameter_ranges": ranges,
            "status": "inconclusive" if len(successful) < BLOCK_COUNT else ("insufficient" if reference["jacobian_rank"] < reference["parameter_count"] else "diagnostic_complete"),
            "all_blocks_successful": len(successful) == BLOCK_COUNT,
            "is_confidence_interval": False, "used_for_model_selection": False,
            "initial_condition_reset": False, "full_boundaries_retained": True}


def diagnose(t, u, dx, model_name, reference_parameters):
    """Run both fixed training-only diagnostics without updating the reference."""
    profile = profile_diffusion(t, u, dx, model_name, reference_parameters)
    blocks = block_delete_diagnostics(t, u, dx, model_name, reference_parameters)
    states = {profile["status"], blocks["status"]}
    status = "inconclusive" if "inconclusive" in states else ("insufficient" if "insufficient" in states else "diagnostic_complete")
    return {"kind": IDENTIFIABILITY_PROTOCOL["kind"], "status": status,
            "model_name": model_name, "reference": profile["reference"], "profile": profile,
            "block_deletion": blocks, "protocol": IDENTIFIABILITY_PROTOCOL,
            "is_confidence_interval": False, "used_for_model_selection": False,
            "unique_physical_equation_established": False}


def _summary_record(case, report):
    reference, profile, blocks = report["reference"], report["profile"], report["block_deletion"]
    interval = profile["near_optimal_D_range_m2_s"]
    spread = blocks["parameter_ranges"]["D"]
    block_range = [spread["min"], spread["max"]] if spread["min"] is not None else None
    return {"case": case, "model_name": report["model_name"], "status": report["status"],
            "reference_D_m2_s": reference["parameters"]["D"], "reference_train_rmse_K": reference["train_rmse_K"],
            "profile_successful_points": profile["successful_points"],
            "profile_near_optimal_D_range_m2_s": interval,
            "profile_near_optimal_D_ratio": interval[1]/interval[0] if interval else None,
            "profile_near_optimal_touches_grid_edge": profile["near_optimal_touches_grid_edge"],
            "profile_near_optimal_touches_parameter_bound": profile["near_optimal_touches_parameter_bound"],
            "block_successful_fits": blocks["successful_fits"], "block_D_range_m2_s": block_range,
            "block_D_ratio": block_range[1]/block_range[0] if block_range else None,
            "reference_jacobian_rank": reference["jacobian_rank"], "parameter_count": reference["parameter_count"],
            "failure_reason": None, "report": f"{case}/report.json"}


def _plot(reports, output):
    import os
    import tempfile
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "pde-lab-matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 3, figsize=(12, 10), layout="constrained")
    for ax, case in zip(axes.ravel(), CASES):
        report = reports.get(case)
        ax.set_title(case)
        if report is None:
            ax.text(.5, .5, "Inconclusive", ha="center", transform=ax.transAxes)
            continue
        points = [p for p in report["profile"]["points"] if p["fit_success"]]
        best = report["profile"]["minimum_train_rmse_K"]
        if points and best is not None and best > 0:
            ax.plot([p["D_over_reference"] for p in points], [p["train_rmse_K"]/best for p in points], "o-", color="#2368a0")
        ax.axhline(NEAR_OPTIMAL_FACTOR, color="#aa513e", linestyle="--", linewidth=1)
        ax.axvline(1, color="#888888", linewidth=.8, linestyle=":")
        ax.set(xscale="log", xlabel="D / frozen D", ylabel="Training RMSE / profile minimum")
        ax.grid(alpha=.2)
    fig.suptitle("Training-only diffusion profiles | 1.05 descriptive loss tolerance, NOT a confidence interval", fontsize=12)
    fig.savefig(Path(output)/"diagnostics.png", dpi=150)
    plt.close(fig)


def run_diagnostics(output=OUTPUT_ROOT, data_root=DATA_ROOT, model_root=MODEL_ROOT):
    """Read only the nine original training CSVs and existing frozen models."""
    output, data_root, model_root = Path(output), Path(data_root), Path(model_root)
    if any(output.resolve().is_relative_to(root.resolve()) or root.resolve().is_relative_to(output.resolve())
           for root in (data_root, model_root)):
        raise ValueError("诊断输出必须与旧数据、冻结模型目录分离。")
    module = Path(__file__).resolve().parent
    protocol = {**IDENTIFIABILITY_PROTOCOL, "cases": list(CASES),
                "source_sha256": {name: _sha(module/name) for name in ("heat_identifiability.py", "real_heat_refined.py")},
                "serialized_before_training_load": True}
    _write_json(output/"protocol.json", protocol)
    index_path = model_root/"freeze_index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    frozen_hashes = {record["case"]: record["frozen_model_sha256"] for record in index["records"]}
    protocol["frozen_index_sha256"] = _sha(index_path)
    _write_json(output/"protocol.json", protocol)
    records, reports = [], {}
    for case in CASES:
        try:
            directory = data_root/case
            manifest_path, model_path = directory/"manifest.json", model_root/case/"frozen_model.json"
            if frozen_hashes.get(case) != _sha(model_path):
                raise ValueError("既有冻结模型与freeze_index校验和不一致。")
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            frozen = json.loads(model_path.read_text(encoding="utf-8"))
            if frozen.get("manifest_sha256") != _sha(manifest_path):
                raise ValueError("既有冻结模型与数据manifest校验和不一致。")
            selected = frozen.get("selected")
            if not selected or not selected.get("fit_success"):
                raise ValueError("没有既有成功冻结支持，不能诊断或替换选择。")
            entry = manifest["splits"]["training"]
            if "rows" in entry:
                raise ValueError("本诊断只允许原完整repeat1训练记录。")
            t, u = load_measurements(directory, entry, manifest["csv_format"])
            report = diagnose(t, u, float(manifest["spacing_m"]), selected["model_name"], selected["parameters"])
            report.update({"case": case, "training_source": entry, "manifest_sha256": _sha(manifest_path),
                           "frozen_model_sha256": _sha(model_path), "numerical_data_roles_loaded": ["training"],
                           "source_sha256": protocol["source_sha256"]})
            reports[case] = report
            _write_json(output/case/"report.json", report)
            record = _summary_record(case, report)
        except (ValueError, KeyError, OSError, RuntimeError, FloatingPointError) as exc:
            record = {"case": case, "status": "inconclusive", "failure_reason": str(exc)}
            _write_json(output/case/"report.json", record)
        records.append(record)
        print(json.dumps({"stage": "training_identifiability", **record}, ensure_ascii=False), flush=True)
    summary = {"kind": IDENTIFIABILITY_PROTOCOL["kind"], "cases": list(CASES), "records": records,
               "is_l1": False, "used_for_model_selection": False, "is_confidence_interval": False,
               "unique_physical_equation_established": False, "protocol": protocol,
               "numerical_data_roles_loaded": ["training"], "generated_at": datetime.now(timezone.utc).isoformat(),
               "all_cases_completed": len(reports) == len(CASES), "plot_available": False}
    _plot(reports, output)
    summary.update({"plot_available": True, "plot_file": "diagnostics.png"})
    _write_json(output/"summary.json", summary)
    return summary
