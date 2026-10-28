"""Physics-prior trajectory fitting and an explicitly retrospective comparison.

Every declared case is frozen before any comparison-repeat numerical values
are parsed. Repeat 3 was already inspected in the first experiment: this is not
a new blind test, even though current candidate selection never accepts it.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .real_benchmark import CASES, DATA_ROOT, PRIMARY_CASE
from .real_experiment import _sha, _write_json, load_measurements
from .real_heat import evaluate
from .real_heat_refined import FIT_PROTOCOL, fit_candidate, fit_candidates, predict

OUTPUT_ROOT = Path("artifacts/refined_heat")
BASELINE_ROOT = Path("artifacts/mendeley_heat")
NEAR_OPTIMAL_FACTOR = 1.05
PROTOCOL_PATH = Path(__file__).resolve().parents[1] / "experiments" / "REFINEMENT_PROTOCOL.md"


def _source_hashes():
    module = Path(__file__).resolve().parent
    return {name: _sha(module / name) for name in ("real_heat_refined.py", "refined_experiment.py",
                                                "real_heat.py", "real_experiment.py")}


def _protocol():
    return {
        "method": "physics_prior_trajectory_fitting",
        "model_family": "theta_t = D theta_xx - h theta + c; D > 0, h >= 0",
        "structure_comparison": ["D", "D+h", "D+c", "D+h+c"],
        "is_l1_optimization": False,
        "physical_prior": "扩散必选；线性衰减和常数源可选；四个预设结构比较，不是无先验方程发现。",
        "fitting": json.loads(json.dumps(FIT_PROTOCOL)),
        "selection_rule": "只用 repeat2 的四内点 RMSE；最优值1.05倍内先最少项，再最低RMSE，最后模型名称。",
        "eligibility_rule": "训练优化收敛、局部参数敏感性Jacobian满列秩且开发预测有限；满秩不证明物理机制唯一。",
        "near_optimal_factor": NEAR_OPTIMAL_FACTOR,
        "fit_data": "完整 repeat1",
        "selection_data": "完整 repeat2",
        "comparison_data": "完整 repeat3（上一轮已查看）",
        "prior_seen_test": True,
        "previous_test_already_seen": True,
        "new_blind_validation": False,
        "final_data_selects_model": False,
        "all_declared_cases_frozen_before_comparison_load": True,
        "prediction_inputs": "两端完整实测温升 + 四个内部测点初始温升；无后续内部实测值。",
        "scoring": "仅四内点；排除初始化首帧。",
        "units": {"time": "s", "space": "m", "temperature": "K above ambient"},
        "spatial_upsampling": False,
        "smoothing": "none",
        "stability_diagnostic": "训练记录按行数分成不重叠前半和后半，分别拟合已选支持；不选模，不作置信区间。",
        "protocol_document_sha256": _sha(PROTOCOL_PATH) if PROTOCOL_PATH.exists() else None,
        "source_sha256": _source_hashes(),
    }


def _validate_manifest(data_dir):
    path = Path(data_dir) / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    entries = manifest["splits"]
    if (manifest["split_kind"] != "independent_repeated_acquisitions"
            or set(entries) != {"training", "development", "final"}
            or any("rows" in value for value in entries.values())
            or len({value["sha256"] for value in entries.values()}) != 3
            or len({value["file"] for value in entries.values()}) != 3):
        raise ValueError("改进协议要求三个不同的完整重复实验；不接受时间裁剪或同文件分组。")
    return manifest, _sha(path)


def _score(terms, t, u, dx):
    trajectory = predict(terms, t, u[:, [0, 5]], u[0, 1:5], dx)
    return trajectory, evaluate(trajectory, u[:, 1:5])


def _stability(selected, t, u, dx):
    midpoint = len(t) // 2
    records = []
    for label, selection in (("first_half", slice(0, midpoint)), ("second_half", slice(midpoint, None))):
        try:
            fit = fit_candidate(t[selection], u[selection], dx, selected["model_name"])
            records.append({"segment": label, "rows": len(t[selection]),
                            "time_start_s": float(t[selection][0]), "time_end_s": float(t[selection][-1]),
                            "fit_success": fit["fit_success"], "terms": fit.get("terms", []),
                            "diagnostics": fit.get("diagnostics", {}),
                            "train_score": fit.get("train_score")})
        except (ValueError, RuntimeError, FloatingPointError) as exc:
            records.append({"segment": label, "fit_success": False, "reason": str(exc)})
    changes = []
    if all(record["fit_success"] for record in records):
        halves = [{term["name"]: term["coefficient"] for term in record["terms"]} for record in records]
        for term in selected["terms"]:
            name, full = term["name"], term["coefficient"]
            first, second = halves[0].get(name, 0.0), halves[1].get(name, 0.0)
            changes.append({"term": name, "whole_training": full, "first_half": first, "second_half": second,
                            "absolute_difference": abs(first-second),
                            "difference_relative_to_full": abs(first-second) / max(abs(full), 1e-12)})
    return {"kind": "descriptive_training_half_refits_not_confidence_intervals",
            "used_for_selection": False, "segments": records, "coefficient_changes": changes}


def freeze_case(case, data_root=DATA_ROOT, output_root=OUTPUT_ROOT):
    """Fit and select using only training/development roles, then save a model."""
    if case not in CASES:
        raise ValueError("不支持的实测工况。")
    data_dir, output = Path(data_root) / case, Path(output_root) / case
    protocol = _protocol()
    candidates, selected, near_supports = [], None, []
    manifest = None
    failure = None
    manifest_hash = None
    try:
        manifest, manifest_hash = _validate_manifest(data_dir)
        protocol.update({"split": manifest["splits"], "split_kind": manifest["split_kind"],
                         "spacing_m": float(manifest["spacing_m"])})
        entries, dx = manifest["splits"], float(manifest["spacing_m"])
        t_train, u_train = load_measurements(data_dir, entries["training"], manifest["csv_format"])
        t_dev, u_dev = load_measurements(data_dir, entries["development"], manifest["csv_format"])
        candidates = fit_candidates(t_train, u_train, dx)
        for candidate in candidates:
            candidate["eligible"] = False
            if not candidate.get("fit_success", False):
                candidate["rejection_reason"] = "训练轨迹拟合失败或优化未收敛。"
                continue
            diagnostics = candidate.get("diagnostics", {})
            rank, parameter_count = diagnostics.get("jacobian_rank"), diagnostics.get("parameter_count")
            if rank is None or parameter_count is None or rank < parameter_count:
                candidate["rejection_reason"] = "训练局部参数敏感性秩不足或缺少秩诊断，不能据此选定系数。"
                continue
            try:
                _, candidate["development"] = _score(candidate["terms"], t_dev, u_dev, dx)
                candidate["eligible"] = True
            except (ValueError, RuntimeError, FloatingPointError) as exc:
                candidate["rejection_reason"] = str(exc)
        eligible = [candidate for candidate in candidates if candidate["eligible"]]
        if eligible:
            best = min(candidate["development"]["rmse"] for candidate in eligible)
            near = [candidate for candidate in eligible
                    if candidate["development"]["rmse"] <= best * NEAR_OPTIMAL_FACTOR + 1e-12]
            selected = min(near, key=lambda c: (c["support_size"], c["development"]["rmse"], c["model_name"]))
            near_supports = sorted({candidate["model_name"] for candidate in near})
            stability = _stability(selected, t_train, u_train, dx)
        else:
            stability = None
            failure = "训练和开发阶段没有可用的稳定候选。"
    except (ValueError, KeyError, OSError, RuntimeError, FloatingPointError) as exc:
        stability = None
        failure = str(exc)
    frozen = {"case": case, "manifest_sha256": manifest_hash, "experiment": (manifest or {}).get("experiment", {}),
              "protocol": protocol, "selected": selected, "candidates": candidates,
              "near_optimal_supports": near_supports, "training_stability": stability,
              "failure_reason": failure}
    _write_json(output / "frozen_model.json", frozen)
    # A rerun cannot advertise a previous success while the new round is pending.
    _write_json(output / "report.json", _base_report(frozen, _sha(output / "frozen_model.json")))
    return frozen


def _base_report(frozen, frozen_hash):
    selected = frozen["selected"]
    near = frozen["near_optimal_supports"]
    reasons = ["开发误差近优池含多个预设结构。"] if len(near) > 1 else []
    discovery = ({**selected, "status": "ambiguous" if reasons else "candidate"}
                 if selected else {"status": "insufficient", "equation": "没有可验证候选", "terms": [], "support_size": 0})
    discovery.update({"near_optimal_supports": near, "ambiguity_reasons": reasons,
                      "unique_physical_equation_established": False})
    return {
        "case": frozen["case"], "experiment": frozen["experiment"], "protocol": frozen["protocol"],
        "prior_seen_test": True, "new_blind_validation": False,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "frozen_model_sha256": frozen_hash, "candidates": frozen["candidates"],
        "discovery": discovery, "training_stability": frozen["training_stability"],
        "validation": {"status": "not_evaluated"},
        "plot_available": False, "trajectory_available": False,
        "comparison": {"kind": "retrospective_method_comparison", "previous_test_already_seen": True},
        "conclusion": frozen.get("failure_reason") or "模型已冻结，等待本轮回顾性比较；repeat3在上一轮已查看。",
        "limitations": [
            "本轮是已见过 repeat3 的回顾性方法比较，不是新独立盲验。",
            "扩散项是物理先验必选项，四支持比较不是 L1 自动发现。",
            "预测依赖两端全程实测温升，只计四个内点误差，不是无边界输入的自由预测。",
            "参数是六原始测点的有效半离散系数，不自动等同连续材料物性。",
            "训练分段稳定性是描述性诊断，不提供统计置信区间。",
            "未自行操作实验设备，未验证跨装置泛化，未接入华为平台。",
        ],
    }


def freeze_benchmark(output_root=OUTPUT_ROOT, *, data_root=DATA_ROOT, cases=CASES, baseline_root=BASELINE_ROOT):
    """Freeze ALL requested cases, then write the index permitting comparison."""
    output_root, baseline_root = Path(output_root), Path(baseline_root)
    if output_root.resolve().is_relative_to(baseline_root.resolve()):
        raise ValueError("改进分支不能覆盖上一轮基线目录。")
    cases = tuple(cases)
    if not cases or len(set(cases)) != len(cases) or any(case not in CASES for case in cases):
        raise ValueError("需指定非空、不重复的合法实测工况。")
    records = []
    for case in cases:
        frozen = freeze_case(case, data_root=data_root, output_root=output_root)
        path = output_root / case / "frozen_model.json"
        records.append({"case": case, "frozen_model_sha256": _sha(path),
                        "manifest_sha256": frozen["manifest_sha256"],
                        "selected_model": frozen["selected"]["model_name"] if frozen["selected"] else None,
                        "failure_reason": frozen["failure_reason"]})
        print(json.dumps({"stage": "frozen", **records[-1]}, ensure_ascii=False), flush=True)
    index = {"schema_version": 1, "cases": list(cases), "records": records,
             "primary_case": PRIMARY_CASE, "all_nine_cases": cases == CASES,
             "source_sha256": _source_hashes(), "protocol": _protocol(),
             "prior_seen_test": True, "new_blind_validation": False,
             "final_numeric_values_loaded_in_freeze": False}
    _write_json(output_root / "freeze_index.json", index)
    return index


def _verify_frozen_index(output_root, data_root):
    index = json.loads((output_root / "freeze_index.json").read_text(encoding="utf-8"))
    cases = index["cases"]
    if (not cases or len(cases) != len(set(cases)) or any(case not in CASES for case in cases)
            or [record["case"] for record in index["records"]] != cases):
        raise ValueError("冻结索引的工况列表无效。")
    if index["source_sha256"] != _source_hashes():
        raise ValueError("代码已发生变化；须重新执行训练/开发冻结后再比较。")
    if index["protocol"] != _protocol():
        raise ValueError("协议已发生变化；须重新冻结。")
    frozen_models = []
    # Verify every model before permitting a single final numerical load.
    for record in index["records"]:
        case = record["case"]
        path = output_root / case / "frozen_model.json"
        if _sha(path) != record["frozen_model_sha256"]:
            raise ValueError(f"冻结模型校验失败：{case}")
        frozen = json.loads(path.read_text(encoding="utf-8"))
        if frozen["case"] != case:
            raise ValueError(f"冻结模型工况不一致：{case}")
        if frozen["manifest_sha256"] is not None:
            if _sha(Path(data_root) / case / "manifest.json") != frozen["manifest_sha256"]:
                raise ValueError(f"数据划分清单已变化：{case}")
        frozen_models.append(frozen)
    return index, frozen_models


def _old_comparison(case, baseline_root):
    path = Path(baseline_root) / case / "report.json"
    result = {"kind": "retrospective_method_comparison", "previous_test_already_seen": True,
              "baseline_relative_l2": None, "baseline_rmse_K": None, "baseline_support_size": None,
              "old_report_sha256": None, "improved": None}
    if path.exists():
        try:
            old = json.loads(path.read_text(encoding="utf-8"))
            validation, discovery = old.get("validation", {}), old.get("discovery", {})
            result.update({"baseline_relative_l2": validation.get("relative_l2"),
                           "baseline_rmse_K": validation.get("rmse_K"),
                           "baseline_support_size": discovery.get("support_size"),
                           "baseline_equation": discovery.get("equation"),
                           "old_report_sha256": _sha(path)})
        except (ValueError, OSError, AttributeError) as exc:
            result["baseline_unavailable_reason"] = str(exc)
    return result


def _evaluate_case(frozen, record, output_root, data_root, baseline_root):
    case, selected = frozen["case"], frozen["selected"]
    output = Path(output_root) / case
    report = _base_report(frozen, record["frozen_model_sha256"])
    report["comparison"] = _old_comparison(case, baseline_root)
    # Save a non-success state first. Old plots remain inaccessible until a
    # successful current comparison finishes and explicitly sets both flags.
    _write_json(output / "report.json", report)
    if selected is None:
        return report
    try:
        data_dir = Path(data_root) / case
        manifest, _ = _validate_manifest(data_dir)
        t, u = load_measurements(data_dir, manifest["splits"]["final"], manifest["csv_format"])
        prediction, score = _score(selected["terms"], t, u, float(manifest["spacing_m"]))
        persistence = np.broadcast_to(u[0, 1:5], prediction.shape).copy()
        weights = np.arange(1, 5)[None, :] / 5
        linear = u[:, :1] * (1-weights) + u[:, -1:] * weights
        baselines = {"persistence": {"label": "初始温升持久值", **evaluate(persistence, u[:, 1:5])},
                     "linear_boundary": {"label": "同边界线性空间插值", **evaluate(linear, u[:, 1:5])}}
        validation = {"status": "evaluated", "kind": "已见重复实验的回顾性条件预测比较",
                      "rmse_K": score["rmse"], "relative_l2": score["relative_error"], **score,
                      "per_sensor": [{"sensor": f"TC{i+2}", "rmse_K": value,
                                      "x_m": (i+1)*manifest["spacing_m"]}
                                     for i, value in enumerate(score["per_sensor_rmse"])],
                      "baselines": baselines,
                      "beats_both_baselines": all(score["rmse"] < value["rmse"] for value in baselines.values()),
                      "below_descriptive_5pct": score["relative_error"] is not None and score["relative_error"] < .05,
                      "final_file": manifest["splits"]["final"]["file"],
                      "scored_sensor_count": 4, "measured_boundary_sensor_count": 2,
                      "rows": len(t), "time_start_s": float(t[0]), "time_end_s": float(t[-1])}
        report["validation"] = validation
        old_error = report["comparison"]["baseline_relative_l2"]
        if old_error is not None and score["relative_error"] is not None:
            report["comparison"]["improved"] = bool(score["relative_error"] < old_error)
            report["comparison"]["relative_l2_difference"] = score["relative_error"] - old_error
        relative_text = f"{100*score['relative_error']:.3f}%" if score["relative_error"] is not None else "未定义"
        report["conclusion"] = (f"物理先验模型 {selected['model_name']} 的回顾性条件预测相对 L2 为 {relative_text}，"
                                f"RMSE 为 {score['rmse']:.4g} K。repeat3 已在上一轮查看；不属于新盲验，"
                                "较少项或较低误差不证明真实 PDE 已唯一确定。")
        np.savez_compressed(output / "trajectories.npz", t=t, observed=u, predicted_interior=prediction,
                            persistence=persistence, linear_boundary=linear)
        _plot_case(t, u, prediction, linear, report, output)
        report["trajectory_available"] = True
        report["plot_available"] = True
    except (ValueError, OSError, RuntimeError, FloatingPointError) as exc:
        report["validation"] = {"status": "failed", "failure_reason": str(exc)}
        report["conclusion"] = "冻结模型的回顾性比较失败；未换模型、重新选模或用比较数据重新拟合。"
    _write_json(output / "report.json", report)
    return report


def evaluate_benchmark(output_root=OUTPUT_ROOT, *, data_root=DATA_ROOT, baseline_root=BASELINE_ROOT):
    """Compare previously frozen models without fitting or candidate selection."""
    output_root = Path(output_root)
    index, frozen_models = _verify_frozen_index(output_root, data_root)
    records = []
    for frozen, entry in zip(frozen_models, index["records"]):
        report = _evaluate_case(frozen, entry, output_root, data_root, baseline_root)
        discovery, validation, comparison = report["discovery"], report["validation"], report["comparison"]
        record = {"case": frozen["case"], "title": report["experiment"].get("title", frozen["case"]),
                  "equation": discovery["equation"], "model_name": discovery.get("model_name"),
                  "status": discovery["status"], "validation_status": validation["status"],
                  "rmse_K": validation.get("rmse_K"), "relative_l2": validation.get("relative_l2"),
                  "support_size": discovery.get("support_size", 0),
                  "baseline_relative_l2": comparison.get("baseline_relative_l2"),
                  "baseline_rmse_K": comparison.get("baseline_rmse_K"),
                  "baseline_support_size": comparison.get("baseline_support_size"),
                  "improved": comparison.get("improved"),
                  "beats_both_baselines": validation.get("beats_both_baselines", False),
                  "below_descriptive_5pct": validation.get("below_descriptive_5pct", False),
                  "frozen_model_sha256": report["frozen_model_sha256"],
                  "failure_reason": validation.get("failure_reason") or frozen["failure_reason"],
                  "report": f"{frozen['case']}/report.json"}
        records.append(record)
        print(json.dumps({"stage": "retrospective_comparison", **record}, ensure_ascii=False), flush=True)
    summary = {"description": "物理先验轨迹拟合与旧 L1 方法的回顾性比较；上一轮已查看 repeat3，不能称新盲验。",
               "prior_seen_test": True, "new_blind_validation": False,
               "prediction_scope": "两端全程实测温升条件下四内点预测；首帧不计分。",
               "method": "physics_prior_trajectory_fitting", "is_l1_optimization": False,
               "selected_case": PRIMARY_CASE if PRIMARY_CASE in index["cases"] else index["cases"][0],
               "all_nine_cases": index["all_nine_cases"], "records": records,
               "freeze_index_sha256": _sha(output_root / "freeze_index.json"),
               "generated_at": datetime.now(timezone.utc).isoformat(),
               "unique_physical_equation_established": False}
    _write_json(output_root / "benchmark.json", summary)
    _plot_summary(summary, output_root)
    return summary


def run_benchmark(output_root=OUTPUT_ROOT, *, data_root=DATA_ROOT, cases=CASES, baseline_root=BASELINE_ROOT):
    freeze_benchmark(output_root, data_root=data_root, cases=cases, baseline_root=baseline_root)
    return evaluate_benchmark(output_root, data_root=data_root, baseline_root=baseline_root)


def run_case(case=PRIMARY_CASE, output_root=OUTPUT_ROOT, *, data_root=DATA_ROOT, baseline_root=BASELINE_ROOT):
    existing = Path(output_root) / "freeze_index.json"
    if existing.exists():
        old_index = json.loads(existing.read_text(encoding="utf-8"))
        if old_index.get("cases") != [case]:
            raise ValueError("单组复现请使用独立输出目录，不能覆盖已有多组冻结索引与汇总。")
    run_benchmark(output_root, data_root=data_root, cases=(case,), baseline_root=baseline_root)
    return json.loads((Path(output_root) / case / "report.json").read_text(encoding="utf-8"))


def _pyplot():
    import os
    import tempfile
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "pde-lab-matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _plot_case(t, u, prediction, linear, report, output):
    plt = _pyplot()
    fig, axes = plt.subplots(2, 3, figsize=(14, 7), layout="constrained")
    for i, ax in enumerate(axes.flat[:4]):
        ax.plot(t, u[:, i+1], color="#19384b", lw=1, label="Measured")
        ax.plot(t, prediction[:, i], color="#d65a38", lw=1.6, label="Physics-prior model")
        ax.plot(t, linear[:, i], color="#70a39a", lw=1, linestyle="--", label="Linear boundary baseline")
        ax.set(title=f"TC{i+2}: comparison interior", xlabel="Time (s)", ylabel="Excess temperature (K)")
        if i == 0:
            ax.legend(fontsize=8)
    axes.flat[4].plot(t, u[:, 0], label="TC1: input")
    axes.flat[4].plot(t, u[:, 5], label="TC6: input")
    axes.flat[4].set(title="Measured boundary forcing (not scored)", xlabel="Time (s)", ylabel="Excess temperature (K)")
    axes.flat[4].legend(fontsize=8)
    ax = axes.flat[5]
    v, comparison = report["validation"], report["comparison"]
    labels, values = ["Refined", "Persistence", "Linear boundary"], [v["rmse_K"], v["baselines"]["persistence"]["rmse"], v["baselines"]["linear_boundary"]["rmse"]]
    if comparison.get("baseline_rmse_K") is not None:
        labels.insert(1, "Old L1")
        values.insert(1, comparison["baseline_rmse_K"])
    ax.bar(labels, values, color="#d65a38")
    ax.set(title="Four interior states; initial frame excluded", ylabel="RMSE (K)")
    ax.tick_params(axis="x", labelrotation=12)
    fig.suptitle(f"{report['case']} | Retrospective comparison; repeat 3 previously seen", fontweight="bold")
    fig.savefig(Path(output) / "validation.png", dpi=160)
    plt.close(fig)


def _plot_summary(summary, output_root):
    plt = _pyplot()
    records = summary["records"]
    x = np.arange(len(records))
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), layout="constrained")
    for offset, key, label, color in ((-.18, "baseline_relative_l2", "Old L1", "#8294a3"),
                                      (.18, "relative_l2", "Physics-prior trajectory fit", "#138a79")):
        vals = [100*record[key] if record.get(key) is not None else np.nan for record in records]
        axes[0].bar(x+offset, vals, width=.36, label=label, color=color)
        for i, value in enumerate(vals):
            axes[0].text(i+offset, (value if np.isfinite(value) else 0)+.5,
                         f"{value:.1f}" if np.isfinite(value) else "N/A", ha="center", fontsize=7)
    axes[0].axhline(5, color="#ba4239", ls="--", lw=1, label="5% descriptive reference")
    axes[0].set(ylabel="Relative L2 prediction error (%)", title="Same previously seen repeat, all cases retained")
    axes[0].legend(fontsize=8)
    axes[1].bar(x-.18, [r.get("baseline_support_size") or 0 for r in records], width=.36, color="#8294a3", label="Old L1")
    axes[1].bar(x+.18, [r["support_size"] for r in records], width=.36, color="#138a79", label="Physics-prior")
    axes[1].set(ylabel="Retained terms", ylim=(0, 5.8), yticks=range(6), title="Different candidate families; diffusion is a prior")
    axes[1].legend(fontsize=8)
    for ax in axes:
        ax.set_xticks(x, [r["case"] for r in records], rotation=35)
    fig.suptitle("Retrospective method comparison, NOT a new blind test", fontweight="bold")
    fig.savefig(Path(output_root) / "benchmark.png", dpi=160)
    plt.close(fig)
