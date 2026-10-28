"""Fixed, specimen-specific identification with twelve project-new holdouts.

This module does not change the previously calibrated numerical method. Four
specimens are fitted on acquisition 1 and selected on acquisition 2. Every
model is frozen before any numerical values from acquisitions 3/4/5 are read.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .real_experiment import _sha, _write_json, load_measurements
from .real_heat import evaluate
from .real_heat_refined import FIT_PROTOCOL, fit_candidate, fit_candidates, predict

GROUPS = ("xn", "xp", "yn", "yp")
HOLDOUT_REPEATS = (3, 4, 5)
DATA_ROOT = Path(__file__).resolve().parent / "data" / "additive_heat"
OUTPUT_ROOT = Path("artifacts/additive_heat")
NEAR_OPTIMAL_FACTOR = 1.05
PROTOCOL_PATH = Path(__file__).resolve().parents[1] / "experiments" / "ADDITIVE_PROTOCOL.md"


def _source_hashes():
    folder = Path(__file__).resolve().parent
    return {name: _sha(folder / name) for name in ("additive_experiment.py", "real_heat_refined.py",
                                                "real_heat.py", "real_experiment.py")}


def _protocol():
    return {
        "method": "unchanged_physics_prior_trajectory_fitting",
        "fitting": json.loads(json.dumps(FIT_PROTOCOL)),
        "groups": list(GROUPS), "holdout_repeats": list(HOLDOUT_REPEATS),
        "is_l1_optimization": False, "is_zero_shot_transfer": False,
        "new_project_holdout": True,
        "validation_scope": "new_to_this_project_holdout_records",
        "claim_scope": "本项目此前未用于建模的记录；不宣称全世界未看过、统计独立、严格同条件或跨装置泛化。",
        "fit_data": "每根杆完整编号1", "selection_data": "每根杆完整编号2",
        "holdout_data": "每根杆完整编号3/4/5，全部报告",
        "selection_rule": "开发RMSE在最优值1.05倍以内先最少项，再较低RMSE，最后模型名称。",
        "near_optimal_factor": NEAR_OPTIMAL_FACTOR,
        "eligibility_rule": "训练优化收敛、参数敏感性Jacobian满列秩、开发预测有限；局部满秩不证明物性。",
        "physical_prior": "正扩散必选，非负线性衰减与常数源可选；四个预设结构比较。",
        "all_declared_groups_frozen_before_any_holdout_load": True,
        "holdout_data_selects_model": False,
        "prediction_inputs": "两端完整实测温升与四内点初始值，不传未来内部测量。",
        "scoring": "仅四内点，排除初始化首帧。",
        "units": {"time": "s", "space": "m", "temperature": "K excess above ambient"},
        "spatial_upsampling": False, "smoothing": "none",
        "stability_diagnostic": "选模后对训练前后两个不重叠半段重拟合该结构，只作描述，不是置信区间。",
        "protocol_document_sha256": _sha(PROTOCOL_PATH) if PROTOCOL_PATH.exists() else None,
        "source_metadata_sha256": {name: _sha(DATA_ROOT / name) for name in
                                   ("metadata_verification.md", "source_excerpts.txt") if (DATA_ROOT / name).exists()},
        "source_sha256": _source_hashes(),
    }


def _validate_manifest(data_dir):
    data_dir = Path(data_dir)
    path = data_dir / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    splits, holdouts = manifest["splits"], manifest["holdouts"]
    if (manifest["split_kind"] != "repeated_acquisitions_same_specimen"
            or set(splits) != {"training", "development"}
            or splits["training"].get("repeat") != 1 or splits["development"].get("repeat") != 2
            or [entry.get("repeat") for entry in holdouts] != list(HOLDOUT_REPEATS)):
        raise ValueError("须固定同杆编号1训练、2开发、3/4/5留出且顺序一致。")
    entries = [splits["training"], splits["development"], *holdouts]
    if any("rows" in entry for entry in entries):
        raise ValueError("协议只接受完整记录，不接受rows裁剪。")
    paths = [(data_dir / entry["file"]).resolve() for entry in entries]
    if any(not path.is_relative_to(data_dir.parent.resolve()) for path in paths):
        raise ValueError("数据路径必须位于本轮数据根目录内。")
    if len(set(paths)) != 5 or len({entry["sha256"] for entry in entries}) != 5:
        raise ValueError("同杆五个角色必须是不同原始文件和不同SHA-256；路径别名也不能重复。")
    dx = float(manifest["spacing_m"])
    if not np.isfinite(dx) or dx <= 0:
        raise ValueError("传感器间距必须是有限正数。")
    return manifest, _sha(data_dir / "manifest.json")


def _preflight(data_root, groups):
    prepared, all_paths, all_hashes = {}, [], []
    for group in groups:
        directory = Path(data_root) / group
        manifest, digest = _validate_manifest(directory)
        prepared[group] = (manifest, digest)
        entries = [manifest["splits"]["training"], manifest["splits"]["development"], *manifest["holdouts"]]
        all_paths.extend((directory / entry["file"]).resolve() for entry in entries)
        all_hashes.extend(entry["sha256"] for entry in entries)
    if len(set(all_paths)) != len(all_paths) or len(set(all_hashes)) != len(all_hashes):
        raise ValueError("不同杆之间也不能重复使用同一原始文件或相同SHA-256记录。")
    return prepared


def _score(terms, t, u, dx):
    predicted = predict(terms, t, u[:, [0, 5]], u[0, 1:5], dx)
    return predicted, evaluate(predicted, u[:, 1:5])


def _stability(selected, t, u, dx):
    records = []
    midpoint = len(t) // 2
    for label, selection in (("first_half", slice(0, midpoint)), ("second_half", slice(midpoint, None))):
        try:
            fit = fit_candidate(t[selection], u[selection], dx, selected["model_name"])
            records.append({"segment": label, "rows": len(t[selection]),
                            "fit_success": fit["fit_success"], "terms": fit.get("terms", []),
                            "diagnostics": fit.get("diagnostics", {}), "train_score": fit.get("train_score")})
        except (ValueError, RuntimeError, FloatingPointError) as exc:
            records.append({"segment": label, "fit_success": False, "reason": str(exc)})
    changes = []
    if all(record["fit_success"] for record in records):
        halves = [{term["name"]: term["coefficient"] for term in record["terms"]} for record in records]
        for term in selected["terms"]:
            name, coefficient = term["name"], term["coefficient"]
            first, second = halves[0].get(name, 0.), halves[1].get(name, 0.)
            changes.append({"term": name, "whole_training": coefficient, "first_half": first, "second_half": second,
                            "difference_relative_to_full": abs(first-second) / max(abs(coefficient), 1e-12)})
    return {"kind": "descriptive_training_half_refits_not_confidence_intervals", "used_for_selection": False,
            "segments": records, "coefficient_changes": changes}


def _freeze_group(group, manifest, manifest_hash, data_root, output_root):
    protocol = _protocol()
    protocol.update({"split_kind": manifest["split_kind"], "split": manifest["splits"],
                     "holdouts": manifest["holdouts"], "spacing_m": manifest["spacing_m"]})
    candidates, selected, near_supports, stability, failure = [], None, [], None, None
    try:
        directory, entries = Path(data_root) / group, manifest["splits"]
        t_train, u_train = load_measurements(directory, entries["training"], manifest["csv_format"])
        t_dev, u_dev = load_measurements(directory, entries["development"], manifest["csv_format"])
        dx = float(manifest["spacing_m"])
        candidates = fit_candidates(t_train, u_train, dx)
        for candidate in candidates:
            candidate["eligible"] = False
            diagnostics = candidate.get("diagnostics", {})
            rank, count = diagnostics.get("jacobian_rank"), diagnostics.get("parameter_count")
            if not candidate.get("fit_success", False):
                candidate["rejection_reason"] = "训练拟合失败或未收敛。"
            elif rank is None or count is None or rank < count:
                candidate["rejection_reason"] = "局部敏感性秩不足或缺少秩诊断。"
            else:
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
            failure = "训练和开发阶段没有可用候选；本组留出不用于补救选模。"
    except (ValueError, OSError, RuntimeError, FloatingPointError, IndexError) as exc:
        failure = str(exc)
    frozen = {"group": group, "manifest_sha256": manifest_hash, "experiment": manifest["experiment"],
              "protocol": protocol, "selected": selected, "candidates": candidates,
              "near_optimal_supports": near_supports, "training_stability": stability, "failure_reason": failure}
    output = Path(output_root) / group
    _write_json(output / "frozen_model.json", frozen)
    digest = _sha(output / "frozen_model.json")
    # Any previous successful reports are invalidated before a rerun can score.
    for repeat in HOLDOUT_REPEATS:
        _write_json(output / f"repeat{repeat}" / "report.json", _base_report(frozen, digest, repeat))
    _write_json(output / "report.json", {**_base_report(frozen, digest, None), "holdout_records": []})
    return frozen


def _base_report(frozen, frozen_hash, repeat):
    selected, near = frozen["selected"], frozen["near_optimal_supports"]
    reasons = ["开发近优池有多个预设支持。"] if len(near) > 1 else []
    discovery = ({**selected, "status": "ambiguous" if reasons else "candidate"} if selected
                 else {"status": "insufficient", "equation": "没有可验证候选", "terms": [], "support_size": 0})
    discovery.update({"near_optimal_supports": near, "ambiguity_reasons": reasons,
                      "unique_physical_equation_established": False})
    return {"group": frozen["group"], "repeat": repeat,
            "case": frozen["group"] + (str(repeat) if repeat is not None else ""),
            "experiment": frozen["experiment"], "protocol": frozen["protocol"],
            "new_project_holdout": True, "is_zero_shot_transfer": False,
            "validation_scope": "new_to_this_project_holdout_records",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "frozen_model_sha256": frozen_hash, "discovery": discovery, "candidates": frozen["candidates"],
            "training_stability": frozen["training_stability"], "validation": {"status": "not_evaluated"},
            "plot_available": False, "trajectory_available": False,
            "conclusion": frozen["failure_reason"] or "本组模型已冻结，等待本项目新增留出记录验证。",
            "limitations": [
                "每根杆重新拟合系数，不是跨材料零样本迁移。",
                "同杆五次测试关系有作者文献支持，但逐次风速、加热与重置条件未完全核实。",
                "本项目此前未用于建模不等于全世界从未看过或统计独立盲验。",
                "new_project_holdout指初次设计时的数据角色；评分后本项目已查看，后续重跑是复现，不形成新的盲验。",
                "给定完整实测两端边界，只评价四内部测点；不是无未来边界的自由预测。",
                "扩散先验必选，四预设结构比较不是 L1 自动发现；简洁或低误差不证明真实 PDE 唯一。",
                "六点有效系数不自动等同连续材料热物性；未进行跨设备验证。",
            ]}


def freeze_benchmark(output_root=OUTPUT_ROOT, *, data_root=DATA_ROOT, groups=GROUPS):
    groups, output_root = tuple(groups), Path(output_root)
    if not groups or len(set(groups)) != len(groups) or any(group not in GROUPS for group in groups):
        raise ValueError("需非空且不重复的合法增材杆组别。")
    for previous in (Path("artifacts/refined_heat"), Path("artifacts/mendeley_heat")):
        if output_root.resolve().is_relative_to(previous.resolve()):
            raise ValueError("新增验证不得覆盖此前实验目录。")
    # No numerical file is parsed until ALL role assignments pass preflight.
    prepared = _preflight(data_root, groups)
    records = []
    for group in groups:
        manifest, digest = prepared[group]
        frozen = _freeze_group(group, manifest, digest, data_root, output_root)
        record = {"group": group, "manifest_sha256": digest,
                  "frozen_model_sha256": _sha(output_root / group / "frozen_model.json"),
                  "selected_model": frozen["selected"]["model_name"] if frozen["selected"] else None,
                  "failure_reason": frozen["failure_reason"]}
        records.append(record)
        print(json.dumps({"stage": "frozen", **record}, ensure_ascii=False), flush=True)
    index = {"schema_version": 1, "groups": list(groups), "records": records,
             "all_four_groups": groups == GROUPS, "protocol": _protocol(), "source_sha256": _source_hashes(),
             "holdout_numeric_values_loaded_in_freeze": False,
             "validation_scope": "new_to_this_project_holdout_records"}
    _write_json(output_root / "freeze_index.json", index)
    return index


def _verify_frozen_index(output_root, data_root):
    index = json.loads((Path(output_root) / "freeze_index.json").read_text(encoding="utf-8"))
    groups = index["groups"]
    if (not groups or len(set(groups)) != len(groups) or any(group not in GROUPS for group in groups)
            or [record["group"] for record in index["records"]] != groups):
        raise ValueError("冻结索引分组无效。")
    if index["source_sha256"] != _source_hashes() or index["protocol"] != _protocol():
        raise ValueError("算法源码或协议已变化；不能继续沿用当前冻结模型。")
    prepared = _preflight(data_root, groups)
    frozen_models = []
    for entry in index["records"]:
        group = entry["group"]
        path = Path(output_root) / group / "frozen_model.json"
        if _sha(path) != entry["frozen_model_sha256"]:
            raise ValueError(f"冻结模型校验失败：{group}")
        frozen = json.loads(path.read_text(encoding="utf-8"))
        if (frozen["group"] != group or prepared[group][1] != frozen["manifest_sha256"]
                or prepared[group][1] != entry["manifest_sha256"]):
            raise ValueError(f"数据划分清单或组别与冻结记录不符：{group}")
        frozen_models.append(frozen)
    return index, frozen_models, prepared


def _evaluate_holdout(frozen, digest, manifest, holdout, output_root, data_root):
    group, repeat = frozen["group"], holdout["repeat"]
    output = Path(output_root) / group / f"repeat{repeat}"
    report = _base_report(frozen, digest, repeat)
    _write_json(output / "report.json", report)
    if frozen["selected"] is None:
        return report
    try:
        t, u = load_measurements(Path(data_root) / group, holdout, manifest["csv_format"])
        prediction, score = _score(frozen["selected"]["terms"], t, u, float(manifest["spacing_m"]))
        persistence = np.broadcast_to(u[0, 1:5], prediction.shape).copy()
        weights = np.arange(1, 5)[None, :] / 5
        linear = u[:, :1] * (1-weights) + u[:, -1:] * weights
        baselines = {"persistence": {"label": "初始温升持久值", **evaluate(persistence, u[:, 1:5])},
                     "linear_boundary": {"label": "同边界线性空间插值", **evaluate(linear, u[:, 1:5])}}
        report["validation"] = {
            "status": "evaluated", "kind": "本项目此前未用于建模的同杆留出记录条件预测",
            "relative_l2": score["relative_error"], "rmse_K": score["rmse"], **score,
            "baselines": baselines, "holdout_file": holdout["file"],
            "per_sensor": [{"sensor": f"TC{i+2}", "rmse_K": value, "x_m": (i+1)*manifest["spacing_m"]}
                           for i, value in enumerate(score["per_sensor_rmse"])],
            "beats_both_baselines": all(score["rmse"] < value["rmse"] for value in baselines.values()),
            "below_descriptive_5pct": score["relative_error"] is not None and score["relative_error"] < .05,
            "scored_sensor_count": 4, "measured_boundary_sensor_count": 2, "rows": len(t),
            "time_start_s": float(t[0]), "time_end_s": float(t[-1])}
        relative = f"{100*score['relative_error']:.3f}%" if score["relative_error"] is not None else "未定义"
        report["conclusion"] = (f"{group} 编号 {repeat} 留出记录四内点相对 L2 为 {relative}，RMSE {score['rmse']:.4g} K。"
                                "使用本杆训练数据拟合及完整实测边界；这是本项目新增留出记录评分，不是零样本迁移或唯一物理方程证明。")
        np.savez_compressed(output / "trajectories.npz", t=t, observed=u, predicted_interior=prediction,
                            persistence=persistence, linear_boundary=linear)
        _plot_case(t, u, prediction, linear, report, output)
        report["plot_available"] = True
        report["trajectory_available"] = True
    except (ValueError, OSError, RuntimeError, FloatingPointError, IndexError) as exc:
        report["validation"] = {"status": "failed", "failure_reason": str(exc), "holdout_file": holdout["file"]}
        report["conclusion"] = "本份留出记录验证失败；失败结果保留，未重新拟合、换模型或调整参数。"
    _write_json(output / "report.json", report)
    return report


def evaluate_benchmark(output_root=OUTPUT_ROOT, *, data_root=DATA_ROOT):
    output_root = Path(output_root)
    index, frozen_models, prepared = _verify_frozen_index(output_root, data_root)
    records = []
    for frozen, entry in zip(frozen_models, index["records"]):
        group = frozen["group"]
        manifest = prepared[group][0]
        group_records = []
        for holdout in manifest["holdouts"]:
            report = _evaluate_holdout(frozen, entry["frozen_model_sha256"], manifest, holdout, output_root, data_root)
            discovery, validation = report["discovery"], report["validation"]
            record = {"case": report["case"], "group": group, "repeat": holdout["repeat"],
                      "title": f"{group} 第{holdout['repeat']}次留出记录",
                      "equation": discovery["equation"], "model_name": discovery.get("model_name"),
                      "support_size": discovery["support_size"], "parameters": discovery.get("parameters"),
                      "status": discovery["status"], "validation_status": validation["status"],
                      "relative_l2": validation.get("relative_l2"), "rmse_K": validation.get("rmse_K"),
                      "baselines": validation.get("baselines", {}),
                      "beats_both_baselines": validation.get("beats_both_baselines", False),
                      "below_descriptive_5pct": validation.get("below_descriptive_5pct", False),
                      "failure_reason": validation.get("failure_reason") or frozen["failure_reason"],
                      "frozen_model_sha256": entry["frozen_model_sha256"],
                      "report": f"{group}/repeat{holdout['repeat']}/report.json"}
            records.append(record)
            group_records.append(record)
            print(json.dumps({"stage": "project_holdout_evaluation", **record}, ensure_ascii=False), flush=True)
        group_report = {**_base_report(frozen, entry["frozen_model_sha256"], None), "holdout_records": group_records}
        group_report["conclusion"] = "本组固定模型的三份留出结果均已记录；任何失败均保留，未依留出重新选择。"
        _write_json(output_root / group / "report.json", group_report)
    summary = {"description": "四根增材304L杆的12份本项目新增留出记录；每杆1训练、2开发、3/4/5留出。",
               "validation_scope": "new_to_this_project_holdout_records", "new_project_holdout": True,
               "is_zero_shot_transfer": False, "is_l1_optimization": False,
               "prediction_scope": "完整实测两端边界条件下预测四内点；排除首帧。",
               "records": records, "groups": index["groups"], "all_four_groups": index["all_four_groups"],
               "selected_case": records[0]["case"], "freeze_index_sha256": _sha(output_root / "freeze_index.json"),
               "generated_at": datetime.now(timezone.utc).isoformat(), "unique_physical_equation_established": False}
    _write_json(output_root / "benchmark.json", summary)
    _plot_summary(summary, output_root)
    return summary


def run_benchmark(output_root=OUTPUT_ROOT, *, data_root=DATA_ROOT, groups=GROUPS):
    freeze_benchmark(output_root, data_root=data_root, groups=groups)
    return evaluate_benchmark(output_root, data_root=data_root)


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
        ax.plot(t, prediction[:, i], color="#d65a38", lw=1.6, label="Frozen physical-prior model")
        ax.plot(t, linear[:, i], color="#70a39a", lw=1, ls="--", label="Linear boundary baseline")
        ax.set(title=f"TC{i+2}: holdout interior", xlabel="Time (s)", ylabel="Excess temperature (K)")
        if i == 0:
            ax.legend(fontsize=8)
    axes.flat[4].plot(t, u[:, 0], label="TC1: input")
    axes.flat[4].plot(t, u[:, 5], label="TC6: input")
    axes.flat[4].set(title="Measured boundaries, not scored", xlabel="Time (s)", ylabel="Excess temperature (K)")
    axes.flat[4].legend(fontsize=8)
    validation = report["validation"]
    labels = ["Frozen model", "Persistence", "Linear boundary"]
    values = [validation["rmse_K"], validation["baselines"]["persistence"]["rmse"], validation["baselines"]["linear_boundary"]["rmse"]]
    axes.flat[5].bar(labels, values, color=["#d65a38", "#8294a3", "#70a39a"])
    axes.flat[5].set(title="Four interiors; initial frame excluded", ylabel="RMSE (K)")
    axes.flat[5].tick_params(axis="x", labelrotation=12)
    fig.suptitle(f"Additive 304L {report['case']} | Project-new holdout, specimen-specific fitting", fontweight="bold")
    fig.savefig(Path(output) / "validation.png", dpi=160)
    plt.close(fig)


def _plot_summary(summary, output_root):
    plt = _pyplot()
    records = summary["records"]
    x = np.arange(len(records))
    colors = {"xn": "#0e8c7f", "xp": "#587ba1", "yn": "#ae783e", "yp": "#8b67a2"}
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), layout="constrained")
    values = [100*r["relative_l2"] if r["relative_l2"] is not None else np.nan for r in records]
    axes[0].bar(x, values, color=[colors[r["group"]] for r in records])
    axes[0].axhline(5, color="#ba4239", ls="--", lw=1, label="5% descriptive reference")
    for i, value in enumerate(values):
        axes[0].text(i, (value if np.isfinite(value) else 0)+.3,
                     f"{value:.2f}" if np.isfinite(value) else "N/A", ha="center", fontsize=8)
    axes[0].set(title="All 12 project-new holdout records retained", ylabel="Conditional prediction relative L2 (%)")
    axes[0].legend(fontsize=8)
    axes[1].bar(x, [r["support_size"] for r in records], color=[colors[r["group"]] for r in records])
    axes[1].set(title="Same frozen model for each specimen's 3 holdouts", ylabel="Retained terms (diffusion is mandatory)", ylim=(0, 3.8), yticks=range(4))
    for ax in axes:
        ax.set_xticks(x, [r["case"] for r in records], rotation=35)
    fig.suptitle("Additive 304L | Train 1, select 2, hold out 3/4/5 | Not zero-shot transfer", fontweight="bold")
    fig.savefig(Path(output_root) / "benchmark.png", dpi=160)
    plt.close(fig)
