"""Reproduce a fixed public-measurement experiment with an auditable holdout.

The manifest fixes raw files, units and split roles. No network access occurs
here. Original CSV checksums are verified before use. Final interior values are
loaded only after the development-selected model has been written to disk.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .real_heat import ALPHA_PATH, build_candidates, evaluate, predict

DATA_DIR = Path(__file__).resolve().parent / "data" / "real_heat"
DEFAULT_OUTPUT = Path("artifacts/real_heat")
WINDOW_FRACTIONS = (0.02, 0.05, 0.10)
NEAR_OPTIMAL_FACTOR = 1.05


def _write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_measurements(data_dir, entry, csv_format):
    """Read one explicitly selected CSV, preserving the six observed columns."""
    path = Path(data_dir) / entry["file"]
    if _sha(path) != entry["sha256"]:
        raise ValueError(f"原始数据校验和不符：{path.name}")
    # Hashing checks bytes for integrity; parsing is restricted to this role's
    # predeclared rows, so final numeric observations are not loaded for fitting.
    first, last = entry.get("rows", [0, None])
    skip = csv_format.get("skiprows", 0)
    with path.open(encoding="utf-8") as source:
        selected_lines = itertools.islice(source, skip + first, None if last is None else skip + last)
        raw = np.loadtxt(selected_lines, delimiter=csv_format.get("delimiter", ","))
    t = raw[:, csv_format.get("time_column", 0)]
    u = raw[:, csv_format.get("temperature_columns", [1, 2, 3, 4, 5, 6])]
    if (len(t) < 8 or u.shape != (len(t), 6) or not np.isfinite(t).all()
            or not np.isfinite(u).all() or np.any(np.diff(t) <= 0)):
        raise ValueError("原始数据不满足六测点、有限值、严格递增时间要求。")
    return t, u


def score_model(terms, t, u, dx):
    prediction = predict(terms, t, u[:, [0, 5]], u[0, 1:5], dx)
    return prediction, evaluate(prediction, u[:, 1:5])


def select_model(t_train, u_train, t_dev, u_dev, dx):
    """Training generates coefficients; development selects; no final data input."""
    duration = float(t_train[-1] - t_train[0])
    windows = sorted(set(max(2 * float(np.median(np.diff(t_train))), duration * f)
                         for f in WINDOW_FRACTIONS))
    candidates = []
    for window in windows:
        for candidate in build_candidates(t_train, u_train, dx, window):
            candidate["window_seconds"] = window
            candidate["candidate_id"] = f"candidate-{len(candidates) + 1:02d}"
            if candidate["diagnostics"]["support_rank"] < candidate["support_size"]:
                candidate["eligible"] = False
                candidate["rejection_reason"] = "支持集秩不足，不能将非唯一系数用于本次选模。"
            elif not candidate["diagnostics"]["lasso_all_support_runs_converged"]:
                candidate["eligible"] = False
                candidate["rejection_reason"] = "该支持集的 L1 优化未全部收敛。"
            else:
                try:
                    _, score = score_model(candidate["terms"], t_dev, u_dev, dx)
                    candidate["development"] = score
                    candidate["eligible"] = True
                except ValueError as exc:
                    candidate["eligible"] = False
                    candidate["rejection_reason"] = str(exc)
            candidates.append(candidate)
    eligible = [c for c in candidates if c["eligible"]]
    if not eligible:
        return None, candidates, []
    best = min(c["development"]["rmse"] for c in eligible)
    near = [c for c in eligible if c["development"]["rmse"] <= best * NEAR_OPTIMAL_FACTOR + 1e-12]
    chosen = min(near, key=lambda c: (c["support_size"], c["development"]["rmse"], c["candidate_id"]))
    supports = {tuple(term["name"] for term in c["terms"]) for c in near}
    return chosen, candidates, [list(s) for s in sorted(supports)]


def run_real_experiment(data_dir=DATA_DIR, output=DEFAULT_OUTPUT):
    data_dir, output = Path(data_dir), Path(output)
    manifest_path = data_dir / "manifest.json"
    if not manifest_path.exists():
        raise ValueError("缺少已核实的真实数据 manifest.json，不能用合成数据替代。")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    dx = float(manifest["spacing_m"])
    entries = manifest["splits"]
    if set(entries) != {"training", "development", "final"}:
        raise ValueError("清单必须分别指定 training、development、final。")
    temporal = manifest["split_kind"] == "chronological_single_acquisition"
    if temporal:
        ordered = [entries[r] for r in ("training", "development", "final")]
        if (len({e["sha256"] for e in ordered}) != 1 or len({e["file"] for e in ordered}) != 1
                or any("rows" not in e or len(e["rows"]) != 2 for e in ordered)):
            raise ValueError("单次时间分割必须指定同一原始文件的三个不重叠行区间。")
        intervals = [e["rows"] for e in ordered]
        if (any(a < 0 or b-a < 8 for a,b in intervals)
                or intervals[0][1] > intervals[1][0] or intervals[1][1] > intervals[2][0]):
            raise ValueError("训练、开发、最终时间区间必须依次排列且不重叠。")
    else:
        if any("rows" in entry for entry in entries.values()):
            raise ValueError("完整重复实验协议不接受 rows 裁剪。")
        if len({entries[r]["sha256"] for r in entries}) != 3:
            raise ValueError("本协议要求三个不同的完整原始实验文件。")
    t_train, u_train = load_measurements(data_dir, entries["training"], manifest["csv_format"])
    t_dev, u_dev = load_measurements(data_dir, entries["development"], manifest["csv_format"])
    selected, candidates, near_supports = select_model(t_train, u_train, t_dev, u_dev, dx)
    protocol = {
        "split": {role: dict(entry) for role, entry in entries.items()},
        "split_kind": manifest["split_kind"],
        "units": {"time": "s", "space": "m", "temperature": "K above ambient"},
        "spacing_m": dx, "library": ["1", "theta", "theta^2", "theta_x", "theta_xx"],
        "alpha_path": list(ALPHA_PATH), "window_fractions_of_training_duration": list(WINDOW_FRACTIONS),
        "selection_rule": "开发集 RMSE 最小值的 1.05 倍内先取最少项，再取最低开发 RMSE；系数仅在训练集拟合。",
        "eligibility_rule": "支持集满秩、该支持集全部 L1 来源收敛，且可在开发实验中稳定正向积分。",
        "freeze_before_final_load": True, "final_data_selects_model": False,
        "smoothing": "none", "spatial_upsampling": False,
        "prediction_inputs": "两端全部实测温度 + 四内点首帧；不给未来内点。",
        "scoring": "仅四内点，排除初始化首帧；不计边界点。",
        "physical_constraint": "负扩散候选拒绝正向积分；未强制扩散项或衰减项必选。",
    }
    frozen = {"manifest_sha256": _sha(manifest_path), "protocol": protocol,
              "selected": selected, "candidates": candidates, "near_optimal_supports": near_supports}
    # An on-disk model artifact exists before the final holdout is opened.
    _write_json(output / "frozen_model.json", frozen)
    frozen_hash = _sha(output / "frozen_model.json")
    base_report = {"experiment": manifest["experiment"], "protocol": protocol,
                   "plot_available": False, "trajectory_available": False,
                   "generated_at": datetime.now(timezone.utc).isoformat(),
                   "frozen_model_sha256": frozen_hash, "candidates": candidates,
                   "limitations": [
                       "公开真实测量数据复现；本项目未自行搭建或操作实验装置。",
                       "预测使用整个验证实验的两端实测边界，不是未知未来边界下的自由预测。",
                       "六点有限差分系数是粗网格有效系数，不是高精度材料物性测量。",
                       "只在固定五项候选库中比较；低误差不能证明唯一真实 PDE。",
                       "未进行跨装置、跨材料泛化验证，未接入华为平台。",
                       manifest.get("provenance_note", "CSV 是作者公开的采集示例，测量属性依据配套代码解释；未独立核实采集日志，不能等同已认证的论文原始数据。"),
                   ]}
    if selected is None:
        report = {**base_report, "discovery": {"status": "insufficient", "equation": "没有可验证候选", "terms": [], "unique_physical_equation_established": False},
                  "validation": {}, "conclusion": "训练与开发阶段没有得到可稳定积分的候选；最终数据未用于补救选模。"}
        _write_json(output / "report.json", report)
        return report
    ambiguity_reasons = []
    if len(near_supports) > 1:
        ambiguity_reasons.append("开发阶段的近优池存在不同支持集。")
    if selected["diagnostics"]["support_rank"] < selected["support_size"]:
        ambiguity_reasons.append("训练支持集秩不足，系数不唯一。")
    ambiguous = bool(ambiguity_reasons)
    status = "ambiguous" if ambiguous else "candidate"
    try:
        t_test, u_test = load_measurements(data_dir, entries["final"], manifest["csv_format"])
        prediction, score = score_model(selected["terms"], t_test, u_test, dx)
    except (ValueError, OSError) as exc:
        report = {**base_report,
                  "discovery": {**selected, "status": status, "near_optimal_supports": near_supports,
                                "ambiguity_reasons": ambiguity_reasons, "unique_physical_equation_established": False},
                  "validation": {"status": "failed", "failure_reason": str(exc),
                                 "final_file": entries["final"]["file"]},
                  "conclusion": "冻结模型在最终实验中预测失败；保留失败结果，未利用最终数据更换候选或调参。"}
        _write_json(output / "report.json", report)
        return report
    persistence = np.broadcast_to(u_test[0, 1:5], prediction.shape).copy()
    weights = np.arange(1, 5)[None, :] / 5
    linear = u_test[:, :1] * (1 - weights) + u_test[:, -1:] * weights
    baselines = {"persistence": {"label": "初始温升持久值", **evaluate(persistence, u_test[:, 1:5])},
                 "linear_boundary": {"label": "同边界线性空间插值", **evaluate(linear, u_test[:, 1:5])}}
    sparse = selected["support_size"] < len(selected["diagnostics"]["library"])
    validation_kind = ("同一次实验的未来时段：给定实测边界的四内点条件预测" if temporal
                       else "独立重复实验：给定实测边界的四内点条件预测")
    if temporal:
        base_report["limitations"].append("仅一条采集记录按60/20/20时间分割，开发和最终段均不足一个完整加热周期；不是独立重复实验或新工况泛化。")
    validation = {"status": "evaluated", "kind": validation_kind,
                  "rmse_K": score["rmse"], "relative_l2": score["relative_error"], **score,
                  "per_sensor": [{"sensor": f"TC{i+2}", "rmse_K": v, "x_m": (i+1)*dx}
                                 for i, v in enumerate(score["per_sensor_rmse"])],
                  "baselines": baselines, "final_file": entries["final"]["file"],
                  "beats_both_baselines": all(score["rmse"] < b["rmse"] for b in baselines.values()),
                  "below_descriptive_5pct": score["relative_error"] is not None and score["relative_error"] < .05,
                  "scored_sensor_count": 4, "measured_boundary_sensor_count": 2,
                  "time_start_s": float(t_test[0]), "time_end_s": float(t_test[-1]), "rows": len(t_test)}
    relative_text = (f"{100*score['relative_error']:.3g}%" if score["relative_error"] is not None
                     else "未定义（观测内点温升的范数为零）")
    conclusion = (f"完成公开测量数据的{'未来时段' if temporal else '重复实验'}检验：四内点 RMSE {score['rmse']:.4g} K，"
                  f"相对 L2 误差 {relative_text}。")
    if ambiguous:
        conclusion += "".join(ambiguity_reasons) + "不能宣称唯一方程已识别。"
    else:
        conclusion += "这是限定候选库中的预测候选，尚不能证明物理机制唯一。"
    if not sparse:
        conclusion += "所选五项全部保留，尚未发现简洁稀疏 PDE；预测有效不等于物理方程恢复成功。"
    if not validation["beats_both_baselines"]:
        conclusion += "模型未同时优于两种简单基线。"
    report = {**base_report, "discovery": {**selected, "status": status, "near_optimal_supports": near_supports,
              "ambiguity_reasons": ambiguity_reasons, "sparsity_achieved": sparse,
              "unique_physical_equation_established": False},
              "validation": validation, "conclusion": conclusion}
    _write_json(output / "report.json", report)
    np.savez_compressed(output / "trajectories.npz", t=t_test, observed=u_test,
                        predicted_interior=prediction, persistence=persistence, linear_boundary=linear)
    _plot(t_test, u_test, prediction, linear, report, output)
    report["plot_available"] = True
    report["trajectory_available"] = True
    _write_json(output / "report.json", report)
    return report


def _plot(t, u, prediction, linear, report, output):
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "pde-lab-matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(14, 7), layout="constrained")
    for i, ax in enumerate(axes.flat[:4]):
        ax.plot(t, u[:, i+1], color="#19384b", lw=1, label="Measured")
        ax.plot(t, prediction[:, i], color="#d65a38", lw=1.6, label="Frozen PDE")
        ax.plot(t, linear[:, i], color="#70a39a", lw=1, linestyle="--", label="Linear boundary baseline")
        ax.set(title=f"TC{i+2}: held-out interior", xlabel="Time (s)", ylabel="Excess temperature (K)")
        if i == 0:
            ax.legend(fontsize=8)
    ax = axes.flat[4]
    ax.plot(t, u[:, 0], label="TC1: input")
    ax.plot(t, u[:, 5], label="TC6: input")
    ax.set(title="Measured boundary forcing (not scored)", xlabel="Time (s)", ylabel="Excess temperature (K)")
    ax.legend(fontsize=8)
    ax = axes.flat[5]
    v = report["validation"]
    labels = ["Frozen PDE", "Persistence", "Linear boundary"]
    values = [v["rmse_K"], v["baselines"]["persistence"]["rmse"], v["baselines"]["linear_boundary"]["rmse"]]
    ax.bar(labels, values, color=["#d65a38", "#8da0ad", "#70a39a"])
    ax.set(title="All four interiors; initial frame excluded", ylabel="RMSE (K)")
    for i, value in enumerate(values):
        ax.text(i, value, f"{value:.3g}", ha="center", va="bottom", fontsize=9)
    split_label = "held-out future segment" if report["protocol"]["split_kind"] == "chronological_single_acquisition" else "held-out repeat"
    fig.suptitle("Public measured heat-rod experiment | frozen model, " + split_label, fontweight="bold")
    fig.savefig(output / "validation.png", dpi=160)
    plt.close(fig)
