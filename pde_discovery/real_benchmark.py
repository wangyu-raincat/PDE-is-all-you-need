"""Predeclared train-1 / select-2 / test-3 measured heat-rod experiments."""
from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime, timezone

from .real_experiment import run_real_experiment, _write_json

DATA_ROOT = Path(__file__).resolve().parent / "data" / "mendeley_heat"
OUTPUT_ROOT = Path("artifacts/mendeley_heat")
CASES = tuple(f"{temperature}{material}" for material in ("ss", "al", "cu") for temperature in (50, 75, 100))
PRIMARY_CASE = "50ss"


def run_case(case=PRIMARY_CASE, output_root=OUTPUT_ROOT):
    if case not in CASES:
        raise ValueError("不支持的真实实验组。")
    return run_real_experiment(DATA_ROOT / case, Path(output_root) / case)


def run_benchmark(output_root=OUTPUT_ROOT):
    output_root = Path(output_root)
    records = []
    for case in CASES:
        result = run_case(case, output_root)
        discovery, validation = result["discovery"], result["validation"]
        record = {"case": case, "title": result["experiment"]["title"],
                  "equation": discovery["equation"], "status": discovery["status"],
                  "validation_status": validation.get("status", "not_evaluated"),
                  "rmse_K": validation.get("rmse_K"), "relative_l2": validation.get("relative_l2"),
                  "support_size": discovery.get("support_size", 0),
                  "sparsity_achieved": discovery.get("sparsity_achieved", False),
                  "beats_both_baselines": validation.get("beats_both_baselines", False),
                  "baselines": validation.get("baselines", {}),
                  "frozen_model_sha256": result["frozen_model_sha256"],
                  "failure_reason": validation.get("failure_reason"),
                  "report": f"{case}/report.json"}
        records.append(record)
        print(json.dumps(record, ensure_ascii=False), flush=True)
    summary = {"description": "九组公开实测热杆实验；固定每组第1次训练、第2次选模、第3次最终检验；所有分组均报告，无按最终误差挑选。",
               "prediction_scope": "两端全程实测边界条件下四内点预测，首帧不计分。",
               "selected_case": PRIMARY_CASE, "records": records,
               "generated_at": datetime.now(timezone.utc).isoformat(),
               "unique_physical_equation_established": False}
    _write_json(output_root / "benchmark.json", summary)
    plot_summary(summary, output_root)
    return summary


def plot_summary(summary, output_root):
    import os
    import tempfile
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "pde-lab-matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    records = summary["records"]
    labels = [r["case"] for r in records]
    colors = ["#596c9a" if "ss" in r["case"] else "#138a79" if "al" in r["case"] else "#bc743d" for r in records]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout="constrained")
    values = [100*r["relative_l2"] if r["relative_l2"] is not None else 0 for r in records]
    axes[0].bar(labels, values, color=colors)
    axes[0].axhline(5, linestyle="--", color="#ba4239", linewidth=1, label="5% descriptive reference")
    for i, (record, value) in enumerate(zip(records, values)):
        axes[0].text(i, value+1, f"{value:.2f}" if record["relative_l2"] is not None else "N/A", ha="center", fontsize=8)
    axes[0].set(ylim=(0, max(values + [10])*1.17), ylabel="Held-out conditional prediction error (%)", title="Repeat 3 was not used to select the equation")
    axes[0].legend(fontsize=8, loc="upper left")
    axes[1].bar(labels, [r["support_size"] for r in records], color=colors)
    axes[1].set(ylim=(0,5.8), yticks=range(0,6), ylabel="Retained candidate terms (out of 5)", title="Fewer terms does not establish a true physical PDE")
    for ax in axes:
        ax.tick_params(axis="x", labelrotation=40)
    fig.suptitle("Mendeley measured heat rods | 1: train, 2: select, 3: evaluate", fontweight="bold")
    fig.savefig(Path(output_root) / "benchmark.png", dpi=160)
    plt.close(fig)
