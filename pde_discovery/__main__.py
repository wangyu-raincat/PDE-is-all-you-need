"""Run the demo server or reproduce all benchmark results from the command line."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="PDE 发现实验室")
    commands = parser.add_subparsers(dest="command", required=True)
    server = commands.add_parser("serve", help="启动中文演示界面")
    server.add_argument("--host", default="127.0.0.1")
    server.add_argument("--port", type=int, default=8000)
    demo = commands.add_parser("demo", help="执行一次实验并保存完整证据")
    demo.add_argument("--scenario", choices=["diffusion", "advection_diffusion", "ambiguity"], default="ambiguity")
    demo.add_argument("--noise", type=float, default=0.0)
    demo.add_argument("--seed", type=int, default=7)
    demo.add_argument("--no-auto-experiment", action="store_true")
    demo.add_argument("--output", default="artifacts/demo.json")
    bench = commands.add_parser("benchmark", help="运行有固定种子的基准实验并导出图表")
    bench.add_argument("--output", default="artifacts")
    real = commands.add_parser("real-experiment", help="复现已归档的公开真实热杆测量实验")
    real.add_argument("--output", default="artifacts/real_heat")
    measured = commands.add_parser("measured-benchmark", help="复现Mendeley九组实测重复实验")
    measured.add_argument("--output", default="artifacts/mendeley_heat")
    measured.add_argument("--case", choices=["50ss", "75ss", "100ss", "50al", "75al", "100al", "50cu", "75cu", "100cu"])
    refined = commands.add_parser("refined-benchmark", help="物理约束轨迹拟合与旧方法的九组回顾性比较")
    refined.add_argument("--output", default="artifacts/refined_heat")
    refined.add_argument("--phase", choices=["all", "freeze", "evaluate"], default="all",
                         help="all完整复现；freeze只训练选模并冻结；evaluate仅评分已冻结模型")
    additive = commands.add_parser("additive-benchmark", help="四根增材杆的十二条预定留出记录检验")
    additive.add_argument("--output", default="artifacts/additive_heat")
    additive.add_argument("--phase", choices=["all", "freeze", "evaluate"], default="all")
    identifiability = commands.add_parser("identifiability", help="旧九组训练数据的参数剖面与删块稳定性诊断")
    identifiability.add_argument("--output", default="artifacts/identifiability")
    external = commands.add_parser("external-benchmark", help="独立来源铝与黄铜高电压留出验证及新旧方法比较")
    external.add_argument("--output", default="artifacts/external_lab")
    external.add_argument("--phase", choices=["all", "download", "freeze", "evaluate"], default="all")
    generic = commands.add_parser("generic-benchmark", help="跨方程类型的共享候选库 L1 发现与方程恢复评分")
    generic.add_argument("--output", default="artifacts/generic_pde")
    generic.add_argument("--phase", choices=["all", "download", "prepare", "freeze", "evaluate"], default="all")
    args = parser.parse_args()
    if args.command == "serve":
        import uvicorn
        uvicorn.run("pde_discovery.app:app", host=args.host, port=args.port)
        return
    if args.command == "measured-benchmark":
        from .real_benchmark import run_benchmark, run_case
        result = run_case(args.case, args.output) if args.case else run_benchmark(args.output)
        print(json.dumps({"output": str(Path(args.output).resolve()),
                          "completed_cases": len(result.get("records", [result]))}, ensure_ascii=False))
        return
    if args.command == "refined-benchmark":
        from .refined_experiment import run_benchmark, freeze_benchmark, evaluate_benchmark
        action = {"all": run_benchmark, "freeze": freeze_benchmark, "evaluate": evaluate_benchmark}[args.phase]
        result = action(output_root=args.output)
        print(json.dumps({"output": str(Path(args.output).resolve()),
                          "completed_cases": len(result.get("records", [])),
                          "phase": args.phase,
                          "new_blind_validation": False}, ensure_ascii=False))
        return
    if args.command == "additive-benchmark":
        from .additive_experiment import run_benchmark, freeze_benchmark, evaluate_benchmark
        action = {"all": run_benchmark, "freeze": freeze_benchmark, "evaluate": evaluate_benchmark}[args.phase]
        result = action(output_root=args.output)
        print(json.dumps({"output": str(Path(args.output).resolve()), "phase": args.phase,
                          "records": len(result.get("records", []))}, ensure_ascii=False))
        return
    if args.command == "identifiability":
        from .heat_identifiability import run_diagnostics
        result = run_diagnostics(output=args.output)
        print(json.dumps({"output": str(Path(args.output).resolve()),
                          "records": len(result.get("records", []))}, ensure_ascii=False))
        return
    if args.command == "external-benchmark":
        from .external_experiment import download_data, run_benchmark, freeze_benchmark, evaluate_benchmark
        if args.phase == "download":
            result = download_data()
        else:
            action = {"all": run_benchmark, "freeze": freeze_benchmark, "evaluate": evaluate_benchmark}[args.phase]
            result = action(output_root=args.output)
        print(json.dumps({"output": str(Path(args.output).resolve()), "phase": args.phase,
                          "records": len(result.get("records", []))}, ensure_ascii=False))
        return
    if args.command == "generic-benchmark":
        from .generic_experiment import download_data, prepare_data, run_benchmark, freeze_benchmark, evaluate_benchmark
        if args.phase in ("download", "prepare"):
            result = {"download": download_data, "prepare": prepare_data}[args.phase]()
        else:
            action = {"all": run_benchmark, "freeze": freeze_benchmark, "evaluate": evaluate_benchmark}[args.phase]
            result = action(output_root=args.output)
        print(json.dumps({"output": str(Path(args.output).resolve()), "phase": args.phase,
                          "records": len(result.get("records", []))}, ensure_ascii=False))
        return
    if args.command == "real-experiment":
        from .real_experiment import run_real_experiment
        result = run_real_experiment(output=args.output)
        print(json.dumps({"equation": result["discovery"]["equation"],
                          "status": result["discovery"]["status"],
                          "validation": result["validation"], "conclusion": result["conclusion"],
                          "output": str(Path(args.output).resolve())}, ensure_ascii=False, indent=2))
        return
    from .app import _json_safe
    from .controller import run_demo
    if args.command == "demo":
        result = run_demo(args.scenario, noise=args.noise, seed=args.seed, auto_experiment=not args.no_auto_experiment)
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(_json_safe(result), ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"equation": result["equation"], "goal": result["goal"], "output": str(output)}, ensure_ascii=False, indent=2))
        return
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for scenario in ("diffusion", "advection_diffusion", "ambiguity"):
        for noise in (0.0, 0.001, 0.01):
            for seed in (7, 19):
                started = time.perf_counter()
                result = run_demo(scenario, noise=noise, seed=seed)
                error = result["diagnostics"].get("validation_relative_error")
                record = {"scenario": scenario, "noise": noise, "seed": seed,
                          "status": result["status"], "equation": result["equation"],
                          "terms": result["terms"], "goal_achieved": result["goal"]["achieved"],
                          "validation_relative_error": error,
                          "extra_experiment": result["experiment"]["executed"],
                          "runtime_seconds": time.perf_counter() - started}
                records.append(record)
                name = f"{scenario}_noise{noise:g}_seed{seed}.json"
                (output / name).write_text(json.dumps(_json_safe(result), ensure_ascii=False), encoding="utf-8")
                print(json.dumps(_json_safe(record), ensure_ascii=False), flush=True)
    stopped = run_demo("ambiguity", auto_experiment=False)
    (output / "ambiguity_without_experiment.json").write_text(json.dumps(_json_safe(stopped), ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {"description": "Synthetic periodic 1D PDE benchmark; not real-world validation.",
               "records": records, "ambiguity_without_experiment": {
                   "status": stopped["status"], "goal_achieved": stopped["goal"]["achieved"]}}
    (output / "benchmark.json").write_text(json.dumps(_json_safe(summary), ensure_ascii=False, indent=2), encoding="utf-8")
    _plot(records, output)
    print(f"Saved benchmark evidence to {output.resolve()}")


def _plot(records, output):
    import os
    import tempfile
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "pde-lab-matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "pde-lab-cache"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout="constrained")
    colors = {"diffusion": "#0e9e95", "advection_diffusion": "#3267c8", "ambiguity": "#ab6b25"}
    for scenario, color in colors.items():
        xs, ys = [], []
        for noise in (0.0, 0.001, 0.01):
            values = [r["validation_relative_error"] for r in records if r["scenario"] == scenario and r["noise"] == noise and r["validation_relative_error"] is not None]
            xs.append(noise * 100)
            ys.append(float(np.mean(values)) * 100 if values else float("nan"))
        axes[0].plot(xs, ys, "o-", label=scenario, color=color)
    axes[0].axhline(5, linestyle="--", color="#a34e4e", linewidth=1, label="5% goal")
    axes[0].set(xlabel="Observation noise (% of trajectory RMS)", ylabel="Held-out rollout relative error (%)", title="Independent initial-condition validation")
    axes[0].legend(fontsize=8)
    labels = list(colors)
    achieved = [sum(r["goal_achieved"] for r in records if r["scenario"] == s) for s in labels]
    axes[1].bar(labels, achieved, color=list(colors.values()), width=.55)
    axes[1].set(ylim=(0, 6.8), ylabel="Runs meeting the configured goal (out of 6)", title="Two seeds × three noise levels per scenario")
    axes[1].tick_params(axis="x", labelrotation=12)
    for i, value in enumerate(achieved):
        axes[1].text(i, value + .12, f"{value}/6", ha="center")
    fig.suptitle("PDE Discovery Lab · Synthetic benchmark", fontweight="bold")
    fig.savefig(output / "benchmark.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
