"""Evidence for the additive-rod equation question, with selection isolated from validation."""
import json

from pde_discovery.real_experiment import load_measurements, select_model
from pde_discovery.real_heat import TERM_NAMES
from competition_agent.scenario_am.catalog import ROOT, evidence
from competition_agent.scenario_am.heat_tools import ADDITIVE, heat_holdout_summary, heat_identify, heat_repeatability


def l1_control(directory, manifest):
    datasets = [load_measurements(directory, manifest["splits"][role], manifest["csv_format"])
                for role in ("training", "development")]
    selected, candidates, near = select_model(*datasets[0], *datasets[1], manifest["spacing_m"])
    return {"library": list(TERM_NAMES), "training_repeat": 1, "selection_repeat": 2,
            "selected": None if selected is None else {
                key: selected[key] for key in ("candidate_id", "equation", "terms", "window_seconds", "development")},
            "near_supports": near, "candidates": [
                {"id": c["candidate_id"], "terms": c["terms"], "eligible": c["eligible"],
                 "rejection": c.get("rejection_reason"), "development_rmse_K": c.get("development", {}).get("rmse")}
                for c in candidates]}


def equation_evidence(diagnostics, q1):
    data = ROOT / "pde_discovery/data/additive_heat"
    rods = {}
    for group in ADDITIVE:
        directory = data / group
        manifest = json.loads((directory / "manifest.json").read_bytes())
        prior = heat_identify(group)
        selected = prior["selected"]
        rods[group] = {
            "candidates": prior["candidate_table"], "near_optimal_factor": prior["near_optimal_factor"],
            "selected": selected["model_name"] if selected else None,
            "equation": selected["equation"] if selected else None,
            "parameters": selected["parameters"] if selected else None,
            "matches_frozen": prior["frozen_consistency"]["matches"],
            "selection_sources": prior["evidence"][-2:], "l1": l1_control(directory, manifest),
            "holdouts": []}

    # Both methods have finished selecting for ALL rods before reading any saved score
    # or calculating descriptive five-repeat correlations.
    report = ROOT / "experiments/ADDITIVE_REPORT.md"
    report_lines = report.read_text(encoding="utf8").splitlines()
    for group, rod in rods.items():
        for record in heat_holdout_summary(group)["records"]:
            val = record["validation"]
            test = record["case"]
            row = next(line for line in report_lines if line.startswith(f"| {test} |"))
            csv = data / "raw" / (test + ".csv")
            with csv.open(encoding="utf8") as stream:
                header = stream.readline().rstrip()
            rod["holdouts"].append({
                "test": test, "relative_l2": val["relative_l2"], "rmse_K": val["rmse_K"],
                "persistence_rmse_K": val["baselines"]["persistence"]["rmse"],
                "linear_boundary_rmse_K": val["baselines"]["linear_boundary"]["rmse"],
                "beats_both": val["beats_both_baselines"], "report_path": record["source_path"],
                "score_evidence": evidence(report, row), "csv_evidence": evidence(csv, header)})
        rod["D_h_correlation"] = heat_repeatability(group)["D_h_correlation"]
        rod["supported"] = (rod["matches_frozen"] and len(rod["holdouts"]) == 3
                            and all(row["beats_both"] for row in rod["holdouts"]))
    code = [
        ("pde_discovery/real_heat_refined.py", "def fit_candidate(t, u, dx, model_name: str)"),
        ("competition_agent/scenario_am/heat_tools.py", "selected = min(near, key=lambda c:"),
        ("pde_discovery/real_experiment.py", "def select_model(t_train, u_train, t_dev, u_dev, dx):"),
        ("pde_discovery/real_heat.py", 'TERM_NAMES = ["1", "theta", "theta^2", "theta_x", "theta_xx"]')]
    return {"rods": rods, "identifiability_evidence": diagnostics,
            "followup": {"question": q1, "tool": "decision_make", "same_case": True},
            "method_sources": [evidence(ROOT / path, quote) for path, quote in code]}


def render_equation(result):
    lines = ["在物理先验约束的候选结构中识别方程，并用留出数据验证。编号 1 拟合，编号 2 选模。"
             "先保留满秩、能预测开发记录的候选，再从开发 RMSE 最优值的 1.05 倍以内选项数最少者，同项数取较低 RMSE。",
             "三项结构写为 θ_t = D·θ_xx − h·θ + c。θ 为相对环境温升（K）；D 为轴向导热的有效热扩散参数（m²/s），"
             "h 为侧面散热衰减率（s⁻¹），c 为加热或环境漂移项（K/s）。",
             "候选结构（各行开发 RMSE 单位为 K）", "",
             "| 杆 | 结构 | 拟合成功 | Jacobian 秩 | 开发 RMSE | 近优 | 选中 |",
             "|---|---|---|---:|---:|---|---|"]
    yes = {True: "是", False: "否"}
    for group, rod in result["rods"].items():
        for c in rod["candidates"]:
            rmse = "—" if c["dev_rmse_K"] is None else f"{c['dev_rmse_K']:.5g}"
            lines.append(f"| {group} | {c['name']} | {yes[c['fit_success']]} | {c['rank']} | {rmse} | "
                         f"{yes[c['near_optimal']]} | {yes[c['selected']]} |")
    lines.extend(["", "历史留出：两端使用实测边界，预测四个内点。下列 RMSE 单位均为 K。", "",
                  "| 记录 | 相对 L2 | 模型 | 初值持久值 | 边界线性插值 |",
                  "|---|---:|---:|---:|---:|"])
    for rod in result["rods"].values():
        for row in rod["holdouts"]:
            lines.append(f"| {row['test']} | {row['relative_l2']:.4%} | {row['rmse_K']:.5g} | "
                         f"{row['persistence_rmse_K']:.5g} | {row['linear_boundary_rmse_K']:.5g} |")
    lines.extend(["", "无物理先验的 L1 对照使用 1、θ、θ²、θ_x、θ_xx 候选库。"])
    for group, rod in result["rods"].items():
        l1 = rod["l1"]["selected"]
        comparison = (f"{l1['equation']}（开发 RMSE {l1['development']['rmse']:.5g} K）"
                      if l1 else "没有合格候选")
        corr = rod["D_h_correlation"]["r"]
        correlation = "未定义" if corr is None else f"{corr:.3f}"
        lines.append(f"- {group}：L1 为 {comparison}；物理先验选中 {rod['selected']}，"
                     f"{rod['equation']}；五次复测的描述性 D–h 相关 r={correlation}。")
    diag = result["identifiability_evidence"]
    lines.extend(["", "采用物理先验的开发选模结果：结构具有传热解释，L1 的支持集随杆变化。"
                  + ("结构与条件预测在此候选族和上述记录内可靠，12 条历史留出均优于两种基线。"
                     if all(r["supported"] for r in result["rods"].values()) else "现有记录不足以完整支持结构与预测。"),
                  f"单个 D、h 受可识别性限制。旧不锈钢训练组三组的 D 近优剖面倍数为 "
                  f"{diag['profile_ratio']['min']:.3f}–{diag['profile_ratio']['max']:.3f}，"
                  f"删块倍数为 {diag['deletion_ratio']['min']:.3f}–{diag['deletion_ratio']['max']:.3f}。"
                  "见 [Q1 两杆差异与几何混杂](trace-1.json)。"])
    return "\n".join(lines)
