"""Independent educational heat-rod measurements with a frozen comparison.

The old L1 method and the physical-prior trajectory method are reused without
changes. Lower-voltage acquisitions supply training/development. Both materials
and both methods are frozen before either higher-voltage record is parsed.
"""
from __future__ import annotations

import hashlib
import io
import json
from datetime import datetime, timezone
from pathlib import Path
import urllib.request

import numpy as np

from .real_experiment import _sha, _write_json, select_model, WINDOW_FRACTIONS
from .real_heat import ALPHA_PATH, evaluate, predict as predict_l1
from .real_heat_refined import FIT_PROTOCOL, fit_candidates, predict as predict_refined

GROUPS = ("aluminum", "brass")
METHODS = ("physical", "l1")
FILES = {
    "aluminum": {"lower": "Aluminum_26V_250mA.txt", "holdout": "Aluminum_28V_269mA.txt"},
    "brass": {"lower": "Brass_26V_245mA.txt", "holdout": "Brass_29V_273mA.txt"},
}
REPOSITORY = "TariqSawaqed1/Heat-Conduction-Aerospace-Lab"
COMMIT = "aba5a8c3c1d52fb879710cdddcfb23d29b079f03"
DATA_ROOT = Path(__file__).resolve().parent / "data" / "external_lab"
OUTPUT_ROOT = Path("artifacts/external_lab")
PROTOCOL_PATH = Path(__file__).resolve().parents[1] / "experiments" / "EXTERNAL_LAB_PROTOCOL.md"
SPACING_M = .0127
TRAIN_FRACTION = .6
NEAR_OPTIMAL_FACTOR = 1.05
DOWNLOAD_TIMEOUT_S = 30
DOWNLOAD_LIMIT_BYTES = 128 * 1024
HEADER = ["Time[s]", *[f"CH{i}[C]" for i in range(1, 9)]]
LIMITATIONS = [
    "独立于旧 Mendeley/ATMC 的公开教学实验来源；原始计量与采集过程未由本项目独立复核。",
    "两种材料各一条高电压条件记录；不等于跨装置重复，也不足以证明一般泛化。",
    "新旧方法均用本轮低电压训练段重新拟合，不是直接迁移旧材料系数。",
    "CH1/CH6 完整实测边界为输入，仅 CH2..CH5 为预测；未知未来边界下自由预测未验证。",
    "CH7/8 事先排除；只用六个原始传感器，无空间升采样或平滑。",
    "Tref 是低电压训练首帧六通道均值，不是实测环境温度；所有分段固定用同一参考。",
    "相对 L2 使用训练参考中心化温度；RMSE 是主要指标，5% 仅描述性参照。",
    "物理先验模型比较不是 L1 自动发现；局部满秩和低预测误差不证明唯一连续物理 PDE。",
    "评分后高电压记录已被本项目查看；之后重跑属于复现，不是新的未见检验。",
    "原仓库未发现明确再分发许可；交付原始文件下载器与派生报告，不打包第三方原始 TXT。",
]


def _now():
    return datetime.now(timezone.utc).isoformat()


def _source_hashes():
    root = Path(__file__).resolve().parent
    return {name: _sha(root / name) for name in
            ("external_experiment.py", "real_heat_refined.py", "real_heat.py", "real_experiment.py")}


def _git_blob_sha1(content):
    return hashlib.sha1(b"blob " + str(len(content)).encode("ascii") + b"\0" + content).hexdigest()


def _manifest(data_root):
    path = Path(data_root) / "source_manifest.json"
    result = json.loads(path.read_text(encoding="utf-8"))
    if result.get("repo") != REPOSITORY or result.get("commit") != COMMIT:
        raise ValueError("本轮只接受协议固定的独立仓库与不可变提交。")
    files = result.get("files", [])
    names = [entry.get("name") for entry in files]
    if len(set(names)) != len(names):
        raise ValueError("来源清单文件名重复。")
    needed = [FILES[group][role] for group in GROUPS for role in ("lower", "holdout")]
    selected = {}
    for name in needed:
        if names.count(name) != 1:
            raise ValueError(f"缺少固定来源文件：{name}")
        entry = files[names.index(name)]
        expected_url = f"https://raw.githubusercontent.com/{REPOSITORY}/{COMMIT}/Part%201/{name}"
        if (entry.get("url") != expected_url or type(entry.get("bytes")) is not int
                or not 0 < entry["bytes"] <= DOWNLOAD_LIMIT_BYTES
                or not isinstance(entry.get("git_blob_sha1"), str)
                or len(entry["git_blob_sha1"]) != 40
                or any(c not in "0123456789abcdef" for c in entry["git_blob_sha1"])):
            raise ValueError(f"来源 URL、大小或 Git blob SHA-1 无效：{name}")
        selected[name] = entry
    if len({entry["git_blob_sha1"] for entry in selected.values()}) != len(needed):
        raise ValueError("不同角色不能重复使用相同原始 Git blob。")
    return result, selected


def _verify_content(content, entry):
    if len(content) != entry["bytes"] or _git_blob_sha1(content) != entry["git_blob_sha1"]:
        raise ValueError(f"原始文件大小或 Git blob 校验失败：{entry['name']}")
    return {"name": entry["name"], "bytes": len(content),
            "git_blob_sha1": entry["git_blob_sha1"],
            "sha256": hashlib.sha256(content).hexdigest(), "url": entry["url"]}


def download_data(data_root=DATA_ROOT):
    """Download only four whitelisted byte files; do not parse observations.

    Existing mismatched files are rejected rather than overwritten. Network
    reads retain Python's normal proxy and TLS verification and are bounded by
    both a timeout and the manifest byte count.
    """
    data_root = Path(data_root)
    _, entries = _manifest(data_root)
    raw = data_root / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    records = []
    for entry in entries.values():
        path = raw / entry["name"]
        if path.exists():
            content = path.read_bytes()
        else:
            request = urllib.request.Request(entry["url"], headers={"User-Agent": "PDE-validation/1.0"})
            with urllib.request.urlopen(request, timeout=DOWNLOAD_TIMEOUT_S) as response:
                content = response.read(entry["bytes"] + 1)
            _verify_content(content, entry)
            # Write only verified complete bytes; no partial files are accepted.
            with path.open("xb") as destination:
                destination.write(content)
        records.append(_verify_content(content, entry))
    receipt = {"schema_version": 1, "repository": REPOSITORY, "commit": COMMIT,
               "source_manifest_sha256": _sha(data_root / "source_manifest.json"),
               "numeric_values_parsed": False, "files": records}
    _write_json(data_root / "download_receipt.json", receipt)
    return receipt


def _preflight(data_root):
    data_root = Path(data_root)
    manifest, entries = _manifest(data_root)
    receipt_path = data_root / "download_receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if (receipt.get("source_manifest_sha256") != _sha(data_root / "source_manifest.json")
            or receipt.get("commit") != COMMIT or receipt.get("repository") != REPOSITORY):
        raise ValueError("下载收据与来源清单或提交不符。")
    actual = [_verify_content((data_root / "raw" / entry["name"]).read_bytes(), entry)
              for entry in entries.values()]
    if actual != receipt.get("files"):
        raise ValueError("原始文件 SHA-256 与下载收据不符。")
    return manifest, receipt


def load_measurements(data_root, filename):
    """Parse one declared acquisition, returning original time and CH1..CH6."""
    _, entries = _manifest(data_root)
    if filename not in entries:
        raise ValueError("不支持未声明的原始文件。")
    content = (Path(data_root) / "raw" / filename).read_bytes()
    _verify_content(content, entries[filename])
    lines = content.decode("utf-8-sig").splitlines()
    if not lines or [cell.strip() for cell in lines[0].split(",")] != HEADER:
        raise ValueError("需原始时间秒与 CH1..CH8 摄氏度列标题。")
    if any(not line.strip() or len(line.split(",")) != 9 for line in lines[1:]):
        raise ValueError("原始记录有空行或不符九列；不自动删行或补值。")
    raw = np.loadtxt(io.StringIO("\n".join(lines[1:])), delimiter=",", usecols=tuple(range(7)), ndmin=2)
    t, u = raw[:, 0], raw[:, 1:7]
    if (len(t) < 8 or u.shape != (len(t), 6) or not np.isfinite(raw).all()
            or np.any(np.diff(t) <= 0)):
        raise ValueError("记录需至少八帧、有限六测点和严格递增时间；不自动修复。")
    return t, u


def _protocol(data_root):
    return {
        "version": 1, "groups": list(GROUPS), "methods": list(METHODS), "files": FILES,
        "training_fraction_of_lower_voltage": TRAIN_FRACTION,
        "training_rows_rule": "[0,floor(0.6*N))", "development_rows_rule": "[floor(0.6*N),N)",
        "holdout_rule": "complete higher-voltage acquisition for each material",
        "temperature_reference": "mean of CH1..CH6 at first lower-voltage training frame, unchanged for all roles",
        "reference_is_measured_ambient": False, "spacing_m": SPACING_M,
        "source_channels": 8, "selected_channels": [1, 2, 3, 4, 5, 6],
        "boundary_channels": [1, 6], "predicted_channels": [2, 3, 4, 5],
        "time_unit": "s", "raw_temperature_unit": "degree C", "centered_temperature_unit": "K difference",
        "smoothing": "none", "spatial_upsampling": False,
        "physical_method": FIT_PROTOCOL,
        "l1_method": {"function": "real_experiment.select_model", "alpha_path": list(ALPHA_PATH),
                      "window_fractions": list(WINDOW_FRACTIONS),
                      "minimum_window": "two median sampling intervals"},
        "near_optimal_factor": NEAR_OPTIMAL_FACTOR,
        "all_groups_and_methods_frozen_before_any_holdout_parse": True,
        "future_interior_is_prediction_input": False, "holdout_selects_model": False,
        "is_zero_shot_transfer": False, "unique_physical_equation_established": False,
        "protocol_document_sha256": _sha(PROTOCOL_PATH),
        "source_manifest_sha256": _sha(Path(data_root) / "source_manifest.json"),
        "download_receipt_sha256": _sha(Path(data_root) / "download_receipt.json"),
        "source_sha256": _source_hashes(),
    }


def _output_guard(output_root, data_root):
    output = Path(output_root).resolve()
    project = Path(__file__).resolve().parents[1]
    if output in (project, project / "artifacts", Path(data_root).resolve()):
        raise ValueError("验证输出必须使用独立子目录。")
    protected = [project / "pde_discovery", project / "experiments", project / "tests",
                 Path(data_root).resolve(), *[project / "artifacts" / name for name in
                 ("real_heat", "mendeley_heat", "refined_heat", "additive_heat", "identifiability", "synthetic-final")]]
    if any(output.is_relative_to(path) for path in protected):
        raise ValueError("独立来源验证不得覆盖原数据、代码或以前实验目录。")


def _source(data_root, group=None):
    result = {"repo": REPOSITORY, "commit": COMMIT,
              "repository_url": f"https://github.com/{REPOSITORY}",
              "source_manifest_sha256": _sha(Path(data_root) / "source_manifest.json")}
    if group:
        result.update({"lower_voltage_filename": FILES[group]["lower"],
                       "holdout_filename": FILES[group]["holdout"]})
    return result


def _empty_method(reason=None):
    return {"selected": None, "candidates": [], "near_optimal_supports": [], "failure_reason": reason}


def _score(terms, t, u, method):
    predictor = predict_refined if method == "physical" else predict_l1
    prediction = predictor(terms, t, u[:, [0, 5]], u[0, 1:5], SPACING_M)
    return prediction, _metrics(prediction, u[:, 1:5])


def _metrics(prediction, observed):
    score = evaluate(prediction, observed)
    return {**score, "rmse_K": score["rmse"], "relative_l2": score["relative_error"],
            "relative_l2_reference": "fixed training-first-frame six-channel mean"}


def _fit_physical(t, u, td, ud):
    candidates = fit_candidates(t, u, SPACING_M)
    for candidate in candidates:
        candidate["eligible"] = False
        diagnostics = candidate.get("diagnostics", {})
        rank, count = diagnostics.get("jacobian_rank"), diagnostics.get("parameter_count")
        if not candidate.get("fit_success"):
            candidate["rejection_reason"] = "训练拟合未收敛。"
        elif rank is None or count is None or rank < count:
            candidate["rejection_reason"] = "训练局部敏感性秩不足。"
        else:
            try:
                _, candidate["development"] = _score(candidate["terms"], td, ud, "physical")
                candidate["eligible"] = True
            except (ValueError, RuntimeError, FloatingPointError) as exc:
                candidate["rejection_reason"] = str(exc)
    eligible = [candidate for candidate in candidates if candidate["eligible"]]
    if not eligible:
        return {**_empty_method("没有训练/开发阶段合格的新模型。"), "candidates": candidates}
    best = min(candidate["development"]["rmse"] for candidate in eligible)
    near = [candidate for candidate in eligible
            if candidate["development"]["rmse"] <= NEAR_OPTIMAL_FACTOR * best + 1e-12]
    selected = min(near, key=lambda item: (item["support_size"], item["development"]["rmse"], item["model_name"]))
    return {"selected": selected, "candidates": candidates,
            "near_optimal_supports": sorted({item["model_name"] for item in near}), "failure_reason": None}


def _freeze_group(group, data_root, protocol):
    models = {method: _empty_method() for method in METHODS}
    reference, split, failure = None, None, None
    try:
        t, raw_u = load_measurements(data_root, FILES[group]["lower"])
        stop = int(np.floor(TRAIN_FRACTION * len(t)))
        if stop < 8 or len(t) - stop < 8:
            raise ValueError("固定 60/40 划分后，训练与开发均需至少八帧。")
        reference = float(raw_u[0].mean())
        u = raw_u - reference
        split = {"total_lower_voltage_rows": len(t), "training_rows": [0, stop],
                 "development_rows": [stop, len(t)],
                 "training_time_s": [float(t[0]), float(t[stop-1])],
                 "development_time_s": [float(t[stop]), float(t[-1])],
                 "sampling_interval_s": {"min": float(np.diff(t).min()),
                                         "max": float(np.diff(t).max()),
                                         "median": float(np.median(np.diff(t)))}}
        for method in METHODS:
            try:
                if method == "physical":
                    models[method] = _fit_physical(t[:stop], u[:stop], t[stop:], u[stop:])
                else:
                    selected, candidates, near = select_model(t[:stop], u[:stop], t[stop:], u[stop:], SPACING_M)
                    models[method] = {"selected": selected, "candidates": candidates,
                                      "near_optimal_supports": near,
                                      "failure_reason": None if selected else "没有训练/开发阶段合格的 L1 候选。"}
            except (ValueError, RuntimeError, FloatingPointError, np.linalg.LinAlgError) as exc:
                models[method] = _empty_method(str(exc))
    except (ValueError, OSError, RuntimeError, FloatingPointError, IndexError) as exc:
        failure = str(exc)
        models = {method: _empty_method(failure) for method in METHODS}
    return {"case": group, "source": _source(data_root, group), "protocol": protocol,
            "Tref_C": reference, "split": split, "models": models, "failure_reason": failure,
            "frozen_at": _now(), "holdout_numeric_values_loaded": False}


def _base_report(frozen, digest):
    models = {}
    for name, model in frozen["models"].items():
        models[name] = {**model, "validation": {"status": "not_evaluated"},
                        "unique_physical_equation_established": False}
    return {"case": frozen["case"], "source": frozen["source"], "protocol": frozen["protocol"],
            "Tref_C": frozen["Tref_C"], "split": frozen["split"], "models": models,
            "frozen_model_sha256": digest, "validation": {"status": "not_evaluated"},
            "baselines": {}, "comparison": {}, "generated_at": _now(),
            "plot_available": False, "trajectory_available": False,
            "is_zero_shot_transfer": False, "unique_physical_equation_established": False,
            "limitations": LIMITATIONS, "conclusion": "两种方法已冻结，等待固定高电压条件测试。"}


def _invalidate(output_root, status="not_evaluated", reason=None):
    records = []
    for group in GROUPS:
        report = {"case": group, "models": {method: {**_empty_method(reason),
                  "validation": {"status": status, "failure_reason": reason}} for method in METHODS},
                  "validation": {"status": status, "failure_reason": reason},
                  "baselines": {}, "comparison": {}, "plot_available": False,
                  "trajectory_available": False, "limitations": LIMITATIONS,
                  "conclusion": reason or "本次运行尚未完成验证；旧图像无效。"}
        _write_json(Path(output_root) / group / "report.json", report)
        records.append({"case": group, "validation_status": status, "failure_reason": reason,
                        "physical": {"status": status}, "l1": {"status": status},
                        "baselines": {}, "comparison": {}, "plot_available": False})
    _write_json(Path(output_root) / "benchmark.json", {"records": records, "plot_available": False,
                "status": status, "failure_reason": reason, "limitations": LIMITATIONS})


def freeze_benchmark(output_root=OUTPUT_ROOT, *, data_root=DATA_ROOT):
    output_root, data_root = Path(output_root), Path(data_root)
    _output_guard(output_root, data_root)
    _invalidate(output_root)
    try:
        _preflight(data_root)
        protocol = _protocol(data_root)
        _write_json(output_root / "protocol.json", protocol)
        records = []
        for group in GROUPS:
            frozen = _freeze_group(group, data_root, protocol)
            path = output_root / group / "frozen_model.json"
            _write_json(path, frozen)
            digest = _sha(path)
            _write_json(output_root / group / "report.json", _base_report(frozen, digest))
            records.append({"case": group, "frozen_model_sha256": digest,
                            "selected": {method: model["selected"].get("equation") if model["selected"] else None
                                         for method, model in frozen["models"].items()},
                            "failure_reason": frozen["failure_reason"]})
            print(json.dumps({"stage": "frozen", **records[-1]}, ensure_ascii=False), flush=True)
        index = {"schema_version": 1, "groups": list(GROUPS), "methods": list(METHODS),
                 "records": records, "protocol": protocol, "source_sha256": _source_hashes(),
                 "protocol_artifact_sha256": _sha(output_root / "protocol.json"),
                 "all_groups_and_methods_frozen": True, "holdout_numeric_values_loaded_in_freeze": False}
        _write_json(output_root / "freeze_index.json", index)
        return index
    except (ValueError, OSError, KeyError, TypeError) as exc:
        _invalidate(output_root, "failed", str(exc))
        raise


def _verify_frozen_index(output_root, data_root):
    output_root = Path(output_root)
    index = json.loads((output_root / "freeze_index.json").read_text(encoding="utf-8"))
    if (index.get("groups") != list(GROUPS) or index.get("methods") != list(METHODS)
            or [item.get("case") for item in index.get("records", [])] != list(GROUPS)
            or not index.get("all_groups_and_methods_frozen")
            or index.get("holdout_numeric_values_loaded_in_freeze") is not False):
        raise ValueError("冻结索引未完整包含两组、两方法和未读测试声明。")
    _preflight(data_root)
    if (index["source_sha256"] != _source_hashes() or index["protocol"] != _protocol(data_root)
            or index["protocol_artifact_sha256"] != _sha(output_root / "protocol.json")):
        raise ValueError("源码、固定协议、来源或下载收据已变化；禁止沿用冻结模型评分。")
    frozen_models = []
    for entry in index["records"]:
        path = output_root / entry["case"] / "frozen_model.json"
        if _sha(path) != entry["frozen_model_sha256"]:
            raise ValueError(f"冻结模型校验失败：{entry['case']}")
        frozen = json.loads(path.read_text(encoding="utf-8"))
        if (frozen.get("case") != entry["case"] or set(frozen.get("models", {})) != set(METHODS)
                or frozen.get("protocol") != index["protocol"] or frozen.get("holdout_numeric_values_loaded") is not False):
            raise ValueError("冻结模型的分组、方法或协议与索引不符。")
        frozen_models.append(frozen)
    return index, frozen_models


def _improvement(new_score, comparator):
    if not new_score or not comparator:
        return None
    new, old = new_score.get("rmse_K"), comparator.get("rmse_K")
    return 1 - new / old if new is not None and old is not None and old > 1e-12 else None


def _evaluate_group(frozen, digest, output_root, data_root):
    group = frozen["case"]
    output = Path(output_root) / group
    report = _base_report(frozen, digest)
    _write_json(output / "report.json", report)
    if all(model["selected"] is None for model in frozen["models"].values()):
        for method in METHODS:
            report["models"][method]["validation"] = {"status": "insufficient", "failure_reason": frozen["models"][method]["failure_reason"]}
        report["validation"] = {"status": "failed", "failure_reason": "两种方法均无冻结候选；不以测试数据补救。"}
        report["conclusion"] = report["validation"]["failure_reason"]
        _write_json(output / "report.json", report)
        return report
    predictions = {}
    try:
        t, raw_u = load_measurements(data_root, FILES[group]["holdout"])
        u = raw_u - frozen["Tref_C"]
        target = u[:, 1:5]
        persistence = np.broadcast_to(target[0], target.shape).copy()
        weights = np.arange(1, 5)[None, :] / 5
        linear = u[:, :1] * (1-weights) + u[:, 5:] * weights
        report["baselines"] = {"persistence": _metrics(persistence, target), "linear_boundary": _metrics(linear, target)}
        for method in METHODS:
            chosen = frozen["models"][method]["selected"]
            if chosen is None:
                report["models"][method]["validation"] = {"status": "insufficient", "failure_reason": frozen["models"][method]["failure_reason"]}
                continue
            try:
                prediction, score = _score(chosen["terms"], t, u, method)
                predictions[method] = prediction
                report["models"][method]["validation"] = {"status": "evaluated", **score,
                    "below_descriptive_5pct": score["relative_l2"] is not None and score["relative_l2"] < .05,
                    "beats_both_baselines": all(score["rmse_K"] < baseline["rmse_K"] for baseline in report["baselines"].values())}
            except (ValueError, RuntimeError, FloatingPointError, np.linalg.LinAlgError) as exc:
                report["models"][method]["validation"] = {"status": "failed", "failure_reason": str(exc)}
        physical = report["models"]["physical"]["validation"]
        report["comparison"] = {
            "new_vs_l1_rmse_improvement_fraction": _improvement(physical, report["models"]["l1"]["validation"]),
            "new_vs_persistence_rmse_improvement_fraction": _improvement(physical, report["baselines"]["persistence"]),
            "new_vs_linear_boundary_rmse_improvement_fraction": _improvement(physical, report["baselines"]["linear_boundary"]),
            "positive_means_new_model_lower_rmse": True}
        report["validation"] = {"status": "evaluated" if predictions else "failed", "rows": len(t),
                                "scored_rows": len(t)-1, "scored_sensor_count": 4, "boundary_sensor_count": 2,
                                "time_start_s": float(t[0]), "time_end_s": float(t[-1]),
                                "sampling_interval_s": {"min": float(np.diff(t).min()), "max": float(np.diff(t).max()),
                                                        "median": float(np.median(np.diff(t)))},
                                "holdout_filename": FILES[group]["holdout"]}
        if predictions:
            arrays = {"t": t, "observed_centered": u, "persistence": persistence, "linear_boundary": linear}
            arrays.update({"predicted_"+method: array for method, array in predictions.items()})
            np.savez_compressed(output / "trajectories.npz", **arrays)
            report["trajectory_available"] = True
            try:
                _plot_case(t, u, predictions, linear, report, output)
                report["plot_available"] = True
            except (ValueError, OSError, RuntimeError) as exc:
                report["plot_failure_reason"] = str(exc)
        report["conclusion"] = "两种预先冻结方法均按原定高电压记录报告；未用测试结果调参或重新选模。"
    except (ValueError, OSError, RuntimeError, FloatingPointError, IndexError) as exc:
        report["validation"] = {"status": "failed", "failure_reason": str(exc)}
        for method in METHODS:
            report["models"][method]["validation"] = {"status": "failed", "failure_reason": str(exc)}
        report["conclusion"] = "本组固定高电压记录验证失败；失败保留，未换数据或重新拟合。"
    _write_json(output / "report.json", report)
    return report


def _summary_record(report):
    record = {"case": report["case"], "title": {"aluminum": "铝杆 28V", "brass": "黄铜杆 29V"}[report["case"]],
              "validation_status": report["validation"]["status"], "Tref_C": report["Tref_C"],
              "baselines": report["baselines"], "comparison": report["comparison"],
              "plot_available": report["plot_available"], "report": f"{report['case']}/report.json",
              "frozen_model_sha256": report["frozen_model_sha256"]}
    for method in METHODS:
        selected = report["models"][method]["selected"] or {}
        record[method] = {**report["models"][method]["validation"], "equation": selected.get("equation"),
                          "support_size": selected.get("support_size"), "model_name": selected.get("model_name"),
                          "parameters": selected.get("parameters"),
                          "near_optimal_supports": report["models"][method]["near_optimal_supports"]}
    return record


def evaluate_benchmark(output_root=OUTPUT_ROOT, *, data_root=DATA_ROOT):
    output_root, data_root = Path(output_root), Path(data_root)
    _output_guard(output_root, data_root)
    # Invalidate display/export status before any integrity check or parsing.
    _invalidate(output_root)
    try:
        index, frozen_models = _verify_frozen_index(output_root, data_root)
    except (ValueError, OSError, KeyError, TypeError) as exc:
        _invalidate(output_root, "failed", str(exc))
        raise
    records = []
    for frozen, entry in zip(frozen_models, index["records"]):
        report = _evaluate_group(frozen, entry["frozen_model_sha256"], output_root, data_root)
        record = _summary_record(report)
        records.append(record)
        print(json.dumps({"stage": "fixed_external_holdout", **record}, ensure_ascii=False), flush=True)
    summary = {"description": "独立教学实验来源：低电压 60/40 训练开发，高电压整条条件测试。",
               "source": _source(data_root), "groups": list(GROUPS), "records": records,
               "status": "evaluated" if any(r["validation_status"] == "evaluated" for r in records) else "failed",
               "primary_metric": "RMSE K across four interior sensors, initial row excluded",
               "prediction_scope": "complete measured CH1/CH6 boundaries and initial CH2..CH5",
               "is_zero_shot_transfer": False, "unique_physical_equation_established": False,
               "freeze_index_sha256": _sha(output_root / "freeze_index.json"), "plot_available": False,
               "generated_at": _now(), "limitations": LIMITATIONS}
    _write_json(output_root / "benchmark.json", summary)
    try:
        _plot_summary(summary, output_root)
        summary["plot_available"] = True
    except (ValueError, OSError, RuntimeError) as exc:
        summary["plot_failure_reason"] = str(exc)
    _write_json(output_root / "benchmark.json", summary)
    return summary


def run_benchmark(output_root=OUTPUT_ROOT, *, data_root=DATA_ROOT):
    download_data(data_root)
    freeze_benchmark(output_root, data_root=data_root)
    return evaluate_benchmark(output_root, data_root=data_root)


def _pyplot():
    import os
    import tempfile
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "pde-lab-matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _plot_case(t, u, predictions, linear, report, output):
    plt = _pyplot()
    fig, axes = plt.subplots(2, 3, figsize=(14, 7), layout="constrained")
    colors = {"physical": "#c44d34", "l1": "#537eaa"}
    labels = {"physical": "Physical-prior model", "l1": "Original L1 method"}
    for i, ax in enumerate(axes.flat[:4]):
        ax.plot(t, u[:, i+1], color="#182d3d", lw=1.6, label="Measured")
        for method, prediction in predictions.items():
            ax.plot(t, prediction[:, i], color=colors[method], lw=1.3, label=labels[method])
        ax.plot(t, linear[:, i], color="#67917b", ls="--", lw=1, label="Linear-boundary baseline")
        ax.set(title=f"CH{i+2}: held-out interior", xlabel="Time (s)", ylabel="T - training reference (K)")
        if i == 0:
            ax.legend(fontsize=7)
    axes.flat[4].plot(t, u[:, 0], label="CH1: supplied boundary")
    axes.flat[4].plot(t, u[:, 5], label="CH6: supplied boundary")
    axes.flat[4].set(title="Measured boundary inputs, not scored", xlabel="Time (s)", ylabel="T - reference (K)")
    axes.flat[4].legend(fontsize=8)
    names, values, palette = [], [], []
    for method in METHODS:
        metric = report["models"][method]["validation"].get("rmse_K")
        names.append("Physical" if method == "physical" else "L1")
        values.append(np.nan if metric is None else metric)
        palette.append(colors[method])
    for key, name, color in (("persistence", "Persistence", "#99a7ad"), ("linear_boundary", "Linear boundary", "#67917b")):
        names.append(name); values.append(report["baselines"][key]["rmse_K"]); palette.append(color)
    axes.flat[5].bar(names, values, color=palette)
    axes.flat[5].set(title="Four interiors; initial frame excluded", ylabel="RMSE (K)")
    axes.flat[5].tick_params(axis="x", labelrotation=15)
    for i, value in enumerate(values):
        axes.flat[5].text(i, value if np.isfinite(value) else 0, f"{value:.3g}" if np.isfinite(value) else "FAILED", ha="center", va="bottom", fontsize=8)
    fig.suptitle(f"Independent source: {report['case']} higher-voltage holdout | coefficients calibrated on lower voltage", fontsize=12)
    fig.savefig(Path(output) / "validation.png", dpi=150)
    plt.close(fig)


def _plot_summary(summary, output_root):
    plt = _pyplot()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout="constrained")
    x = np.arange(len(summary["records"]))
    styles = [("physical", "Physical prior", "#c44d34"), ("l1", "Original L1", "#537eaa"),
              ("persistence", "Persistence", "#99a7ad"), ("linear_boundary", "Linear boundary", "#67917b")]
    for j, (key, label, color) in enumerate(styles):
        values = [r.get(key, r.get("baselines", {}).get(key, {})).get("rmse_K") for r in summary["records"]]
        ys = [np.nan if value is None else value for value in values]
        axes[0].bar(x + (j-1.5)*.19, ys, width=.18, color=color, label=label)
    axes[0].set(ylabel="RMSE (K)", title="All prespecified high-voltage tests")
    axes[0].set_xticks(x, [r["case"] for r in summary["records"]])
    axes[0].legend(fontsize=8)
    for j, (key, label, color) in enumerate(styles[:2]):
        values = [r.get(key, {}).get("relative_l2") for r in summary["records"]]
        axes[1].bar(x + (j-.5)*.3, [np.nan if v is None else 100*v for v in values], width=.28, color=color, label=label)
    axes[1].axhline(5, color="#777", ls="--", lw=1, label="5% descriptive reference")
    axes[1].set(ylabel="Centered relative L2 (%)", title="Same training-only temperature reference")
    axes[1].set_xticks(x, [r["case"] for r in summary["records"]])
    axes[1].legend(fontsize=8)
    fig.suptitle("External teaching-lab measurements | supplied boundary histories | two held-out conditions", fontsize=12)
    fig.savefig(Path(output_root) / "benchmark.png", dpi=150)
    plt.close(fig)
