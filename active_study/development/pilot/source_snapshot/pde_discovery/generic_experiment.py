"""Shared-library PDE discovery on three public numerical benchmarks.

MAT ingestion mechanically decodes each full field before writing chronological
roles. Scientific fitting reads training/development roles only. Final arrays
and reference equations are scored only after every candidate has been frozen.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import urllib.request

import numpy as np
from scipy.io import loadmat

from .generic_engine import FIT_PROTOCOL, TERM_NAMES, fit_candidates, forecast
from .real_experiment import _sha, _write_json

GROUPS = ("burgers", "kdv", "ks")
ROLES = ("training", "development", "final")
REPOSITORY = "snagcliffs/PDE-FIND"
COMMIT = "86911349b9fac82996e35465a13d06205730cc05"
FILENAMES = {"burgers": "burgers.mat", "kdv": "kdv.mat", "ks": "kuramoto_sivishinky.mat"}
DATA_ROOT = Path(__file__).resolve().parent / "data" / "generic_pde"
OUTPUT_ROOT = Path("artifacts/generic_pde")
PROJECT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = PROJECT / "experiments" / "GENERIC_PROTOCOL.md"
REFERENCE_PATH = PROJECT / "experiments" / "generic_reference_equations.json"
NEAR_OPTIMAL_FACTOR = 1.05
DOWNLOAD_TIMEOUT_S = 30
DOWNLOAD_LIMIT_BYTES = 8 * 1024 * 1024
LIMITATIONS = [
    "三个数据集均为外部公开数值模拟基准，不是真实仪器测量。",
    "同一轨迹按时间 60/20/20 分割；最终段不是新初始条件的独立实验。",
    "MAT 导入必须机械解码完整数组，再按预定行号拆分；不声称最终数值从未被程序解码。",
    "同一九项候选库用于所有数据；未强制热方程、扩散项或参考方程支持。",
    "模型只由训练与开发数据确定；参考方程仅在全部模型冻结后参与评分。",
    "范围为一维、标量、常系数、均匀周期网格；任意 PDE、多场耦合和非周期边界未覆盖。",
    "最终预测只输入该段首帧；不注入未来场或逐时实测边界。",
    "结构恢复、系数误差与未来预测误差是不同指标；混沌系统长期相位误差不等同结构错误。",
    "系数向量误差依赖作者坐标与变量缩放，不跨不同物理单位合并解释。",
    "公开原仓库未发现明确再分发许可；交付下载器及派生结果，不复制原始 MAT 或观测数组。",
    "首次评分后最终段已被查看；之后重跑是复现，不产生新的未见检验。",
]


def _source_hashes():
    root = Path(__file__).resolve().parent
    return {name: _sha(root / name) for name in
            ("generic_experiment.py", "generic_engine.py", "real_experiment.py")}


def _blob_digest(content):
    return hashlib.sha1(b"blob " + str(len(content)).encode("ascii") + b"\0" + content).hexdigest()


def _manifest(data_root):
    result = json.loads((Path(data_root) / "source_manifest.json").read_text(encoding="utf-8"))
    if result.get("repo") != REPOSITORY or result.get("commit") != COMMIT:
        raise ValueError("只接受预定作者仓库与固定提交。")
    groups = result.get("groups", {})
    if set(groups) != set(GROUPS):
        raise ValueError("来源清单必须完整包含三个预定数据集。")
    blobs = []
    for group in GROUPS:
        entry = groups[group]
        filename = FILENAMES[group]
        url = f"https://raw.githubusercontent.com/{REPOSITORY}/{COMMIT}/Datasets/{filename}"
        digest = entry.get("git_blob_sha1")
        if (entry.get("filename") != filename or entry.get("url") != url
                or type(entry.get("bytes")) is not int or not 0 < entry["bytes"] <= DOWNLOAD_LIMIT_BYTES
                or not isinstance(digest, str) or len(digest) != 40
                or any(c not in "0123456789abcdef" for c in digest)
                or entry.get("orientation") != "space_time"
                or set(entry.get("mat_keys", {})) != {"x", "t", "u"}):
            raise ValueError(f"来源文件、坐标或校验元数据无效：{group}")
        blobs.append(digest)
    if len(set(blobs)) != len(blobs):
        raise ValueError("三个数据集不能重复使用同一原始文件。")
    return result


def _verify_bytes(content, entry):
    if len(content) != entry["bytes"] or _blob_digest(content) != entry["git_blob_sha1"]:
        raise ValueError(f"原始 MAT 文件大小或 Git blob 校验失败：{entry['filename']}")
    digest = hashlib.sha256(content).hexdigest()
    if entry.get("sha256") is not None and entry["sha256"] != digest:
        raise ValueError(f"原始 MAT SHA-256 与来源清单不符：{entry['filename']}")
    return {"filename": entry["filename"], "bytes": len(content), "sha256": digest,
            "git_blob_sha1": entry["git_blob_sha1"], "url": entry["url"]}


def download_data(data_root=DATA_ROOT):
    """Obtain immutable raw bytes without loading numeric arrays or author code."""
    data_root = Path(data_root)
    manifest = _manifest(data_root)
    raw = data_root / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    records = {}
    for group in GROUPS:
        entry = manifest["groups"][group]
        path = raw / entry["filename"]
        if path.exists():
            content = path.read_bytes()
        else:
            request = urllib.request.Request(entry["url"], headers={"User-Agent": "PDE-discovery-validation/1.0"})
            with urllib.request.urlopen(request, timeout=DOWNLOAD_TIMEOUT_S) as response:
                content = response.read(entry["bytes"] + 1)
            _verify_bytes(content, entry)
            with path.open("xb") as destination:
                destination.write(content)
        records[group] = _verify_bytes(content, entry)
    receipt = {"repo": REPOSITORY, "commit": COMMIT,
               "source_manifest_sha256": _sha(data_root / "source_manifest.json"),
               "numeric_arrays_loaded": False, "records": records}
    _write_json(data_root / "download_receipt.json", receipt)
    return receipt


def _raw_preflight(data_root):
    data_root = Path(data_root)
    manifest = _manifest(data_root)
    receipt = json.loads((data_root / "download_receipt.json").read_text(encoding="utf-8"))
    actual = {group: _verify_bytes((data_root / "raw" / entry["filename"]).read_bytes(), entry)
              for group, entry in ((g, manifest["groups"][g]) for g in GROUPS)}
    if (receipt.get("records") != actual or receipt.get("source_manifest_sha256")
            != _sha(data_root / "source_manifest.json")):
        raise ValueError("下载收据与固定来源或原始文件不一致。")
    return manifest, receipt


def _validate_arrays(x, t, u):
    x, t, u = np.asarray(x), np.asarray(t), np.asarray(u)
    if any(np.iscomplexobj(array) for array in (x, t, u)):
        raise ValueError("基准适配器只接受实数场，不静默丢弃复数分量。")
    x, t, u = np.asarray(x, float), np.asarray(t, float), np.asarray(u, float)
    if (x.ndim != 1 or t.ndim != 1 or u.shape != (len(t), len(x))
            or not 16 <= len(x) <= 2048 or not 16 <= len(t) <= 5001
            or u.size > 3_000_000 or not all(np.isfinite(a).all() for a in (x, t, u))):
        raise ValueError("需有限实数时空场 u[时间,空间]，坐标与维度须符合固定预算。")
    for array in (x, t):
        delta = np.diff(array)
        if np.any(delta <= 0) or not np.allclose(delta, delta[0], rtol=1e-6, atol=1e-10):
            raise ValueError("坐标必须严格递增且均匀；不自动重采样。")
    return x, t, u


def prepare_data(data_root=DATA_ROOT):
    """Mechanically decode complete compressed MAT fields into fixed roles.

    The final field is necessarily decoded during this conversion. No model,
    reference equation, performance metric, smoothing or data-dependent split
    is computed here. This is ingestion, not an assertion of unread raw bytes.
    """
    data_root = Path(data_root)
    manifest, receipt = _raw_preflight(data_root)
    records = {}
    for group in GROUPS:
        entry = manifest["groups"][group]
        keys = entry["mat_keys"]
        parsed = loadmat(data_root / "raw" / entry["filename"], variable_names=list(keys.values()))
        x, t = np.asarray(parsed[keys["x"]]).reshape(-1), np.asarray(parsed[keys["t"]]).reshape(-1)
        field = np.asarray(parsed[keys["u"]])
        expected_shape = entry.get("field_shape")
        if expected_shape is not None and list(field.shape) != expected_shape:
            raise ValueError(f"原始矩阵形状不符来源清单：{group}")
        if not np.isfinite(field).all():
            raise ValueError("原始场包含非有限实数或复数分量。")
        imaginary_max = float(np.max(np.abs(field.imag))) if np.iscomplexobj(field) else 0.0
        was_complex = bool(np.iscomplexobj(field))
        common_scale = max(1.0, float(np.max(np.abs(field.real))), imaginary_max)
        real_norm = float(np.linalg.norm(field.real/common_scale))
        imaginary_norm = float(np.linalg.norm(field.imag/common_scale)) if was_complex else 0.0
        imaginary_to_real = imaginary_norm/real_norm if real_norm > 0 else None
        if was_complex:
            if entry.get("take_real_part") is not True:
                raise ValueError("源场是复数，但作者元数据未定义取实部；拒绝静默转换。")
            field = field.real
        x, t, u = _validate_arrays(x, t, field.T)
        first, second = int(np.floor(.6 * len(t))), int(np.floor(.8 * len(t)))
        intervals = ((0, first), (first, second), (second, len(t)))
        if any(stop-start < 16 for start, stop in intervals):
            raise ValueError("固定 60/20/20 划分的每段均需至少十六帧。")
        parts = {}
        for role, (start, stop) in zip(ROLES, intervals):
            path = data_root / "prepared" / group / f"{role}.npz"
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(path, x=x, t=t[start:stop], u=u[start:stop])
            parts[role] = {"file": str(path.relative_to(data_root)), "sha256": _sha(path),
                           "rows": [start, stop], "frames": stop-start,
                           "time_range": [float(t[start]), float(t[stop-1])]}
        records[group] = {"raw_sha256": receipt["records"][group]["sha256"],
                          "raw_field_shape_space_time": list(field.shape),
                          "space_points": len(x), "total_frames": len(t),
                          "x_first": float(x[0]), "x_last": float(x[-1]),
                          "dx": float(x[1]-x[0]), "dt": float(t[1]-t[0]),
                          "periodic_domain_length": float((x[1]-x[0])*len(x)),
                          "real_component_conversion": {"source_array_was_complex": was_complex,
                              "applied": was_complex, "maximum_absolute_imaginary_component": imaginary_max,
                              "imaginary_to_real_l2_ratio": imaginary_to_real,
                              "rule": "author-defined real(usol) when source take_real_part=true; no magnitude gate",
                              "imaginary_component_is_not_asserted_to_be_roundoff": True},
                          "parts": parts}
    record = {"source_manifest_sha256": _sha(data_root / "source_manifest.json"),
              "download_receipt_sha256": _sha(data_root / "download_receipt.json"),
              "protocol_document_sha256": _sha(PROTOCOL_PATH), "records": records,
              "ingestion_decoded_full_original_arrays": True,
              "final_values_used_for_modeling_or_selection": False,
              "reference_equations_loaded": False, "smoothing_or_spatial_resampling": False,
              "split_rule": "chronological rows [0,floor(.6N)),[floor(.6N),floor(.8N)),[floor(.8N),N)"}
    _write_json(data_root / "preparation_record.json", record)
    return record


def _prepared_preflight(data_root):
    data_root = Path(data_root)
    _raw_preflight(data_root)
    record = json.loads((data_root / "preparation_record.json").read_text(encoding="utf-8"))
    if (record.get("source_manifest_sha256") != _sha(data_root / "source_manifest.json")
            or record.get("download_receipt_sha256") != _sha(data_root / "download_receipt.json")
            or record.get("protocol_document_sha256") != _sha(PROTOCOL_PATH)
            or set(record.get("records", {})) != set(GROUPS)):
        raise ValueError("准备记录与来源、下载或预定协议不一致。")
    for group in GROUPS:
        details = record["records"][group]
        n = details["total_frames"]
        endpoints = ((0, int(np.floor(.6*n))), (int(np.floor(.6*n)), int(np.floor(.8*n))),
                     (int(np.floor(.8*n)), n))
        if set(details["parts"]) != set(ROLES):
            raise ValueError("缺少训练、开发或最终角色。")
        for role, interval in zip(ROLES, endpoints):
            part = details["parts"][role]
            expected_file = f"prepared/{group}/{role}.npz"
            if (part.get("file") != expected_file or part.get("rows") != list(interval)
                    or part.get("frames") != interval[1]-interval[0]
                    or _sha(data_root / expected_file) != part["sha256"]):
                raise ValueError(f"预定角色、划分或准备数组校验失败：{group}/{role}")
    return record


def load_partition(data_root, group, role):
    if group not in GROUPS or role not in ROLES:
        raise ValueError("未知的数据集或数据角色。")
    data_root = Path(data_root)
    record = json.loads((data_root / "preparation_record.json").read_text(encoding="utf-8"))
    part = record["records"][group]["parts"][role]
    path = data_root / f"prepared/{group}/{role}.npz"
    if _sha(path) != part["sha256"]:
        raise ValueError("准备数组哈希不符。")
    with np.load(path, allow_pickle=False) as parsed:
        x, t, u = _validate_arrays(parsed["x"], parsed["t"], parsed["u"])
    if len(t) != part["frames"]:
        raise ValueError("准备数组帧数不符预定角色。")
    return {"x": x, "t": t, "u": u, "boundary": "periodic"}


def _protocol(data_root):
    data_root = Path(data_root)
    manifest = _manifest(data_root)
    return {"groups": list(GROUPS), "split": "chronological 60/20/20; same original trajectory",
            "library": list(TERM_NAMES), "engine": FIT_PROTOCOL,
            "candidate_selection": "development RMSE <= 1.05*best, then fewest terms, RMSE, candidate_id",
            "near_optimal_factor": NEAR_OPTIMAL_FACTOR,
            "is_measured_data": False, "is_l1_discovery": True,
            "mandatory_physical_terms": [], "reference_equations_used_for_selection": False,
            "final_selects_model": False, "all_three_models_frozen_before_final_scoring": True,
            "prediction_inputs": "final-segment initial spatial field, x and requested times only",
            "future_field_or_boundary_input": False,
            "boundary_evidence": {group: manifest["groups"][group].get("boundary_evidence", {}) for group in GROUPS},
            "reference_role": "structure/coefficient scoring only, parsed after global freeze verification",
            "ingestion_full_array_decode_disclosed": True,
            "protocol_document_sha256": _sha(PROTOCOL_PATH),
            "ingestion_amendment_sha256": _sha(PROJECT / "experiments" / "GENERIC_INGESTION_AMENDMENT.json"),
            "reference_file_sha256_not_parsed": _sha(REFERENCE_PATH),
            "source_manifest_sha256": _sha(data_root / "source_manifest.json"),
            "download_receipt_sha256": _sha(data_root / "download_receipt.json"),
            "preparation_record_sha256": _sha(data_root / "preparation_record.json"),
            "source_sha256": _source_hashes()}


def _guard_output(output_root, data_root):
    output = Path(output_root).resolve()
    protected = [PROJECT / "pde_discovery", PROJECT / "experiments", PROJECT / "tests",
                 Path(data_root).resolve(), *[PROJECT / "artifacts" / name for name in
                 ("real_heat", "mendeley_heat", "refined_heat", "additive_heat", "identifiability",
                  "external_lab", "synthetic-final")]]
    if output in (PROJECT, PROJECT / "artifacts") or any(output.is_relative_to(p) for p in protected):
        raise ValueError("通用 PDE 输出必须使用独立目录，不得覆盖代码、数据或旧实验。")


def forecast_metrics(predicted, observed):
    predicted, observed = np.asarray(predicted, float), np.asarray(observed, float)
    if (predicted.shape != observed.shape or predicted.ndim != 2 or len(observed) < 2
            or observed.shape[1] == 0
            or not np.isfinite(predicted).all() or not np.isfinite(observed).all()):
        raise ValueError("预测与观测必须是匹配的有限时空矩阵。")
    residual = predicted[1:] - observed[1:]
    error, denominator = float(np.linalg.norm(residual)), float(np.linalg.norm(observed[1:]))
    return {"rmse": float(np.sqrt(np.mean(residual**2))), "mae": float(np.mean(np.abs(residual))),
            "max_absolute_error": float(np.max(np.abs(residual))),
            "relative_l2": error/denominator if denominator > 1e-12 else (0.0 if error < 1e-12 else None),
            "excluded_initial_frame": True, "scored_frames": len(residual),
            "space_points": observed.shape[1]}


def reference_metrics(terms, reference_terms):
    def coefficients(items):
        result, seen = {}, set()
        for item in items:
            name, value = item["name"], float(item["coefficient"])
            if name not in TERM_NAMES or not np.isfinite(value) or name in seen:
                raise ValueError("参考或恢复项需有限、唯一，且位于共同候选库。")
            seen.add(name)
            if value != 0:
                result[name] = value
        return result
    found, reference = coefficients(terms), coefficients(reference_terms)
    if not reference:
        raise ValueError("参考方程不能是空支持。")
    selected, truth = set(found), set(reference)
    true_positive = len(selected & truth)
    precision = true_positive/len(selected) if selected else 0.0
    recall = true_positive/len(truth)
    rows = [{"name": name, "reference": reference.get(name, 0.0), "discovered": found.get(name, 0.0),
             "absolute_error": abs(found.get(name, 0.0)-reference.get(name, 0.0)),
             "relative_error": abs(found.get(name, 0.0)-reference[name])/abs(reference[name]) if name in reference else None}
            for name in TERM_NAMES if name in selected | truth]
    denominator = np.linalg.norm(list(reference.values()))
    numerator = np.linalg.norm([row["discovered"]-row["reference"] for row in rows])
    return {"status": "evaluated", "exact_support": selected == truth, "precision": precision,
            "recall": recall, "f1": 2*precision*recall/(precision+recall) if precision+recall else 0.0,
            "true_positive_terms": sorted(selected & truth), "missing_terms": sorted(truth-selected),
            "spurious_terms": sorted(selected-truth), "coefficient_relative_l2": float(numerator/denominator),
            "coefficient_comparison": rows, "reference_terms": reference_terms,
            "coefficient_metric_scope": "author coordinate/field scaling; not unit-invariant"}


def load_reference():
    reference = json.loads(REFERENCE_PATH.read_text(encoding="utf-8"))
    if set(reference.get("references", {})) != set(GROUPS):
        raise ValueError("参考文件必须完整包含三个预定方程。")
    return reference["references"]


def _source(group):
    return {"repo": REPOSITORY, "commit": COMMIT, "filename": FILENAMES[group],
            "url": f"https://github.com/{REPOSITORY}", "kind": "public numerical simulation benchmark",
            "is_measured_data": False}


def _base_report(frozen, digest):
    selected = frozen.get("selected")
    near = frozen.get("near_optimal_supports", [])
    diagnostics = selected.get("diagnostics", {}) if selected else {}
    reasons = ["开发近优池存在多个支持。"] if len(near) > 1 else []
    if diagnostics.get("correlated_columns"):
        reasons.append("训练候选库存在相关系数绝对值大于0.995的列；有限L1路径可能遗漏等价解释。")
    active = diagnostics.get("active_library_count")
    rank = diagnostics.get("full_library_rank")
    if active is not None and rank is not None and rank < active:
        reasons.append("训练候选库的活跃列不满秩，存在结构不可识别风险。")
    return {**frozen, "frozen_model_sha256": digest,
            "discovery": {"status": "insufficient" if selected is None else ("ambiguous" if reasons else "candidate"),
                          "equation": selected.get("equation") if selected else None,
                          "terms": selected.get("terms", []) if selected else [],
                          "near_optimal_supports": near,
                          "ambiguity_reasons": reasons,
                          "correlated_columns": diagnostics.get("correlated_columns", []),
                          "full_library_rank": rank, "active_library_count": active,
                          "finite_candidate_path_is_not_exhaustive": True,
                          "unique_physical_equation_established": False},
            "validation": {"status": "not_evaluated"}, "equation_recovery": {"status": "not_evaluated"},
            "plot_available": False, "trajectory_available": False, "limitations": LIMITATIONS}


def _invalidate(output_root, status="not_evaluated", reason=None):
    records = []
    for group in GROUPS:
        report = {"case": group, "source": _source(group), "selected": None, "candidates": [],
                  "validation": {"status": status, "failure_reason": reason},
                  "equation_recovery": {"status": status, "failure_reason": reason},
                  "discovery": {"status": "insufficient", "equation": None, "terms": []},
                  "plot_available": False, "trajectory_available": False, "limitations": LIMITATIONS}
        _write_json(Path(output_root) / group / "report.json", report)
        records.append({"case": group, "validation": report["validation"],
                        "equation_recovery": report["equation_recovery"], "plot_available": False})
    _write_json(Path(output_root) / "benchmark.json", {"status": status, "records": records,
                "plot_available": False, "failure_reason": reason, "limitations": LIMITATIONS})


def _freeze_group(group, data_root, protocol, split):
    candidates, selected, near, failure, seam = [], None, [], None, None
    try:
        training = load_partition(data_root, group, "training")
        development = load_partition(data_root, group, "development")
        field = training["u"]
        seam_rms = float(np.sqrt(np.mean((field[:, 0]-field[:, -1])**2)))
        adjacent_rms = float(np.sqrt(np.mean(np.diff(field, axis=1)**2)))
        seam = {"training_only": True, "used_for_selection": False,
                "seam_step_rms": seam_rms, "adjacent_step_rms": adjacent_rms,
                "seam_to_adjacent_ratio": seam_rms/adjacent_rms if adjacent_rms > 1e-12 else None,
                "interpretation": "A periodic-grid seam is one spatial step; this diagnostic does not certify the boundary condition."}
        candidates = fit_candidates(training)
        for index, candidate in enumerate(candidates):
            candidate.setdefault("candidate_id", f"candidate-{index:02d}")
            candidate["eligible"] = False
            if not candidate.get("fit_success"):
                candidate["rejection_reason"] = "训练拟合、L1 收敛或支持秩未通过。"
                continue
            try:
                prediction = forecast(candidate["terms"], development["x"], development["t"], development["u"][0])
                candidate["development"] = forecast_metrics(prediction, development["u"])
                candidate["eligible"] = True
            except (ValueError, RuntimeError, FloatingPointError, np.linalg.LinAlgError) as exc:
                candidate["rejection_reason"] = str(exc)
        eligible = [c for c in candidates if c["eligible"]]
        if eligible:
            best = min(c["development"]["rmse"] for c in eligible)
            pool = [c for c in eligible if c["development"]["rmse"] <= NEAR_OPTIMAL_FACTOR*best + 1e-12]
            selected = min(pool, key=lambda c: (c["support_size"], c["development"]["rmse"], c["candidate_id"]))
            near = [list(support) for support in sorted({tuple(sorted(t["name"] for t in c["terms"])) for c in pool})]
        else:
            failure = "训练与开发阶段没有可用候选；不按最终数据补救。"
    except (ValueError, OSError, RuntimeError, FloatingPointError, np.linalg.LinAlgError) as exc:
        failure = str(exc)
    return {"case": group, "source": {**_source(group), "boundary_evidence": protocol["boundary_evidence"][group]},
            "protocol": protocol, "split": split,
            "selected": selected, "candidates": candidates, "near_optimal_supports": near,
            "failure_reason": failure, "training_periodic_seam": seam, "final_role_loaded_during_freeze": False,
            "reference_equations_parsed_during_freeze": False}


def freeze_benchmark(output_root=OUTPUT_ROOT, *, data_root=DATA_ROOT):
    output_root, data_root = Path(output_root), Path(data_root)
    _guard_output(output_root, data_root)
    _invalidate(output_root)
    try:
        preparation = _prepared_preflight(data_root)
        protocol = _protocol(data_root)
        _write_json(output_root / "protocol.json", protocol)
        records = []
        for group in GROUPS:
            frozen = _freeze_group(group, data_root, protocol, preparation["records"][group])
            path = output_root / group / "frozen_model.json"
            _write_json(path, frozen)
            digest = _sha(path)
            _write_json(output_root / group / "report.json", _base_report(frozen, digest))
            records.append({"case": group, "frozen_model_sha256": digest,
                            "selected_equation": frozen["selected"]["equation"] if frozen["selected"] else None,
                            "failure_reason": frozen["failure_reason"]})
            print(json.dumps({"stage": "frozen", **records[-1]}, ensure_ascii=False), flush=True)
        index = {"groups": list(GROUPS), "records": records, "protocol": protocol,
                 "protocol_artifact_sha256": _sha(output_root / "protocol.json"),
                 "all_three_models_frozen": True, "final_role_loaded_during_freeze": False,
                 "reference_equations_parsed_during_freeze": False}
        _write_json(output_root / "freeze_index.json", index)
        return index
    except (ValueError, OSError, KeyError, TypeError) as exc:
        _invalidate(output_root, "failed", str(exc))
        raise


def _verify_frozen_index(output_root, data_root):
    output_root = Path(output_root)
    index = json.loads((output_root / "freeze_index.json").read_text(encoding="utf-8"))
    _prepared_preflight(data_root)
    if (index.get("groups") != list(GROUPS)
            or [r.get("case") for r in index.get("records", [])] != list(GROUPS)
            or index.get("all_three_models_frozen") is not True
            or index.get("final_role_loaded_during_freeze") is not False
            or index.get("reference_equations_parsed_during_freeze") is not False
            or index.get("protocol") != _protocol(data_root)
            or index.get("protocol_artifact_sha256") != _sha(output_root / "protocol.json")):
        raise ValueError("全局冻结索引、源码、协议或数据校验失败。")
    models = []
    for entry in index["records"]:
        path = output_root / entry["case"] / "frozen_model.json"
        if _sha(path) != entry["frozen_model_sha256"]:
            raise ValueError("冻结模型校验失败；在读取任何最终角色前停止。")
        frozen = json.loads(path.read_text(encoding="utf-8"))
        if frozen.get("case") != entry["case"] or frozen.get("protocol") != index["protocol"]:
            raise ValueError("冻结模型角色或协议不一致。")
        models.append(frozen)
    return index, models


def _evaluate_group(frozen, digest, reference, output_root, data_root):
    report = _base_report(frozen, digest)
    output = Path(output_root) / frozen["case"]
    _write_json(output / "report.json", report)
    selected = frozen["selected"]
    if selected is None:
        report["validation"] = {"status": "insufficient", "failure_reason": frozen["failure_reason"]}
        report["equation_recovery"] = {"status": "insufficient", "failure_reason": "没有冻结候选，不计为恢复成功。"}
        _write_json(output / "report.json", report)
        return report
    try:
        report["equation_recovery"] = reference_metrics(selected["terms"], reference["terms"])
        final = load_partition(data_root, frozen["case"], "final")
        observed = final["u"]
        persistence = np.broadcast_to(observed[0], observed.shape).copy()
        baseline = forecast_metrics(persistence, observed)
        report["validation"] = {"status": "failed", "persistence": baseline,
                                "time_range": [float(final["t"][0]), float(final["t"][-1])]}
        prediction = forecast(selected["terms"], final["x"], final["t"], observed[0])
        score = forecast_metrics(prediction, observed)
        report["validation"].update({"status": "evaluated", **score,
            "beats_persistence": score["rmse"] < baseline["rmse"],
            "future_field_is_input": False,
            "relative_rmse_improvement": 1-score["rmse"]/baseline["rmse"] if baseline["rmse"] > 1e-12 else None})
        np.savez_compressed(output / "trajectories.npz", x=final["x"], t=final["t"],
                            observed=observed, prediction=prediction, persistence=persistence)
        report["trajectory_available"] = True
        try:
            _plot_case(final, prediction, report, output)
            report["plot_available"] = True
        except (ValueError, OSError, RuntimeError) as exc:
            report["plot_failure_reason"] = str(exc)
    except (ValueError, OSError, RuntimeError, FloatingPointError, np.linalg.LinAlgError) as exc:
        report["validation"].update({"status": "failed", "failure_reason": str(exc)})
    _write_json(output / "report.json", report)
    return report


def _summary_record(report):
    selected = report["selected"] or {}
    return {"case": report["case"], "equation": selected.get("equation"),
            "terms": selected.get("terms", []), "support_size": selected.get("support_size"),
            "discovery_status": report["discovery"]["status"],
            "near_optimal_supports": report.get("near_optimal_supports", []),
            "validation": report["validation"], "equation_recovery": report["equation_recovery"],
            "plot_available": report["plot_available"], "report": f"{report['case']}/report.json",
            "frozen_model_sha256": report["frozen_model_sha256"]}


def evaluate_benchmark(output_root=OUTPUT_ROOT, *, data_root=DATA_ROOT):
    output_root, data_root = Path(output_root), Path(data_root)
    _guard_output(output_root, data_root)
    _invalidate(output_root)
    try:
        index, frozen_models = _verify_frozen_index(output_root, data_root)
        references = load_reference()
    except (ValueError, OSError, KeyError, TypeError) as exc:
        _invalidate(output_root, "failed", str(exc))
        raise
    records = []
    for frozen, entry in zip(frozen_models, index["records"]):
        report = _evaluate_group(frozen, entry["frozen_model_sha256"], references[frozen["case"]], output_root, data_root)
        record = _summary_record(report)
        records.append(record)
        print(json.dumps({"stage": "fixed_final_score", **record}, ensure_ascii=False), flush=True)
    summary = {"description": "公开数值基准上的共同九项 L1 PDE 发现；不是热方程先验校准。",
               "records": records, "groups": list(GROUPS), "is_measured_data": False,
               "status": "evaluated" if any(r["validation"]["status"] == "evaluated" for r in records) else "failed",
               "freeze_index_sha256": _sha(output_root / "freeze_index.json"),
               "exact_support_recovered_count": sum(r["equation_recovery"].get("exact_support", False) for r in records),
               "declared_case_count": len(GROUPS), "plot_available": False, "limitations": LIMITATIONS}
    _write_json(output_root / "benchmark.json", summary)
    try:
        _plot_summary(summary, output_root)
        summary["plot_available"] = True
    except (ValueError, OSError, RuntimeError) as exc:
        summary["plot_failure_reason"] = str(exc)
    _write_json(output_root / "benchmark.json", summary)
    return summary


def run_benchmark(output_root=OUTPUT_ROOT, *, data_root=DATA_ROOT):
    _guard_output(output_root, data_root)
    _invalidate(output_root)
    try:
        download_data(data_root)
        prepare_data(data_root)
    except (ValueError, OSError, KeyError, TypeError) as exc:
        _invalidate(output_root, "failed", str(exc))
        raise
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


def _plot_case(data, prediction, report, output):
    plt = _pyplot()
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    rows = report["equation_recovery"]["coefficient_comparison"]
    index = np.arange(len(rows))
    axes[0, 0].bar(index-.18, [r["reference"] for r in rows], width=.35, label="Reference (scoring only)", color="#254b63")
    axes[0, 0].bar(index+.18, [r["discovered"] for r in rows], width=.35, label="Discovered", color="#cf7048")
    axes[0, 0].set_xticks(index, [r["name"] for r in rows])
    axes[0, 0].set(title="Term and coefficient recovery", ylabel="Coefficient in author coordinates")
    axes[0, 0].legend(fontsize=8)
    observed = data["u"]
    vmin, vmax = float(observed.min()), float(observed.max())
    extent = [float(data["x"][0]), float(data["x"][-1]), float(data["t"][-1]), float(data["t"][0])]
    for ax, field, title in ((axes[0, 1], observed, "External numerical field: final segment"),
                              (axes[1, 0], prediction, "Frozen PDE free rollout from initial frame")):
        im = ax.imshow(field, aspect="auto", extent=extent, vmin=vmin, vmax=vmax, cmap="viridis")
        ax.set(title=title, xlabel="x (author coordinates)", ylabel="t (author coordinates)")
        fig.colorbar(im, ax=ax, shrink=.85)
    error = prediction-observed
    maximum = max(float(np.max(np.abs(error))), 1e-12)
    im = axes[1, 1].imshow(error, aspect="auto", extent=extent, vmin=-maximum, vmax=maximum, cmap="RdBu_r")
    axes[1, 1].set(title="Prediction minus numerical observation", xlabel="x", ylabel="t")
    fig.colorbar(im, ax=axes[1, 1], shrink=.85)
    fig.suptitle(f"{report['case']} | shared-library L1 discovery | chronological holdout, simulated data", fontsize=12)
    fig.savefig(Path(output) / "validation.png", dpi=140)
    plt.close(fig)


def _plot_summary(summary, output_root):
    plt = _pyplot()
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), layout="constrained")
    records = summary["records"]
    fields = [("equation_recovery", "f1", "Term-support F1", 1),
              ("equation_recovery", "coefficient_relative_l2", "Coefficient vector relative error (%)", 100),
              ("validation", "relative_l2", "Future field relative L2 error (%)", 100)]
    for ax, (section, field, title, scale) in zip(axes, fields):
        values = [r.get(section, {}).get(field) for r in records]
        ax.bar(np.arange(len(records)), [np.nan if v is None else scale*v for v in values], color="#237568")
        ax.set_xticks(np.arange(len(records)), [r["case"] for r in records])
        ax.set(title=title)
        if field == "f1":
            ax.set_ylim(0, 1.13)
        for i, value in enumerate(values):
            ax.text(i, 0 if value is None else scale*value,
                    "FAILED" if value is None else f"{scale*value:.3g}", ha="center", va="bottom", fontsize=9)
    fig.suptitle("Three public numerical PDE benchmarks | structure, coefficients and prediction reported separately", fontsize=12)
    fig.savefig(Path(output_root) / "benchmark.png", dpi=140)
    plt.close(fig)
