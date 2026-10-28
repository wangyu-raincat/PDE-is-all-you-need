"""Local demonstration and small, bounded tool API for agent-platform integration."""
from __future__ import annotations

import json
import math
import threading
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, Response
from starlette.concurrency import run_in_threadpool

from . import __version__
from .controller import SCENARIOS, discover_uploaded, run_demo
from .simulator import make_dataset
from .real_benchmark import OUTPUT_ROOT, PRIMARY_CASE, CASES, run_case

DEFAULT_OUTPUT = OUTPUT_ROOT / PRIMARY_CASE
REFINED_OUTPUT = Path("artifacts/refined_heat")
ADDITIVE_OUTPUT = Path("artifacts/additive_heat")
IDENTIFIABILITY_OUTPUT = Path("artifacts/identifiability")
EXTERNAL_OUTPUT = Path("artifacts/external_lab")
EXTERNAL_CASES = ("aluminum", "brass")
GENERIC_OUTPUT = Path("artifacts/generic_pde")
GENERIC_CASES = ("burgers", "kdv", "ks")
ADDITIVE_CASES = {f"{group}{repeat}": (group, repeat)
                  for group in ("xn", "xp", "yn", "yp") for repeat in (3, 4, 5)}

PACKAGE = Path(__file__).resolve().parent
MAX_BODY = 6 * 1024 * 1024
WORKER = threading.BoundedSemaphore(1)
app = FastAPI(title="PDE 发现与验证工具", version=__version__, description="周期仿真 PDE 发现与六测点实测热杆条件预测。固定候选库、训练/选模/留出隔离；未接入大模型。")
DATASET_SCHEMA = {"type": "object", "required": ["x", "t", "u"], "properties": {
    "x": {"type": "array", "items": {"type": "number"}, "minItems": 32, "maxItems": 192},
    "t": {"type": "array", "items": {"type": "number"}, "minItems": 16, "maxItems": 401},
    "u": {"type": "array", "items": {"type": "array", "items": {"type": "number"}}},
    "boundary": {"type": "string", "enum": ["periodic"], "default": "periodic"},
}}


def _body_schema(schema):
    return {"requestBody": {"required": True, "content": {"application/json": {"schema": schema}}}}


def _json_safe(value):
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


async def _read_json(request):
    body = bytearray()
    async for chunk in request.stream():
        body.extend(chunk)
        if len(body) > MAX_BODY:
            raise HTTPException(413, "JSON 文件最大为 6 MB。")
    try:
        value = json.loads(body)
    except (ValueError, UnicodeError, RecursionError) as exc:
        raise HTTPException(400, "无法解析 JSON，请检查文件格式。") from exc
    if not isinstance(value, dict):
        raise HTTPException(422, "请求必须是 JSON 对象。")
    return value


def _compute(function, *args, **kwargs):
    if not WORKER.acquire(blocking=False):
        raise HTTPException(429, "已有实验正在运行，请完成后重试。")
    try:
        return _json_safe(function(*args, **kwargs))
    except (ValueError, TypeError, KeyError) as exc:
        raise HTTPException(422, str(exc)) from exc
    finally:
        WORKER.release()


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(PACKAGE / "static" / "index.html")


@app.get("/real", include_in_schema=False)
def real_experiment_page():
    return FileResponse(PACKAGE / "static" / "real.html")


@app.get("/refined", include_in_schema=False)
def refined_experiment_page():
    return FileResponse(PACKAGE / "static" / "refined.html")


@app.get("/validation", include_in_schema=False)
def validation_page():
    return FileResponse(PACKAGE / "static" / "validation.html")


@app.get("/discovery", include_in_schema=False)
def generic_page():
    return FileResponse(PACKAGE / "static" / "discovery.html")


def _generic_output(case):
    if case not in GENERIC_CASES:
        raise HTTPException(422, "跨方程基准仅包含 burgers、kdv 与 ks。")
    return GENERIC_OUTPUT / case


@app.get("/api/generic-experiment", operation_id="read_generic_discovery")
def generic_experiment_report(case: str = "burgers"):
    path = _generic_output(case) / "report.json"
    if not path.exists():
        raise HTTPException(404, "该方程类型的发现报告尚未生成。")
    return JSONResponse(json.loads(path.read_text(encoding="utf-8")))


@app.get("/api/generic-experiment/benchmark", operation_id="read_generic_benchmark")
def generic_benchmark_report():
    path = GENERIC_OUTPUT / "benchmark.json"
    if not path.exists():
        raise HTTPException(404, "跨方程基准汇总尚未生成。")
    return JSONResponse(json.loads(path.read_text(encoding="utf-8")))


@app.get("/api/generic-experiment/plot", operation_id="generic_discovery_plot")
def generic_experiment_plot(case: str = "burgers"):
    folder = _generic_output(case)
    path, report_path = folder / "validation.png", folder / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else {}
    if not path.exists() or not report.get("plot_available"):
        raise HTTPException(404, "当前跨方程报告没有可用图像。")
    return FileResponse(path, media_type="image/png")


@app.get("/external", include_in_schema=False)
def external_page():
    return FileResponse(PACKAGE / "static" / "external.html")


def _external_output(case):
    if case not in EXTERNAL_CASES:
        raise HTTPException(422, "独立来源实验仅包含 aluminum 与 brass。")
    return EXTERNAL_OUTPUT / case


@app.get("/api/external-experiment", operation_id="read_external_holdout")
def external_experiment_report(case: str = "aluminum"):
    path = _external_output(case) / "report.json"
    if not path.exists():
        raise HTTPException(404, "独立来源留出报告尚未生成。")
    return JSONResponse(json.loads(path.read_text(encoding="utf-8")))


@app.get("/api/external-experiment/benchmark", operation_id="read_external_benchmark")
def external_benchmark_report():
    path = EXTERNAL_OUTPUT / "benchmark.json"
    if not path.exists():
        raise HTTPException(404, "独立来源比较汇总尚未生成。")
    return JSONResponse(json.loads(path.read_text(encoding="utf-8")))


@app.get("/api/external-experiment/plot", operation_id="external_holdout_plot")
def external_experiment_plot(case: str = "aluminum"):
    directory = _external_output(case)
    path, report_path = directory / "validation.png", directory / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else {}
    if (not path.exists() or not report.get("plot_available")
            or report.get("validation", {}).get("status") != "evaluated"):
        raise HTTPException(404, "当前独立来源报告没有可用图像。")
    return FileResponse(path, media_type="image/png")


def _additive_output(case):
    if case not in ADDITIVE_CASES:
        raise HTTPException(422, "不支持的增材杆留出记录。")
    group, repeat = ADDITIVE_CASES[case]
    return ADDITIVE_OUTPUT / group / f"repeat{repeat}"


@app.get("/api/additive-experiment", operation_id="read_additive_holdout")
def additive_experiment_report(case: str = "xn3"):
    path = _additive_output(case) / "report.json"
    if not path.exists():
        raise HTTPException(404, "尚无该条增材杆留出报告。")
    return JSONResponse(json.loads(path.read_text(encoding="utf-8")))


@app.get("/api/additive-experiment/benchmark", operation_id="read_additive_benchmark")
def additive_benchmark():
    path = ADDITIVE_OUTPUT / "benchmark.json"
    if not path.exists():
        raise HTTPException(404, "十二条增材杆留出汇总尚未生成。")
    return JSONResponse(json.loads(path.read_text(encoding="utf-8")))


@app.get("/api/additive-experiment/plot", operation_id="additive_holdout_plot")
def additive_experiment_plot(case: str = "xn3"):
    root = _additive_output(case)
    path, report_path = root / "validation.png", root / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else {}
    if (not path.exists() or not report.get("plot_available")
            or report.get("validation", {}).get("status") != "evaluated"):
        raise HTTPException(404, "当前留出报告没有可用图像。")
    return FileResponse(path, media_type="image/png")


@app.get("/api/identifiability/summary", operation_id="read_identifiability_summary")
def identifiability_summary():
    path = IDENTIFIABILITY_OUTPUT / "summary.json"
    if not path.exists():
        raise HTTPException(404, "训练数据可辨识性诊断尚未生成。")
    return JSONResponse(json.loads(path.read_text(encoding="utf-8")))


@app.get("/api/identifiability/plot", operation_id="identifiability_plot")
def identifiability_plot():
    path = IDENTIFIABILITY_OUTPUT / "diagnostics.png"
    report_path = IDENTIFIABILITY_OUTPUT / "summary.json"
    report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else {}
    if not path.exists() or not report.get("plot_available"):
        raise HTTPException(404, "当前训练数据诊断没有可用图像。")
    return FileResponse(path, media_type="image/png")


def _refined_output(case):
    if case not in CASES:
        raise HTTPException(422, "不支持的真实实验组。")
    return REFINED_OUTPUT / case


@app.get("/api/refined-experiment", operation_id="read_refined_measurement_comparison")
def refined_experiment_report(case: str = PRIMARY_CASE):
    path = _refined_output(case) / "report.json"
    if not path.exists():
        raise HTTPException(404, "尚无改进方法报告，请先运行 refined-benchmark。")
    return JSONResponse(json.loads(path.read_text(encoding="utf-8")))


@app.get("/api/refined-experiment/benchmark", operation_id="read_refined_benchmark")
def refined_measurement_benchmark():
    path = REFINED_OUTPUT / "benchmark.json"
    if not path.exists():
        raise HTTPException(404, "改进方法汇总尚未生成。")
    return JSONResponse(json.loads(path.read_text(encoding="utf-8")))


@app.get("/api/refined-experiment/plot", operation_id="refined_measurement_comparison_plot")
def refined_experiment_plot(case: str = PRIMARY_CASE):
    root = _refined_output(case)
    path, report_path = root / "validation.png", root / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else {}
    if (not path.exists() or not report.get("plot_available")
            or report.get("validation", {}).get("status") != "evaluated"):
        raise HTTPException(404, "当前改进方法报告没有可用图像。")
    return FileResponse(path, media_type="image/png")


def _real_output(case):
    if case is None:
        return DEFAULT_OUTPUT
    if case not in CASES:
        raise HTTPException(422, "不支持的真实实验组。")
    return OUTPUT_ROOT / case


@app.get("/api/real-experiment", operation_id="read_real_measurement_experiment")
def real_experiment_report(case: str | None = None):
    path = _real_output(case) / "report.json"
    if not path.exists():
        raise HTTPException(404, "尚无真实实验报告，请先复现公开数据实验。")
    return JSONResponse(json.loads(path.read_text(encoding="utf-8")))


@app.post("/api/real-experiment", operation_id="reproduce_real_measurement_experiment")
async def reproduce_real_experiment(case: str = PRIMARY_CASE):
    _real_output(case)
    return JSONResponse(await run_in_threadpool(_compute, run_case, case))


@app.get("/api/real-experiment/benchmark", operation_id="read_measured_benchmark")
def real_measurement_benchmark():
    path = OUTPUT_ROOT / "benchmark.json"
    if not path.exists():
        raise HTTPException(404, "九组实测汇总尚未生成。")
    return JSONResponse(json.loads(path.read_text(encoding="utf-8")))


@app.get("/api/real-experiment/plot", operation_id="real_measurement_validation_plot")
def real_experiment_plot(case: str | None = None):
    path = _real_output(case) / "validation.png"
    report_path = _real_output(case) / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else {}
    if (not path.exists() or not report.get("plot_available")
            or report.get("validation", {}).get("status") != "evaluated"):
        raise HTTPException(404, "尚无真实实验验证图。")
    return FileResponse(path, media_type="image/png")


@app.get("/api/health", operation_id="health")
def health():
    return {"status": "ok", "version": __version__, "llm_connected": False,
            "huawei_platform_connected": False, "controller": "deterministic_evidence_driven"}


@app.post("/api/run", operation_id="run_simulated_discovery", openapi_extra=_body_schema({
    "type": "object", "properties": {
        "scenario": {"type": "string", "enum": sorted(SCENARIOS), "default": "diffusion"},
        "noise": {"type": "number", "minimum": 0, "maximum": .03, "default": 0},
        "seed": {"type": "integer", "minimum": 0, "maximum": 2147483647, "default": 7},
        "auto_experiment": {"type": "boolean", "default": True},
    }, "additionalProperties": False,
}))
async def run(request: Request):
    """Run a bounded simulated goal-completion experiment. Never invokes real hardware."""
    payload = await _read_json(request)
    unknown = set(payload) - {"scenario", "noise", "seed", "auto_experiment"}
    if unknown:
        raise HTTPException(422, "不支持的字段：" + ", ".join(sorted(unknown)))
    if not isinstance(payload.get("auto_experiment", True), bool):
        raise HTTPException(422, "auto_experiment 必须是布尔值。")
    return JSONResponse(await run_in_threadpool(_compute, run_demo, **payload))


@app.post("/api/discover", operation_id="discover_pde_from_observations", openapi_extra=_body_schema({
    "type": "object", "required": ["training"], "properties": {
        "training": {"type": "array", "items": DATASET_SCHEMA, "minItems": 1, "maxItems": 3},
        "validation": DATASET_SCHEMA,
    },
}))
async def uploaded(request: Request):
    """Identify PDE terms from observed periodic trajectories and independently validate."""
    payload = await _read_json(request)
    return JSONResponse(await run_in_threadpool(_compute, discover_uploaded, payload))


@app.get("/api/example", operation_id="download_example_observations")
def example(scenario: str = "diffusion", kind: str = "train", seed: int = 7):
    if scenario not in SCENARIOS or kind not in {"train", "validation", "probe", "bundle"}:
        raise HTTPException(422, "scenario 或 kind 不支持。")
    if not 0 <= seed <= 2**31 - 203:
        raise HTTPException(422, "seed 超出范围。")
    if kind == "bundle":
        data = {"training": [make_dataset(scenario, seed=seed, initial_condition="train")],
                "validation": make_dataset(scenario, seed=seed + 202, initial_condition="validation")}
    else:
        data = make_dataset(scenario, seed=seed, initial_condition=kind)
    return Response(json.dumps(_json_safe(data), ensure_ascii=False), media_type="application/json",
                    headers={"Content-Disposition": f'attachment; filename="{scenario}_{kind}.json"'})


@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    return Response(status_code=204)
