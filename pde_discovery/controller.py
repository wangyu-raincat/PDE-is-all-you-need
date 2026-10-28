"""Bounded numerical goal controller; no LLM or hidden ground truth is consulted."""
from __future__ import annotations

import copy
import hashlib
import uuid
from datetime import datetime, timezone

import numpy as np

from .engine import discover, forecast
from .simulator import make_dataset

SCENARIOS = {"diffusion", "advection_diffusion", "ambiguity"}
GOAL_ERROR = 0.05
GOAL_TEXT = "在限定候选库中识别无已检测歧义的 PDE，并在独立初始条件下达到相对预测误差 ≤ 5%。"


def _step(title, detail, state="complete"):
    return {"title": title, "detail": detail, "state": state}


def _fingerprint(dataset):
    digest = hashlib.sha256()
    # Absolute coordinate origins do not turn the same observed field into
    # independent evidence for an autonomous constant-coefficient equation.
    for key in ("u",):
        value = np.asarray(dataset[key], dtype=np.float64)
        digest.update(str(value.shape).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def _choose_experiment(result, observed):
    """Choose a bounded probe by disagreement of discovered models, not oracle truth."""
    x, t = np.asarray(observed["x"]), np.asarray(observed["t"])
    phase = 2 * np.pi * (x - x[0]) / (len(x) * (x[1] - x[0]))
    probes = [("amplitude", "1.5 sin(x)", 1.5 * np.sin(phase)),
              ("probe2", "sin(2x)", np.sin(2 * phase)),
              ("probe3", "sin(3x)", np.sin(3 * phase))]
    candidates = result.get("candidates", [])[:4]
    scores = []
    for key, label, initial in probes:
        predictions = []
        for candidate in candidates:
            try:
                prediction = forecast(candidate["terms"], x.tolist(), t.tolist(), initial.tolist())
                if np.isfinite(prediction).all():
                    predictions.append(np.asarray(prediction))
            except (ValueError, FloatingPointError, RuntimeError):
                continue
        score = None
        if len(predictions) >= 2:
            normalizer = max(np.linalg.norm(initial) * np.sqrt(len(t)), 1e-12)
            score = max(float(np.linalg.norm(a - b) / normalizer)
                        for i, a in enumerate(predictions) for b in predictions[i + 1:])
        scores.append({"initial_condition": key, "label": label, "disagreement": score})
    available = [s for s in scores if s["disagreement"] is not None]
    selected = max(available, key=lambda s: s["disagreement"]) if available else scores[1]
    return selected, scores


def validate_dataset(dataset):
    """Reject unsupported geometry and oversized inputs before numerical work."""
    if not isinstance(dataset, dict) or not all(k in dataset for k in ("x", "t", "u")):
        raise ValueError("每条轨迹必须包含 x、t、u；u 的形状为 [时间点数][空间点数]。")
    try:
        x, t, u = (np.asarray(dataset[k], dtype=float) for k in ("x", "t", "u"))
    except (TypeError, ValueError) as exc:
        raise ValueError("x、t、u 必须是规则排列的数值数组。") from exc
    if x.ndim != 1 or t.ndim != 1 or u.ndim != 2 or u.shape != (len(t), len(x)):
        raise ValueError("u 的形状必须是 [len(t), len(x)]，x、t 必须是一维数组。")
    if not 32 <= len(x) <= 192 or not 16 <= len(t) <= 401:
        raise ValueError("原型支持 32–192 个空间点、16–401 个时间点。")
    if not all(np.isfinite(a).all() for a in (x, t, u)):
        raise ValueError("数据不得包含 NaN 或无穷大。")
    for axis, label in ((x, "x"), (t, "t")):
        diffs = np.diff(axis)
        if not (diffs > 0).all() or not np.allclose(diffs, diffs[0], rtol=1e-5, atol=1e-10):
            raise ValueError(f"{label} 必须严格递增且均匀采样。")
    if np.max(np.abs(u)) > 1e6:
        raise ValueError("数值幅度过大，请先进行无量纲化。")
    if dataset.get("boundary", "periodic") != "periodic":
        raise ValueError("当前版本仅支持周期边界；空间网格不要重复右端点。")
    return {"x": x.tolist(), "t": t.tolist(), "u": u.tolist(), "boundary": "periodic"}


def _complete(result, *, scenario, steps, experiment, validation_present):
    result = copy.deepcopy(result)
    diagnostics = result.setdefault("diagnostics", {})
    error = diagnostics.get("validation_relative_error", result.get("validation_relative_error"))
    if error is not None:
        diagnostics["validation_relative_error"] = float(error)
    status = result.get("status", "insufficient")
    achieved = bool(validation_present and status == "identified" and error is not None
                    and np.isfinite(error) and error <= GOAL_ERROR)
    if achieved:
        reason = "独立轨迹预测达标；在当前候选库与已测试工况内，未检测到未解除的结构歧义。"
    elif status == "ambiguous":
        reason = "多个候选方程能够解释当前数据，需要有辨识力的新轨迹。"
    elif not validation_present and status in {"identified", "candidate"}:
        status = "waiting_for_data"
        reason = "已有候选方程，但缺少独立验证轨迹，不能宣布目标完成。"
    else:
        reason = "数据证据或独立预测精度未达到目标；请检查噪声、采样和候选库范围。"
        if status == "identified":
            status = "insufficient"
    result.update({
        "run_id": uuid.uuid4().hex[:12],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scenario": scenario,
        "status": status,
        "steps": steps,
        "experiment": experiment,
        "goal": {"description": GOAL_TEXT, "achieved": achieved, "reason": reason,
                 "validation_error_threshold": GOAL_ERROR},
        "controller": {"type": "deterministic_evidence_driven", "llm_connected": False,
                       "max_discovery_rounds": 3,
                       "scope": "一维周期边界、常系数、固定低阶候选库；非任意 PDE 发现器"},
    })
    result.setdefault("warnings", [])
    result.setdefault("prediction", None)
    result.setdefault("candidates", [])
    result.setdefault("terms", [])
    result.setdefault("equation", "尚无可靠方程")
    steps.append(_step("目标检查", reason, "complete" if achieved else "warning"))
    return result


def run_demo(scenario="diffusion", noise=0.0, seed=7, auto_experiment=True):
    if scenario not in SCENARIOS:
        raise ValueError("未知演示场景。")
    if not isinstance(seed, int) or not 0 <= seed <= 2**31 - 1:
        raise ValueError("seed 必须是 0–2147483647 的整数。")
    if not np.isfinite(noise) or not 0 <= noise <= 0.03:
        raise ValueError("noise 必须在 0–0.03 之间。")
    train = make_dataset(scenario, seed=seed, noise=noise, initial_condition="train")
    steps = [
        _step("明确目标与范围", GOAL_TEXT),
        _step("取得观测数据", f"独立隐藏方程仿真器提供 {len(train['t'])}×{len(train['x'])} 个观测值；识别器不读取真方程。"),
    ]
    first = discover([train])
    steps.append(_step("稀疏识别", "构造候选项，使用 L1 正则筛选，再对保留项重估系数。"))
    experiment = {"proposed": False, "executed": False, "description": "本次不需要追加训练实验。"}
    initial_evidence = {
        "status": first.get("status"), "equation": first.get("equation"),
        "candidates": copy.deepcopy(first.get("candidates", [])),
        "diagnostics": copy.deepcopy(first.get("diagnostics", {})),
    }
    training = [train]
    if first.get("status") == "ambiguous":
        steps.append(_step("检测结构歧义", "候选项共线或存在等价解释；当前数据不足以唯一选择方程。", "warning"))
        selected, scores = _choose_experiment(first, train)
        experiment = {
            "proposed": True, "executed": False,
            "description": f"建议追加 {selected['label']} 初始状态：在有限备选实验中比较候选模型预测差异，优先获取更有辨识力的数据。",
            "policy": "对 3 个预设初始状态进行候选模型分歧评分；数值策略控制器，非大模型决策。",
            "selected_initial_condition": selected["initial_condition"],
            "probe_scores": scores,
            "source": "独立隐藏真值仿真器；用户数据模式仅给建议，不自动制造证据。",
        }
        steps.append(_step("提出追加实验", experiment["description"]))
        if not auto_experiment:
            first["initial_evidence"] = initial_evidence
            return _complete(first, scenario=scenario, steps=steps, experiment=experiment, validation_present=False)
        probe = make_dataset(scenario, seed=seed + 101, noise=noise, initial_condition=selected["initial_condition"])
        training.append(probe)
        experiment["executed"] = True
        steps.append(_step("执行仿真追加实验", "从同一隐藏系统取得新初始条件的数据，并与第一条轨迹共同重新识别。"))
        second = discover(training)
        state_labels = {"candidate": "已形成候选方程，等待独立验证", "ambiguous": "仍有候选歧义",
                        "insufficient": "证据仍不足", "identified": "已通过验证"}
        steps.append(_step("重新识别", "追加证据后的结果：" + state_labels.get(second.get("status"), "需继续检查") + "。"))
    validation = make_dataset(scenario, seed=seed + 202, noise=noise, initial_condition="validation")
    result = discover(training, validation=validation)
    steps.append(_step("独立预测验证", "从独立保留的初始状态出发积分候选 PDE；最终验证轨迹不参与系数拟合或方程选择。"))
    result["initial_evidence"] = initial_evidence
    return _complete(result, scenario=scenario, steps=steps, experiment=experiment, validation_present=True)


def discover_uploaded(payload):
    if not isinstance(payload, dict):
        raise ValueError("上传内容必须是 JSON 对象。")
    if all(key in payload for key in ("x", "t", "u")):
        payload = {"training": [payload]}
    raw_training = payload.get("training")
    if not isinstance(raw_training, list) or not 1 <= len(raw_training) <= 3:
        raise ValueError("training 必须包含 1–3 条来自同一物理系统的轨迹。")
    training = [validate_dataset(item) for item in raw_training]
    validation = validate_dataset(payload["validation"]) if payload.get("validation") is not None else None
    fingerprints = [_fingerprint(d) for d in training]
    if len(set(fingerprints)) != len(fingerprints):
        raise ValueError("训练轨迹重复；重复数据不能提供额外的独立证据。")
    if validation is not None and _fingerprint(validation) in fingerprints:
        raise ValueError("验证轨迹与训练轨迹相同。请提供不同初始条件下的独立轨迹。")
    datasets = training + ([validation] if validation is not None else [])
    if sum(len(d['x']) * len(d['t']) for d in datasets) > 300_000:
        raise ValueError("数据总量超过原型限制（300000 个场值）。")
    for d in datasets[1:]:
        if len(d['x']) != len(training[0]['x']) or not np.allclose(d['x'], training[0]['x']):
            raise ValueError("所有轨迹必须使用相同的空间网格与坐标单位。")
    result = discover(training, validation=validation)
    steps = [
        _step("检查输入", f"收到 {len(training)} 条训练轨迹；周期边界和常系数是本次识别假设。"),
        _step("稀疏识别", "按轨迹构建候选矩阵，执行 L1 筛项与系数去偏重估。"),
        _step("验证证据", "独立验证轨迹不参与拟合。" if validation else "未提供独立验证轨迹。", "complete" if validation else "warning"),
    ]
    ambiguous = result.get("status") == "ambiguous"
    experiment = {
        "proposed": ambiguous, "executed": False,
        "description": "请追加不同空间模态或有区分力的初始条件实验，并保留另外一条轨迹作最终验证。" if ambiguous else "如尚未提供，请补充独立初始条件的验证轨迹。",
        "source": "真实实验或用户认可的独立仿真器；系统不会生成数据来证明自己的猜测。",
    }
    return _complete(result, scenario="uploaded", steps=steps, experiment=experiment, validation_present=validation is not None)
