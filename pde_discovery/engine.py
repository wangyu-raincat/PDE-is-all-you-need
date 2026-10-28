"""Sparse PDE identification from observed fields only.

Scope: a scalar, one-dimensional, constant-coefficient equation on a uniform
periodic grid.  A finite candidate library and fixed low-pass filter are
assumptions, not a universal PDE search.  No simulator or ground truth is used.
"""
from __future__ import annotations

import warnings as pywarnings
import numpy as np
from scipy.integrate import cumulative_simpson, solve_ivp
from scipy.signal import savgol_filter
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Lasso

TERM_NAMES = ["1", "u", "u²", "u³", "u_x", "u_xx", "u·u_x"]
MAX_VALIDATION_ERROR = 0.05


def _validate(data: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not isinstance(data, dict):
        raise ValueError("数据应为包含 x、t、u 的对象。")
    try:
        x, t, u = (np.asarray(data[k], dtype=float) for k in ("x", "t", "u"))
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("数据必须包含数值数组 x、t、u。") from exc
    if x.ndim != 1 or t.ndim != 1 or u.shape != (len(t), len(x)):
        raise ValueError("u 的形状必须为 [时间点数, 空间点数]。")
    if not 16 <= len(x) <= 256 or not 16 <= len(t) <= 1001:
        raise ValueError("原型支持 16–256 个空间点及 16–1001 个时间点。")
    if not all(np.isfinite(a).all() for a in (x, t, u)):
        raise ValueError("数据中不能包含 NaN 或无穷值。")
    for a, name in ((x, "空间"), (t, "时间")):
        da = np.diff(a)
        if np.any(da <= 0) or not np.allclose(da, da[0], rtol=1e-5, atol=1e-10):
            raise ValueError(f"{name}坐标必须严格递增并均匀采样。")
    if data.get("boundary", "periodic") != "periodic":
        raise ValueError("当前原型仅支持周期边界，x 不应重复包含周期终点。")
    if np.max(np.abs(u)) > 1e6:
        raise ValueError("观测幅度过大，请先无量纲化。")
    return x, t, u


def _spatial(u: np.ndarray, x: np.ndarray, cutoff: int = 6):
    k = 2 * np.pi * np.fft.fftfreq(len(x), d=x[1] - x[0])
    mode = np.fft.fftfreq(len(x)) * len(x)
    retained = np.abs(mode) <= min(cutoff, len(x) // 3)
    spectrum = np.fft.fft(u, axis=-1) * retained
    return (np.fft.ifft(spectrum, axis=-1).real,
            np.fft.ifft(1j * k * spectrum, axis=-1).real,
            np.fft.ifft(-k**2 * spectrum, axis=-1).real)


def _regression_data(data: dict):
    x, t, raw = _validate(data)
    spatial, _, _ = _spatial(raw, x)
    high_frequency = np.sqrt(np.mean((raw - spatial)**2)) / max(np.sqrt(np.mean(raw**2)), 1e-12)
    # Smoothing assumptions are surfaced in diagnostics.  The integral
    # formulation avoids dividing pointwise measurement noise by small dt.
    window = min(9, len(t) if len(t) % 2 else len(t) - 1)
    smoothed = savgol_filter(spatial, window, 3, axis=0, mode="interp")
    u, ux, uxx = _spatial(smoothed, x)
    theta = np.stack([np.ones_like(u), u, u**2, u**3, ux, uxx, u * ux], axis=-1)
    integrated = cumulative_simpson(theta, x=t, axis=0, initial=0)
    width = max(3, min(12, len(t) // 10))
    starts = np.arange(2, len(t) - width - 2, 2)
    ends = starts + width
    duration = (t[ends] - t[starts])[:, None]
    target = (u[ends] - u[starts]) / duration
    features = (integrated[ends] - integrated[starts]) / duration[:, :, None]
    return features.reshape(-1, len(TERM_NAMES)), target.ravel(), float(high_frequency)


def _equation(terms: list[dict]) -> str:
    if not terms:
        return "u_t = 0"
    parts = []
    for term in terms:
        c, name = term["coefficient"], term["name"]
        body = f"{abs(c):.6g}" + ("" if name == "1" else f" {name}")
        parts.append(("- " if c < 0 else "+ ") + body)
    rhs = " ".join(parts)
    if rhs.startswith("+ "):
        rhs = rhs[2:]
    return "u_t = " + rhs


def forecast(terms: list[dict], x: list, t: list, u0: list) -> np.ndarray:
    """Simulate a candidate, including every nonzero listed term.

    Linear equations use exact Fourier evolution.  Nonlinear candidates use
    dealiased spectral spatial derivatives and an adaptive ODE solver.
    Non-finite, unstable, and unsupported candidates fail explicitly.
    """
    x, t, u0 = np.asarray(x, float), np.asarray(t, float), np.asarray(u0, float)
    if (x.ndim != 1 or t.ndim != 1 or u0.shape != x.shape or len(x) < 4
            or len(t) < 2 or not all(np.isfinite(a).all() for a in (x, t, u0))
            or np.any(np.diff(x) <= 0) or np.any(np.diff(t) <= 0)
            or not np.allclose(np.diff(x), x[1] - x[0])):
        raise ValueError("预测坐标、初值或采样间隔无效。")
    coef = {name: 0.0 for name in TERM_NAMES}
    for item in terms:
        if item.get("name") not in coef:
            raise ValueError("候选方程含不支持的项。")
        value = float(item["coefficient"])
        if not np.isfinite(value):
            raise ValueError("候选方程系数必须有限。")
        coef[item["name"]] += value
    if coef["u_xx"] < -1e-8:
        raise ValueError("候选含负扩散系数，当前前向验证拒绝不适定的反扩散模型。")
    k = 2 * np.pi * np.fft.fftfreq(len(x), d=x[1] - x[0])
    elapsed = t - t[0]
    if not any(abs(coef[n]) > 1e-12 for n in ("u²", "u³", "u·u_x")):
        exponent = coef["u"] + 1j * k * coef["u_x"] - k**2 * coef["u_xx"]
        if np.max(exponent.real) * elapsed[-1] > 30:
            raise ValueError("候选模型在验证区间过度增长。")
        spectrum = np.exp(elapsed[:, None] * exponent) * np.fft.fft(u0)
        if abs(coef["1"]) > 0:
            reaction = coef["u"]
            forcing = (np.expm1(reaction * elapsed) / reaction
                       if abs(reaction) > 1e-10 else elapsed)
            spectrum[:, 0] += len(x) * coef["1"] * forcing
        result = np.fft.ifft(spectrum, axis=1).real
    else:
        evaluations = 0
        def rhs(_time, field):
            nonlocal evaluations
            evaluations += 1
            if evaluations > 12000:
                raise ValueError("非线性验证超出 12000 次求值预算；请缩短时间区间或重新缩放问题。")
            if not np.isfinite(field).all() or np.max(np.abs(field)) > 1e4:
                raise ValueError("非线性候选模型积分发散。")
            smooth, ux, uxx = _spatial(field, x, cutoff=len(x) // 3)
            return (coef["1"] + coef["u"] * smooth + coef["u²"] * smooth**2
                    + coef["u³"] * smooth**3 + coef["u_x"] * ux
                    + coef["u_xx"] * uxx + coef["u·u_x"] * smooth * ux)
        solution = solve_ivp(rhs, (float(t[0]), float(t[-1])), u0, t_eval=t,
                             method="RK45", rtol=2e-6, atol=1e-8, max_step=(t[-1]-t[0])/40)
        if not solution.success or solution.y.shape != (len(x), len(t)):
            raise ValueError("非线性候选模型数值验证未收敛。")
        result = solution.y.T
    if not np.isfinite(result).all() or np.max(np.abs(result)) > 1e4:
        raise ValueError("候选模型预测无效或发散。")
    return result


def discover(training: list[dict], validation: dict | None = None) -> dict:
    """Identify sparse candidate PDEs; no simulator truth enters this call."""
    if not isinstance(training, list) or not 1 <= len(training) <= 8:
        raise ValueError("训练数据应为包含 1–8 条完整轨迹的列表。")
    parsed = [_validate(d) for d in training]
    refx = parsed[0][0]
    for x, _, _ in parsed[1:]:
        if x.shape != refx.shape or not np.allclose(x, refx):
            raise ValueError("同一次发现中的轨迹应使用相同空间网格。")
    if validation is not None:
        vx, vt, vu = _validate(validation)
        if vx.shape != refx.shape or not np.allclose(vx, refx):
            raise ValueError("验证轨迹须与训练轨迹使用相同空间网格。")
    rows = [_regression_data(d) for d in training]
    theta, y = np.concatenate([r[0] for r in rows]), np.concatenate([r[1] for r in rows])
    noise = max(r[2] for r in rows)
    scale = np.sqrt(np.mean(theta**2, axis=0))
    target_scale = float(np.sqrt(np.mean(y**2)))
    diagnostics = {
        "training_trajectories": len(training), "regression_samples": len(y),
        "candidate_library": TERM_NAMES, "high_frequency_fraction": float(noise),
        "derivative_method": "空间低通谱导数（保留前6个模态）+ 时间平滑与局部积分回归",
        "boundary": "periodic", "ambiguous_pairs": [],
        "alias_correlation_threshold": 0.995,
        "alias_tolerance_rule": "保留等稀疏替换：训练误差增加不超过 max(基础容差, 2√(1−相关系数²))",
        "needs_validation": validation is None, "validation_relative_error": None,
        "validation_threshold": MAX_VALIDATION_ERROR,
        "validation_role": "最终留出评估；不参与拟合、候选排序或解除训练歧义",
    }
    notes = ["结果仅适用于当前有限候选库、周期边界和数据覆盖范围，不是唯一真实规律的证明。"]
    if noise > 0.01:
        notes.append("观测含较明显高频成分；低通平滑可能同时滤除真实高频动力学。")
    base = {"equation": "尚未识别", "terms": [], "status": "insufficient", "candidates": [],
            "diagnostics": diagnostics, "warnings": notes, "prediction": None,
            "validation_relative_error": None}
    if target_scale < 1e-7:
        notes.append("观测几乎没有时间变化，无法从静态轨迹确定动力学；请增加激励或新的初始条件。")
        return base
    active = scale > max(float(scale.max()) * 1e-10, 1e-12)
    scales = np.where(active, scale, 1.0)
    X, yn = theta / scales, y / target_scale
    norm = np.linalg.norm(X, axis=0)
    similarity = X.T @ X / np.maximum(np.outer(norm, norm), 1e-15)
    alias_pairs = []
    for i in range(len(TERM_NAMES)):
        for j in range(i + 1, len(TERM_NAMES)):
            if active[i] and active[j] and abs(similarity[i, j]) > 0.995:
                alias_pairs.append((i, j))
                diagnostics["ambiguous_pairs"].append({"terms": [TERM_NAMES[i], TERM_NAMES[j]],
                                                        "correlation": float(similarity[i, j])})
    singular = np.linalg.svd(X[:, active], compute_uv=False)
    condition = float(singular[0] / max(singular[-1], 1e-30))
    diagnostics["condition_number"] = condition if condition < 1e16 else None
    supports = set()
    for alpha in (0.00003, 0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1):
        model = Lasso(alpha=alpha, fit_intercept=False, max_iter=30000, tol=1e-9)
        with pywarnings.catch_warnings():
            pywarnings.simplefilter("ignore", ConvergenceWarning)
            model.fit(X, yn)
        support = tuple(np.flatnonzero((np.abs(model.coef_) > 1e-5) & active))
        if support:
            supports.add(support)
    # Explicitly retain observationally indistinguishable alternative supports.
    for support in list(supports):
        for i, j in alias_pairs:
            for old, new in ((i, j), (j, i)):
                if old in support and new not in support:
                    supports.add(tuple(sorted((set(support) - {old}) | {new})))
    candidates = []
    seen = set()
    for support in supports:
        indices = np.asarray(support, dtype=int)
        coeff = np.linalg.lstsq(X[:, indices], yn, rcond=1e-9)[0]
        # Refit on selected support removes L1 shrinkage. Drop numerical zero.
        keep = np.abs(coeff) > 1e-6
        indices, coeff = indices[keep], coeff[keep]
        if not len(indices) or tuple(indices) in seen:
            continue
        seen.add(tuple(indices))
        coefficients = coeff * target_scale / scales[indices]
        terms = [{"name": TERM_NAMES[int(i)], "coefficient": float(c)}
                 for i, c in zip(indices, coefficients)]
        error = float(np.linalg.norm(y - theta[:, indices] @ coefficients) / np.linalg.norm(y))
        candidates.append({"equation": _equation(terms), "terms": terms,
                           "fit_error": error, "validation_error": None,
                           "validation_relative_error": None, "validation_failure": None})
    if not candidates:
        notes.append("当前候选库与正则化范围内未得到可用方程。")
        return base
    min_error = min(c["fit_error"] for c in candidates)
    allowance = max(0.002, min_error * 0.05)
    close = [c for c in candidates if c["fit_error"] <= min_error + allowance]
    smallest = min(len(c["terms"]) for c in close)
    eligible = [c for c in close if len(c["terms"]) == smallest]
    # Near-collinear alternatives must not disappear merely because noise in
    # a second derivative makes their training residual slightly worse.  Keep
    # equally sparse substitutions within the observed collinearity tolerance
    # and let a genuinely different experiment distinguish their predictions.
    for original in list(eligible):
        names = {term["name"] for term in original["terms"]}
        for i, j in alias_pairs:
            for old, new in ((TERM_NAMES[i], TERM_NAMES[j]),
                             (TERM_NAMES[j], TERM_NAMES[i])):
                if old not in names or new in names:
                    continue
                alternate = (names - {old}) | {new}
                tolerance = max(allowance, 2 * np.sqrt(max(0, 1 - similarity[i, j]**2)))
                for candidate in candidates:
                    if ({term["name"] for term in candidate["terms"]} == alternate
                            and candidate["fit_error"] <= original["fit_error"] + tolerance
                            and candidate not in eligible):
                        eligible.append(candidate)
    eligible.sort(key=lambda c: (c["fit_error"], c["equation"]))
    # A small visible set is sufficient to expose competing sparse explanations.
    eligible = eligible[:8]
    predictions = {}
    if validation is not None:
        for index, candidate in enumerate(eligible):
            try:
                pred = forecast(candidate["terms"], vx, vt, vu[0])
                err = float(np.linalg.norm(pred[1:] - vu[1:]) / max(np.linalg.norm(vu[1:]), 1e-12))
                candidate["validation_error"] = err
                candidate["validation_relative_error"] = err
                predictions[candidate["equation"]] = pred
            except (ValueError, FloatingPointError, OverflowError) as exc:
                candidate["validation_failure"] = str(exc)
    best = eligible[0]
    fit_error = best["fit_error"]
    validation_error = best["validation_error"]
    competing = len(eligible) > 1
    if fit_error > 0.15:
        status = "insufficient"
        notes.append("候选方程未能充分解释观测变化；请检查采样、噪声或候选库是否缺项。")
    elif competing:
        status = "ambiguous"
        notes.append("多个稀疏方程仍可解释数据；应改变初始空间模态进行独立实验，不应直接宣布发现唯一规律。")
        if validation is not None:
            notes.append("留出验证不会用于挑选候选方程；若将区分性观测转入训练，须另备新的最终留出轨迹。")
    elif validation is None:
        status = "candidate"
        notes.append("尚无独立验证轨迹，当前结果只是候选方程。")
    elif validation_error is None or validation_error > MAX_VALIDATION_ERROR:
        status = "insufficient"
        notes.append("候选方程未通过独立轨迹预测验证。")
    else:
        status = "identified"
    diagnostics.update({"fit_relative_error": fit_error, "sparse_candidates_evaluated": len(candidates),
                        "validation_relative_error": validation_error,
                        "competing_candidates": len(eligible), "unresolved_ambiguity": competing})
    prediction = None
    if validation is not None and best["equation"] in predictions:
        predicted = predictions[best["equation"]]
        prediction = {"x": vx.tolist(), "t": vt.tolist(), "observed": vu.tolist(),
                      "predicted": predicted.tolist(), "absolute_error": np.abs(predicted-vu).tolist()}
    return {"equation": best["equation"], "terms": best["terms"], "status": status,
            "candidates": eligible, "diagnostics": diagnostics, "warnings": notes,
            "prediction": prediction, "validation_relative_error": validation_error}
