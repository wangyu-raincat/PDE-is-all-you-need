"""Shared-library sparse discovery for scalar periodic one-dimensional PDEs.

No dataset identifier, reference equation, mandatory physical term, or test
field enters candidate generation. This clean-data protocol is separate from
the archived demonstration and measured-rod numerical engines.
"""
from __future__ import annotations

import warnings

import numpy as np
from scipy.integrate import cumulative_simpson
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Lasso

TERM_NAMES = ["1", "u", "u^2", "u^3", "u_x", "u_xx", "u_xxx", "u_xxxx", "u*u_x"]
ALPHA_PATH = (3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1)
SUBSTEPS = 32
MAX_STEPS = 20000
FIT_PROTOCOL = {
    "version": 1,
    "scope": "scalar autonomous constant-coefficient 1D PDE, uniform periodic spatial grid",
    "left_hand_side": "u_t with coefficient fixed to 1",
    "candidate_library": TERM_NAMES,
    "mandatory_right_hand_side_terms": [],
    "data_class": "clean external numerical reference benchmark; not real measurements",
    "spatial_projection": "retain integer Fourier modes |m| <= floor(Nx/3)",
    "polynomial_products": "2*Nx zero-padded multiplication then truncate and project",
    "nonlinear_transport": "u*u_x represented equivalently as one half d_x(u^2)",
    "time_smoothing": "none",
    "time_regression": "local integrated equations using original-time cumulative Simpson quadrature",
    "window_intervals": "max(3,min(12,Nt//10))",
    "window_starts": "4,Nt-window-4 exclusive, stride 2",
    "overlapping_windows_are_independent_experiments": False,
    "feature_normalization": "training column RMS and training target RMS",
    "alpha_path": list(ALPHA_PATH),
    "lasso": {"fit_intercept": False, "max_iter": 100000, "tol": 1e-10},
    "normalized_support_threshold": 1e-6,
    "debias": "least-squares on selected normalized support, rcond=1e-10",
    "normalized_debiased_zero_threshold": 1e-8,
    "eligibility": "all contributing LASSO fits converged, support full column rank, finite coefficients",
    "selection_in_this_module": False,
    "reference_equation_used": False,
    "forecast": "Cox-Matthews ETDRK4, diagonal Fourier linear operator, stable complex phi functions",
    "forecast_substeps_per_observation_interval": SUBSTEPS,
    "forecast_max_steps": MAX_STEPS,
    "forecast_initial_state": "same fixed 2/3 Fourier projection of supplied initial frame",
    "forecast_inputs": "coefficients, coordinates, initial frame only; no future observations",
    "forward_operator_guard": "highest nonzero even spatial derivative must be dissipative; negative second derivative allowed with negative fourth derivative",
    "forward_growth_guard": "finite states, maxabs <= max(100,1000*initial_maxabs); no truth-specific coefficient bounds",
}


def _coordinates(x, t):
    x, t = np.asarray(x, float), np.asarray(t, float)
    if (x.ndim != 1 or t.ndim != 1 or not 16 <= len(x) <= 2048
            or not 2 <= len(t) <= 2000 or not np.isfinite(x).all() or not np.isfinite(t).all()):
        raise ValueError("Need finite 1D coordinates, 16..2048 spatial and 2..2000 temporal points.")
    for values, name in ((x, "space"), (t, "time")):
        delta = np.diff(values)
        if np.any(delta <= 0) or not np.allclose(delta, delta[0], rtol=1e-5, atol=0.0):
            raise ValueError(f"The {name} grid must be strictly increasing and uniform.")
    return x, t


def _validate(data):
    if not isinstance(data, dict) or not all(key in data for key in ("x", "t", "u")):
        raise ValueError("Need x,t,u observations.")
    x, t = _coordinates(data["x"], data["t"])
    u = np.asarray(data["u"], float)
    if (len(t) < 16 or u.shape != (len(t), len(x)) or not np.isfinite(u).all()
            or u.size > 3_000_000 or np.max(np.abs(u)) > 1e6):
        raise ValueError("Training requires finite Nt-by-Nx fields, >=16 frames, at most 3 million values.")
    if data.get("boundary", "periodic") != "periodic":
        raise ValueError("Only periodic spatial grids without a duplicated endpoint are supported.")
    return x, t, u


def _geometry(x):
    n = len(x)
    modes = np.rint(np.fft.fftfreq(n) * n).astype(int)
    k = 2 * np.pi * np.fft.fftfreq(n, d=x[1]-x[0])
    return k, np.abs(modes) <= n//3, modes % (2*n)


def _products(spectrum, retained, padded_indices, *, square=True, cube=True):
    """Polynomial spectra evaluated on a 2N grid to prevent cubic aliasing."""
    n = spectrum.shape[-1]
    padded = np.zeros((*spectrum.shape[:-1], 2*n), dtype=complex)
    padded[..., padded_indices] = 2 * spectrum
    field = np.fft.ifft(padded, axis=-1).real
    result = []
    for needed, power in ((square, 2), (cube, 3)):
        if needed:
            values = np.fft.fft(field**power, axis=-1)[..., padded_indices] / 2
            result.append(values * retained)
        else:
            result.append(None)
    return result


def _features(x, raw):
    k, retained, indices = _geometry(x)
    spectrum = np.fft.fft(raw, axis=-1) * retained
    u = np.fft.ifft(spectrum, axis=-1).real
    square, cube = _products(spectrum, retained, indices)
    features = [np.ones_like(u), u, np.fft.ifft(square, axis=-1).real,
                np.fft.ifft(cube, axis=-1).real]
    features.extend(np.fft.ifft((1j*k)**order * spectrum, axis=-1).real for order in range(1, 5))
    features.append(np.fft.ifft(.5j*k * square, axis=-1).real)
    fraction = float(np.linalg.norm(raw-u) / max(np.linalg.norm(raw), 1e-15))
    return u, np.stack(features, axis=-1), fraction


def _condition(matrix):
    singular = np.linalg.svd(matrix, compute_uv=False)
    if not len(singular) or singular[0] <= 0:
        return None, 0
    rank = int(np.sum(singular > singular[0] * 1e-10))
    condition = float(singular[0] / max(singular[-1], 1e-30))
    return condition if condition < 1e16 else None, rank


def _equation(terms):
    parts = []
    for term in terms:
        value, name = term["coefficient"], term["name"]
        body = f"{abs(value):.10g}" + ("" if name == "1" else f" {name}")
        parts.append(("- " if value < 0 else "+ ") + body)
    return "u_t = " + (" ".join(parts).removeprefix("+ ") or "0")


def fit_candidates(data):
    """Return a fixed L1 path's debiased supports, without choosing a winner."""
    x, t, raw = _validate(data)
    u, library, projection = _features(x, raw)
    integral = cumulative_simpson(library, x=t, axis=0, initial=0)
    width = max(3, min(12, len(t)//10))
    starts = np.arange(4, len(t)-width-4, 2, dtype=int)
    ends = starts + width
    if len(starts) < 2:
        raise ValueError("Too few complete original-grid integration windows.")
    durations = (t[ends]-t[starts])[:, None]
    features = ((integral[ends]-integral[starts]) / durations[:, :, None]).reshape(-1, len(TERM_NAMES))
    target = ((u[ends]-u[starts]) / durations).ravel()
    if not np.isfinite(features).all() or not np.isfinite(target).all():
        raise ValueError("Candidate library or integrated targets overflowed.")
    scales = np.sqrt(np.mean(features**2, axis=0))
    target_scale = float(np.sqrt(np.mean(target**2)))
    if target_scale <= 1e-12:
        raise ValueError("No measurable temporal excitation; dynamics cannot be identified.")
    active = scales > max(1e-15, float(scales.max()) * 1e-12)
    safe_scales = np.where(active, scales, 1.0)
    X, y = features / safe_scales, target / target_scale
    full_condition, full_rank = _condition(X[:, active])
    norms = np.linalg.norm(X, axis=0)
    correlations = X.T @ X / np.maximum(np.outer(norms, norms), 1e-30)
    pairs = [{"terms": [TERM_NAMES[i], TERM_NAMES[j]], "correlation": float(correlations[i,j])}
             for i in range(len(TERM_NAMES)) for j in range(i+1, len(TERM_NAMES))
             if active[i] and active[j] and abs(correlations[i,j]) > .995]
    paths, runs = {}, {}
    for alpha in ALPHA_PATH:
        model = Lasso(alpha=alpha, fit_intercept=False, max_iter=100000, tol=1e-10)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            model.fit(X, y)
        runs[alpha] = {"alpha": alpha, "iterations": int(model.n_iter_), "dual_gap": float(model.dual_gap_),
                       "converged": not any(issubclass(item.category, ConvergenceWarning) for item in caught)}
        support = tuple(int(i) for i in np.flatnonzero((np.abs(model.coef_) > 1e-6) & active))
        if support:
            paths.setdefault(support, []).append(alpha)
    candidates = []
    for support, alphas in sorted(paths.items()):
        indices = np.asarray(support, int)
        normalized = np.linalg.lstsq(X[:, indices], y, rcond=1e-10)[0]
        keep = np.abs(normalized) > 1e-8
        indices, normalized = indices[keep], normalized[keep]
        if not len(indices):
            continue
        condition, rank = _condition(X[:, indices])
        coefficients = normalized * target_scale / safe_scales[indices]
        terms = [{"name": TERM_NAMES[i], "coefficient": float(c)} for i,c in zip(indices, coefficients)]
        residual = target-features[:, indices] @ coefficients
        converged = all(runs[alpha]["converged"] for alpha in alphas)
        success = bool(converged and rank == len(indices) and np.isfinite(coefficients).all())
        candidates.append({
            "candidate_id": "lasso-" + "-".join(str(i) for i in indices) + "-" + f"{alphas[0]:g}",
            "equation": _equation(terms), "terms": terms, "support_size": len(indices),
            "fit_success": success, "fit_relative_error": float(np.linalg.norm(residual)/np.linalg.norm(target)),
            "fit_rmse": float(np.sqrt(np.mean(residual**2))), "alpha_values": alphas,
            "failure_reason": None if success else "Unconverged LASSO path or rank-deficient selected support.",
            "diagnostics": {"support_rank": rank, "parameter_count": len(indices), "condition_number": condition,
                "full_library_rank": full_rank, "full_library_condition_number": full_condition,
                "lasso_all_support_runs_converged": converged, "lasso_runs": [runs[a] for a in alphas],
                "projection_energy_fraction": projection, "correlated_columns": pairs,
                "regression_rows": len(target), "window_intervals": width, "integration_windows": len(starts),
                "training_frames": len(t), "spatial_points": len(x), "overlapping_windows": True,
                "reference_equation_used": False, "time_smoothing": "none"},
        })
    return sorted(candidates, key=lambda item: (item["support_size"], item["fit_relative_error"], item["candidate_id"]))


def _phi(z, order):
    """Entire phi_j(z), stable at zero and on purely imaginary Fourier modes."""
    z = np.asarray(z, complex)
    result = np.empty_like(z)
    small = np.abs(z) < .5
    zs = z[small]
    factorial = 1
    for i in range(2, order+1):
        factorial *= i
    term = np.full(zs.shape, 1.0/factorial, complex)
    value = term.copy()
    for k in range(1, 24):
        term = term * zs / (order+k)
        value += term
    result[small] = value
    zl = z[~small]
    numerator = np.expm1(zl)
    if order >= 2:
        numerator -= zl
    if order >= 3:
        numerator -= .5*zl**2
    result[~small] = numerator / zl**order
    return result


def _etd_coefficients(linear, step):
    z = linear * step
    if not np.isfinite(z).all() or np.max(z.real) > 40:
        raise ValueError("Candidate linear growth exceeds the bounded forward solver.")
    p1, p2, p3 = (_phi(z, j) for j in (1, 2, 3))
    return (np.exp(z), np.exp(z/2), step*.5*_phi(z/2, 1),
            step*(p1-3*p2+4*p3), step*(p2-2*p3), step*(-p2+4*p3))


def forecast(terms, x, t, u0):
    """Roll out a discovered PDE using only a supplied initial frame."""
    x, t = _coordinates(x, t)
    u0 = np.asarray(u0, float)
    if u0.shape != x.shape or not np.isfinite(u0).all():
        raise ValueError("Initial field must be finite and match the spatial grid.")
    if (len(t)-1)*SUBSTEPS > MAX_STEPS:
        raise ValueError("Forecast exceeds the fixed ETDRK4 integration budget.")
    coefficients = {name: 0.0 for name in TERM_NAMES}
    for term in terms:
        if not isinstance(term, dict) or term.get("name") not in coefficients:
            raise ValueError("Unsupported PDE term.")
        value = float(term["coefficient"])
        if not np.isfinite(value):
            raise ValueError("PDE coefficients must be finite.")
        coefficients[term["name"]] += value
    fourth, second = coefficients["u_xxxx"], coefficients["u_xx"]
    if fourth > 0 or (fourth == 0 and second < 0):
        raise ValueError("Highest even derivative is antidissipative; unbounded continuum forward problem rejected.")
    k, retained, indices = _geometry(x)
    linear = np.full(len(x), coefficients["u"], complex)
    for order, name in enumerate(("u_x", "u_xx", "u_xxx", "u_xxxx"), start=1):
        linear += coefficients[name] * (1j*k)**order
    if not np.isfinite(linear).all():
        raise ValueError("Fourier operator overflowed.")
    v = np.fft.fft(u0) * retained
    prediction = np.empty((len(t), len(x)))
    prediction[0] = np.fft.ifft(v).real
    limit = max(100.0, 1000*float(np.max(np.abs(u0))))
    square_needed = coefficients["u^2"] != 0 or coefficients["u*u_x"] != 0
    cube_needed = coefficients["u^3"] != 0

    def nonlinear(spectrum):
        if not np.isfinite(spectrum).all() or np.max(np.abs(spectrum)) > len(x)*limit:
            raise ValueError("Nonlinear candidate exceeded the finite-state growth budget.")
        result = np.zeros(len(x), complex)
        result[0] = len(x) * coefficients["1"]
        if square_needed or cube_needed:
            square, cube = _products(spectrum*retained, retained, indices, square=square_needed, cube=cube_needed)
            if square_needed:
                result += (coefficients["u^2"] + .5j*k*coefficients["u*u_x"]) * square
            if cube_needed:
                result += coefficients["u^3"] * cube
        return result*retained

    cache = {}
    for row, delta in enumerate(np.diff(t), start=1):
        step = float(delta/SUBSTEPS)
        # Cache genuinely identical intervals without rounding or resampling
        # the original temporal coordinates.
        key = step
        if key not in cache:
            cache[key] = _etd_coefficients(linear, step)
        E, E2, Q, f1, f2, f3 = cache[key]
        for _ in range(SUBSTEPS):
            Nv = nonlinear(v)
            a = E2*v + Q*Nv
            Na = nonlinear(a)
            b = E2*v + Q*Na
            Nb = nonlinear(b)
            c = E2*a + Q*(2*Nb-Nv)
            Nc = nonlinear(c)
            v = (E*v + f1*Nv + 2*f2*(Na+Nb) + f3*Nc)*retained
        field = np.fft.ifft(v).real
        if not np.isfinite(field).all() or np.max(np.abs(field)) > limit:
            raise ValueError("Candidate forecast diverged or exceeded the finite-state growth budget.")
        prediction[row] = field
    return prediction
