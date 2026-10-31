"""Independent references with new initial conditions for the coefficient reliability study.

The reference integrator never imports discovery or forecasting code.  Contour
ETDRK4 advances a Fourier Galerkin system; quadratic transport uses two-thirds
truncation and cubic reaction is evaluated on a doubled spatial grid.  These
are the same three known numerical benchmark families as the prior round,
with a new predeclared seed allocation; not new physical systems, measurements
or certified chaotic trajectories.
"""
from __future__ import annotations

import hashlib
from numbers import Integral
from pathlib import Path

import numpy as np

FAMILIES = ("allen_cahn", "low_viscosity_burgers", "ks_unstable")
ROLE_COUNTS = {"calibration": 1, "training": 3, "development": 2, "final": 3}
ROLE_SEEDS = {"calibration": 168000, "training": 168100, "development": 168200, "final": 168300}
FAMILY_IDS = {family: i + 1 for i, family in enumerate(FAMILIES)}
CONTOUR_POINTS = 64
CALIBRATION_RELATIVE_LIMIT = 1e-5
CALIBRATION_MAX_ABSOLUTE_LIMIT = 1e-4
GENERATOR_PROTOCOL = {
    "version": 1,
    "kind": "new initial conditions in the same known numerical families; not measured data",
    "method": "contour ETDRK4; 64 fixed upper-half unit-circle roots",
    "spatial_discretization": "Fourier Galerkin, |mode| <= floor(N/3); quadratic 2/3 truncation; cubic product on 2N grid then project",
    "observation_rule": "direct reference-grid subsampling; no interpolated observations",
    "time_rule": "ceil(each observation interval/max_dt) equal substeps, exact output times",
    "observed_nt": 111,
    "prefix_intervals": 10,
    "forecast_intervals": 100,
    "prefix_duration": "0.1 times the unchanged forecast horizon; observations precede forecast start",
    "role_counts": ROLE_COUNTS,
    "role_seed_bases": ROLE_SEEDS,
    "family_ids": FAMILY_IDS,
    "seed_rule": "SeedSequence([role_seed_base + zero_based_index, family_id]); refinement preserves seed",
    "initial_mean": 0.1,
    "initial_amplitude_factor": "independent uniform [0.8,1.2), followed by phases uniform [-pi,pi)",
    "families": {
        "allen_cahn": {"period": 2.0, "x_left": -1.0, "reference_n": 1024,
            "observed_nx": 256, "max_dt": 0.001, "end_time": 0.5,
            "modes": [1, 2, 4, 8, 16, 24], "base_amplitudes": [0.25, 0.15, 0.1, 0.05, 0.03, 0.02]},
        "low_viscosity_burgers": {"period": float(2*np.pi), "x_left": 0.0, "reference_n": 512,
            "observed_nx": 128, "max_dt": 0.002, "end_time": 1.5,
            "modes": [1, 2, 3, 4], "base_amplitudes": [0.9, 0.3, 0.15, 0.08]},
        "ks_unstable": {"period": float(32*np.pi), "x_left": 0.0, "reference_n": 256,
            "observed_nx": 128, "max_dt": 0.01, "end_time": 20.0,
            "modes": [1, 2, 4, 8, 12, 16], "base_amplitudes": [0.5, 0.3, 0.2, 0.15, 0.12, 0.08]},
    },
    "refinement": "double reference_N and halve max_dt, same initial Fourier coefficients",
    "calibration_role_only": True,
    "calibration_relative_l2_limit": CALIBRATION_RELATIVE_LIMIT,
    "calibration_max_absolute_limit": CALIBRATION_MAX_ABSOLUTE_LIMIT,
    "failure_policy": "retain source_quality_failure; never replace failed trajectories or relax thresholds",
    "formal_final_generation": "caller must verify global model freeze before requesting final roles",
    "certified_chaos": False,
    "units": "declared dimensionless benchmark coordinates",
}


def _contour_coefficients(symbol, step):
    roots = np.exp(1j*np.pi*(np.arange(1, CONTOUR_POINTS+1)-0.5)/CONTOUR_POINTS)
    z = step*np.asarray(symbol, float)[:, None] + roots[None, :]
    ez = np.exp(z)
    q = step*np.real(np.mean((np.exp(z/2)-1)/z, axis=1))
    f1 = step*np.real(np.mean((-4-z+ez*(4-3*z+z*z))/(z*z*z), axis=1))
    f2 = step*np.real(np.mean((2+z+ez*(-2+z))/(z*z*z), axis=1))
    f3 = step*np.real(np.mean((-4-3*z-z*z+ez*(4-z))/(z*z*z), axis=1))
    return np.exp(step*symbol), np.exp(step*symbol/2), q, f1, f2, f3


def _integrate(initial, period, times, linear_coefficients, transport, cubic, max_dt):
    """Independent Galerkin reference; linear orders 0/2/4, u*u_x and u^3.

    Physical coefficients are fixed by generate(), never read from a fitted
    model.  The general private interface supports independent solver tests.
    """
    initial, times = np.asarray(initial), np.asarray(times)
    if np.iscomplexobj(initial) or np.iscomplexobj(times):
        raise ValueError("Reference integration requires real fields and times.")
    initial, times = np.asarray(initial, float), np.asarray(times, float)
    # Power-of-two sizes avoid the ambiguous exactly-N/3 quadratic boundary.
    # Every declared source and refinement uses this reference-grid class.
    if (initial.ndim != 1 or len(initial) < 16 or len(initial) > 4096
            or len(initial) & (len(initial)-1)
            or times.ndim != 1 or not 2 <= len(times) <= 2000
            or not np.isfinite(initial).all() or not np.isfinite(times).all()
            or np.any(np.diff(times) <= 0) or not np.isfinite(period) or period <= 0
            or not np.isfinite(max_dt) or max_dt <= 0 or not np.isfinite(transport)
            or not np.isfinite(cubic)):
        raise ValueError("Invalid reference grid, field, coefficients or step.")
    if any(order not in (0, 2, 4) or not np.isfinite(value)
           for order, value in linear_coefficients.items()):
        raise ValueError("Only finite linear orders 0, 2 and 4 are supported.")
    n = len(initial)
    modes = np.rint(np.fft.fftfreq(n)*n).astype(int)
    retained = np.abs(modes) <= n//3
    wave = modes*(2*np.pi/period)
    padded_indices = modes % (2*n)
    symbol = np.zeros(n)
    for order, coefficient in linear_coefficients.items():
        symbol += coefficient*np.real((1j*wave)**order)
    state = np.fft.fft(initial)*retained
    result = np.empty((len(times), n), float)
    result[0] = np.fft.ifft(state).real
    cache = {}
    total_steps = 0

    def nonlinear(spectrum):
        spectrum = spectrum*retained
        out = np.zeros(n, complex)
        if transport:
            field = np.fft.ifft(spectrum).real
            out += transport*0.5j*wave*np.fft.fft(field*field)*retained
        if cubic:
            padded = np.zeros(2*n, complex)
            padded[padded_indices] = 2*spectrum
            fine_field = np.fft.ifft(padded).real
            out += cubic*np.fft.fft(fine_field**3)[padded_indices]/2*retained
        return out

    for output_index, interval in enumerate(np.diff(times), start=1):
        steps = int(np.ceil(interval/max_dt))
        total_steps += steps
        if total_steps > 50000:
            raise ValueError("Reference integration exceeds 50000 substeps.")
        step = float(interval/steps)
        if step not in cache:
            cache[step] = _contour_coefficients(symbol, step)
        e, e2, q, f1, f2, f3 = cache[step]
        with np.errstate(over="raise", invalid="raise"):
            for _ in range(steps):
                nv = nonlinear(state)
                a = e2*state + q*nv
                na = nonlinear(a)
                b = e2*state + q*na
                nb = nonlinear(b)
                c = e2*a + q*(2*nb-nv)
                nc = nonlinear(c)
                state = (e*state + f1*nv + 2*f2*(na+nb) + f3*nc)*retained
        values = np.fft.ifft(state).real
        if not np.isfinite(values).all() or np.max(np.abs(values)) > 1e6:
            raise FloatingPointError("Reference exceeded finite-state limits.")
        result[output_index] = values
    return result


def _initial_parameters(family, role, index):
    if family not in FAMILIES or role not in ROLE_COUNTS:
        raise ValueError("Unknown reference family or role.")
    if (isinstance(index, (bool, np.bool_)) or not isinstance(index, Integral)
            or not 0 <= index < ROLE_COUNTS[role]):
        raise ValueError("Trajectory index is outside the declared allocation.")
    spec = GENERATOR_PROTOCOL["families"][family]
    seed = [ROLE_SEEDS[role]+int(index), FAMILY_IDS[family]]
    rng = np.random.default_rng(np.random.SeedSequence(seed))
    amplitudes = np.asarray(spec["base_amplitudes"])*rng.uniform(0.8, 1.2, len(spec["modes"]))
    phases = rng.uniform(-np.pi, np.pi, len(spec["modes"]))
    return spec, seed, amplitudes, phases


def generate(family, role, index=0, refined=False):
    """Generate one planned source; callers enforce final-role freeze access."""
    if type(refined) is not bool:
        raise ValueError("refined must be a bool.")
    spec, seed, amplitudes, phases = _initial_parameters(family, role, index)
    n = spec["reference_n"]*(2 if refined else 1)
    observed_n = spec["observed_nx"]
    if n % observed_n:
        raise ValueError("Reference grid must contain all observed points.")
    period = spec["period"]
    x = spec["x_left"] + np.arange(n)*period/n
    t = np.linspace(0.0, 1.1*spec["end_time"], 111)
    phase = 2*np.pi*(x-spec["x_left"])[:, None]/period*np.asarray(spec["modes"])[None, :]+phases
    initial = 0.1+np.sum(amplitudes[None, :]*np.sin(phase), axis=1)
    if family == "allen_cahn":
        linear, transport, cubic = {0: 5.0, 2: 0.0001}, 0.0, -5.0
    elif family == "low_viscosity_burgers":
        linear, transport, cubic = {2: 0.03}, -1.0, 0.0
    else:
        linear, transport, cubic = {2: -1.0, 4: -1.0}, -1.0, 0.0
    dt = spec["max_dt"]/(2 if refined else 1)
    field = _integrate(initial, period, t, linear, transport, cubic, dt)
    stride = n//observed_n
    observed = field[:, ::stride].copy()
    return {"x": x[::stride].copy(), "t": t, "u": observed, "boundary": "periodic",
        "metadata": {"kind": "fresh independent numerical reference", "is_measured_data": False,
            "family": family, "role": role, "index": int(index), "seed_sequence": seed,
            "reference_n": n, "max_dt": dt, "refined": refined, "observed_nx": observed_n,
            "initial_observed_sha256": hashlib.sha256(observed[0].tobytes()).hexdigest(),
            "certified_chaos": False, "independent_of_discovery_forecast": True}}


def reference_terms(family):
    """Scoring-only reference; never used by fitting or model selection."""
    if family == "allen_cahn":
        return [{"name": "u_xx", "coefficient": 0.0001}, {"name": "u", "coefficient": 5.0},
                {"name": "u^3", "coefficient": -5.0}]
    if family == "low_viscosity_burgers":
        return [{"name": "u_xx", "coefficient": 0.03}, {"name": "u*u_x", "coefficient": -1.0}]
    if family == "ks_unstable":
        return [{"name": "u_xx", "coefficient": -1.0}, {"name": "u_xxxx", "coefficient": -1.0},
                {"name": "u*u_x", "coefficient": -1.0}]
    raise ValueError("Unknown reference family.")


def calibration():
    """Check only reserved calibration ICs, retaining every failed family."""
    records = []
    for family in FAMILIES:
        try:
            base = generate(family, "calibration", 0, refined=False)
            fine = generate(family, "calibration", 0, refined=True)
            np.testing.assert_array_equal(base["x"], fine["x"])
            np.testing.assert_array_equal(base["t"], fine["t"])
            difference = base["u"][1:]-fine["u"][1:]
            relative = float(np.linalg.norm(difference)/np.linalg.norm(fine["u"][1:]))
            maximum = float(np.max(np.abs(difference)))
            passed = relative <= CALIBRATION_RELATIVE_LIMIT and maximum <= CALIBRATION_MAX_ABSOLUTE_LIMIT
            records.append({"family": family, "role": "calibration", "relative_l2": relative,
                "max_absolute_error": maximum, "relative_limit": CALIBRATION_RELATIVE_LIMIT,
                "max_absolute_limit": CALIBRATION_MAX_ABSOLUTE_LIMIT,
                "status": "passed" if passed else "source_quality_failure",
                "coarse_metadata": base["metadata"], "refined_metadata": fine["metadata"]})
        except (ValueError, RuntimeError, FloatingPointError, AssertionError) as exc:
            records.append({"family": family, "role": "calibration", "status": "source_quality_failure",
                "failure_reason": str(exc)})
    return {"records": records, "all_passed": all(r["status"] == "passed" for r in records),
        "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "formal_training_development_final_generated": False,
        "thresholds_changed_after_results": False, "protocol": GENERATOR_PROTOCOL}
