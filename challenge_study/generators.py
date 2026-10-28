"""Independent clean numerical references for the frozen challenge suite.

This module does not import the discovery or forecast implementation. It uses
the Kassam--Trefethen contour construction of ETDRK4 and a two-thirds spectral
Galerkin truncation. Generated fields are synthetic, not instrument readings.
The KS family is called nonlinearly unstable; no chaos diagnostic is asserted.
"""
from __future__ import annotations

import hashlib
from numbers import Integral

import numpy as np

FAMILIES = ("low_viscosity_burgers", "ks_unstable")
ROLE_COUNTS = {"calibration": 1, "training": 3, "development": 2, "final": 2}
ROLE_SEEDS = {"calibration": 44000, "training": 44100, "development": 44200, "final": 44300}
OBSERVED_NX = 128
CONTOUR_POINTS = 64
CALIBRATION_RELATIVE_LIMIT = 1e-5
CALIBRATION_MAX_ABSOLUTE_LIMIT = 1e-4
GENERATOR_PROTOCOL = {
    "version": 1,
    "kind": "independent clean numerical simulation; not measured data",
    "method": "Kassam-Trefethen contour ETDRK4, 64 fixed upper-half unit-circle roots",
    "spatial_discretization": "Fourier Galerkin with |integer mode| <= floor(N/3), quadratic two-thirds dealiasing",
    "observation_rule": "retain every reference_N/128 spatial point; no interpolated observations",
    "time_rule": "each observation interval has ceil(interval/max_dt) equal fixed substeps",
    "observed_nx": OBSERVED_NX,
    "observed_nt": 101,
    "role_counts": ROLE_COUNTS,
    "role_seed_bases": ROLE_SEEDS,
    "seed_rule": "role seed base plus zero-based trajectory index; refinement preserves seed",
    "initial_mean": 0.1,
    "initial_amplitude_factor": "independent uniform [0.8,1.2), followed by phases uniform [-pi,pi)",
    "families": {
        "low_viscosity_burgers": {"period": float(2*np.pi), "reference_n": 512,
            "max_dt": 0.002, "end_time": 1.5, "modes": [1,2,3,4],
            "base_amplitudes": [0.9,0.3,0.15,0.08]},
        "ks_unstable": {"period": float(32*np.pi), "reference_n": 256,
            "max_dt": 0.01, "end_time": 20.0, "modes": [1,2,4,8,12,16],
            "base_amplitudes": [0.5,0.3,0.2,0.15,0.12,0.08]},
    },
    "refinement": "double reference_N and halve max_dt, identical initial Fourier coefficients",
    "calibration_role_only": True,
    "calibration_relative_l2_limit": CALIBRATION_RELATIVE_LIMIT,
    "calibration_max_absolute_limit": CALIBRATION_MAX_ABSOLUTE_LIMIT,
    "failure_policy": "report source_quality_failure; do not change parameters or count failed references as scientific evidence",
    "certified_chaos": False,
    "units": "declared dimensionless benchmark coordinates",
}


def _contour_coefficients(symbol, step):
    """Contour averages avoid cancellation near a zero real linear symbol."""
    roots = np.exp(1j*np.pi*(np.arange(1, CONTOUR_POINTS+1)-0.5)/CONTOUR_POINTS)
    z = step*np.asarray(symbol, float)[:, None] + roots[None, :]
    ez = np.exp(z)
    q = step*np.real(np.mean((np.exp(z/2)-1)/z, axis=1))
    f1 = step*np.real(np.mean((-4-z+ez*(4-3*z+z*z))/(z*z*z), axis=1))
    f2 = step*np.real(np.mean((2+z+ez*(-2+z))/(z*z*z), axis=1))
    f3 = step*np.real(np.mean((-4-3*z-z*z+ez*(4-z))/(z*z*z), axis=1))
    return np.exp(step*symbol), np.exp(step*symbol/2), q, f1, f2, f3


def _integrate(initial, period, times, linear_coefficients, nonlinear_coefficient, max_dt):
    """Integrate a real periodic scalar field independently of the PDE engine.

    ``linear_coefficients`` maps even derivative orders 2/4 to coefficients;
    ``nonlinear_coefficient`` multiplies u*u_x. This private interface permits
    independent numerical tests without using a discovery candidate or label.
    """
    initial, times = np.asarray(initial), np.asarray(times)
    if np.iscomplexobj(initial) or np.iscomplexobj(times):
        raise ValueError("Reference integration needs real fields and times.")
    initial, times = np.asarray(initial, float), np.asarray(times, float)
    if (initial.ndim != 1 or len(initial) < 16 or len(initial) > 2048
            or times.ndim != 1 or len(times) < 2 or not np.isfinite(initial).all()
            or not np.isfinite(times).all() or np.any(np.diff(times) <= 0)
            or not np.isfinite(period) or period <= 0 or not np.isfinite(max_dt)
            or max_dt <= 0 or not np.isfinite(nonlinear_coefficient)):
        raise ValueError("Invalid reference grid, field, coefficients or time step.")
    if any(order not in (2,4) or not np.isfinite(value)
           for order,value in linear_coefficients.items()):
        raise ValueError("Only finite even-order linear coefficients 2 and 4 are supported.")
    n = len(initial)
    integer_modes = np.rint(np.fft.fftfreq(n)*n).astype(int)
    retained = np.abs(integer_modes) <= n//3
    wave = integer_modes*(2*np.pi/period)
    symbol = np.zeros(n)
    for order,coefficient in linear_coefficients.items():
        symbol += coefficient*np.real((1j*wave)**order)
    state = np.fft.fft(initial)*retained
    result = np.empty((len(times),n),float)
    result[0] = np.fft.ifft(state).real
    coefficients_cache = {}
    total_steps = 0

    def nonlinear(spectrum):
        field = np.fft.ifft(spectrum*retained).real
        return nonlinear_coefficient*0.5j*wave*np.fft.fft(field*field)*retained

    for output_index,interval in enumerate(np.diff(times), start=1):
        steps = int(np.ceil(interval/max_dt))
        total_steps += steps
        if total_steps > 50000:
            raise ValueError("Reference integration exceeds the declared 50000-step budget.")
        step = float(interval/steps)
        if step not in coefficients_cache:
            coefficients_cache[step] = _contour_coefficients(symbol,step)
        e,e2,q,f1,f2,f3 = coefficients_cache[step]
        with np.errstate(over="raise",invalid="raise"):
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
            raise FloatingPointError("Reference field failed finite-state limits.")
        result[output_index] = values
    return result


def _initial_parameters(family,role,index):
    if family not in FAMILIES or role not in ROLE_COUNTS:
        raise ValueError("Unknown reference family or data role.")
    if isinstance(index,(bool,np.bool_)) or not isinstance(index,Integral) or not 0 <= index < ROLE_COUNTS[role]:
        raise ValueError("Trajectory index is outside the fixed role allocation.")
    spec = GENERATOR_PROTOCOL["families"][family]
    seed = ROLE_SEEDS[role]+int(index)
    rng = np.random.default_rng(seed)
    amplitudes = np.asarray(spec["base_amplitudes"])*rng.uniform(0.8,1.2,len(spec["modes"]))
    phases = rng.uniform(-np.pi,np.pi,len(spec["modes"]))
    return spec,seed,amplitudes,phases


def generate(family,role,index=0,refined=False):
    """Generate one fixed numerical reference; final roles require caller freeze."""
    if type(refined) is not bool:
        raise ValueError("refined must be a bool, not a data-dependent resolution.")
    spec,seed,amplitudes,phases = _initial_parameters(family,role,index)
    n = spec["reference_n"]*(2 if refined else 1)
    if n % OBSERVED_NX:
        raise ValueError("Reference grid must contain every observation point exactly.")
    period = spec["period"]
    x = np.arange(n)*period/n
    t = np.linspace(0.0,spec["end_time"],101)
    phase = 2*np.pi*x[:,None]/period*np.asarray(spec["modes"])[None,:]+phases
    initial = 0.1+np.sum(amplitudes[None,:]*np.sin(phase),axis=1)
    linear = {2:0.03} if family == "low_viscosity_burgers" else {2:-1.0,4:-1.0}
    dt = spec["max_dt"]/(2 if refined else 1)
    field = _integrate(initial,period,t,linear,-1.0,dt)
    stride = n//OBSERVED_NX
    observed = field[:,::stride].copy()
    return {"x":x[::stride].copy(),"t":t,"u":observed,"boundary":"periodic",
        "metadata":{"kind":"independent numerical reference","is_measured_data":False,
            "family":family,"role":role,"index":int(index),"seed":seed,
            "reference_n":n,"max_dt":dt,"refined":refined,"observed_nx":OBSERVED_NX,
            "initial_observed_sha256":hashlib.sha256(observed[0].tobytes()).hexdigest(),
            "certified_chaos":False,"independent_of_discovery_forecast":True}}


def reference_terms(family):
    """Scoring-only truth; generate() never consults this function."""
    if family == "low_viscosity_burgers":
        return [{"name":"u_xx","coefficient":0.03},{"name":"u*u_x","coefficient":-1.0}]
    if family == "ks_unstable":
        return [{"name":"u_xx","coefficient":-1.0},{"name":"u_xxxx","coefficient":-1.0},
                {"name":"u*u_x","coefficient":-1.0}]
    raise ValueError("Unknown reference family.")


def calibration():
    """Run only reserved calibration ICs; failure does not change the protocol."""
    records=[]
    for family in FAMILIES:
        try:
            coarse=generate(family,"calibration",0,refined=False)
            fine=generate(family,"calibration",0,refined=True)
            np.testing.assert_allclose(coarse["x"],fine["x"],rtol=0,atol=0)
            np.testing.assert_allclose(coarse["t"],fine["t"],rtol=0,atol=0)
            difference=coarse["u"][1:]-fine["u"][1:]
            relative=float(np.linalg.norm(difference)/np.linalg.norm(fine["u"][1:]))
            maximum=float(np.max(np.abs(difference)))
            passed=relative <= CALIBRATION_RELATIVE_LIMIT and maximum <= CALIBRATION_MAX_ABSOLUTE_LIMIT
            records.append({"family":family,"role":"calibration","relative_l2":relative,
                "max_absolute_error":maximum,"relative_limit":CALIBRATION_RELATIVE_LIMIT,
                "max_absolute_limit":CALIBRATION_MAX_ABSOLUTE_LIMIT,
                "status":"passed" if passed else "source_quality_failure",
                "coarse_metadata":coarse["metadata"],"refined_metadata":fine["metadata"]})
        except (ValueError,RuntimeError,FloatingPointError,AssertionError) as exc:
            records.append({"family":family,"role":"calibration","status":"source_quality_failure",
                "failure_reason":str(exc)})
    return {"records":records,"all_passed":all(r["status"]=="passed" for r in records),
        "formal_training_development_final_generated":False,
        "thresholds_changed_after_results":False,"protocol":GENERATOR_PROTOCOL}
