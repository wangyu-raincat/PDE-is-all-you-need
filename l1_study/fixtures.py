"""Independent, deterministic analytic trajectories for the isolated L1 study.

These are generated numerical benchmarks, never real measurements.  Exact
derivatives are privileged analytic information: they provide an upper-bound
control and are not evidence that derivatives are available from observations.
No discovery or forecast implementation is used to generate the fields.

The scoring reference is exposed separately.  In particular, the Cole--Hopf
time derivative differentiates its analytic quotient, not an evaluated PDE
right-hand side.  Roles select distinct initial conditions at fixed physical
parameters; they are not adjacent slices of a shared trajectory.
"""
from __future__ import annotations

import hashlib
from math import comb

import numpy as np

FAMILIES = ("advdiff", "linear_high_order", "burgers_cole_hopf")
TERM_NAMES = ("1", "u", "u^2", "u^3", "u_x", "u_xx", "u_xxx", "u_xxxx", "u*u_x")
ROLE_COUNTS = {"training": 3, "development": 2, "final": 2}
ROLE_SEEDS = {"training": 1101, "development": 2201, "final": 3301}
NU = 0.12
FIXTURE_PROTOCOL = {
    "families": list(FAMILIES),
    "kind": "deterministic analytic synthetic trajectories; not measurements",
    "spatial_grid": {"points": 64, "start": 0.0, "period": float(2*np.pi), "duplicated_endpoint": False},
    "temporal_grid": {"points": 81, "start": 0.0, "end": 0.8},
    "role_counts": ROLE_COUNTS,
    "role_seed_base": ROLE_SEEDS,
    "seed_rule": "role_seed_base + zero-based trajectory index",
    "initial_modes": [1, 2, 3, 4],
    "linear_base_amplitudes": [0.65, 0.3, 0.17, 0.09],
    "cole_hopf_q_base_amplitudes": [0.32, 0.15, 0.07, 0.03],
    "amplitude_factor": "independent uniform [0.8,1.2) from declared generator seed",
    "phase": "independent uniform [-pi,pi) after amplitude draws",
    "linear_mean": 0.15,
    "cole_hopf_q_mean": 1.0,
    "noise": "none",
    "all_initial_conditions_contain_all_four_modes": True,
    "same_parameters_across_roles_and_initial_conditions": True,
    "exact_derivatives": "analytic Fourier differentiation or Cole-Hopf quotient differentiation",
    "exact_time_derivative_uses_discovered_pde": False,
    "oracle_control": True,
    "independent_final_initial_conditions": True,
}


def _initial_parameters(family, role, index):
    if family not in FAMILIES:
        raise ValueError(f"Unknown analytic family: {family!r}")
    if role not in ROLE_COUNTS:
        raise ValueError(f"Unknown trajectory role: {role!r}")
    if (isinstance(index, (bool, np.bool_)) or not isinstance(index, (int, np.integer))
            or not 0 <= int(index) < ROLE_COUNTS[role]):
        raise ValueError("Trajectory index lies outside the predeclared role allocation.")
    seed = ROLE_SEEDS[role] + int(index)
    rng = np.random.default_rng(seed)
    base = (np.array([0.32, 0.15, 0.07, 0.03]) if family == "burgers_cole_hopf"
            else np.array([0.65, 0.3, 0.17, 0.09]))
    amplitudes = base * rng.uniform(0.8, 1.2, 4)
    phases = rng.uniform(-np.pi, np.pi, 4)
    return amplitudes, phases, seed


def _linear_fields(family, x, t, amplitudes, phases):
    """Closed-form Fourier evolution, differentiated before taking real parts."""
    modes = np.arange(1, 5, dtype=float)
    ik = 1j*modes
    if family == "advdiff":
        eigenvalues = -0.65*ik + 0.12*ik**2
    elif family == "linear_high_order":
        eigenvalues = -0.4*ik - 0.04*ik**2 - 0.07*ik**3 - 0.004*ik**4
    else:
        raise ValueError("A linear family is required.")
    spectrum = (amplitudes[None, None, :]
                * np.exp(1j*(x[None, :, None]*modes + phases)
                         + t[:, None, None]*eigenvalues))
    fields = [np.real(np.sum(spectrum, axis=-1)) + 0.15]
    fields.extend(np.real(np.sum(spectrum*ik**order, axis=-1)) for order in range(1, 5))
    ut = np.real(np.sum(spectrum*eigenvalues, axis=-1))
    return fields, ut


def _cole_hopf_fields(x, t, amplitudes, phases):
    """Evaluate u=-2 nu q_x/q and its analytic derivatives without a PDE RHS."""
    modes = np.arange(1, 5, dtype=float)
    arguments = x[None, :, None]*modes + phases
    decay = np.exp(-NU*t[:, None, None]*modes**2)
    q_derivatives = []
    for order in range(6):
        q_derivatives.append(np.sum(amplitudes*decay*modes**order
                                    * np.cos(arguments + order*np.pi/2), axis=-1))
    q_derivatives[0] += 1.0
    q = q_derivatives[0]
    if np.min(q) <= 0:
        raise ValueError("The predeclared Cole-Hopf potential must remain positive.")

    # Differentiate r*q=q_x.  This quotient recurrence does not use the PDE.
    quotient_derivatives = []
    for order in range(5):
        numerator = q_derivatives[order+1].copy()
        for lower_order in range(order):
            numerator -= (comb(order, lower_order) * quotient_derivatives[lower_order]
                          * q_derivatives[order-lower_order])
        quotient_derivatives.append(numerator/q)
    fields = [-2*NU*value for value in quotient_derivatives]

    # Directly differentiate the exponential factors in q and q_x with time.
    qt = np.sum(amplitudes*decay*(-NU*modes**2)*np.cos(arguments), axis=-1)
    qxt = np.sum(amplitudes*decay*(-NU*modes**2)*modes
                 * np.cos(arguments + np.pi/2), axis=-1)
    ut = -2*NU*(qxt*q - q_derivatives[1]*qt)/q**2
    return fields, ut


def generate(family, role, index=0):
    """Return a fresh trajectory with distinct role/index initial conditions.

    Callers must generate/read final roles only after every model is frozen.
    Exact features and u_t may only enter explicitly named oracle-control arms.
    """
    amplitudes, phases, seed = _initial_parameters(family, role, index)
    x = np.linspace(0.0, 2*np.pi, 64, endpoint=False)
    t = np.linspace(0.0, 0.8, 81)
    if family == "burgers_cole_hopf":
        fields, ut = _cole_hopf_fields(x, t, amplitudes, phases)
    else:
        fields, ut = _linear_fields(family, x, t, amplitudes, phases)
    u, ux, uxx, uxxx, uxxxx = fields
    exact_features = np.stack((np.ones_like(u), u, u**2, u**3,
                               ux, uxx, uxxx, uxxxx, u*ux), axis=-1)
    return {
        "x": x, "t": t, "u": u, "boundary": "periodic",
        "exact_features": exact_features, "exact_ut": ut,
        "metadata": {"kind": "analytic synthetic", "is_measured_data": False,
                     "family": family, "role": role, "index": int(index), "seed": seed,
                     "initial_field_sha256": hashlib.sha256(u[0].tobytes()).hexdigest(),
                     "exact_derivatives_are_privileged": True,
                     "time_derivative_from_analytic_solution": True},
    }


def reference_terms(family):
    """Scoring-only reference coefficients; never consulted by generate()."""
    references = {
        "advdiff": {"u_x": -0.65, "u_xx": 0.12},
        "linear_high_order": {"u_x": -0.4, "u_xx": -0.04, "u_xxx": -0.07, "u_xxxx": -0.004},
        "burgers_cole_hopf": {"u_xx": NU, "u*u_x": -1.0},
    }
    if family not in references:
        raise ValueError(f"Unknown analytic family: {family!r}")
    return [{"name": name, "coefficient": coefficient}
            for name, coefficient in references[family].items()]
