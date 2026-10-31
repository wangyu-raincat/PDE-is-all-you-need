"""Equivalent closed-form acceleration of the frozen prefix forecast.

Only constant-coefficient linear candidates use this path.  The spatial
interpolation, projection, prefix filter, and observation times are unchanged.
Nonlinear candidates use the existing prefix/ETDRK4 implementation verbatim.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import resample

from five_study.initial import denoise_initial
from pde_discovery import generic_engine as baseline
from prefix_study import forecast as original_prefix
from resolved_study import forecast as resolved


NONLINEAR_TERMS = ('u^2', 'u^3', 'u*u_x')
PROTOCOL = {
    'version': 1,
    'name': 'equivalent_linear_prefix_acceleration',
    'baseline': original_prefix.PROTOCOL,
    'change': 'mathematically equivalent computation optimization, not new physics or model selection',
    'eligibility': 'accumulated coefficients of u^2, u^3 and u*u_x are exactly zero',
    'linear_solver': 'exact Fourier exponential and stable constant-DC forcing integral',
    'spatial_factor': resolved.DEFAULT_FACTOR,
    'interpolation_projection': 'unchanged scipy.signal.resample then original internal-grid Fourier projection',
    'prefix': 'same per-frame filter, propagation to cutoff, and equal mean of restricted terminal states',
    'nonlinear_solver': 'unchanged prefix_study.forecast and resolved factor-4 ETDRK4',
    'guards': 'same coordinates, term accumulation, dissipativity, integration budget, linear-step and finite-state growth limits',
    'linear_stage_guard': 'each non-DC modal magnitude and absolute real DC state attain their maximum at a time endpoint',
    'agreement': 'mathematical equivalence; ordinary floating-point differences from repeated ETDRK4 products remain',
    'future_observations_input': False,
    'additional_measurements': False,
}


def _coefficients(terms):
    """Use precisely the baseline term convention, including repeated terms."""
    coefficients = {name: 0.0 for name in baseline.TERM_NAMES}
    for term in terms:
        if not isinstance(term, dict) or term.get('name') not in coefficients:
            raise ValueError('Unsupported PDE term.')
        value = float(term['coefficient'])
        if not np.isfinite(value):
            raise ValueError('PDE coefficients must be finite.')
        coefficients[term['name']] += value
    return coefficients


def _is_linear(coefficients):
    return all(coefficients[name] == 0 for name in NONLINEAR_TERMS)


def _forcing_integral(rate, elapsed):
    """Integral of exp(rate*s), including zero and subnormal reaction rates."""
    z = rate * elapsed
    small = np.abs(z) < 1e-6
    result = np.empty_like(elapsed)
    zs = z[small]
    result[small] = elapsed[small] * (1 + zs/2 + zs*zs/6 + zs**3/24 + zs**4/120)
    if np.any(~small):
        result[~small] = np.expm1(z[~small]) / rate
    return result


def _linear_forecast(coefficients, x, t, u0):
    if any(np.iscomplexobj(np.asarray(value)) for value in (x, t, u0)):
        raise ValueError('Coordinates and initial observations must be real-valued.')
    x, t = baseline._coordinates(x, t)
    u0 = np.asarray(u0, float)
    if u0.shape != x.shape or not np.isfinite(u0).all():
        raise ValueError('Initial field must be finite and match the observed grid.')
    internal_n = resolved.DEFAULT_FACTOR * len(x)
    if internal_n > resolved.MAX_INTERNAL_POINTS:
        raise ValueError('Declared spatial refinement exceeds the 2048-point computational budget.')
    if (len(t)-1)*baseline.SUBSTEPS > baseline.MAX_STEPS:
        raise ValueError('Forecast exceeds the fixed ETDRK4 integration budget.')
    fourth, second = coefficients['u_xxxx'], coefficients['u_xx']
    if fourth > 0 or (fourth == 0 and second < 0):
        raise ValueError('Highest even derivative is antidissipative; unbounded continuum forward problem rejected.')

    period = len(x)*(x[1]-x[0])
    internal_x = x[0] + np.arange(internal_n)*period/internal_n
    internal_initial = resample(u0, internal_n)
    k, retained, _ = baseline._geometry(internal_x)
    linear = np.full(internal_n, coefficients['u'], complex)
    with np.errstate(over='ignore', invalid='ignore'):
        for order, name in enumerate(('u_x', 'u_xx', 'u_xxx', 'u_xxxx'), start=1):
            linear += coefficients[name]*(1j*k)**order
    if not np.isfinite(linear).all():
        raise ValueError('Fourier operator overflowed.')
    # Preserve the original per-step validity checks, including pathological
    # phi-function overflow, although no time stepping is needed for linearity.
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        for step in set(float(delta/baseline.SUBSTEPS) for delta in np.diff(t)):
            factors = baseline._etd_coefficients(linear, step)
            if any(not np.isfinite(factor).all() for factor in factors):
                raise ValueError('Nonlinear candidate exceeded the finite-state growth budget.')

    v0 = np.fft.fft(internal_initial)*retained
    limit = max(100.0, 1000*float(np.max(np.abs(internal_initial))))
    elapsed = t-t[0]
    spectrum = np.zeros((len(t), internal_n), complex)
    active = v0 != 0
    with np.errstate(over='ignore', invalid='ignore', under='ignore'):
        spectrum[:, active] = np.exp(elapsed[:, None]*linear[None, active])*v0[None, active]
        if coefficients['1'] != 0:
            spectrum[:, 0] += internal_n*coefficients['1']*_forcing_integral(coefficients['u'], elapsed)
    # Non-DC |v_k(t)| is exponential and DC is a real monotone solution of
    # v'_0 = rate*v_0 + forcing.  Endpoint maxima therefore also bound every
    # half/full ETDRK4 stage; this replaces repeated checks, not their budget.
    if not np.isfinite(spectrum).all() or np.max(np.abs(spectrum)) > internal_n*limit:
        raise ValueError('Nonlinear candidate exceeded the finite-state growth budget.')
    prediction = np.fft.ifft(spectrum, axis=1).real
    if not np.isfinite(prediction).all() or np.max(np.abs(prediction[1:])) > limit:
        raise ValueError('Candidate forecast diverged or exceeded the finite-state growth budget.')
    return prediction[:, ::resolved.DEFAULT_FACTOR].copy()


def raw_forecast(terms, x, t, u0):
    """Same factor-4 supplied-frame forecast, with a linear-only fast path."""
    terms = list(terms)
    coefficients = _coefficients(terms)
    if not _is_linear(coefficients):
        return resolved.forecast(terms, x, t, u0)
    return _linear_forecast(coefficients, x, t, u0)


def forecast_with_diagnostics(terms, x, t, prefix_t, prefix_u):
    """Keep the existing conditioning protocol and full post-cutoff horizon."""
    terms = list(terms)
    coefficients = _coefficients(terms)
    if not _is_linear(coefficients):
        return original_prefix.forecast_with_diagnostics(terms, x, t, prefix_t, prefix_u)
    x, t, pt, pu = original_prefix.validate_prefix(x, t, prefix_t, prefix_u)
    states = []
    filter_evidence = []
    for i in range(len(pt)):
        filtered, diagnostic = denoise_initial(x, pu[i])
        state = (_linear_forecast(coefficients, x, pt[i:], filtered)[-1]
                 if i < len(pt)-1 else filtered)
        states.append(state)
        filter_evidence.append(diagnostic)
    state = np.mean(np.asarray(states), axis=0)
    if not np.isfinite(state).all():
        raise ValueError('Prefix estimate is not finite.')
    diagnostic = {
        'conditioning': 'observed_prefix', 'prefix_frames': len(pt),
        'prefix_start': float(pt[0]), 'forecast_start': float(t[0]),
        'prefix_duration': float(pt[-1]-pt[0]), 'forecast_duration': float(t[-1]-t[0]),
        'future_observations_used': False, 'equal_frame_weights': True,
        'candidate_PDE_used_for_state_estimation': True,
        'last_observation_relative_change': float(np.linalg.norm(state-pu[-1])/max(np.linalg.norm(pu[-1]), 1e-30)),
        'per_frame_filter': filter_evidence, 'independent_estimates_claimed': False,
    }
    return _linear_forecast(coefficients, x, t, state), diagnostic


def forecast(terms, x, t, prefix_t, prefix_u):
    return forecast_with_diagnostics(terms, x, t, prefix_t, prefix_u)[0]
