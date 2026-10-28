"""Forecast a candidate PDE on a fixed finer internal periodic grid.

Only the supplied observed initial frame is spectrally interpolated. The extra
grid points are computational degrees of freedom, never additional evidence.
The original ETDRK4 implementation and its physical/growth guards are reused.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import resample

from pde_discovery import generic_engine as baseline

DEFAULT_FACTOR = 4
ALLOWED_FACTORS = (1, 2, 4)
MAX_INTERNAL_POINTS = 2048
PROTOCOL = {
    'version': 1,
    'default_spatial_factor': DEFAULT_FACTOR,
    'diagnostic_spatial_factors': list(ALLOWED_FACTORS),
    'factor_selection': 'fixed before new training/development generation; not family dependent',
    'initial_interpolation': 'real periodic Fourier interpolation of the observed initial frame; scipy.signal.resample handles even-grid Nyquist splitting',
    'internal_grid': 'Nx*factor equally spaced points on the same period, starting at the observed x[0]',
    'solver': 'unchanged pde_discovery.generic_engine.forecast on the internal grid',
    'output': 'direct restriction to every factor-th internal point at the original output times',
    'time_substeps': baseline.SUBSTEPS,
    'maximum_internal_points': MAX_INTERNAL_POINTS,
    'budget_failure': 'reject; never silently reduce factor',
    'extra_points_are_measurements': False,
    'truth_or_future_input': False,
    'scope': 'uniform periodic scalar autonomous constant-coefficient nine-term PDE',
}


def forecast(terms, x, t, u0, factor=DEFAULT_FACTOR):
    """Use a fixed computational refinement, with no future or truth input."""
    if type(factor) is not int or factor not in ALLOWED_FACTORS:
        raise ValueError('Spatial factor must be one of the fixed integers 1, 2, 4.')
    if any(np.iscomplexobj(np.asarray(v)) for v in (x, t, u0)):
        raise ValueError('Coordinates and initial observations must be real-valued.')
    x, t = baseline._coordinates(x, t)
    u0 = np.asarray(u0, float)
    if u0.shape != x.shape or not np.isfinite(u0).all():
        raise ValueError('Initial field must be finite and match the observed grid.')
    internal_n = factor*len(x)
    if internal_n > MAX_INTERNAL_POINTS:
        raise ValueError('Declared spatial refinement exceeds the 2048-point computational budget.')
    if factor == 1:
        return baseline.forecast(terms, x, t, u0)
    period = len(x)*(x[1]-x[0])
    internal_x = x[0]+np.arange(internal_n)*period/internal_n
    internal_initial = resample(u0, internal_n)
    prediction = baseline.forecast(terms, internal_x, t, internal_initial)
    return prediction[:, ::factor].copy()
