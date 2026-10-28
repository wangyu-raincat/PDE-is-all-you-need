"""Refined-grid forecasting after one-frame, observation-only filtering."""
from copy import deepcopy

from resolved_study import forecast as refined
from .initial import denoise_initial, PROTOCOL as INITIAL_PROTOCOL

PROTOCOL = {
    'version': 1,
    'name': 'denoised_resolved_factor4',
    'initial_filter': 'spectral_hard',
    'initial_filter_protocol': deepcopy(INITIAL_PROTOCOL),
    'refined_forecast_protocol': deepcopy(refined.PROTOCOL),
    'first_prediction_frame': 'filtered observed first frame, not an additional measured frame',
    'future_or_true_noise_input': False,
    'rule_selection': 'fixed filter, consistently applied in development and final forecasting',
}


def forecast_with_diagnostics(terms, x, t, u0):
    """Return prediction and first-frame-only filter evidence in one pass."""
    initial, diagnostics = denoise_initial(x, u0)
    return refined.forecast(terms, x, t, initial), diagnostics


def forecast(terms, x, t, u0):
    """The production interface accepts only a candidate and observed first frame."""
    return forecast_with_diagnostics(terms, x, t, u0)[0]
