"""Single-frame spectral hard threshold with an estimated noise scale.

This is a sparse-spectrum initial-condition prior, not a general denoising
guarantee. Genuine broadband high-frequency content can contaminate the scale
estimate and small physical modes can be removed. Development scoring must
judge that tradeoff. No future frames, family labels or true noise levels enter.
"""
from __future__ import annotations

import numpy as np
from pde_discovery import generic_engine as baseline

MAX_POINTS = 512
MAX_ABSOLUTE_STATE = 1e6
PROTOCOL = {
    'version': 1,
    'input': 'one observed initial frame and its uniform periodic spatial grid only',
    'spectrum': 'real FFT divided by N',
    'scale_band': 'integer modes floor(N/4) <= k < floor(N/2); DC and real Nyquist excluded',
    'coefficient_noise_rms': 'median(abs(normalized FFT in scale band))/sqrt(log(2))',
    'estimated_spatial_sigma': 'coefficient_noise_rms*sqrt(N)',
    'threshold': 'coefficient_noise_rms*sqrt(2*log(N))',
    'rule': 'retain coefficients with magnitude >= threshold, always preserve DC; hard threshold only',
    'retained_modes': 'data dependent, no fixed physical modes and no family-specific rule',
    'phase': 'retained complex coefficients unchanged; irfft handles real Nyquist',
    'maximum_observed_points': MAX_POINTS,
    'maximum_absolute_state': MAX_ABSOLUTE_STATE,
    'scale_assumption': 'additive spatial white Gaussian noise; upper-band majority is noise rather than signal',
    'limitations': 'single-frame sparse-spectrum prior; broadband signal can bias scale upward and weak physical modes can be removed',
    'future_input': False,
    'reference_or_true_noise_input': False,
    'additional_measurements': False,
}


def denoise_initial(x, u0):
    """Return a filtered real first frame plus auditable estimated quantities.

For white spatial noise of standard deviation sigma, an interior normalized
complex FFT coefficient has radial RMS sigma/sqrt(N), and Rayleigh median
sigma*sqrt(log(2)/N). The threshold above uses that radial RMS rather than
mistaking a complex coefficient's variance for one real component's variance.
The one real Nyquist coefficient is excluded from estimating a Rayleigh scale,
but receives the same conservative magnitude threshold when present.
"""
    if any(np.iscomplexobj(np.asarray(value)) for value in (x, u0)):
        raise ValueError('Initial observations and coordinates must be real-valued.')
    x, _ = baseline._coordinates(x, np.array([0., 1.]))
    u = np.asarray(u0, float)
    if len(x) > MAX_POINTS:
        raise ValueError('Initial filtering exceeds the fixed 512-point observed-grid budget.')
    if (u.shape != x.shape or not np.isfinite(u).all()
            or np.max(np.abs(u)) > MAX_ABSOLUTE_STATE):
        raise ValueError('Initial observations must be finite, match x, and stay within the state budget.')
    n = len(u)
    spectrum = np.fft.rfft(u)/n
    band = np.arange(n//4, n//2)
    radial_rms = float(np.median(np.abs(spectrum[band]))/np.sqrt(np.log(2.)))
    sigma = radial_rms*np.sqrt(n)
    threshold = radial_rms*np.sqrt(2*np.log(n))
    keep = np.abs(spectrum) >= threshold
    keep[0] = True
    filtered = np.fft.irfft(spectrum*keep, n=n)*n
    if not np.isfinite(filtered).all():
        raise ValueError('Initial filtering did not produce a finite field.')
    difference = filtered-u
    raw_norm = float(np.linalg.norm(u))
    centered_norm = float(np.linalg.norm(u-u.mean()))
    change_norm = float(np.linalg.norm(difference))
    diagnostics = {
        'method': 'single_frame_radial_fourier_hard_threshold',
        'observed_points': n,
        'estimated_spatial_noise_sigma': float(sigma),
        'estimated_coefficient_noise_radial_rms': radial_rms,
        'coefficient_threshold': float(threshold),
        'scale_band_start': int(band[0]),
        'scale_band_stop_exclusive': int(band[-1]+1),
        'scale_band_count': len(band),
        'retained_positive_modes': np.flatnonzero(keep)[1:].tolist(),
        'removed_positive_mode_count': int(np.count_nonzero(~keep[1:])),
        'relative_initial_change': change_norm/raw_norm if raw_norm else 0.,
        'removed_non_dc_energy_fraction': min(1., (change_norm/centered_norm)**2) if centered_norm else 0.,
        'mean_before': float(u.mean()),
        'mean_after': float(filtered.mean()),
        'noise_scale_estimated_from_this_frame_only': True,
        'reference_used': False,
        'future_used': False,
        'broadband_signal_may_be_removed': True,
    }
    return filtered.copy(), diagnostics
