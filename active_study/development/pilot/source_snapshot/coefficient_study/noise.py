"""Observation-only Gaussian moment correction for the existing weak library.

No filtering is applied to the field. A single trajectory-level variance is
estimated from pooled high-frequency Fourier magnitudes. Under spatially iid,
zero-mean Gaussian measurement noise, second and third moments are corrected
on the same 2N real trigonometric interpolant used by the original library.

The scale estimate assumes the upper frequency band is mostly noise. Physical
signal in that band, colored noise, or changing noise variance can invalidate
the correction. Estimated-variance corrections are plug-in estimates; exact
unbiasedness only holds for the mathematical moments with the true variance.
This does not remove all errors-in-variables or shared X/y noise bias.
"""
from __future__ import annotations

from copy import deepcopy
import numpy as np
from scipy.integrate import newton_cotes

from five_study import initial
from l1_study import engine as sparse
from pde_discovery import generic_engine as baseline
from weak_study import engine as original


VARIANTS = deepcopy(original.VARIANTS)
PROTOCOL = {
    'version': 1,
    'base_weak_protocol': deepcopy(original.PROTOCOL),
    'input': 'one observed periodic x,t,u trajectory; no family, reference, or true noise scale',
    'noise_model': 'additive zero-mean spatially iid Gaussian noise, constant variance within a trajectory',
    'noise_estimation': {
        'spectrum': initial.PROTOCOL['spectrum'],
        'scale_band': initial.PROTOCOL['scale_band'],
        'pooling': 'pool magnitudes across all supplied observed frames and upper-band modes',
        'radial_rms': 'median(pooled abs(rFFT(u)/N))/sqrt(log(2))',
        'spatial_variance': 'N * radial_rms**2',
        'maximum_observed_points': initial.MAX_POINTS,
        'additional_filtering': False,
        'reference_or_true_noise_used': False,
    },
    'interpolation': 'unchanged 2N real trigonometric interpolation, split even-grid real Nyquist equally',
    'interpolated_variance': 'odd N: sigma2 everywhere; even N: sigma2 at original points, sigma2*(1-1/N) at midpoints',
    'second_moment': 'interpolated_u**2 - interpolated_noise_variance',
    'third_moment': 'interpolated_u**3 - 3*interpolated_noise_variance*interpolated_u',
    'advection_product': 'one half spatial derivative of the corrected second moment',
    'linear_features_and_target': 'unchanged raw observed Fourier coefficients and original weak integration',
    'quadrature_and_windows': 'unchanged original weak_block rule for each variant',
    'known_zero_variance': 'exact original spectra; correction=False returns the original weak block',
    'general_errors_in_variables_bias_removed': False,
    'plug_in_estimated_variance_is_exactly_unbiased': False,
    'overlapping_windows_are_independent': False,
    'limits': 'upper-band physical signal may bias the scale; colored, heteroskedastic or non-Gaussian noise is not covered',
}


def interpolation_variance_factors(n):
    """Variance of each 2N interpolated value for unit iid original noise.

For even N the N-1 non-Nyquist real degrees of freedom contribute (N-1)/N.
The single real Nyquist degree contributes cos(pi*j/2)**2/N, hence it vanishes
at the inserted midpoints. For odd N all Fourier degrees occur in conjugate
pairs, and the point variance is constant. Interpolated points remain
correlated; these diagonal factors are not an independence claim.
"""
    if type(n) is not int or not 16 <= n <= initial.MAX_POINTS:
        raise ValueError('Need an integer observed-grid size between 16 and 512.')
    factors = np.ones(2*n)
    if n % 2 == 0:
        factors[1::2] = 1.-1./n
    return factors


def estimate_noise_variance(x, u):
    """Estimate one variance from the supplied observations, without filtering.

All frames contribute to the pooled Rayleigh scale. Pooling reduces noise in
the scale estimate for the declared constant-variance model, but does not
make signal-contaminated spectra safe or imply independent physical trials.
"""
    if any(np.iscomplexobj(np.asarray(value)) for value in (x, u)):
        raise ValueError('Noise estimation requires real coordinates and observations.')
    x, _ = baseline._coordinates(x, np.array([0., 1.]))
    u = np.asarray(u, float)
    n = len(x)
    if n > initial.MAX_POINTS:
        raise ValueError('Noise estimation exceeds the fixed 512-point observed-grid budget.')
    if (u.ndim != 2 or not 1 <= len(u) <= 2000 or u.shape[1] != n
            or not np.isfinite(u).all() or np.max(np.abs(u)) > initial.MAX_ABSOLUTE_STATE
            or u.size > 3_000_000):
        raise ValueError('Need finite Nt-by-Nx observations within the declared state and array budgets.')
    band = np.arange(n//4, n//2)
    magnitudes = np.abs((np.fft.rfft(u, axis=1)/n)[:, band])
    radial_rms = float(np.median(magnitudes)/np.sqrt(np.log(2.)))
    variance = float(n*radial_rms**2)
    if not np.isfinite(variance):
        raise ValueError('Estimated measurement variance is not finite.')
    diagnostics = {
        'method': 'pooled_observed_upper_band_rayleigh_scale',
        'estimated_spatial_noise_variance': variance,
        'estimated_spatial_noise_sigma': float(np.sqrt(variance)),
        'estimated_coefficient_noise_radial_rms': radial_rms,
        'observed_frames': len(u),
        'observed_points': n,
        'scale_band_start': int(band[0]),
        'scale_band_stop_exclusive': int(band[-1]+1),
        'pooled_magnitude_count': int(magnitudes.size),
        'observations_filtered': False,
        'constant_variance_assumed': True,
        'spatial_white_gaussian_noise_assumed': True,
        'upper_band_signal_can_bias_scale': True,
        'independent_physical_trials_claimed': False,
        'reference_used': False,
        'true_noise_used': False,
    }
    return variance, diagnostics


def _spectra(u, variance):
    """Private mathematical primitive; the public fit API estimates variance.

The variance argument exists for internal composition and manufactured tests,
not as a production interface for supplying an oracle noise level.
"""
    if (isinstance(variance, (bool, complex)) or not np.isscalar(variance)
            or not np.isfinite(variance) or variance < 0):
        raise ValueError('Moment variance must be a finite nonnegative scalar.')
    if variance == 0:
        return original._spectra(u)
    n = u.shape[1]
    spectrum = np.fft.fft(u, axis=1)/n
    padded = np.zeros((len(u), 2*n), complex)
    positive = (n+1)//2
    padded[:, :positive] = spectrum[:, :positive]
    negative = (n-1)//2
    if negative:
        padded[:, -negative:] = spectrum[:, -negative:]
    if n % 2 == 0:
        padded[:, n//2] = spectrum[:, n//2]/2
        padded[:, -n//2] = spectrum[:, n//2]/2
    interpolated = np.fft.ifft(padded*(2*n), axis=1).real
    point_variance = variance*interpolation_variance_factors(n)[None, :]
    square = interpolated**2-point_variance
    cube = interpolated**3-3*point_variance*interpolated
    return spectrum, np.fft.fft(square, axis=1)/(2*n), np.fft.fft(cube, axis=1)/(2*n)


def weak_block(data, variant='compact', noise_correction=True):
    """Construct an original-geometry weak block with optional moment correction.

The method never changes the observed field, temporal target, or derivative
columns. There is no reference-equation, true-variance, or family argument.
"""
    if type(noise_correction) is not bool:
        raise ValueError('noise_correction must be a boolean.')
    if not noise_correction:
        return original.weak_block(data, variant)
    if variant not in VARIANTS:
        raise ValueError('Unknown weak-block variant.')
    if not isinstance(data, dict) or not all(k in data for k in ('x', 't', 'u')):
        raise ValueError('Need x,t,u observations.')
    if any(np.iscomplexobj(np.asarray(data.get(k))) for k in ('x', 't', 'u')):
        raise ValueError('Weak observations must be real-valued.')
    x, t, u = baseline._validate(data)
    variance, noise_diagnostics = estimate_noise_variance(x, u)
    n = len(x)
    spec = VARIANTS[variant]
    width = max(8, 8*int(round((len(t)-1)*spec['time_fraction']/8)))
    width = min(width, 8*((len(t)-1)//8))
    starts = np.arange(0, len(t)-width, max(1, width//4))
    max_mode = min(n//4, max(8, int(n*spec['mode_fraction'])))
    modes = np.arange(0, max_mode+1)
    wave = 2*np.pi*modes/(n*(x[1]-x[0]))
    uh, square, cube = _spectra(u, variance)
    uh, square, cube = uh[:, modes], square[:, modes], cube[:, modes]
    constant = np.zeros_like(uh)
    constant[:, 0] = 1
    library = np.stack([constant, uh, square, cube,
                       *[(1j*wave)**order*uh for order in range(1, 5)],
                       .5j*wave*square], axis=-1)
    features, targets = [], []
    for start in starts:
        local_t = t[start:start+width+1]
        half = (local_t[-1]-local_t[0])/2
        z = (local_t-(local_t[-1]+local_t[0])/2)/half
        phi = (1-z*z)**4
        derivative = -8*z*(1-z*z)**3/half
        weights = np.zeros(width+1)
        for offset in range(0, width, 8):
            weights[offset:offset+9] += newton_cotes(8, equal=1)[0]*(local_t[1]-local_t[0])
        norm = float(weights@phi)
        X = np.einsum('t,tmj->mj', weights*phi, library[start:start+width+1])/norm
        y = -np.einsum('t,tm->m', weights*derivative, uh[start:start+width+1])/norm
        features.extend([X[0:1].real, np.sqrt(2)*X[1:].real, np.sqrt(2)*X[1:].imag])
        targets.extend([y[0:1].real, np.sqrt(2)*y[1:].real, np.sqrt(2)*y[1:].imag])
    factors = interpolation_variance_factors(n)
    block = {'X': np.vstack(features), 'y': np.concatenate(targets), 'metadata': {
        'derivative_mode': 'space_time_weak_fourier', 'variant': variant,
        'window_intervals': width, 'integration_windows': len(starts),
        'max_spatial_mode': int(modes[-1]), 'time_test_degree': 8,
        'no_observed_derivatives': True, 'training_frames': len(t), 'spatial_points': n,
        'overlapping_windows': True,
        'noise_correction': 'gaussian_second_and_third_moments_with_estimated_variance',
        'noise_estimation': noise_diagnostics,
        'interpolated_variance_factor_min': float(factors.min()),
        'interpolated_variance_factor_max': float(factors.max()),
        'interpolated_noise_points_independent': False,
        'all_errors_in_variables_bias_removed': False,
        'reference_used': False,
    }}
    sparse.validate_block(block)
    return block
