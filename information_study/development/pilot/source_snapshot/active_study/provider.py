"""Private synthetic experiment provider, separate from discovery and policy.

Only the controller may request a selected experiment. The six-array
``observation`` view contains no equation label, reference coefficients, clean
field, or true noise level. Clean references and task specifications are kept
separate for source calibration and post-freeze scoring.
"""
from __future__ import annotations

import hashlib
from numbers import Integral

import numpy as np

from coefficient_study.generators import _integrate as _independent_integrate
from . import actions

FAMILIES = ('decay', 'diffusion', 'hyperdiffusion', 'advection_diffusion',
            'reaction_diffusion', 'burgers')
ROLE_SEEDS = {'calibration': 851000, 'development': 852000, 'formal': 853000}
NOISE_SEED_BASE = 854000
OBSERVED_NX = 64
FULL_FRAMES = 111
PREFIX_INTERVALS = 10
FORECAST_DURATION = 1.0
REFERENCE_N = 512
REFERENCE_MAX_DT = .001
CALIBRATION_RELATIVE_LIMIT = 1e-5
CALIBRATION_MAX_ABSOLUTE_LIMIT = 1e-4
COEFFICIENT_RANGES = {
    'decay': {'u': [-.4, -.15]},
    'diffusion': {'u_xx': [.05, .15]},
    'hyperdiffusion': {'u_xxxx': [-.005, -.001]},
    'advection_diffusion': {'u_x': [-.7, -.3], 'u_xx': [.05, .15]},
    'reaction_diffusion': {'u': [-.4, -.15], 'u_xx': [.05, .15]},
    'burgers': {'u_xx': [.04, .08], 'u*u_x': [-1.2, -.6]},
}
PROTOCOL = {
    'version': 1,
    'kind': 'controlled synthetic initial-condition queries, not measured data',
    'families': list(FAMILIES),
    'coefficient_ranges': COEFFICIENT_RANGES,
    'role_seed_bases': ROLE_SEEDS,
    'task_index_rule': 'family=index%6, coefficient replicate=index//6; same family/action spaces for all policies',
    'parameter_seed': 'SeedSequence([role_seed_base, task_index])',
    'noise_seed_base': NOISE_SEED_BASE,
    'noise_rule': '0.4*noise_level*standard_normal(111,64), caller supplies query seed; no trajectory-RMS oracle',
    'noise_limits': [0., .03],
    'observed_nx': OBSERVED_NX,
    'full_frames': FULL_FRAMES,
    'prefix_intervals': PREFIX_INTERVALS,
    'prefix_frames': PREFIX_INTERVALS + 1,
    'future_frames': FULL_FRAMES - PREFIX_INTERVALS,
    'full_time_end': 1.1,
    'future_duration': FORECAST_DURATION,
    'measurement_values_per_experiment': FULL_FRAMES * OBSERVED_NX,
    'measurement_count': '111*64 distinct observations; shared history endpoint counted once',
    'linear_reference': 'exact Fourier evolution under hidden linear symbol; independent of discovery forecast',
    'nonlinear_reference': 'independent contour ETDRK4 Fourier Galerkin Burgers reference',
    'reference_n': REFERENCE_N,
    'reference_max_dt': REFERENCE_MAX_DT,
    'refinement': 'double spatial reference grid and halve timestep; same prescribed initial field',
    'observation_rule': 'direct reference-grid samples, never sparse-sensor interpolation',
    'calibration_relative_l2_limit': CALIBRATION_RELATIVE_LIMIT,
    'calibration_max_absolute_limit': CALIBRATION_MAX_ABSOLUTE_LIMIT,
    'calibration_roles_only': True,
    'formal_generation_guard': 'controller freezes protocol before formal tasks; provider does not enforce global lifecycle',
    'final_generation_guard': 'controller freezes all policy snapshots before final queries',
    'scope': 'six fixed one-dimensional periodic scalar equation families, varying coefficients; not arbitrary PDE discovery',
}


def _integer(value, name, maximum=2**31):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or not 0 <= value < maximum:
        raise ValueError(f'{name} must be a nonnegative integer below {maximum}.')
    return int(value)


def get_task_spec(role, index):
    """Controller/scoring-only task definition; never pass this into a learner."""
    if role not in ROLE_SEEDS:
        raise ValueError('Task role must be calibration, development or formal.')
    index = _integer(index, 'Task index', 6000)
    family = FAMILIES[index % len(FAMILIES)]
    seed = [ROLE_SEEDS[role], index]
    rng = np.random.default_rng(np.random.SeedSequence(seed))
    coefficients = {name: float(rng.uniform(*limits))
                    for name, limits in COEFFICIENT_RANGES[family].items()}
    return {'role': role, 'index': index, 'family': family,
            'parameter_seed': seed, 'coefficients': coefficients}


def _validate_spec(spec):
    if not isinstance(spec, dict) or 'role' not in spec or 'index' not in spec:
        raise ValueError('A declared private task specification is required.')
    expected = get_task_spec(spec['role'], spec['index'])
    if spec != expected:
        raise ValueError('Task specification differs from its declared role/index seed.')
    return expected


def reference_terms(spec):
    """Scoring-only coefficients; no discovery module imports this provider."""
    spec = _validate_spec(spec)
    return [{'name': name, 'coefficient': value} for name, value in spec['coefficients'].items()]


def generate_clean(spec, action, refined=False):
    """Return all 111 directly sampled clean frames for one requested action."""
    spec = _validate_spec(spec)
    action = actions.validate_action(action)
    if type(refined) is not bool:
        raise ValueError('refined must be a boolean.')
    n = REFERENCE_N * (2 if refined else 1)
    x = np.arange(n) * actions.PERIOD / n
    t = np.linspace(0., 1.1, FULL_FRAMES)
    initial = actions.action_field(action, x)
    coefficients = spec['coefficients']
    if spec['family'] == 'burgers':
        field = _independent_integrate(initial, actions.PERIOD, t,
            {2: coefficients['u_xx']}, coefficients['u*u_x'], 0.,
            REFERENCE_MAX_DT / (2 if refined else 1))
    else:
        wave = 2 * np.pi * np.fft.fftfreq(n, d=actions.PERIOD / n)
        symbol = np.full(n, coefficients.get('u', 0.), dtype=complex)
        for name, order in (('u_x', 1), ('u_xx', 2), ('u_xxxx', 4)):
            symbol += coefficients.get(name, 0.) * (1j * wave)**order
        field = np.fft.ifft(np.exp(t[:, None] * symbol) * np.fft.fft(initial), axis=1).real
    observed = field[:, ::n // OBSERVED_NX].copy()
    if observed.shape != (FULL_FRAMES, OBSERVED_NX) or not np.isfinite(observed).all():
        raise FloatingPointError('Independent reference produced invalid observations.')
    return {'x': x[::n // OBSERVED_NX].copy(), 't': t, 'u': observed, 'boundary': 'periodic'}


def observation_view(clean, observed):
    """Separate fitting/future and preceding observed history without metadata."""
    if not isinstance(clean, dict) or not all(k in clean for k in ('x', 't')):
        raise ValueError('Clean coordinates are required.')
    raw_x, raw_t = np.asarray(clean['x']), np.asarray(clean['t'])
    if np.iscomplexobj(raw_x) or np.iscomplexobj(raw_t):
        raise ValueError('Observation coordinates must be real.')
    try:
        x, t = np.asarray(raw_x, float), np.asarray(raw_t, float)
    except (ValueError, TypeError) as exc:
        raise ValueError('Observation coordinates must be numeric.') from exc
    if (x.shape != (OBSERVED_NX,) or t.shape != (FULL_FRAMES,)
            or not np.isfinite(x).all() or not np.isfinite(t).all()
            or not np.allclose(x, np.arange(OBSERVED_NX) * actions.PERIOD / OBSERVED_NX, rtol=0, atol=1e-12)
            or not np.allclose(t, np.linspace(0., 1.1, FULL_FRAMES), rtol=0, atol=1e-12)):
        raise ValueError('Observation coordinates differ from the declared full grid.')
    field = np.asarray(observed)
    if np.iscomplexobj(field) or field.shape != (FULL_FRAMES, OBSERVED_NX) or not np.isfinite(field).all():
        raise ValueError('Need the complete finite 111-by-64 observed field.')
    return {'x': x.copy(),
            't': t[PREFIX_INTERVALS:].copy(),
            'u': np.asarray(field, float)[PREFIX_INTERVALS:].copy(), 'boundary': 'periodic',
            'prefix_t': t[:PREFIX_INTERVALS + 1].copy(),
            'prefix_u': np.asarray(field, float)[:PREFIX_INTERVALS + 1].copy()}


def generate_observation(spec, action, noise_seed, noise_level):
    """Provide exactly one selected experiment, keeping truth in a separate view.

``receipt`` provides measurement accounting and byte identity only. Noise
parameters, coefficients and labels never appear inside ``observation``.
The clean return value must remain in the controller/provider or final scorer.
"""
    noise_seed = _integer(noise_seed, 'Noise seed')
    if (isinstance(noise_level, (bool, complex, np.bool_))
            or not isinstance(noise_level, (int, float, np.integer, np.floating))
            or not np.isfinite(noise_level) or not 0 <= noise_level <= .03):
        raise ValueError('Noise level must be a finite scalar in [0,.03].')
    clean = generate_clean(spec, action)
    rng = np.random.default_rng(np.random.SeedSequence([NOISE_SEED_BASE, noise_seed]))
    observed = clean['u'] + actions.INITIAL_RMS * float(noise_level) * rng.standard_normal(clean['u'].shape)
    observation = observation_view(clean, observed)
    receipt = {'experiments': 1, 'measurement_values': FULL_FRAMES * OBSERVED_NX,
               'measured_frames': FULL_FRAMES, 'measured_spatial_points': OBSERVED_NX,
               'prefix_frames': PREFIX_INTERVALS + 1, 'future_frames': FULL_FRAMES - PREFIX_INTERVALS,
               'history_endpoint_counted_once': True, 'future_duration': FORECAST_DURATION,
               'action_sha256': actions.action_digest(action),
               'observed_field_sha256': hashlib.sha256(observed.tobytes()).hexdigest(),
               'is_measured_physical_data': False}
    return {'observation': observation, 'clean': clean, 'receipt': receipt}


def calibration():
    """Use reserved calibration parameters and lawful ICs; never formal fields."""
    records = []
    pool = actions.make_action_pool(855000)
    # The rich and highest-frequency single mode exercise nonlinear resolution
    # and the steepest linear decay in the declared finite action space.
    probes = [actions.fixed_rich_action(), next(a for a in pool if a['id'] == 'single-6-0')]
    for index, family in enumerate(FAMILIES):
        spec = get_task_spec('calibration', index)
        for action in probes:
            try:
                coarse = generate_clean(spec, action)
                fine = generate_clean(spec, action, refined=True)
                np.testing.assert_array_equal(coarse['x'], fine['x'])
                np.testing.assert_array_equal(coarse['t'], fine['t'])
                difference = coarse['u'][1:] - fine['u'][1:]
                relative = float(np.linalg.norm(difference) / max(np.linalg.norm(fine['u'][1:]), 1e-30))
                maximum = float(np.max(np.abs(difference)))
                passed = relative <= CALIBRATION_RELATIVE_LIMIT and maximum <= CALIBRATION_MAX_ABSOLUTE_LIMIT
                records.append({'family': family, 'action_id': action['id'], 'role': 'calibration',
                    'relative_l2': relative, 'max_absolute_error': maximum,
                    'status': 'passed' if passed else 'source_quality_failure'})
            except (ValueError, FloatingPointError, RuntimeError, AssertionError) as exc:
                records.append({'family': family, 'action_id': action['id'], 'role': 'calibration',
                    'status': 'source_quality_failure', 'failure_reason': str(exc)})
    return {'records': records, 'all_passed': all(r['status'] == 'passed' for r in records),
            'formal_data_generated': False, 'calibration_only': True,
            'relative_limit': CALIBRATION_RELATIVE_LIMIT,
            'maximum_absolute_limit': CALIBRATION_MAX_ABSOLUTE_LIMIT,
            'protocol': PROTOCOL, 'action_protocol': actions.PROTOCOL}
