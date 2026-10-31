"""A common, observation-independent periodic initial-condition action space.

Action proposals use no PDE label, measured trajectory, or reference solution.
Every action has equal RMS energy; large-amplitude proposals cannot win an
acquisition score merely by using a stronger experiment.
"""
from __future__ import annotations

import hashlib
import json
from numbers import Integral

import numpy as np

PERIOD = float(2 * np.pi)
MODES = tuple(range(1, 7))
INITIAL_RMS = .4
MAX_ABSOLUTE_INITIAL = 1.2
POOL_SIZE = 32
CERTIFICATION_POINTS = 4096
PROTOCOL = {
    'version': 1,
    'parameterization': 'sum(a[k-1]*sin(k*x)+b[k-1]*cos(k*x)), k=1..6; zero constant',
    'period': PERIOD,
    'modes': list(MODES),
    'initial_rms': INITIAL_RMS,
    'maximum_absolute_initial': MAX_ABSOLUTE_INITIAL,
    'amplitude_certificate': 'maximum on 4096-point periodic grid plus sum(k*hypot(a,b))*pi/4096',
    'candidate_pool_size': POOL_SIZE,
    'candidate_pool': 'four fixed all-six-mode rich phase variants, twelve single-mode actions, sixteen random 2..6-mode mixtures',
    'initial_training_actions': 3,
    'initial_development_actions': 2,
    'initial_excitation': 'mode one only, five separately drawn phases; same physical excitation spectrum',
    'fixed_baseline': 'four predefined all-six-mode rich phase variants; no truth-dependent selection',
    'pool_seed_input': 'nonnegative integer seed; no family, coefficient or observation input',
    'candidate_predictions_are_measurements': False,
    'units': 'dimensionless benchmark coordinates',
}


def _seed(value):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or not 0 <= value < 2**31:
        raise ValueError('Action seed must be an integer in [0,2**31).')
    return int(value)


def _raw_field(a, b, x):
    phase = np.asarray(x, float)[:, None] * np.asarray(MODES)[None, :]
    return np.sin(phase) @ a + np.cos(phase) @ b


def validate_action(action):
    """Return a canonical finite action, certifying its continuous amplitude.

The derivative magnitude is bounded by sum(k*hypot(a,b)). Every point on
the periodic domain lies at most pi/4096 from a certification-grid point.
Thus the reported bound covers the continuous initial condition, not only
the 64 measured points or a possibly coarser acquisition grid.
"""
    if not isinstance(action, dict) or set(action) != {'id', 'a', 'b'}:
        raise ValueError('An action must contain exactly id, a and b.')
    identity = action['id']
    if not isinstance(identity, str) or not 1 <= len(identity) <= 96:
        raise ValueError('Action id must be a nonempty string of at most 96 characters.')
    arrays = []
    for key in ('a', 'b'):
        raw = np.asarray(action[key])
        if np.iscomplexobj(raw) or raw.dtype.kind == 'b':
            raise ValueError('Fourier coefficients must be real numbers, not booleans.')
        try:
            value = np.asarray(raw, float)
        except (ValueError, TypeError) as exc:
            raise ValueError('Fourier coefficients must be numeric arrays.') from exc
        if value.shape != (6,) or not np.isfinite(value).all():
            raise ValueError('Each action requires six finite sine and cosine coefficients.')
        arrays.append(value)
    a, b = arrays
    rms = float(np.sqrt(.5 * (a @ a + b @ b)))
    if not np.isclose(rms, INITIAL_RMS, rtol=0, atol=1e-10):
        raise ValueError('Action RMS must equal the fixed 0.4 energy budget.')
    grid = np.arange(CERTIFICATION_POINTS) * PERIOD / CERTIFICATION_POINTS
    derivative_bound = float(np.asarray(MODES) @ np.hypot(a, b))
    bound = float(np.max(np.abs(_raw_field(a, b, grid))) + derivative_bound * np.pi / CERTIFICATION_POINTS)
    if not np.isfinite(bound) or bound > MAX_ABSOLUTE_INITIAL:
        raise ValueError('Continuous initial-condition amplitude exceeds the 1.2 bound.')
    return {'id': identity, 'a': a.tolist(), 'b': b.tolist()}


def action_field(action, x):
    action = validate_action(action)
    raw = np.asarray(x)
    if np.iscomplexobj(raw):
        raise ValueError('Action coordinates must be real.')
    x = np.asarray(raw, float)
    if x.ndim != 1 or not 1 <= len(x) <= 16384 or not np.isfinite(x).all():
        raise ValueError('Need a finite one-dimensional spatial grid.')
    return _raw_field(np.asarray(action['a']), np.asarray(action['b']), x)


def action_digest(action):
    """Canonical action identity, retaining the declared human-readable id."""
    action = validate_action(action)
    payload = json.dumps(action, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    return hashlib.sha256(payload).hexdigest()


def _from_amplitude_phase(identity, amplitudes, phases):
    return validate_action({'id': identity, 'a': (amplitudes * np.cos(phases)).tolist(),
                            'b': (amplitudes * np.sin(phases)).tolist()})


def fixed_rich_action(index=0):
    """A strong predetermined baseline with nonzero amplitude in all six modes."""
    if isinstance(index, (bool, np.bool_)) or not isinstance(index, Integral) or not 0 <= index < 4:
        raise ValueError('Fixed-rich index must be an integer from zero to three.')
    index = int(index)
    k = np.asarray(MODES)
    phases = np.pi * k * (k - 1) / len(k) + .7 * index
    amplitudes = np.full(6, INITIAL_RMS * np.sqrt(2 / 6))
    return _from_amplitude_phase('fixed-rich' if index == 0 else f'fixed-rich-{index}', amplitudes, phases)


def initial_actions(seed):
    rng = np.random.default_rng(np.random.SeedSequence([841000, _seed(seed)]))
    actions = []
    for i in range(5):
        amplitudes = np.zeros(6)
        amplitudes[0] = np.sqrt(2) * INITIAL_RMS
        phases = np.zeros(6)
        phases[0] = float(rng.uniform(-np.pi, np.pi))
        actions.append(_from_amplitude_phase(f'initial-{i}', amplitudes, phases))
    return {'training': actions[:3], 'development': actions[3:]}


def make_action_pool(seed):
    rng = np.random.default_rng(np.random.SeedSequence([842000, _seed(seed)]))
    actions = [fixed_rich_action(i) for i in range(4)]
    for mode in MODES:
        for repeat in range(2):
            amplitudes = np.zeros(6)
            amplitudes[mode - 1] = np.sqrt(2) * INITIAL_RMS
            phases = np.zeros(6)
            phases[mode - 1] = float(rng.uniform(-np.pi, np.pi))
            actions.append(_from_amplitude_phase(f'single-{mode}-{repeat}', amplitudes, phases))
    attempts = 0
    while len(actions) < POOL_SIZE:
        attempts += 1
        if attempts > 1000:
            raise RuntimeError('Unable to construct the declared finite action pool.')
        size = int(rng.integers(2, 7))
        modes = rng.choice(6, size=size, replace=False)
        amplitudes = np.zeros(6)
        amplitudes[modes] = rng.uniform(.25, 1., size)
        amplitudes *= np.sqrt(2) * INITIAL_RMS / np.linalg.norm(amplitudes)
        phases = rng.uniform(-np.pi, np.pi, 6)
        try:
            action = _from_amplitude_phase(f'mixture-{len(actions)-16:02}', amplitudes, phases)
        except ValueError:
            continue
        actions.append(action)
    return actions
