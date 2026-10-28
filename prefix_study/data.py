"""Fresh synthetic observations; only corruption sees clean fields/noise scales."""
from pathlib import Path
import numpy as np
from challenge_study.data import _validated, training_scale as _training_scale

FAMILY_IDS = {'allen_cahn': 0, 'low_viscosity_burgers': 1, 'ks_unstable': 2}
ROLE_IDS = {'training': 0, 'development': 1, 'final': 2}
ROLE_COUNTS = {'training': 3, 'development': 2, 'final': 3}
PROTOCOL = {
    'kind': 'synthetic observations; no measured data',
    'role_counts': ROLE_COUNTS,
    'family_ids': FAMILY_IDS,
    'role_ids': ROLE_IDS,
    'noise_levels': [0., .001, .01],
    'nonzero_noise_seeds': [71, 103],
    'clean_noise_seed': 0,
    'seed_sequence': '[159000, family_id, role_id, trajectory_index, noise_seed]',
    'scale': 'noise_level * full-grid clean training FORECAST-segment equal-trajectory RMS',
    'sampling_order': 'full-role noise first, then stride 1 or 2 in space and time',
    'coarse_is_nested_observation': True,
    'observation_interface': 'x,t,u,boundary,prefix_t,prefix_u; prefix ends at forecast t[0]; no noise oracle',
    'prefix_intervals_full': 10,
    'forecast_intervals_full': 100,
    'forecast_horizon_unchanged': True,
    'prefix_endpoint_shared_observation': True,
}


PREFIX_INTERVALS = 10


def training_scale(clean_fields):
    segments=[]
    for clean in clean_fields:
        x,t,u=_validated(clean)
        if len(t)!=111:raise ValueError('Need the declared 10-prefix + 100-forecast intervals.')
        segments.append({'x':x,'t':t[PREFIX_INTERVALS:],'u':u[PREFIX_INTERVALS:]})
    return _training_scale(segments)


def identity(family, role, index):
    if family not in FAMILY_IDS or role not in ROLE_IDS:
        raise ValueError('Unknown family or role.')
    if type(index) is not int or not 0 <= index < ROLE_COUNTS[role]:
        raise ValueError('Invalid trajectory index.')


def write_clean(root, family, role, index, clean):
    identity(family, role, index)
    x, t, u = _validated(clean)
    path = Path(root)/'clean'/family/role/f'{index}.npz'
    if path.exists(): raise RuntimeError('Clean role already exists; preserve it.')
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, x=x, t=t, u=u)


def read_clean(root, family, role, index):
    identity(family, role, index)
    with np.load(Path(root)/'clean'/family/role/f'{index}.npz', allow_pickle=False) as z:
        result = {k:z[k].copy() for k in ('x','t','u')}
    result['boundary'] = 'periodic'
    _validated(result)
    return result


def corrupt(clean, family, role, index, sampling, level, seed, scale):
    identity(family, role, index)
    if (sampling not in ('full','coarse') or isinstance(level, bool)
            or level not in (0., .001, .01) or type(seed) is not int
            or seed not in ((0,) if level == 0 else (71,103))):
        raise ValueError('Undeclared observation condition.')
    if isinstance(scale,bool) or not np.isfinite(scale) or scale <= 0:
        raise ValueError('Training-only RMS must be positive and finite.')
    x,t,u = _validated(clean)
    rng = np.random.default_rng(np.random.SeedSequence([159000,FAMILY_IDS[family],ROLE_IDS[role],index,seed]))
    noisy = u + level*scale*rng.standard_normal(u.shape)
    stride = 1 if sampling == 'full' else 2
    if len(t)!=111:raise ValueError('Need the full declared prefix and forecast grid.')
    return {'x':x[::stride].copy(),'t':t[PREFIX_INTERVALS::stride].copy(),
            'u':noisy[PREFIX_INTERVALS::stride,::stride].copy(),'boundary':'periodic',
            'prefix_t':t[:PREFIX_INTERVALS+1:stride].copy(),
            'prefix_u':noisy[:PREFIX_INTERVALS+1:stride,::stride].copy()}
