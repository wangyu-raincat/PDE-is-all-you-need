"""Fresh synthetic observations; only corruption sees clean fields/noise scales."""
from pathlib import Path
import numpy as np
from challenge_study.data import _validated, training_scale

FAMILY_IDS = {'allen_cahn': 0, 'low_viscosity_burgers': 1, 'ks_unstable': 2}
ROLE_IDS = {'training': 0, 'development': 1, 'final': 2}
ROLE_COUNTS = {'training': 3, 'development': 2, 'final': 3}


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
            or seed not in ((0,) if level == 0 else (43,71))):
        raise ValueError('Undeclared observation condition.')
    if isinstance(scale,bool) or not np.isfinite(scale) or scale <= 0:
        raise ValueError('Training-only RMS must be positive and finite.')
    x,t,u = _validated(clean)
    rng = np.random.default_rng(np.random.SeedSequence([77000,FAMILY_IDS[family],ROLE_IDS[role],index,seed]))
    noisy = u + level*scale*rng.standard_normal(u.shape)
    stride = 1 if sampling == 'full' else 2
    return {'x':x[::stride].copy(),'t':t[::stride].copy(),
            'u':noisy[::stride,::stride].copy(),'boundary':'periodic'}
