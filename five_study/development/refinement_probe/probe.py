"""One retrospective training-only coefficient refinement probe.

Uses the first quarter of three OLD observed training trajectories. Neither
development, reference coefficients nor final fields enter fit_coefficients.
The two OLD development trajectories are evaluated only after fit is saved.
No formal new data or old final fields are loaded/generated.
"""
from pathlib import Path
import hashlib
import json
import sys
import time
from copy import deepcopy
import numpy as np
from scipy.optimize import least_squares
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from five_study.forecast import forecast
from fusion_study import data
from pde_discovery.generic_experiment import forecast_metrics, reference_metrics

OUT = Path(__file__).resolve().parent
OLD = Path('/workspace/pde-discovery-fused-weak')
MODEL = ROOT/'five_study/development/retrospective/models/ks_unstable--coarse--noise1_s97--denoised_median.json'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n')


def fit_coefficients(training, initial_terms):
    """Same-support bounded coefficient fit using training observations only."""
    initial_terms = deepcopy(initial_terms)
    base = np.asarray([term['coefficient'] for term in initial_terms], float)
    if not np.isfinite(base).all() or np.any(base == 0):
        raise ValueError('Need finite nonzero initial coefficients.')
    windows = []
    for observation in training:
        x, t, u = [np.asarray(observation[key], float) for key in ('x','t','u')]
        end = t[0] + .25*(t[-1]-t[0])
        indices = np.flatnonzero(t <= end + 1e-12)
        if len(indices) < 3:
            raise ValueError('First quarter must contain at least three observations.')
        windows.append({'x':x.copy(), 't':t[indices].copy(), 'u':u[indices].copy()})
    observed_rms = float(np.sqrt(np.mean([np.mean(obs['u']**2) for obs in training])))
    if not np.isfinite(observed_rms) or observed_rms <= 1e-12:
        raise ValueError('Need observed temporal excitation.')
    calls = 0
    failures = []
    history = []

    def terms_at(delta):
        return [dict(name=term['name'], coefficient=float(value))
                for term, value in zip(initial_terms, base*(1+delta))]

    def residual(delta):
        nonlocal calls
        calls += 1
        pieces = []
        for window in windows:
            predicted = forecast(terms_at(delta), window['x'], window['t'], window['u'][0])
            difference = predicted[1:]-window['u'][1:]
            pieces.append(difference.ravel()/np.sqrt(len(windows)*difference.size)/observed_rms)
        values = np.concatenate(pieces)
        if not np.isfinite(values).all():
            raise FloatingPointError('Nonfinite training objective.')
        score = float(values@values)
        history.append({'call':calls, 'delta':delta.tolist(), 'normalized_equal_trajectory_mse':score})
        if calls == 1 or calls % 5 == 0:
            print(f'training residual call={calls} objective={score:.9g}', flush=True)
        return values

    start = time.perf_counter()
    before = residual(np.zeros_like(base))
    result = least_squares(residual, np.zeros_like(base), bounds=(-.25,.25),
                           max_nfev=20, jac='2-point', xtol=1e-6, ftol=1e-6, gtol=1e-6)
    singular = np.linalg.svd(result.jac, compute_uv=False)
    condition = float(singular[0]/singular[-1]) if singular[-1] > 0 else None
    rank = int(np.linalg.matrix_rank(result.jac))
    return {
        'terms':terms_at(result.x), 'initial_terms':initial_terms,
        'delta':result.x.tolist(), 'success':bool(result.success), 'status':int(result.status),
        'message':str(result.message), 'nfev':int(result.nfev),
        'njev':int(result.njev) if result.njev is not None else None,
        'objective_calls_including_finite_difference':calls,
        'before_objective':float(before@before), 'after_objective':float(result.fun@result.fun),
        'observed_training_rms_for_objective_only':observed_rms,
        'optimality':float(result.optimality), 'jacobian_condition':condition,
        'jacobian_rank':rank, 'jacobian_singular_values':singular.tolist(),
        'active_mask':result.active_mask.tolist(), 'elapsed_seconds':time.perf_counter()-start,
        'training_window_frames':[len(w['t']) for w in windows],
        'training_window_time_ranges':[[float(w['t'][0]),float(w['t'][-1])] for w in windows],
        'objective_history':history, 'objective_failures':failures,
        'protocol':{'time_fraction':.25, 'delta_bounds':[-.25,.25], 'max_nfev':20,
                    'xtol':1e-6, 'ftol':1e-6, 'gtol':1e-6, 'finite_difference':'2-point',
                    'residual':'future observed spatial field residual, equal trajectory MSE / observed training RMS squared',
                    'initialization':'filtered observed first frame, never later-window restarts',
                    'support_fixed':True, 'development_used_for_fit':False,
                    'reference_coefficients_used_for_fit':False, 'true_noise_used_for_fit':False},
    }


def main():
    if (OUT/'result.json').exists() or (OUT/'fitted.json').exists():
        raise RuntimeError('Preserve existing probe evidence.')
    observed_source_files = [ROOT/'five_study/forecast.py',ROOT/'five_study/initial.py',
        ROOT/'resolved_study/forecast.py',ROOT/'pde_discovery/generic_engine.py', Path(__file__).resolve()]
    source_hashes = {str(p.relative_to(ROOT)):sha(p) for p in observed_source_files}
    model = json.loads(MODEL.read_text())
    clean_paths = []

    def load_observations(role, count):
        if role not in ('training','development'):
            raise ValueError('No final fields allowed.')
        observations = []
        for index in range(count):
            path = OLD/'fusion_study/data/clean'/model['family']/role/f'{index}.npz'
            clean_paths.append(path)
            clean = data.read_clean(OLD/'fusion_study/data',model['family'],role,index)
            observations.append(data.corrupt(clean,model['family'],role,index,model['sampling'],
                model['noise_level'],model['noise_seed'],model['training_rms_for_corruption_only']))
        return observations

    training = load_observations('training',3)
    fit = fit_coefficients(training,model['selected']['terms'])
    assert all(sha(ROOT/path)==digest for path,digest in source_hashes.items())
    fit.update(source_hashes=source_hashes, source_model_sha256=sha(MODEL),
        input_training_hashes={str(p.relative_to(OLD)):sha(p) for p in clean_paths},
        retrospective=True, old_final_fields_loaded=False, new_fields_generated=False)
    write(OUT/'fitted.json',fit)
    print('Fit frozen before development read',flush=True)
    development = load_observations('development',2)
    scores = {}
    for label,terms in [('original',fit['initial_terms']),('refined',fit['terms'])]:
        individual = []
        for obs in development:
            prediction = forecast(terms,obs['x'],obs['t'],obs['u'][0])
            individual.append(forecast_metrics(prediction,obs['u']))
        scores[label] = {'per_trajectory':individual,
                         'mean_relative_l2':float(np.mean([x['relative_l2'] for x in individual])),
                         'equal_trajectory_rmse':float(np.sqrt(np.mean([x['rmse']**2 for x in individual])))}
    # Reference coefficients are used only after fitting and development scoring.
    references = json.loads((ROOT/'five_study/references.json').read_text())['references'][model['family']]
    reference_diagnostics = {label:reference_metrics(terms,references)
        for label,terms in [('original',fit['initial_terms']),('refined',fit['terms'])]}
    assert all(sha(ROOT/path)==digest for path,digest in source_hashes.items())
    report = {'fit':fit,'development':scores,'post_fit_reference_diagnostics':reference_diagnostics,
        'retrospective':True,'single_probe_no_hyperparameter_search':True,
        'old_final_fields_loaded':False,'formal_fields_generated':False,
        'input_hashes':{str(p.relative_to(OLD)):sha(p) for p in clean_paths},
        'fitted_json_sha256':sha(OUT/'fitted.json')}
    write(OUT/'result.json',report)
    print(json.dumps({k:fit[k] for k in ['terms','success','nfev','before_objective','after_objective',
                    'elapsed_seconds','jacobian_condition','jacobian_rank']},indent=2),flush=True)
    print(json.dumps(scores,indent=2),flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=1): main()
