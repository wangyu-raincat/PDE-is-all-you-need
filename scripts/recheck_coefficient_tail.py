"""Reproduce each coefficient-study method/family's worst scored condition.

This is an integrity check on already evaluated data, never a new holdout or a
way to choose a model. All four arms use the same frozen prefix forecaster.
Only its exact frozen terms and the observed historical prefix enter that
forecaster. Relative L2 is recomputed directly with NumPy, excluding the initial
frame; this script does not call the study's forecast_metrics or score helpers.
No fitting, source generation, correction re-estimation or model edits occur.
"""
from pathlib import Path
import argparse
import math
import sys

import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from coefficient_study import data, run
from prefix_study import forecast

HORIZONS = {'allen_cahn': .5, 'low_viscosity_burgers': 1.5, 'ks_unstable': 20.}
PREFIX_FRAMES = {'full': 11, 'coarse': 6}
FUTURE_FRAMES = {'full': 101, 'coarse': 51}
RTOL = 2e-11
ATOL = 2e-13


def finite_nonnegative(value):
    return (not isinstance(value, bool) and isinstance(value, (int, float))
            and math.isfinite(value) and value >= 0)


def require_close(value, saved, label):
    if (not finite_nonnegative(value) or not finite_nonnegative(saved)
            or not np.isclose(value, saved, rtol=RTOL, atol=ATOL)):
        raise ValueError(f'{label}: reproduced {value!r} differs from saved {saved!r}.')


def recheck(output):
    output = Path(output)
    comparison = output/'comparison.json'
    if not comparison.is_file():
        raise FileNotFoundError('Completed comparison.json is required before any final-data recheck.')
    receipt_path = output/'tail_reproduction.json'
    if receipt_path.exists():
        raise RuntimeError('Tail receipt already exists; preserve the original verification record.')
    index = run.verify_freeze(output)
    index_sha = run.sha(output/'freeze_index.json')
    report_sha = run.sha(comparison)
    report = run.read(comparison)
    records = report['records']
    if not run.complete(records) or report.get('all_declared_models_reported') is not True:
        raise ValueError('Require all 120 unique scored configurations; do not omit failures.')
    if report.get('freeze_index_sha256') != index_sha:
        raise ValueError('Comparison does not refer to the verified model freeze.')
    frozen_hashes = {r['id']: r['sha256'] for r in index['records']}
    for result in records:
        if (result.get('forecast_failures') != 0 or len(result.get('forecasts', [])) != 3
                or not finite_nonnegative(result.get('mean_clean_relative_l2'))
                or any(f.get('status') != 'evaluated'
                       or not finite_nonnegative(f.get('clean', {}).get('relative_l2'))
                       for f in result.get('forecasts', []))):
            raise ValueError('Tail recheck requires the complete successful run; retain failure slots separately.')
        if result.get('frozen_model_sha256') != frozen_hashes[result['id']]:
            raise ValueError('Scored model digest differs from global freeze.')
    rows = []
    with threadpool_limits(limits=1):
        for family in run.FAMILIES:
            for method in run.METHODS:
                group = [r for r in records if r['family'] == family and r['method'] == method]
                if len(group) != 10:
                    raise ValueError('Each method/family must retain all ten declared conditions.')
                # Selection here identifies a verification case only. It never
                # changes the saved model, method, coefficient or promotion gate.
                result = max(group, key=lambda r: (r['mean_clean_relative_l2'], r['id']))
                model_path = output/'models'/f"{result['id']}.json"
                if run.sha(model_path) != result['frozen_model_sha256']:
                    raise ValueError('Frozen model changed.')
                model = run.read(model_path)
                run.validate_configuration(model, index['quality'], index['training_rms'])
                if (model['selected'] is None or model['selected']['terms'] != result['terms']
                        or model['final_predictor'] != 'prefix_aligned_resolved_factor4'
                        or model['conditioning'] != 'observed_prefix'):
                    raise ValueError('Frozen terms or common prefix predictor metadata differ.')
                terms = model['selected']['terms']
                sampling = model['sampling']
                errors = []
                for i in range(3):
                    final = output/'final_reference'/family/f'{i}.npz'
                    relative = str(final.relative_to(output))
                    if run.sha(final) != report['final_reference_sha256'][relative]:
                        raise ValueError('Final reference bytes changed.')
                    with np.load(final, allow_pickle=False) as archive:
                        clean = {k: archive[k].copy() for k in ('x', 't', 'u')}
                    observed = data.corrupt(clean, family, 'final', i, sampling,
                        model['noise_level'], model['noise_seed'], model['training_rms_for_corruption_only'])
                    horizon = HORIZONS[family]
                    if (len(observed['t']) != FUTURE_FRAMES[sampling]
                            or len(observed['prefix_t']) != PREFIX_FRAMES[sampling]
                            or not np.isclose(observed['t'][-1]-observed['t'][0], horizon, rtol=0, atol=1e-10)
                            or not np.isclose(observed['prefix_t'][-1]-observed['prefix_t'][0], .1*horizon,
                                              rtol=0, atol=1e-10)
                            or observed['t'][0] != observed['prefix_t'][-1]
                            or not np.array_equal(observed['u'][0], observed['prefix_u'][-1])):
                        raise ValueError('Historical information or full future horizon differs.')
                    prediction = forecast.forecast(terms, observed['x'], observed['t'],
                                                   observed['prefix_t'], observed['prefix_u'])
                    stride = 1 if sampling == 'full' else 2
                    truth = clean['u'][data.PREFIX_INTERVALS::stride, ::stride]
                    if prediction.shape != truth.shape or not np.isfinite(prediction).all():
                        raise ValueError('Recomputed forecast shape or finite-value check failed.')
                    denominator = float(np.linalg.norm(truth[1:]))
                    if not np.isfinite(denominator) or denominator <= 0:
                        raise ValueError('Relative L2 reference denominator must be positive and finite.')
                    error = float(np.linalg.norm(prediction[1:]-truth[1:])/denominator)
                    saved_forecast = result['forecasts'][i]
                    if (saved_forecast.get('prefix_frames_used') != PREFIX_FRAMES[sampling]
                            or saved_forecast.get('conditioning') != 'observed_prefix'
                            or not np.isclose(saved_forecast.get('forecast_duration', float('nan')),
                                              horizon, rtol=0, atol=1e-10)
                            or saved_forecast.get('clean', {}).get('excluded_initial_frame') is not True):
                        raise ValueError('Saved score conditioning or horizon metadata differs.')
                    require_close(error, saved_forecast['clean']['relative_l2'], result['id']+f':trajectory{i}')
                    errors.append(error)
                mean = float(np.mean(errors))
                require_close(mean, result['mean_clean_relative_l2'], result['id']+':mean')
                row = {'id': result['id'], 'family': family, 'method': method,
                       'per_trajectory_relative_l2': errors, 'mean_relative_l2': mean,
                       'saved_mean': result['mean_clean_relative_l2'],
                       'difference': mean-result['mean_clean_relative_l2'],
                       'forecast_duration': HORIZONS[family], 'prefix_frames': PREFIX_FRAMES[sampling],
                       'frozen_model_sha256': result['frozen_model_sha256']}
                rows.append(row)
                print(result['id'], mean, flush=True)
    run.verify_freeze(output)
    if run.sha(output/'freeze_index.json') != index_sha or run.sha(comparison) != report_sha:
        raise ValueError('Frozen index or original comparison changed during recheck.')
    for relative, digest in report['final_reference_sha256'].items():
        if run.sha(output/relative) != digest:
            raise ValueError('Final reference bytes changed during recheck.')
    if len(rows) != 12:
        raise ValueError('Require all twelve method/family verification cases.')
    receipt = {'passed': True, 'post_evaluation_reproduction_not_new_holdout': True,
        'models_rechecked': len(rows), 'forecasts_recomputed': 3*len(rows),
        'metric_recalculated_with_numpy': True, 'excluded_initial_frame': True,
        'no_fit_or_selection': True, 'no_model_selection_or_retuning': True,
        'selection_of_checks': 'worst saved condition for each family/method; verification only',
        'conditioning': 'observed prefix, 11 full or 6 coarse frames',
        'full_future_durations': HORIZONS, 'metric_tolerance': {'rtol': RTOL, 'atol': ATOL},
        'comparison_sha256': report_sha, 'freeze_index_sha256': index_sha, 'records': rows}
    run.write(receipt_path, receipt)
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--output', type=Path, default=run.OUTPUT)
    recheck(parser.parse_args().output)
