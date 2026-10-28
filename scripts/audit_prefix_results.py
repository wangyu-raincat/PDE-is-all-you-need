#!/usr/bin/env python3
"""Independently audit a completed prefix study; never fit, forecast or generate.

Run only after the final comparison has been completed and its inspection is
authorized. This script does not import study code or call its metric/gate
functions. It rederives term metrics and gates, but only reaggregates saved
prediction errors: prediction arrays are not saved by the study. NPZ payloads
are hashed and their x/t coordinates checked; u arrays are never loaded.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
FAMILIES = ('allen_cahn', 'low_viscosity_burgers', 'ks_unstable')
METHODS = ('weak_resolved', 'denoised_median', 'prefix_weak', 'prefix_median')
SAMPLINGS = ('full', 'coarse')
NOISES = {'clean': (0., 0), 'noise0p1_s71': (.001, 71),
          'noise0p1_s103': (.001, 103), 'noise1_s71': (.01, 71),
          'noise1_s103': (.01, 103)}
HORIZONS = {'allen_cahn': .5, 'low_viscosity_burgers': 1.5, 'ks_unstable': 20.}
SPACE = {'allen_cahn': 256, 'low_viscosity_burgers': 128, 'ks_unstable': 128}
TERMS = ('1', 'u', 'u^2', 'u^3', 'u_x', 'u_xx', 'u_xxx', 'u_xxxx', 'u*u_x')
BASELINE = '0b32f670841452a5ecc7e18a2c82c689bd14da59'
METHOD_COMMIT = '51c151dea5b34708f27117efcc6e690cdcbbd298'
EXPECTED_IDS = {'--'.join(v) for v in itertools.product(FAMILIES, SAMPLINGS, NOISES, METHODS)}


class AuditError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise AuditError(message)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def finite(value):
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value) and value >= 0


def close(actual, expected, label):
    require(finite(actual) and finite(expected) and math.isclose(actual, expected, rel_tol=2e-11, abs_tol=2e-13),
            f'{label}: {actual!r} != {expected!r}')


def metadata(method):
    prefix = method.startswith('prefix_')
    predictor = ('prefix_aligned_resolved_factor4' if prefix else
                 'denoised_resolved_factor4' if method == 'denoised_median' else 'resolved_factor4')
    return {'structural_selection_predictor': predictor, 'coefficient_selection_predictor': predictor,
            'final_predictor': predictor, 'initial_filter': 'prefix_propagate_average' if prefix else
            'spectral_hard' if method == 'denoised_median' else 'none',
            'conditioning': 'observed_prefix' if prefix else 'last_frame_only'}


def coefficients(items):
    result = {}
    seen = set()
    for item in items:
        name, value = item['name'], item['coefficient']
        require(name in TERMS and name not in seen, 'Unknown or repeated PDE term')
        require(not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value),
                'Nonfinite PDE coefficient')
        seen.add(name)
        if value != 0:
            result[name] = value
    return result


def recovery(found_terms, reference_terms):
    found, truth = coefficients(found_terms), coefficients(reference_terms)
    require(bool(truth), 'Empty reference PDE')
    found_set, truth_set = set(found), set(truth)
    common = len(found_set & truth_set)
    precision = common / len(found_set) if found_set else 0.
    recall = common / len(truth_set)
    rows = []
    for name in TERMS:
        if name in found_set | truth_set:
            error = abs(found.get(name, 0.) - truth.get(name, 0.))
            rows.append({'name': name, 'reference': truth.get(name, 0.), 'discovered': found.get(name, 0.),
                         'absolute_error': error, 'relative_error': error / abs(truth[name]) if name in truth else None})
    return {'exact_support': found_set == truth_set, 'precision': precision, 'recall': recall,
            'f1': 2 * precision * recall / (precision + recall) if precision + recall else 0.,
            'true_positive_terms': sorted(found_set & truth_set), 'missing_terms': sorted(truth_set - found_set),
            'spurious_terms': sorted(found_set - truth_set), 'coefficient_comparison': rows,
            'coefficient_relative_l2': math.sqrt(sum(r['absolute_error'] ** 2 for r in rows) /
                                               sum(c ** 2 for c in truth.values()))}


def compare_recovery(saved, computed, label):
    for key in ('exact_support', 'true_positive_terms', 'missing_terms', 'spurious_terms'):
        require(saved[key] == computed[key], f'{label}: incorrect {key}')
    for key in ('precision', 'recall', 'f1', 'coefficient_relative_l2'):
        close(saved[key], computed[key], f'{label}:{key}')
    require(len(saved['coefficient_comparison']) == len(computed['coefficient_comparison']), label + ': term-row count')
    for a, b in zip(saved['coefficient_comparison'], computed['coefficient_comparison']):
        for key in ('name', 'reference', 'discovered'):
            require(a[key] == b[key], f'{label}: incorrect term {key}')
        close(a['absolute_error'], b['absolute_error'], label + ':absolute_error')
        if b['relative_error'] is None:
            require(a['relative_error'] is None, label + ': spurious relative error')
        else:
            close(a['relative_error'], b['relative_error'], label + ':relative_error')


def verify_digest_map(root, mapping, label):
    require(bool(mapping), label + ': empty hash map')
    for relative, digest in mapping.items():
        path = (root / relative).resolve()
        require(path.is_relative_to(root.resolve()), label + ': path escapes root')
        require(path.is_file() and sha(path) == digest, f'{label}: changed/missing {relative}')


def check_selection(model):
    candidates = model['candidates']
    ids = [c['candidate_id'] for c in candidates]
    require(len(ids) == len(set(ids)), model['id'] + ': duplicate candidates')
    eligible = [c for c in candidates if c.get('eligible')]
    for candidate in eligible:
        scores = candidate['development']['per_trajectory']
        require(candidate.get('fit_success') is True and len(scores) == 2, model['id'] + ': development count')
        require(candidate['support_size'] == len(coefficients(candidate['terms'])), model['id'] + ': candidate support')
        require(all(finite(s['rmse']) for s in scores), model['id'] + ': invalid development error')
        close(candidate['development']['rmse'], math.sqrt(statistics.mean(s['rmse'] ** 2 for s in scores)),
              model['id'] + ': development aggregate')
    if not eligible:
        require(model['selected'] is None, model['id'] + ': selected ineligible candidate')
    else:
        threshold = min(c['development']['rmse'] for c in eligible) * 1.05 + 1e-12
        pool = [c for c in eligible if c['development']['rmse'] <= threshold]
        best = min(pool, key=lambda c: (c['support_size'], c['development']['rmse'], c['candidate_id']))
        require(model['selected'] == best, model['id'] + ': fixed selector mismatch')
        supports = [list(s) for s in sorted({tuple(sorted(t['name'] for t in c['terms'])) for c in pool})]
        require(model['near_optimal_supports'] == supports, model['id'] + ': near-optimal supports mismatch')
    return len(eligible)


def independently_compute_gate(rows, quality):
    by = {r['id']: r for r in rows}
    maxima = {f: {m: max(v['relative_error'] for r in rows if r['family'] == f and r['method'] == m
                        for v in r['recovery']['coefficient_comparison'] if v['reference'] != 0)
                   for m in METHODS} for f in FAMILIES}
    checks = [{'name': 'source_quality', 'passed': bool(quality)}]
    targets = []
    for row in rows:
        if row['method'] != 'prefix_median':
            continue
        value, identity = row['recovery'], row['id']
        checks.append({'name': identity + ':exact_structure', 'passed': value['exact_support'] and
                       not value['missing_terms'] and not value['spurious_terms']})
        errors = [v['relative_error'] for v in value['coefficient_comparison'] if v['reference'] != 0]
        limit = .25 if row['noise_level'] == .01 else .10
        checks.append({'name': identity + ':each_true_term_accuracy', 'passed': bool(errors) and all(v <= limit for v in errors)})
        target = row['forecast_failures'] == 0 and finite(row['mean_clean_relative_l2']) and row['mean_clean_relative_l2'] <= .05
        targets.append(target)
        checks.append({'name': identity + ':mean_prediction_at_most_5pct', 'passed': target})
        for other in ('weak_resolved', 'denoised_median'):
            key = '--'.join((row['family'], row['sampling'], row['condition'], other))
            checks.append({'name': identity + ':vs_' + other + ':coefficient', 'passed':
                           value['coefficient_relative_l2'] <= max(1.25 * by[key]['recovery']['coefficient_relative_l2'], .01)})
    for family in FAMILIES:
        for other in ('weak_resolved', 'denoised_median'):
            checks.append({'name': f'{family}:vs_{other}:worst_true_term_nonregression',
                           'passed': maxima[family]['prefix_median'] <= max(1.25 * maxima[family][other], .01)})
    require(len(checks) == 157 and len(targets) == 30, 'Independent gate denominator incorrect')
    return checks, maxima, targets


def check_baseline(root, snapshot_path):
    git = lambda *args: subprocess.check_output(['git', '-C', str(root), *args], text=True).strip()
    require(git('rev-parse', BASELINE) == BASELINE, 'Baseline commit unavailable')
    require(git('rev-parse', METHOD_COMMIT) == METHOD_COMMIT, 'Method freeze commit unavailable')
    subprocess.run(['git', '-C', str(root), 'merge-base', '--is-ancestor', METHOD_COMMIT, 'HEAD'], check=True)
    # New study files and the appended ignore rules do not alter baseline science.
    changes = git('diff', '--name-status', BASELINE, 'HEAD').splitlines()
    require(all(line.startswith('A\t') or line == 'M\t.gitignore' for line in changes),
            'Existing baseline tracked files changed beyond the declared .gitignore addition')
    snapshot = read(snapshot_path)
    previous = Path(snapshot['directory'])
    verify_digest_map(previous, snapshot['files'], 'Preserved previous worktree')
    require(subprocess.check_output(['git', '-C', str(previous), 'rev-parse', 'HEAD'], text=True).strip() == BASELINE,
            'Previous worktree moved away from rollback baseline')
    return {'baseline_commit': BASELINE, 'method_commit': METHOD_COMMIT,
            'snapshot_sha256': sha(snapshot_path), 'previous_directory': str(previous),
            'previous_files_verified': len(snapshot['files']), 'existing_tracked_files_changed': ['.gitignore']}


def audit(root, output, snapshot):
    study, data = root / 'prefix_study', root / 'prefix_study' / 'data'
    # Fail before opening final references if the completed comparison is absent.
    require((output / 'comparison.json').is_file(), 'No completed comparison; do not inspect partial final data')
    report, index, prepared = read(output / 'comparison.json'), read(output / 'freeze_index.json'), read(data / 'prepared.json')
    protocol, references = read(study / 'protocol.json'), read(study / 'references.json')['references']
    require(protocol['variant_menu_status'] == 'frozen', 'Protocol not frozen')
    require(index['protocol'] == prepared['protocol'] == protocol, 'Protocol differs across preparation/freeze')
    require(index['preparation_sha256'] == sha(data / 'prepared.json'), 'Preparation receipt changed')
    require(report['freeze_index_sha256'] == sha(output / 'freeze_index.json'), 'Global freeze changed')
    for key in ('sources', 'input_hashes', 'quality', 'training_rms', 'all_source_quality_passed'):
        require(index[key] == prepared[key], 'Freeze/prepared mismatch: ' + key)
    expected_sources = {str(p.relative_to(root)) for p in study.glob('*.py')} | {
        'prefix_study/protocol.json', 'prefix_study/references.json',
        'challenge_study/data.py', 'challenge_study/generators.py', 'l1_study/engine.py',
        'weak_study/engine.py', 'weak_study/data.py', 'weak_study/generators.py',
        'resolved_study/forecast.py', 'resolved_study/selection.py', 'fusion_study/engine.py',
        'five_study/engine.py', 'five_study/initial.py', 'five_study/forecast.py',
        'pde_discovery/generic_engine.py', 'pde_discovery/generic_experiment.py'}
    expected_inputs = {str(p.relative_to(root)) for p in (data / 'clean').glob('*/*/*.npz')} | {
        str(p.relative_to(root)) for p in (data / 'raw').glob('*.mat')} | {'prefix_study/data/calibration.json'}
    require(set(index['sources']) == expected_sources, 'Source dependency inventory incomplete')
    require(set(index['input_hashes']) == expected_inputs, 'Prepared input inventory incomplete')
    verify_digest_map(root, index['sources'], 'Frozen sources')
    verify_digest_map(root, index['input_hashes'], 'Prepared inputs')
    require(index['all_120_frozen_before_final'] is True and index['synthetic_final_generated'] is False,
            'Missing pre-final freeze declaration')
    require(prepared['synthetic_final_generated'] is False, 'Preparation contains final generation')
    require(report['all_declared_models_reported'] is True and report['final_scores_used_for_tuning'] is False,
            'Missing isolation/completeness declaration')
    require(report['all_data_numerical_not_real_measurement'] is True, 'Synthetic scope declaration missing')
    expected_target = protocol['target_protocol']
    require(report['prediction_target_protocol'] == expected_target == protocol['gate']['target'], 'Target protocol mismatch')
    require(expected_target['relative_l2_limit'] == .05 and expected_target['condition_count'] == 30 and
            expected_target['trajectories_per_condition'] == 3 and expected_target['individual_trajectory_limit'] is None and
            expected_target['forecast_duration'] == HORIZONS, 'Declared target differs from authorized design')
    rows = report['records']
    for label, items in [('result', rows), ('freeze', index['records'])]:
        ids = [r['id'] for r in items]
        require(len(ids) == 120 and len(set(ids)) == 120 and set(ids) == EXPECTED_IDS, label + ': incorrect 120 configuration set')
    frozen = {r['id']: r for r in index['records']}
    require({p.name for p in (output / 'models').glob('*.json')} == {key + '.json' for key in EXPECTED_IDS},
            'Model directory has missing or undeclared models')
    eligibility_count = 0
    computed_rows = []
    all_slots = successful = 0
    for row in rows:
        identity = row['id']
        require(identity == '--'.join(row[k] for k in ('family', 'sampling', 'condition', 'method')), 'Result identity mismatch')
        record = frozen[identity]
        require(record['file'] == 'models/' + identity + '.json', 'Unexpected frozen model path')
        model_path = output / record['file']
        require(sha(model_path) == record['sha256'] == row['frozen_model_sha256'], identity + ': model hash mismatch')
        model = read(model_path)
        for key in ('sources', 'input_hashes'):
            require(model[key] == index[key], identity + ': dependency mismatch')
        for flag in ('reference_used_for_selection', 'clean_future_used_for_selection', 'final_loaded_during_selection'):
            require(model[flag] is False, identity + ': isolation flag invalid')
        for key in ('id', 'family', 'sampling', 'condition', 'method', 'noise_level', 'noise_seed',
                    'source_quality_passed', 'training_trajectories', 'development_trajectories',
                    'training_rms_for_corruption_only'):
            require(row[key] == model[key], identity + ': result/model ' + key)
        for key, value in metadata(row['method']).items():
            require(row[key] == model[key] == value, identity + ': conditioning/predictor ' + key)
        require((row['noise_level'], row['noise_seed']) == NOISES[row['condition']], identity + ': noise mismatch')
        source_ok = index['quality'][row['family']]['status'] == 'passed'
        require(row['source_quality_passed'] is source_ok, identity + ': source status mismatch')
        require(row['training_trajectories'] == (3 if source_ok else 0) and
                row['development_trajectories'] == (2 if source_ok else 0), identity + ': role count mismatch')
        require(model['training_rms_for_corruption_only'] == index['training_rms'][row['family']], identity + ': scale changed')
        eligibility_count += check_selection(model)
        selected = model['selected']
        require(row['terms'] == (selected['terms'] if selected else []), identity + ': final terms differ from frozen selection')
        require(row['selected'] is (selected is not None), identity + ': selection status mismatch')
        metrics = recovery(row['terms'], references[row['family']])
        compare_recovery(row['recovery'], metrics, identity)
        require(row['recovery']['reference_terms'] == references[row['family']], identity + ': reference mismatch')
        forecasts = row['forecasts']
        require(len(forecasts) == 3, identity + ': failed/absent trajectory removed from denominator')
        failures = sum(f['status'] != 'evaluated' for f in forecasts)
        require(row['forecast_failures'] == failures, identity + ': failure count mismatch')
        all_slots += len(forecasts)
        successful += len(forecasts) - failures
        stride = 1 if row['sampling'] == 'full' else 2
        for f in forecasts:
            if f['status'] != 'evaluated':
                require(f['status'] == 'failed' and bool(f.get('reason')), identity + ': unexplained failed slot')
                continue
            close(f['forecast_duration'], HORIZONS[row['family']], identity + ': full future horizon')
            expected_frames = 11 if stride == 1 else 6
            require(f['prefix_frames_used'] == (expected_frames if row['method'].startswith('prefix_') else 1), identity + ': prefix count')
            require(f['conditioning'] == metadata(row['method'])['conditioning'], identity + ': forecast conditioning')
            require(f['initial_is_noisy_input'] is True, identity + ': initial observation origin')
            for target in ('clean', 'noisy'):
                m = f[target]
                require(m['excluded_initial_frame'] is True and m['scored_frames'] == 100 // stride and
                        m['space_points'] == SPACE[row['family']] // stride, identity + ': scored grid/horizon')
                require(all(finite(m[k]) for k in ('rmse', 'mae', 'max_absolute_error', 'relative_l2')), identity + ': invalid trajectory metrics')
            if row['method'].startswith('prefix_'):
                d = f['initial_denoise_diagnostics']
                require(d['future_observations_used'] is False and d['equal_frame_weights'] is True and
                        d['independent_estimates_claimed'] is False and d['prefix_frames'] == expected_frames,
                        identity + ': prefix diagnostics mismatch')
                close(d['prefix_duration'], .1 * HORIZONS[row['family']], identity + ': history duration')
                close(d['forecast_duration'], HORIZONS[row['family']], identity + ': diagnostic future duration')
        derived = dict(row, recovery=metrics)
        for target in ('clean', 'noisy'):
            key = 'mean_' + target + '_relative_l2'
            if failures:
                require(row[key] is None, identity + ': successful-only mean hides failures')
                derived[key] = None
            else:
                derived[key] = statistics.mean(f[target]['relative_l2'] for f in forecasts)
                close(row[key], derived[key], identity + ':' + key)
        computed_rows.append(derived)
    require(all_slots == 360, 'Incomplete 360 forecast slots')
    require(report['configurations_with_forecast_failure'] == sum(r['forecast_failures'] > 0 for r in rows), 'Aggregate failure mismatch')
    require(report['prediction_above_5pct_all_arms'] == sum(r['mean_clean_relative_l2'] is not None and
            r['mean_clean_relative_l2'] > .05 for r in computed_rows), 'Above-target aggregate mismatch')

    final_failures = report['final_reference_failures']
    require(set(final_failures) == set(FAMILIES) and all(len(v) == 3 for v in final_failures.values()), 'Final source slot denominator')
    require(all(error is None or isinstance(error, str) and error for errors in final_failures.values() for error in errors),
            'Invalid final source failure evidence')
    for row in rows:
        for slot, error in enumerate(final_failures[row['family']]):
            if error is not None:
                require(row['forecasts'][slot]['status'] == 'failed', row['id'] + ': absent final source scored as success')
    expected_final = {f'final_reference/{f}/{i}.npz' for f in FAMILIES for i in range(3) if final_failures[f][i] is None}
    require(set(report['final_reference_sha256']) == expected_final, 'Final hash map does not retain source-failure slots')
    require({str(p.relative_to(output)) for p in (output / 'final_reference').rglob('*.npz')} == expected_final,
            'Unexpected/missing final reference file')
    if expected_final:
        verify_digest_map(output, report['final_reference_sha256'], 'Final reference')
    for relative in sorted(expected_final):
        family = Path(relative).parts[1]
        with np.load(output / relative, allow_pickle=False) as field:
            require(set(field.files) == {'x', 't', 'u'}, relative + ': unexpected array names')
            x, t = field['x'], field['t']  # Deliberately do not access the future field u.
            require(x.shape == (SPACE[family],) and t.shape == (111,) and np.isfinite(x).all() and
                    np.isfinite(t).all() and np.all(np.diff(x) > 0) and np.all(np.diff(t) > 0), relative + ': coordinates')
            close(float(t[10] - t[0]), .1 * HORIZONS[family], relative + ': prefix duration')
            close(float(t[-1] - t[10]), HORIZONS[family], relative + ': full future duration')
    quality = all(v['status'] == 'passed' for v in index['quality'].values()) and len(expected_final) == 9
    require(index['all_source_quality_passed'] is all(v['status'] == 'passed' for v in index['quality'].values()), 'Source aggregate mismatch')
    checks, maxima, targets = independently_compute_gate(computed_rows, quality)
    saved_gate = report['gate']
    require(saved_gate['checks'] == checks, 'Independent 157 gate checks differ from recorded gates')
    require(saved_gate['passed'] is all(c['passed'] for c in checks) and saved_gate['prediction_target_passed'] is all(targets), 'Gate conclusion mismatch')
    require(saved_gate['target_conditions_passed'] == sum(targets) and saved_gate['target_conditions_total'] == 30, 'Gate denominator mismatch')
    require(saved_gate['merged'] is False and saved_gate['regression_required_separately'] is True, 'Result claims unauthorized automatic merge')
    require(saved_gate['decision'] == ('eligible_for_review_only' if all(c['passed'] for c in checks) else 'keep_baseline_do_not_merge'), 'Gate decision mismatch')
    for f in FAMILIES:
        for m in METHODS:
            close(saved_gate['family_max_true_error'][f][m], maxima[f][m], 'Family coefficient-tail aggregate')
    baseline = check_baseline(root, snapshot)
    arms = {}
    for method in METHODS:
        group = [r for r in computed_rows if r['method'] == method]
        means = [r['mean_clean_relative_l2'] for r in group if r['mean_clean_relative_l2'] is not None]
        individual = [f['clean']['relative_l2'] for r in group for f in r['forecasts'] if f['status'] == 'evaluated']
        arms[method] = {'conditions': len(group), 'exact_support': sum(r['recovery']['exact_support'] for r in group),
            'spurious_terms': sum(len(r['recovery']['spurious_terms']) for r in group),
            'missing_terms': sum(len(r['recovery']['missing_terms']) for r in group),
            'median_coefficient_relative_l2': statistics.median(r['recovery']['coefficient_relative_l2'] for r in group),
            'maximum_true_term_relative_error': max(maxima[f][method] for f in FAMILIES),
            'condition_means_available': len(means), 'condition_means_at_most_5pct': sum(v <= .05 for v in means),
            'median_condition_mean_relative_l2': statistics.median(means) if means else None,
            'maximum_condition_mean_relative_l2': max(means) if means else None,
            'individual_forecasts_available': len(individual), 'individual_forecasts_above_5pct': sum(v > .05 for v in individual),
            'maximum_individual_relative_l2': max(individual) if individual else None,
            'forecast_failure_slots': sum(r['forecast_failures'] for r in group)}
    return {'audit_passed': True, 'comparison_sha256': sha(output / 'comparison.json'),
            'independent_gate_recalculation': True, 'raw_prediction_errors_recomputed': False,
            'runtime_order_independently_authenticated': False,
            'source_hashes_checked': len(index['sources']), 'prepared_input_hashes_checked': len(index['input_hashes']),
            'configurations': len(rows), 'forecast_slots': all_slots, 'successful_forecasts': successful,
            'final_source_slots': 9, 'final_source_hashes_verified': len(expected_final),
            'eligible_development_aggregates_checked': eligibility_count,
            'gate_checks_passed': sum(c['passed'] for c in checks), 'gate_checks_total': len(checks),
            'failed_gates': [c['name'] for c in checks if not c['passed']],
            'prediction_target_conditions_passed': sum(targets), 'prediction_target_conditions_total': 30,
            'promotion_gate_passed': all(c['passed'] for c in checks), 'baseline_preservation': baseline,
            'arms': arms, 'scope': 'Known synthetic families; conditional prefix prediction, unchanged full future horizon.',
            'limits': ['Saved per-trajectory prediction metrics were aggregated, not recomputed from raw predictions.',
                       'Hashes and code-order review do not independently authenticate runtime chronology.',
                       'Prefix methods use additional history; across-round cutoff-state distributions differ.',
                       'Three final trajectories per family are shared across conditions and arms; forecasts are not independent.']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--output', type=Path, help='Completed study output; default ROOT/prefix_study/results/first_run')
    parser.add_argument('--baseline-snapshot', type=Path, default=Path('/workspace/pde-prefix-previous-snapshot.json'))
    parser.add_argument('--write-json', type=Path, help='Optional new audit result file; existing files are never overwritten')
    args = parser.parse_args()
    output = args.output or args.root / 'prefix_study' / 'results' / 'first_run'
    try:
        result = audit(args.root.resolve(), output.resolve(), args.baseline_snapshot)
        text = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
        if args.write_json:
            with args.write_json.open('x') as handle:
                handle.write(text)
        print(text, end='')
    except (AuditError, OSError, KeyError, TypeError, ValueError, subprocess.CalledProcessError) as exc:
        print('AUDIT FAILED: ' + str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
