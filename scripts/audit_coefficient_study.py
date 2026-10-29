#!/usr/bin/env python3
"""Independently audit completed, saved coefficient-study results.

No experiment modules are imported, no fields are generated, and no fitting or
prediction is performed. The comparison file must exist before any formal
input/model/final artifact is read. Hashes establish saved-artifact consistency;
they do not by themselves prove an unobserved process's execution history.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import statistics

FAMILIES = ('allen_cahn', 'low_viscosity_burgers', 'ks_unstable')
METHODS = ('prefix_baseline', 'stable_selection', 'noise_corrected', 'noise_stable')
SAMPLINGS = ('full', 'coarse')
CONDITIONS = {'clean': (0., 0), 'noise0p1_s79': (.001, 79),
              'noise0p1_s109': (.001, 109), 'noise1_s79': (.01, 79),
              'noise1_s109': (.01, 109)}
HORIZONS = {'allen_cahn': .5, 'low_viscosity_burgers': 1.5, 'ks_unstable': 20.}
SPACES = {'allen_cahn': 256, 'low_viscosity_burgers': 128, 'ks_unstable': 128}
TERMS = ('1', 'u', 'u^2', 'u^3', 'u_x', 'u_xx', 'u_xxx', 'u_xxxx', 'u*u_x')


def identity(family, sampling, condition, method):
    return '--'.join((family, sampling, condition, method))


def expected_ids():
    return {identity(f, s, n, m) for f in FAMILIES for s in SAMPLINGS
            for n in CONDITIONS for m in METHODS}


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def nonnegative(value):
    return finite(value) and value >= 0


def close(a, b):
    return finite(a) and finite(b) and math.isclose(a, b, rel_tol=2e-11, abs_tol=2e-13)


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def load(path):
    with Path(path).open(encoding='utf-8') as stream:
        return json.load(stream)


def safe_path(root, relative):
    if not isinstance(relative, str) or Path(relative).is_absolute():
        raise ValueError('Manifest paths must be relative.')
    root = Path(root).resolve()
    value = (root / relative).resolve()
    if not value.is_relative_to(root):
        raise ValueError('Manifest path escapes its artifact root.')
    return value


def full_forecast_mean(record):
    """Return None rather than average away any missing/failed trajectory."""
    rows = record.get('forecasts', [])
    if len(rows) != 3 or record.get('forecast_failures') != 0:
        return None
    values = [r.get('clean', {}).get('relative_l2') for r in rows]
    if any(r.get('status') != 'evaluated' for r in rows) or not all(nonnegative(v) for v in values):
        return None
    mean = statistics.fmean(values)
    saved = record.get('mean_clean_relative_l2')
    # Verify the arithmetic independently, then preserve the declared saved
    # value at an exact 5% boundary; alternate summation orders differ by ulps.
    return saved if close(mean, saved) else None


def rederive_recovery(terms, references):
    def mapping(items):
        result = {}
        for item in items:
            name, coefficient = item['name'], item['coefficient']
            if name not in TERMS or name in result or not finite(coefficient):
                raise ValueError('Unknown, duplicate or nonfinite coefficient.')
            result[name] = coefficient
        return {name: value for name, value in result.items() if value != 0}
    found, truth = mapping(terms), mapping(references)
    if not truth:
        raise ValueError('Reference support cannot be empty.')
    support = set(found)
    true_support = set(truth)
    comparisons = []
    for name in TERMS:
        if name in support | true_support:
            actual, target = found.get(name, 0.), truth.get(name, 0.)
            difference = abs(actual - target)
            comparisons.append({'name': name, 'reference': target, 'discovered': actual,
                'absolute_error': difference,
                'relative_error': difference / abs(target) if target else None})
    norm_error = math.sqrt(sum(row['absolute_error'] ** 2 for row in comparisons))
    norm_truth = math.sqrt(sum(value ** 2 for value in truth.values()))
    return {'exact_support': support == true_support, 'missing_terms': sorted(true_support-support),
        'spurious_terms': sorted(support-true_support), 'coefficient_relative_l2': norm_error/norm_truth,
        'coefficient_comparison': comparisons}


def rederive_gate(records, source_quality=True):
    """Independent expression of the prespecified 126 saved-result checks."""
    failure = {'passed': False, 'checks': [], 'prediction_target_passed': False,
        'coefficient_improvement_passed': False, 'candidate': 'noise_stable'}
    try:
        by_id = {row['id']: row for row in records}
        if len(records) != 120 or len(by_id) != 120 or set(by_id) != expected_ids():
            return {**failure, 'reason': 'Require exactly 120 unique declared configurations.'}
        tails = {}
        for family in FAMILIES:
            tails[family] = {}
            for method in METHODS:
                group = [r for r in records if r['family'] == family and r['method'] == method]
                errors = [[c['relative_error'] for c in r['recovery']['coefficient_comparison']
                           if c['reference'] != 0] for r in group]
                if len(group) != 10 or any(not e for e in errors) or not all(nonnegative(v) for e in errors for v in e):
                    raise ValueError('Incomplete or invalid coefficient-error rows.')
                tails[family][method] = max(v for e in errors for v in e)
        checks = [{'name': 'source_quality', 'passed': source_quality is True}]
        prediction_checks = []
        for family in FAMILIES:
            for sampling in SAMPLINGS:
                for condition, (level, _) in CONDITIONS.items():
                    name = identity(family, sampling, condition, 'noise_stable')
                    row = by_id[name]
                    recovery = row['recovery']
                    baseline = by_id[identity(family, sampling, condition, 'prefix_baseline')]['recovery']
                    true_errors = [c['relative_error'] for c in recovery['coefficient_comparison'] if c['reference'] != 0]
                    mean = full_forecast_mean(row)
                    prediction = mean is not None and mean <= .05
                    prediction_checks.append(prediction)
                    vector, base_vector = recovery['coefficient_relative_l2'], baseline['coefficient_relative_l2']
                    checks.extend([
                        {'name': name+':exact_structure', 'passed': recovery.get('exact_support') is True
                         and not recovery.get('spurious_terms') and not recovery.get('missing_terms')},
                        {'name': name+':each_true_term_accuracy', 'passed': bool(true_errors) and all(
                         nonnegative(v) and v <= (.25 if level == .01 else .10) for v in true_errors)},
                        {'name': name+':mean_prediction_at_most_5pct', 'passed': prediction},
                        {'name': name+':vs_prefix_baseline:coefficient', 'passed': nonnegative(vector)
                         and nonnegative(base_vector) and vector <= max(1.25*base_vector, .01)}])
        for family in FAMILIES:
            checks.append({'name': family+':vs_prefix_baseline:worst_true_term_nonregression',
                'passed': tails[family]['noise_stable'] <= max(1.25*tails[family]['prefix_baseline'], .01)})
        ac_before, ac_after = tails['allen_cahn']['prefix_baseline'], tails['allen_cahn']['noise_stable']
        global_before = max(tails[f]['prefix_baseline'] for f in FAMILIES)
        global_after = max(tails[f]['noise_stable'] for f in FAMILIES)
        gain = ac_before > 0 and ac_after <= .8*ac_before
        global_ok = global_after <= global_before
        checks.extend([
            {'name': 'allen_cahn:worst_true_term_relative_reduction_at_least_20pct', 'passed': gain},
            {'name': 'all_families:maximum_true_term_error_not_increased', 'passed': global_ok}])
        if len(checks) != 126:
            raise ValueError('Independent gate count is not 126.')
        return {'passed': all(c['passed'] for c in checks), 'checks': checks,
            'candidate': 'noise_stable', 'prediction_target_passed': all(prediction_checks),
            'coefficient_improvement_passed': gain and global_ok,
            'target_conditions_passed': sum(prediction_checks), 'target_conditions_total': 30,
            'family_max_true_error': tails, 'coefficient_improvement': {
                'ac_baseline_worst': ac_before, 'ac_candidate_worst': ac_after,
                'ac_relative_reduction': 1-ac_after/ac_before if ac_before else None,
                'global_baseline_worst': global_before, 'global_candidate_worst': global_after}}
    except (KeyError, ValueError, TypeError) as exc:
        return {**failure, 'reason': str(exc)}


def summarize(records):
    result = {}
    for method in METHODS:
        group = [r for r in records if r['method'] == method]
        good = [r for r in group if full_forecast_mean(r) is not None]
        mean_worst = max(good, key=lambda r: full_forecast_mean(r)) if good else None
        true_errors = [(c['relative_error'], r['id'], c['name']) for r in group
            for c in r['recovery']['coefficient_comparison'] if c['reference'] != 0]
        worst = max(true_errors) if true_errors else None
        result[method] = {'conditions': len(group),
            'exact_support': sum(r['recovery']['exact_support'] is True for r in group),
            'extra_terms': sum(len(r['recovery']['spurious_terms']) for r in group),
            'missing_terms': sum(len(r['recovery']['missing_terms']) for r in group),
            'complete_forecast_conditions': len(good),
            'conditions_mean_at_most_5pct': sum(full_forecast_mean(r) <= .05 for r in good),
            'worst_condition_mean_relative_l2': full_forecast_mean(mean_worst) if mean_worst else None,
            'worst_prediction_condition': mean_worst['id'] if mean_worst else None,
            'worst_individual_relative_l2': max((f['clean']['relative_l2'] for r in group
                for f in r['forecasts'] if f['status'] == 'evaluated'), default=None),
            'worst_true_coefficient_error': worst[0] if worst else None,
            'worst_coefficient_condition': worst[1] if worst else None,
            'worst_coefficient_term': worst[2] if worst else None,
            'max_coefficient_vector_error': max((r['recovery']['coefficient_relative_l2'] for r in group), default=None),
            'reliability_statuses': dict(Counter((r.get('reliability_selection') or {}).get('status', 'not_applicable') for r in group)),
            'fallback_conditions': [r['id'] for r in group if (r.get('reliability_selection') or {}).get('fallback_used') is True]}
    return result


def audit(root, output):
    root, output = Path(root).resolve(), Path(output).resolve()
    comparison_path = output/'comparison.json'
    # This must be the first artifact access. Never inspect an in-progress final.
    if not comparison_path.is_file():
        return {'passed': False, 'errors': ['comparison.json is not complete; no formal artifacts were read.']}
    errors = []
    checks_done = Counter()

    def require(value, message):
        if not value:
            errors.append(message)

    def same(actual, expected, name):
        if type(expected) in (float, int) and type(expected) is not bool:
            require(close(actual, expected), name+' numeric mismatch')
        elif isinstance(expected, dict):
            require(isinstance(actual, dict) and set(actual) == set(expected), name+' key mismatch')
            if isinstance(actual, dict):
                for key in expected:
                    same(actual.get(key), expected[key], name+'.'+key)
        elif isinstance(expected, list):
            require(isinstance(actual, list) and len(actual) == len(expected), name+' length mismatch')
            if isinstance(actual, list):
                for i, (a, b) in enumerate(zip(actual, expected)):
                    same(a, b, name+'.'+str(i))
        else:
            require(type(actual) is type(expected) and actual == expected, name+' mismatch')

    try:
        comparison = load(comparison_path)
        records = comparison.get('records', [])
        if len(records) != 120 or {r.get('id') for r in records} != expected_ids():
            return {'passed': False, 'errors': ['comparison.json lacks 120 distinct declared records; no other artifacts were read.']}
        index = load(output/'freeze_index.json')
        prepared_path = root/'coefficient_study/data/prepared.json'
        prepared = load(prepared_path)
        protocol = load(root/'coefficient_study/protocol.json')
        references = load(root/'coefficient_study/references.json')['references']
        require(digest(output/'freeze_index.json') == comparison['freeze_index_sha256'], 'Final report freeze-index hash mismatch')
        require(digest(prepared_path) == index['preparation_sha256'], 'Preparation hash mismatch')
        require(index.get('all_120_frozen_before_final') is True and index.get('synthetic_final_generated') is False,
                'Global pre-final freeze flags invalid')
        require(prepared.get('synthetic_final_generated') is False, 'Preparation claims final generation')
        for key in ('sources', 'input_hashes', 'quality', 'training_rms', 'all_source_quality_passed', 'protocol'):
            same(index[key], prepared[key], 'freeze/preparation.'+key)
        same(protocol, index['protocol'], 'frozen protocol')
        require(protocol.get('variant_menu_status') == 'frozen', 'Protocol is not frozen')
        require(len(index['sources']) == 25, 'Expected 25 scientific source hashes')
        require({'coefficient_study/protocol.json', 'coefficient_study/references.json',
                 'coefficient_study/generators.py'} <= set(index['sources']),
                'Protocol, reference coefficients or generator missing from source hash binding')
        for relative, expected in index['sources'].items():
            require(digest(safe_path(root, relative)) == expected, 'Scientific source hash mismatch: '+relative)
            checks_done['source_hashes'] += 1
        expected_inputs = {'coefficient_study/data/calibration.json'} | {
            f'coefficient_study/data/clean/{family}/{role}/{i}.npz' for family in FAMILIES
            for role, count in (('training', 3), ('development', 2)) for i in range(count)}
        require(set(index['input_hashes']) == expected_inputs, 'Expected 15 train/dev fields plus one calibration input')
        for relative, expected in index['input_hashes'].items():
            require(digest(safe_path(root, relative)) == expected, 'Prepared input hash mismatch: '+relative)
            checks_done['input_hashes'] += 1
        final_hashes = comparison['final_reference_sha256']
        expected_finals = {f'final_reference/{f}/{i}.npz' for f in FAMILIES for i in range(3)}
        require(set(final_hashes) == expected_finals, 'Expected exactly nine final reference files')
        for relative, expected in final_hashes.items():
            require(digest(safe_path(output, relative)) == expected, 'Final reference hash mismatch: '+relative)
            checks_done['final_hashes'] += 1
        require(set(index['input_hashes'].values()).isdisjoint(final_hashes.values()), 'Final bytes overlap prepared input bytes')
        require(not any('/final/' in key or 'final_reference/' in key for key in index['input_hashes']), 'Final role included in prepared inputs')
        require(comparison.get('final_scores_used_for_tuning') is False, 'Final tuning flag is not false')
        require(comparison.get('all_data_numerical_not_real_measurement') is True, 'Synthetic source scope not explicit')
        frozen = index['records']
        require(len(frozen) == 120 and {r['id'] for r in frozen} == expected_ids(), 'Frozen model identities incomplete')
        manifest = {r['id']: r for r in frozen}
        rebuilt = []
        for record in records:
            name = record['id']
            family, sampling, condition, method = (record[k] for k in ('family', 'sampling', 'condition', 'method'))
            require(name == identity(family, sampling, condition, method), name+': identity metadata mismatch')
            require((record['noise_level'], record['noise_seed']) == CONDITIONS[condition], name+': noise metadata mismatch')
            entry = manifest[name]
            require(entry['file'] == 'models/'+name+'.json', name+': unexpected model path')
            model_path = safe_path(output, entry['file'])
            require(digest(model_path) == entry['sha256'] == record['frozen_model_sha256'], name+': model hash mismatch')
            model = load(model_path)
            checks_done['model_hashes'] += 1
            same(model['sources'], index['sources'], name+': model source binding')
            same(model['input_hashes'], index['input_hashes'], name+': model input binding')
            for flag in ('reference_used_for_selection', 'clean_future_used_for_selection', 'final_loaded_during_selection'):
                require(model.get(flag) is False, name+': selection isolation flag '+flag)
            for role, count in (('training', 3), ('development', 2)):
                require(model[role+'_trajectories'] == count, name+': '+role+' trajectory count')
            require(model.get('source_quality_passed') is (index['quality'][family]['status'] == 'passed'),
                    name+': model source quality mismatch')
            same(model.get('training_rms_for_corruption_only'), index['training_rms'][family], name+': frozen corruption scale')
            same(record.get('training_rms_for_corruption_only'), index['training_rms'][family], name+': scored corruption scale')
            require(model.get('id') == name, name+': frozen identity mismatch')
            stable = method in ('stable_selection', 'noise_stable')
            corrected = method in ('noise_corrected', 'noise_stable')
            stage = {'conditioning': 'observed_prefix', 'initial_filter': 'prefix_propagate_average',
                'coefficient_noise_correction': corrected,
                'coefficient_selection': 'weak_residual_then_trajectory_stability' if stable else 'forecast_only',
                **{key: 'prefix_aligned_resolved_factor4' for key in (
                    'structural_selection_predictor', 'coefficient_selection_predictor', 'final_predictor')}}
            for key, value in stage.items():
                same(model.get(key), value, name+': frozen '+key)
                same(record.get(key), value, name+': scored '+key)
            selected = model['selected']
            terms = selected['terms'] if selected else []
            same(record['terms'], terms, name+': unchanged selected coefficients')
            same(record.get('reliability_selection'), model.get('reliability_selection'), name+': reliability receipt')
            _audit_selection(model, name, require, same)
            recovery = rederive_recovery(terms, references[family])
            for key, value in recovery.items():
                same(record['recovery'].get(key), value, name+': rederived '+key)
            independent = deepcopy(record)
            independent['recovery'] = recovery
            rebuilt.append(independent)
            forecasts = record['forecasts']
            require(len(forecasts) == 3, name+': final trajectory slots missing')
            failure_count = sum(f.get('status') != 'evaluated' for f in forecasts)
            require(record['forecast_failures'] == failure_count, name+': forecast failure count')
            if failure_count:
                require(record['mean_clean_relative_l2'] is None, name+': failure omitted from mean')
            else:
                require(full_forecast_mean(record) is not None, name+': invalid condition mean')
            for trajectory, forecast in enumerate(forecasts):
                if forecast.get('status') != 'evaluated':
                    require(bool(forecast.get('reason')), name+': failure reason missing')
                    continue
                prefix_count = 11 if sampling == 'full' else 6
                same(forecast.get('forecast_duration'), HORIZONS[family], name+': full horizon')
                require(forecast.get('prefix_frames_used') == prefix_count, name+': prefix length')
                require(forecast.get('conditioning') == 'observed_prefix', name+': forecast conditioning')
                diagnostic = forecast['initial_denoise_diagnostics']
                require(diagnostic.get('conditioning') == 'observed_prefix', name+': prefix diagnostic conditioning')
                for key in ('future_observations_used', 'independent_estimates_claimed'):
                    require(diagnostic.get(key) is False, name+': prefix '+key)
                for key in ('equal_frame_weights', 'candidate_PDE_used_for_state_estimation'):
                    require(diagnostic.get(key) is True, name+': prefix '+key)
                require(diagnostic.get('prefix_frames') == prefix_count and len(diagnostic['per_frame_filter']) == prefix_count,
                        name+': history diagnostic length')
                same(diagnostic.get('prefix_duration'), .1*HORIZONS[family], name+': historical duration')
                same(diagnostic.get('forecast_duration'), HORIZONS[family], name+': diagnostic horizon')
                for truth in ('clean', 'noisy'):
                    score = forecast[truth]
                    require(score.get('excluded_initial_frame') is True, name+': scored initial frame')
                    require(score.get('scored_frames') == (100 if sampling == 'full' else 50), name+': scored frame count')
                    require(score.get('space_points') == SPACES[family]//(1 if sampling == 'full' else 2), name+': space count')
                    require(nonnegative(score.get('relative_l2')), name+': invalid relative L2')
                checks_done['forecast_receipts'] += 1
        failures = comparison['final_reference_failures']
        require(set(failures) == set(FAMILIES) and all(len(v) == 3 for v in failures.values()),
                'Final reference failure slots incomplete')
        final_quality = all(v is None for rows in failures.values() for v in rows)
        gate = rederive_gate(rebuilt, index['all_source_quality_passed'] and final_quality)
        saved_gate = comparison['gate']
        same({c['name']: c['passed'] for c in saved_gate['checks']},
             {c['name']: c['passed'] for c in gate['checks']}, '126 independently rederived checks')
        for key in ('passed', 'prediction_target_passed', 'coefficient_improvement_passed', 'target_conditions_passed',
                    'target_conditions_total', 'family_max_true_error', 'coefficient_improvement'):
            same(saved_gate.get(key), gate.get(key), 'rederived gate '+key)
        require(saved_gate.get('candidate') == 'noise_stable' and saved_gate.get('merged') is False, 'Candidate changed or merged flag set')
        require(comparison.get('all_declared_models_reported') is True, 'Missing all-configurations receipt')
        require(comparison.get('configurations_with_forecast_failure') == sum(r['forecast_failures'] > 0 for r in records),
                'Forecast-failure aggregate mismatch')
        require(comparison.get('prediction_above_5pct_all_arms') == sum(
            full_forecast_mean(r) is not None and full_forecast_mean(r) > .05 for r in records), 'Prediction exceedance aggregate mismatch')
        result = {'passed': not errors, 'errors': errors, 'verified_counts': dict(checks_done),
            'gate': gate, 'aggregates': summarize(rebuilt), 'comparison_sha256': digest(comparison_path),
            'meaning_of_passed': 'Artifact integrity and independent recalculation agree; study promotion is gate.passed separately.',
            'read_numeric_arrays': False, 'generated_or_fitted_anything': False,
            'production_gate_called': False,
            'limits': ['Recomputes equation recovery and saved-score aggregation, not predictions from raw fields.',
                'Hashes and receipts verify artifact consistency; static design review supports execution-isolation claims.',
                'Same-family synthetic final trajectories are reused across conditions; not independent physical experiments.']}
        return result
    except (KeyError, ValueError, TypeError, OSError, ArithmeticError, IndexError, AttributeError) as exc:
        return {'passed': False, 'errors': errors+[type(exc).__name__+': '+str(exc)], 'verified_counts': dict(checks_done)}


def _audit_selection(model, name, require, same):
    selected = model['selected']
    candidates = {c['candidate_id']: c for c in model['candidates']}
    require(len(candidates) == len(model['candidates']), name+': duplicate candidate identities')
    if selected:
        require(selected['candidate_id'] in candidates, name+': selected candidate absent')
        same(selected['terms'], candidates[selected['candidate_id']]['terms'], name+': candidate coefficients')
    eligible = [c for c in candidates.values() if c.get('eligible')]
    near = [c for c in eligible if c['development']['rmse'] <= 1.05*min(
        p['development']['rmse'] for p in eligible)+1e-12] if eligible else []
    baseline = min(near, key=lambda c: (c['support_size'], c['development']['rmse'], c['candidate_id'])) if near else None
    if model['method'] not in ('stable_selection', 'noise_stable'):
        require((selected or {}).get('candidate_id') == (baseline or {}).get('candidate_id'), name+': prediction baseline selection')
        require(model.get('reliability_selection') is None, name+': unexpected reliability receipt on baseline arm')
        return
    receipt = model.get('reliability_selection')
    if receipt is None and selected is None:
        require(bool(model.get('failure_reason')), name+': missing model failure reason')
        return
    require(isinstance(receipt, dict), name+': missing reliability receipt')
    if not isinstance(receipt, dict):
        return
    for flag in ('reference_used', 'final_used', 'prediction_recomputed', 'confidence_interval_claimed'):
        require(receipt.get(flag) is False, name+': reliability '+flag)
    require(receipt.get('whole_trajectory_omissions') is True, name+': omission unit')
    require(receipt.get('baseline_candidate_id') == (baseline or {}).get('candidate_id'), name+': receipt baseline identity')
    status = receipt.get('status')
    if status == 'fallback_to_prediction':
        require(receipt.get('fallback_used') is True and bool(receipt.get('fallback_reason')), name+': missing explicit fallback')
        require((selected or {}).get('candidate_id') == (baseline or {}).get('candidate_id'), name+': fallback changed baseline')
    elif status == 'no_baseline_candidate':
        require(selected is None and baseline is None, name+': false no-candidate status')
    elif status == 'selected':
        require(receipt.get('fallback_used') is False and receipt.get('fallback_reason') is None, name+': contradictory success receipt')
        same(receipt.get('prediction_pool_ids'), [c['candidate_id'] for c in near], name+': prediction pool')
        minimum = min(c['support_size'] for c in near)
        pool = [c for c in near if c['support_size'] == minimum]
        same(receipt.get('minimum_support_pool_ids'), [c['candidate_id'] for c in pool], name+': support pool')
        diagnostics = receipt['candidate_diagnostics']
        require(set(diagnostics) == {c['candidate_id'] for c in pool}, name+': diagnostic pool')
        for candidate in pool:
            detail = diagnostics[candidate['candidate_id']]
            require(detail.get('status') == 'evaluated', name+': required diagnostic not evaluated')
            folds = detail['leave_out_folds']
            require(len(folds) == 3 and {f['omitted_training_trajectory'] for f in folds} == {0, 1, 2}, name+': LOO folds')
            names = detail['support']
            coefficients = {t['name']: t['coefficient'] for t in candidate['terms']}
            denominator = detail['coefficient_denominators']
            target_scale = receipt['common_training_target_rms']
            feature_scale = receipt['common_training_feature_rms']
            floors = [1e-6*target_scale/feature_scale[TERMS.index(term)] for term in names]
            same(detail['coefficient_floors'], floors, name+': normalized coefficient floors')
            same(denominator, [max(abs(coefficients[term]), floor) for term, floor in zip(names, floors)],
                 name+': stability denominators')
            require(len(detail['weak_per_development_trajectory']) == 2, name+': weak development trajectory count')
            per_term = [math.sqrt(statistics.fmean(((fold['coefficients'][i]-coefficients[term])/denominator[i])**2
                for fold in folds)) for i, term in enumerate(names)]
            same(detail['per_term_relative_omission_rms'], per_term, name+': rederived LOO displacement')
            same(detail['worst_relative_omission_rms'], max(per_term), name+': rederived LOO maximum')
            same(detail['weak_score'], math.sqrt(statistics.fmean(v*v for v in detail['weak_per_development_trajectory'])), name+': weak residual aggregation')
            if detail.get('median_reconstructed'):
                for fold in folds:
                    groups = fold['distinct_vector_groups']
                    vectors = [tuple(g['coefficients']) for g in groups]
                    aliases = [v for g in groups for v in g['variants']]
                    require(len(vectors) == len(set(vectors)) and len(aliases) == len(set(aliases)), name+': fold alias deduplication')
                    same(fold['coefficients'], [statistics.median(v[i] for v in vectors) for i in range(len(names))], name+': fold distinct-vector median')
        threshold = 1.05*min(diagnostics[c['candidate_id']]['weak_score'] for c in pool)+1e-12
        weak_pool = [c for c in pool if diagnostics[c['candidate_id']]['weak_score'] <= threshold]
        same(receipt.get('weak_pool_ids'), [c['candidate_id'] for c in weak_pool], name+': weak pool')
        winner = min(weak_pool, key=lambda c: (diagnostics[c['candidate_id']]['worst_relative_omission_rms'],
            diagnostics[c['candidate_id']]['weak_score'], c['development']['rmse'], c['candidate_id']))
        require(selected is not None and selected['candidate_id'] == winner['candidate_id'] == receipt.get('selected_candidate_id'),
                name+': reliability selection order mismatch')
    else:
        require(False, name+': unknown reliability status '+str(status))


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--output', type=Path)
    parser.add_argument('--write', type=Path, help='Optional audit JSON; scientific files are never modified.')
    args = parser.parse_args()
    result = audit(args.root, args.output or args.root/'coefficient_study/results/first_run')
    serialized = json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False)+'\n'
    if args.write:
        args.write.parent.mkdir(parents=True, exist_ok=True)
        args.write.write_text(serialized, encoding='utf-8')
    print(serialized, end='')
    raise SystemExit(0 if result['passed'] else 1)


if __name__ == '__main__':
    main()
