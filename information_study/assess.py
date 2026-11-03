"""Predeclared, offline assessment of the coefficient-information comparison.

No model, provider, field cache or training module is imported. The CLI reads
comparison.json, its independent audit and the source-binding freeze receipt;
it checks its own code/criteria hashes and publishes one assessment.json
exclusively. Passing all gates permits review only.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CRITERIA_FILE = Path(__file__).with_name('decision_protocol.json')
SOURCE_FILES = {'information_study/assess.py': Path(__file__).resolve(),
                'information_study/decision_protocol.json': CRITERIA_FILE}
FAMILIES = ('decay', 'diffusion', 'hyperdiffusion', 'advection_diffusion',
            'reaction_diffusion', 'burgers')
TERMS = ('1', 'u', 'u^2', 'u^3', 'u_x', 'u_xx', 'u_xxx', 'u_xxxx', 'u*u_x')
CONTEXT_KEY = '_assessment_context'
PAIRING_CHECK = 'paired_rich_first_acquisition'


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'),
                      ensure_ascii=False, allow_nan=False).encode('utf-8')


def _digest(payload):
    return hashlib.sha256(payload).hexdigest()


def _finite(value):
    return (type(value) in (int, float) and math.isfinite(value))


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _criteria(value):
    result = deepcopy(json.loads(CRITERIA_FILE.read_text()) if value is None else value)
    _require(isinstance(result, dict), 'Criteria must be a predeclared dictionary.')
    for key, count in (('indices', 18), ('policies', 6), ('noise_labels', 2), ('budgets', 5)):
        values = result.get(key)
        _require(isinstance(values, list) and len(values) == count and len(set(values)) == count,
                 f'Expected exactly {count} distinct {key}.')
    _require(all(type(i) is int and i >= 0 for i in result['indices']), 'Invalid parameter-task indices.')
    _require(all(sum(i % 6 == j for i in result['indices']) == 3 for j in range(6)),
             'Exactly three independent parameter tasks per each of six families are required.')
    _require(result['budgets'] == [5, 6, 7, 8, 9], 'The original five saved budgets must remain unchanged.')
    _require(result['noise_labels'] == ['noise1', 'noise3'], 'Expected noise1 and noise3.')
    _require(result['candidate'] in result['policies'], 'Candidate policy is absent.')
    comparators = result['required_improvement_comparators'] + result['no_worse_primary_comparators']
    _require(len(comparators) == 5 and set(comparators) == set(result['policies'])-{result['candidate']},
             'Every retained comparator must have exactly one primary comparison.')
    _require(len(result['required_improvement_comparators']) == 2,
             'Both predeclared improvement controls are required.')
    _require(result['primary_budget'] == 9 and result['primary_noise'] == 'noise1'
             and result['stress_noise'] == 'noise3' and result['structure_guard_budgets'] == [6, 9],
             'Primary/stress allocation or structure guard budgets changed.')
    for key in ('minimum_primary_relative_improvement', 'absolute_roundoff_tolerance',
                'terminal_tail_multiplier', 'stress_mean_multiplier', 'forecast_limit', 'cost_target_fraction'):
        _require(_finite(result.get(key)) and result[key] >= 0, 'Invalid criterion: '+key)
    _require(0 <= result['minimum_primary_relative_improvement'] <= 1
             and result['cost_target_fraction'] == .8, 'Invalid improvement or changed old cost target.')
    resampling = result['paired_resampling']
    _require(resampling['replicates'] == 2000 and resampling['seed'] == 970000
             and resampling['quantiles'] == [.025, .975], 'Predeclared descriptive resampling changed.')
    _require(result['automatic_merge'] is False, 'This assessment can never automatically merge.')
    return result


def _coefficient_map(terms, label):
    _require(isinstance(terms, list), label+' must be a list.')
    values = {}
    for term in terms:
        _require(isinstance(term, dict), label+' contains a malformed term.')
        name, value = term.get('name'), term.get('coefficient')
        _require(name in TERMS and name not in values and _finite(value), label+' has an invalid coefficient.')
        values[name] = float(value)
    return {name: value for name, value in values.items() if value != 0}


def _row_metrics(row, tolerance):
    """Recompute from saved coefficients, retaining a missing model's 100% loss."""
    recovery = row.get('recovery')
    _require(isinstance(recovery, dict), 'Missing saved recovery record.')
    found = _coefficient_map(row.get('terms') or [], 'Selected terms')
    truth = _coefficient_map(recovery.get('reference_terms'), 'Saved reference terms')
    _require(bool(truth), 'No nonzero true coefficient is available for assessment.')
    selected, true = set(found), set(truth)
    relative = {name: abs(found.get(name, 0.)-value)/abs(value) for name, value in truth.items()}
    _require(all(math.isfinite(value) for value in relative.values()), 'Coefficient-error calculation overflowed.')
    worst = max(relative.values())
    components = recovery.get('coefficient_comparison')
    _require(isinstance(components, list), 'Missing saved termwise coefficient errors.')
    compared = {}
    for component in components:
        _require(isinstance(component, dict), 'Malformed termwise coefficient error.')
        name, reference = component.get('name'), component.get('reference')
        _require(name in TERMS and name not in compared and _finite(reference), 'Invalid termwise coefficient identity.')
        compared[name] = component
        _require(math.isclose(reference, truth.get(name, 0.), rel_tol=0, abs_tol=tolerance),
                 'Saved termwise reference differs from saved true equation.')
    _require(set(compared) == selected | true, 'Termwise error inventory differs from selected/reference union.')
    for name, value in relative.items():
        saved = compared[name].get('relative_error')
        _require(_finite(saved) and saved >= 0 and math.isclose(saved, value, rel_tol=0, abs_tol=tolerance),
                 'Saved true-term relative error differs from coefficient arithmetic.')
    if 'worst_true_coefficient_relative_error' in row:
        _require(_finite(row['worst_true_coefficient_relative_error']) and math.isclose(
            row['worst_true_coefficient_relative_error'], worst, rel_tol=0, abs_tol=tolerance),
            'Saved worst true-term relative error differs.')
    exact, missing, extra = selected == true, sorted(true-selected), sorted(selected-true)
    _require(type(recovery.get('exact_support')) is bool and recovery['exact_support'] == exact,
             'Saved exact-support flag differs from coefficient supports.')
    _require(recovery.get('missing_terms') == missing and recovery.get('spurious_terms') == extra,
             'Saved missing/extra term inventory differs.')
    forecasts = row.get('forecasts')
    _require(isinstance(forecasts, list) and len(forecasts) == 3,
             'Exactly three forecast slots, including explicit failures, are required.')
    _require(all(isinstance(f, dict) and type(f.get('index')) is int for f in forecasts)
             and {f['index'] for f in forecasts} == {0, 1, 2}, 'Forecast identities are missing/duplicated.')
    values, failed = [], 0
    for forecast in forecasts:
        if forecast.get('status') == 'evaluated':
            _require(isinstance(forecast.get('clean'), dict), 'Missing saved forecast metric object.')
            value = forecast['clean'].get('relative_l2')
            _require(_finite(value) and value >= 0, 'Invalid saved forecast error.')
            values.append(float(value))
        else:
            _require(forecast.get('status') == 'failed', 'Unknown prediction-slot status.')
            failed += 1
    mean = None if failed else float(np.mean(values))
    _require(type(row.get('forecast_failures')) is int and row['forecast_failures'] == failed,
             'Saved prediction-failure count differs from the complete three slots.')
    saved_mean = row.get('mean_clean_relative_l2')
    _require(saved_mean is None if failed else _finite(saved_mean) and math.isclose(
        saved_mean, mean, rel_tol=0, abs_tol=tolerance), 'Saved prediction mean differs or omits a failure.')
    success = bool(exact and worst <= .1 and mean is not None and mean <= .05)
    _require(type(row.get('offline_success')) is bool and row['offline_success'] == success,
             'Saved offline success differs from the unchanged joint definition.')
    return {'worst_true_coefficient_relative_error': float(worst), 'true_term_errors': relative,
            'missing_model': not found, 'exact_support': exact, 'missing_terms': len(missing),
            'extra_terms': len(extra), 'forecast_mean': mean, 'forecast_failures': failed,
            'offline_success': success, 'charged_measurement_scalars': row.get('charged_measurement_scalars')}


def _audit_evidence(comparison, criteria):
    context = comparison.get(CONTEXT_KEY, {})
    if not isinstance(context, dict):
        context = {}
    audit = context.get('audit', {})
    if not isinstance(audit, dict):
        audit = {}
    checks = audit.get('checks', [])
    checks = checks if isinstance(checks, list) else []
    pairs = [check for check in checks if isinstance(check, dict) and check.get('name') == PAIRING_CHECK]
    payload = {key: value for key, value in comparison.items() if key != CONTEXT_KEY}
    try:
        content_matches = _digest(_canonical(payload)) == context.get('comparison_content_sha256')
    except (ValueError, TypeError):
        content_matches = False
    raw_sha = context.get('comparison_sha256')
    bound = (isinstance(raw_sha, str) and len(raw_sha) == 64
             and audit.get('comparison_sha256') == raw_sha and content_matches)
    freeze_sha = context.get('freeze_index_sha256')
    frozen_sources = context.get('frozen_sources', {})
    current_sources = context.get('current_source_sha256', {})
    source_binding = (isinstance(freeze_sha, str) and len(freeze_sha) == 64
        and comparison.get('freeze_index_sha256') == freeze_sha
        and isinstance(frozen_sources, dict) and isinstance(current_sources, dict)
        and all(isinstance(current_sources.get(name), str) and len(current_sources[name]) == 64
                and frozen_sources.get(name) == current_sources[name] for name in SOURCE_FILES)
        and context.get('criteria_content_sha256') == _digest(_canonical(criteria)))
    return {'audit_passed': audit.get('status') == 'passed' and bool(checks)
            and all(isinstance(c, dict) and c.get('status') == 'passed' for c in checks),
            'comparison_binding_passed': bool(bound),
            'paired_first_rich_passed': len(pairs) == 1 and pairs[0].get('status') == 'passed',
            'source_binding_passed': bool(source_binding),
            'comparison_sha256': raw_sha, 'audit_sha256': context.get('audit_sha256'),
            'freeze_index_sha256': freeze_sha,
            'comparison_claimed_freeze_index_sha256': comparison.get('freeze_index_sha256'),
            'frozen_assessment_sources': {name: frozen_sources.get(name) for name in SOURCE_FILES}
            if isinstance(frozen_sources, dict) else {},
            'current_assessment_sources': {name: current_sources.get(name) for name in SOURCE_FILES}
            if isinstance(current_sources, dict) else {},
            'first_rich_check_name': PAIRING_CHECK,
            'rich_opening_pairing': audit.get('summary', {}).get('rich_opening_pairing')
            if isinstance(audit.get('summary'), dict) else None}


def assess(comparison, criteria=None):
    """Return an offline decision without modifying the supplied comparison.

The optional reserved ``_assessment_context`` is supplied by the CLI from the
independent audit and exact comparison bytes. Without it, numerical summaries
remain available but audit gates fail closed. No successful-subset means exist.
"""
    criteria = _criteria(criteria)
    _require(isinstance(comparison, dict), 'Comparison must be a dictionary.')
    indices, policies = criteria['indices'], criteria['policies']
    noises, budgets = criteria['noise_labels'], criteria['budgets']
    tolerance = criteria['absolute_roundoff_tolerance']
    expected = {f'task-{i:02d}--{n}--{p}--budget-{b}': (i, n, p, b)
                for i in indices for n in noises for p in policies for b in budgets}
    records = comparison.get('records', [])
    records = records if isinstance(records, list) else []
    ids = [row.get('id') if isinstance(row, dict) else None for row in records]
    string_ids = [identity for identity in ids if isinstance(identity, str)]
    counts = Counter(string_ids)
    missing = sorted(set(expected)-set(counts))
    unexpected = sorted(set(counts)-set(expected))
    duplicate = sorted(identity for identity, count in counts.items() if count != 1)
    identity_errors, metric_errors, parsed = [], [], {}
    for row in records:
        if not isinstance(row, dict) or not isinstance(row.get('id'), str) or row['id'] not in expected:
            continue
        identity = row['id']; index, noise, policy, budget = expected[identity]
        if (type(row.get('spec_index')) is not int or row['spec_index'] != index
                or row.get('task_id') != f'task-{index:02d}--{noise}'
                or row.get('noise_label') != noise or row.get('policy') != policy
                or type(row.get('budget')) is not int or row['budget'] != budget
                or row.get('family') != FAMILIES[index % 6]):
            identity_errors.append(identity)
            continue
        if counts[identity] != 1:
            continue
        try:
            parsed[identity] = _row_metrics(row, tolerance)
        except (ValueError, TypeError, KeyError, OverflowError, ZeroDivisionError) as exc:
            metric_errors.append({'id': identity, 'reason': str(exc)})
    identity_complete = (len(records) == len(expected) == 1080 and len(string_ids) == len(records)
                         and not missing and not unexpected and not duplicate and not identity_errors)
    metrics_complete = identity_complete and not metric_errors and len(parsed) == len(expected)
    groups, by_group = [], {}
    for policy in policies:
        for noise in noises:
            for budget in budgets:
                keys = [f'task-{i:02d}--{noise}--{policy}--budget-{budget}' for i in indices]
                members = [parsed[k] for k in keys if k in parsed]
                complete = metrics_complete and len(members) == 18
                all_predictions = complete and all(m['forecast_mean'] is not None for m in members)
                group = {'policy': policy, 'noise_label': noise, 'budget': budget,
                    'expected_tasks': 18, 'present_valid_tasks': len(members), 'complete': complete,
                    'mean_worst_true_coefficient_error': float(np.mean([m['worst_true_coefficient_relative_error'] for m in members])) if complete else None,
                    'max_worst_true_coefficient_error': max(m['worst_true_coefficient_relative_error'] for m in members) if complete else None,
                    'exact_support_count': sum(m['exact_support'] for m in members) if complete else None,
                    'missing_term_total': sum(m['missing_terms'] for m in members) if complete else None,
                    'extra_term_total': sum(m['extra_terms'] for m in members) if complete else None,
                    'missing_model_count': sum(m['missing_model'] for m in members) if complete else None,
                    'prediction_complete': all_predictions,
                    'available_condition_forecast_means': sum(m['forecast_mean'] is not None for m in members),
                    'failed_prediction_slots': sum(m['forecast_failures'] for m in members) if complete else None,
                    'mean_condition_forecast_error': float(np.mean([m['forecast_mean'] for m in members])) if all_predictions else None,
                    'max_condition_forecast_error': max(m['forecast_mean'] for m in members) if all_predictions else None,
                    'offline_success_count': sum(m['offline_success'] for m in members) if complete else None}
                groups.append(group); by_group[policy, noise, budget] = group
    gates = []
    def gate(name, passed, **details):
        gates.append({'name': name, 'passed': bool(passed), **details})
    gate('complete_1080_declared_identities', identity_complete, expected=1080, present=len(records))
    gate('complete_saved_metric_arithmetic', metrics_complete, invalid_rows=len(metric_errors))
    evidence = _audit_evidence(comparison, criteria)
    for key in ('audit_passed', 'comparison_binding_passed', 'paired_first_rich_passed', 'source_binding_passed'):
        gate(key, evidence[key])
    candidate = criteria['candidate']; controls = criteria['required_improvement_comparators']
    candidate_primary = by_group[candidate, criteria['primary_noise'], 9]['mean_worst_true_coefficient_error']
    for control in controls + criteria['no_worse_primary_comparators']:
        value = by_group[control, criteria['primary_noise'], 9]['mean_worst_true_coefficient_error']
        relative_required = criteria['minimum_primary_relative_improvement'] if control in controls else 0.
        limit = None if value is None else (1.-relative_required)*value+tolerance
        zero_control = control in controls and value is not None and value <= tolerance
        gate('primary_vs_'+control, not zero_control and candidate_primary is not None and limit is not None and candidate_primary <= limit,
             candidate_mean=candidate_primary, comparator_mean=value, maximum_allowed=limit,
             minimum_relative_improvement=relative_required,
             observed_relative_improvement=None if value is None or value <= tolerance or candidate_primary is None else (value-candidate_primary)/value,
             reason='control_zero_no_relative_improvement_defined' if zero_control else None,
             zero_comparator_handling='required-improvement control <= roundoff tolerance fails; relative improvement remains undefined')
    for noise in noises:
        for budget in criteria['structure_guard_budgets']:
            c = by_group[candidate, noise, budget]
            for control in controls:
                other = by_group[control, noise, budget]
                for metric in ('exact_support_count', 'missing_term_total', 'extra_term_total'):
                    a, b = c[metric], other[metric]
                    ok = a is not None and b is not None and (a >= b if metric == 'exact_support_count' else a <= b)
                    gate(f'structure_{noise}_budget{budget}_{metric}_vs_{control}', ok,
                         candidate=a, comparator=b, direction='at_least' if metric == 'exact_support_count' else 'at_most')
        c = by_group[candidate, noise, 9]
        tails = [by_group[p, noise, 9]['max_worst_true_coefficient_error'] for p in controls]
        limit = criteria['terminal_tail_multiplier']*min(tails)+tolerance if all(v is not None for v in tails) else None
        gate('terminal_tail_'+noise, c['max_worst_true_coefficient_error'] is not None and limit is not None
             and c['max_worst_true_coefficient_error'] <= limit,
             candidate_maximum=c['max_worst_true_coefficient_error'], control_maxima=tails, maximum_allowed=limit)
        gate('complete_forecast_'+noise, c['prediction_complete'] and c['failed_prediction_slots'] == 0
             and c['max_condition_forecast_error'] <= criteria['forecast_limit'],
             complete_condition_means=c['available_condition_forecast_means'], required_conditions=18,
             failed_prediction_slots=c['failed_prediction_slots'], maximum_condition_mean=c['max_condition_forecast_error'],
             maximum_allowed=criteria['forecast_limit'], rule='every one of 18 three-trajectory means; zero failed slots')
    stress = by_group[candidate, criteria['stress_noise'], 9]['mean_worst_true_coefficient_error']
    means = [by_group[p, criteria['stress_noise'], 9]['mean_worst_true_coefficient_error'] for p in controls]
    limit = criteria['stress_mean_multiplier']*min(means)+tolerance if all(v is not None for v in means) else None
    gate('stress_mean', stress is not None and limit is not None and stress <= limit,
         candidate_mean=stress, control_means=means, maximum_allowed=limit)
    paired, intervals = [], []
    if metrics_complete:
        rng = np.random.default_rng(criteria['paired_resampling']['seed'])
        # Reuse the same paired, family-stratified draws for every comparator.
        draws = np.concatenate([rng.choice([j for j, i in enumerate(indices) if i % 6 == family],
            size=(2000, 3), replace=True) for family in range(6)], axis=1)
        for control in [p for p in policies if p != candidate]:
            primary_delta = []
            for index in indices:
                for noise in noises:
                    for budget in budgets:
                        base = f'task-{index:02d}--{noise}--'
                        a = parsed[base+candidate+f'--budget-{budget}']; b = parsed[base+control+f'--budget-{budget}']
                        delta = a['worst_true_coefficient_relative_error']-b['worst_true_coefficient_relative_error']
                        paired.append({'spec_index': index, 'family': FAMILIES[index % 6], 'noise_label': noise,
                            'budget': budget, 'comparator': control, 'candidate_coefficient_error': a['worst_true_coefficient_relative_error'],
                            'comparator_coefficient_error': b['worst_true_coefficient_relative_error'],
                            'coefficient_error_difference': delta, 'exact_support_difference': int(a['exact_support'])-int(b['exact_support']),
                            'missing_term_difference': a['missing_terms']-b['missing_terms'],
                            'extra_term_difference': a['extra_terms']-b['extra_terms'],
                            'forecast_mean_difference': None if a['forecast_mean'] is None or b['forecast_mean'] is None else a['forecast_mean']-b['forecast_mean'],
                            'offline_success_difference': int(a['offline_success'])-int(b['offline_success'])})
                        if noise == criteria['primary_noise'] and budget == 9:
                            primary_delta.append(delta)
            values = np.asarray(primary_delta)
            sampled = np.mean(values[draws], axis=1)
            intervals.append({'comparator': control, 'candidate_minus_comparator_mean': float(np.mean(values)),
                'percentile_interval': np.quantile(sampled, criteria['paired_resampling']['quantiles']).tolist(),
                'parameter_tasks': 18, 'families': 6, 'tasks_per_family': 3,
                'replicates': 2000, 'seed': 970000, 'same_paired_draws_across_comparators': True,
                'used_for_promotion': False, 'calibrated_probability_claimed': False,
                'limitation': criteria['paired_resampling']['interpretation']})
    cost_rows, first = [], {}
    for policy in policies:
        first[policy] = None
        for budget in budgets:
            keys = [f'task-{i:02d}--{n}--{policy}--budget-{budget}' for i in indices for n in noises]
            members = [parsed[k] for k in keys if k in parsed]
            complete = metrics_complete and len(members) == 36
            count = sum(m['offline_success'] for m in members) if complete else None
            fraction = count/36 if complete else None
            if first[policy] is None and fraction is not None and fraction >= criteria['cost_target_fraction']:
                first[policy] = budget
            charges = [m['charged_measurement_scalars'] for m in members]
            cost_rows.append({'policy': policy, 'budget': budget, 'expected_conditions': 36,
                'complete': complete, 'joint_success_count': count, 'joint_success_fraction': fraction,
                'total_declared_experiment_attempts': 36*budget,
                'total_charged_measurement_scalars': sum(charges) if complete and all(type(v) is int and v >= 0 for v in charges) else None})
    passed = all(g['passed'] for g in gates)
    return {'schema_version': 1, 'decision': criteria['all_gates_passed_decision'] if passed else criteria['any_gate_failed_decision'],
        'all_gates_passed': passed, 'automatic_merge': False, 'candidate': candidate,
        'gates': gates, 'groups': groups, 'paired_condition_deltas': paired,
        'primary_paired_intervals': intervals, 'paired_intervals_available': metrics_complete,
        'costs': {'target_fraction': .8, 'unchanged_joint_success_definition': 'exact support AND maximum true-term coefficient error<=10% AND complete three-trajectory mean prediction error<=5%',
            'denominator_per_policy_budget': 36, 'rows': cost_rows, 'first_budget_at_target': first,
            'candidate_fraction_saved_vs': {p: None if first[candidate] is None or first[p] is None else 1-first[candidate]/first[p] for p in policies if p != candidate},
            'used_for_promotion': False, 'online_stopping': False, 'success_target_changed': False},
        'completeness': {'expected_conditions': 36, 'expected_policies': 6, 'expected_budgets': 5,
            'expected_records': 1080, 'present_records': len(records), 'identity_complete': identity_complete,
            'metrics_complete': metrics_complete, 'missing_ids': missing, 'unexpected_ids': unexpected,
            'duplicate_ids': duplicate, 'invalid_identity_rows': identity_errors, 'invalid_metric_rows': metric_errors,
            'means_over_successful_subsets': False},
        'audit_evidence': evidence, 'criteria': criteria, 'criteria_content_sha256': _digest(_canonical(criteria)),
        'read_only_comparison': True, 'model_fitting_performed': False, 'final_fields_loaded': False,
        'scope': 'Offline predeclared assessment of controlled synthetic tasks; no real-world or Huawei integration claim.',
        'old_KS_failure_fixed': False}


def _write_exclusive(path, value):
    """Publish only a complete file; an existing destination is never replaced."""
    payload = (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n').encode('utf-8')
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix='.assessment-', dir=path.parent)
    try:
        with os.fdopen(descriptor, 'wb') as handle:
            handle.write(payload); handle.flush(); os.fsync(handle.fileno())
        os.link(name, path)
    finally:
        os.unlink(name)


def main(argv=None):
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'information_study/results/first_run')
    args = parser.parse_args(argv)
    output = args.output.resolve()
    comparison_bytes = (output/'comparison.json').read_bytes()
    comparison = json.loads(comparison_bytes)
    _require(isinstance(comparison, dict) and CONTEXT_KEY not in comparison,
             'Comparison contains a reserved assessment context or is not an object.')
    audit_path = output/'audit.json'
    audit_bytes = audit_path.read_bytes() if audit_path.exists() else None
    audit = json.loads(audit_bytes) if audit_bytes is not None else {}
    freeze_path = output/'freeze_index.json'
    freeze_bytes = freeze_path.read_bytes() if freeze_path.exists() else None
    freeze = json.loads(freeze_bytes) if freeze_bytes is not None else {}
    source_bytes = {name: path.read_bytes() for name, path in SOURCE_FILES.items()}
    criteria = json.loads(source_bytes['information_study/decision_protocol.json'])
    comparison[CONTEXT_KEY] = {'audit': audit, 'comparison_sha256': _digest(comparison_bytes),
        'comparison_content_sha256': _digest(_canonical(comparison)),
        'audit_sha256': _digest(audit_bytes) if audit_bytes is not None else None,
        'freeze_index_sha256': _digest(freeze_bytes) if freeze_bytes is not None else None,
        'frozen_sources': freeze.get('sources', {}) if isinstance(freeze, dict) else {},
        'current_source_sha256': {name: _digest(payload) for name, payload in source_bytes.items()},
        'criteria_content_sha256': _digest(_canonical(criteria))}
    result = assess(comparison, criteria)
    _require((output/'comparison.json').read_bytes() == comparison_bytes, 'Comparison changed during assessment.')
    _require(audit_bytes is None or audit_path.read_bytes() == audit_bytes, 'Audit changed during assessment.')
    _require(freeze_bytes is None or freeze_path.read_bytes() == freeze_bytes, 'Freeze receipt changed during assessment.')
    _require(all(path.read_bytes() == source_bytes[name] for name, path in SOURCE_FILES.items()),
             'Assessment code or predeclared criteria changed during assessment.')
    destination = output/'assessment.json'
    _write_exclusive(destination, result)
    print(json.dumps({'assessment': str(destination), 'decision': result['decision'],
                      'all_gates_passed': result['all_gates_passed'], 'automatic_merge': False}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
