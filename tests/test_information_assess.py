"""Manufactured assessment records only; no formal providers or result files."""
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from information_study import assess as evaluator


TRUTHS = ({'u': -.2}, {'u_xx': .1}, {'u_xxxx': -.01},
          {'u_x': -.5, 'u_xx': .1}, {'u': -.2, 'u_xx': .1}, {'u_xx': .1, 'u*u_x': -.5})


def set_metrics(row, error=.08, *, missing=False, extra=False, forecast=.01, failed_forecasts=0):
    truth = TRUTHS[row['spec_index'] % 6]
    found = {} if missing else {name: value*(1+error) for name, value in truth.items()}
    if extra:
        found['1'] = .01
    components = []
    for name in evaluator.TERMS:
        if name not in set(truth) | set(found):
            continue
        absolute = abs(found.get(name, 0)-truth.get(name, 0))
        components.append({'name': name, 'reference': truth.get(name, 0),
                           'discovered': found.get(name, 0), 'absolute_error': absolute,
                           'relative_error': absolute/abs(truth[name]) if name in truth else None})
    worst = max(c['relative_error'] for c in components if c['reference'] != 0)
    exact = set(found) == set(truth)
    mean = None if failed_forecasts else float(np.mean([forecast]*3))
    row.update(terms=[{'name': name, 'coefficient': value} for name, value in found.items()],
        recovery={'reference_terms': [{'name': name, 'coefficient': value} for name, value in truth.items()],
                  'coefficient_comparison': components, 'exact_support': exact,
                  'missing_terms': sorted(set(truth)-set(found)), 'spurious_terms': sorted(set(found)-set(truth))},
        worst_true_coefficient_relative_error=worst,
        forecasts=[{'index': i, 'status': 'failed', 'reason': 'manufactured failure'} if i < failed_forecasts
                   else {'index': i, 'status': 'evaluated', 'clean': {'relative_l2': forecast}} for i in range(3)],
        forecast_failures=failed_forecasts, mean_clean_relative_l2=mean,
        offline_success=bool(exact and worst <= .1 and mean is not None and mean <= .05))


def fixture():
    criteria = json.loads(evaluator.CRITERIA_FILE.read_text())
    rows = []
    for index in criteria['indices']:
        for noise in criteria['noise_labels']:
            for policy in criteria['policies']:
                for budget in criteria['budgets']:
                    row = {'id': f'task-{index}--{noise}--{policy}--budget-{budget}',
                           'task_id': f'task-{index}--{noise}', 'spec_index': index,
                           'family': evaluator.FAMILIES[index % 6], 'noise_label': noise,
                           'policy': policy, 'budget': budget, 'status': 'evaluated',
                           'charged_measurement_scalars': 7104*budget}
                    error = .08 if policy == criteria['candidate'] else .1 if policy in criteria['required_improvement_comparators'] else .09
                    set_metrics(row, error)
                    rows.append(row)
    return {'records': rows, 'audit': {'status': 'pending_independent_audit'}}


def bind(comparison, *, status='passed', paired=True, bad_hash=False):
    result = deepcopy(comparison)
    result.pop(evaluator.CONTEXT_KEY, None)
    frozen_sources = {name: hashlib.sha256(path.read_bytes()).hexdigest()
                      for name, path in evaluator.SOURCE_FILES.items()}
    freeze_sha = hashlib.sha256(evaluator._canonical({'sources': frozen_sources})).hexdigest()
    result['freeze_index_sha256'] = freeze_sha
    payload = evaluator._canonical(result)
    raw_hash = hashlib.sha256(payload).hexdigest()
    result[evaluator.CONTEXT_KEY] = {
        'audit': {'status': status, 'comparison_sha256': '0'*64 if bad_hash else raw_hash,
                  'checks': [{'name': evaluator.PAIRING_CHECK, 'status': 'passed' if paired else 'failed'}],
                  'summary': {'rich_opening_pairing': {'paired_conditions': 36}}},
        'comparison_sha256': raw_hash, 'comparison_content_sha256': raw_hash,
        'audit_sha256': 'a'*64, 'freeze_index_sha256': freeze_sha,
        'frozen_sources': frozen_sources, 'current_source_sha256': frozen_sources.copy(),
        'criteria_content_sha256': hashlib.sha256(evaluator._canonical(
            json.loads(evaluator.CRITERIA_FILE.read_text()))).hexdigest()}
    return result


def group(result, policy='rich_then_information', noise='noise1', budget=9):
    return next(g for g in result['groups'] if (g['policy'], g['noise_label'], g['budget']) == (policy, noise, budget))


def gate(result, name):
    return next(g for g in result['gates'] if g['name'] == name)


class InformationAssessmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base = fixture()

    def test_all_gates_pass_only_permits_review_without_merge_or_mutation(self):
        comparison = bind(self.base); before = deepcopy(comparison)
        result = evaluator.assess(comparison)
        self.assertTrue(result['all_gates_passed'])
        self.assertEqual(result['decision'], 'eligible_for_review_only')
        self.assertFalse(result['automatic_merge'])
        self.assertEqual(comparison, before)
        self.assertEqual(len(result['groups']), 60)
        self.assertEqual(len(result['paired_condition_deltas']), 900)
        self.assertEqual(result['completeness']['expected_records'], 1080)
        json.dumps(result, allow_nan=False)

    def test_primary_exact_ten_percent_boundary_and_beyond(self):
        comparison = deepcopy(self.base)
        for row in comparison['records']:
            if row['policy'] == 'rich_then_information' and row['noise_label'] == 'noise1' and row['budget'] == 9:
                set_metrics(row, .09)
        result = evaluator.assess(bind(comparison))
        self.assertTrue(gate(result, 'primary_vs_fixed_rich')['passed'])
        self.assertTrue(gate(result, 'primary_vs_rich_then_variance')['passed'])
        for row in comparison['records']:
            if row['policy'] == 'rich_then_information' and row['noise_label'] == 'noise1' and row['budget'] == 9:
                set_metrics(row, .090001)
        self.assertFalse(gate(evaluator.assess(bind(comparison)), 'primary_vs_fixed_rich')['passed'])

    def test_zero_baseline_never_divides_by_zero_or_reports_relative_gain(self):
        comparison = deepcopy(self.base)
        for row in comparison['records']:
            set_metrics(row, 0.)
        result = evaluator.assess(bind(comparison))
        primary = gate(result, 'primary_vs_fixed_rich')
        self.assertFalse(primary['passed'])
        self.assertEqual(primary['reason'], 'control_zero_no_relative_improvement_defined')
        self.assertIsNone(primary['observed_relative_improvement'])
        self.assertEqual(primary['comparator_mean'], 0.)
        json.dumps(result, allow_nan=False)
        for row in comparison['records']:
            if row['policy'] == 'rich_then_information':
                set_metrics(row, 1e-5)
        self.assertFalse(gate(evaluator.assess(bind(comparison)), 'primary_vs_fixed_rich')['passed'])

    def test_missing_identity_nulls_every_group_mean_and_disables_intervals(self):
        comparison = deepcopy(self.base); comparison['records'].pop()
        result = evaluator.assess(bind(comparison))
        self.assertFalse(result['completeness']['identity_complete'])
        self.assertTrue(all(g['mean_worst_true_coefficient_error'] is None for g in result['groups']))
        self.assertFalse(result['paired_intervals_available'])
        self.assertEqual(result['primary_paired_intervals'], [])
        self.assertTrue(all(value is None for value in result['costs']['first_budget_at_target'].values()))
        self.assertEqual(result['decision'], 'retain_baseline_do_not_merge')

    def test_duplicate_unexpected_or_mislabelled_slots_are_not_complete(self):
        for variant in ('duplicate', 'unexpected', 'family', 'budget', 'task', 'unhashable_id'):
            comparison = deepcopy(self.base)
            if variant == 'duplicate': comparison['records'][-1] = deepcopy(comparison['records'][0])
            elif variant == 'unexpected': comparison['records'][-1]['id'] = 'undeclared'
            elif variant == 'family': comparison['records'][-1]['family'] = 'invented'
            elif variant == 'budget': comparison['records'][-1]['budget'] = True
            elif variant == 'task': comparison['records'][-1]['task_id'] = 'task-999--noise3'
            else: comparison['records'][-1]['id'] = ['not a scalar identity']
            with self.subTest(variant=variant):
                self.assertFalse(evaluator.assess(bind(comparison))['completeness']['identity_complete'])

    def test_missing_model_counts_one_in_all_eighteen_not_success_subset(self):
        comparison = deepcopy(self.base)
        target = next(r for r in comparison['records'] if r['policy'] == 'rich_then_information'
                      and r['noise_label'] == 'noise1' and r['budget'] == 9)
        set_metrics(target, missing=True, failed_forecasts=3)
        result = evaluator.assess(bind(comparison)); metrics = group(result)
        self.assertTrue(result['completeness']['metrics_complete'])
        self.assertEqual(metrics['missing_model_count'], 1)
        self.assertAlmostEqual(metrics['mean_worst_true_coefficient_error'], (17*.08+1)/18)
        self.assertEqual(metrics['max_worst_true_coefficient_error'], 1.)
        self.assertEqual(metrics['failed_prediction_slots'], 3)
        self.assertIsNone(metrics['mean_condition_forecast_error'])
        self.assertFalse(gate(result, 'complete_forecast_noise1')['passed'])

    def test_invalid_missing_reference_or_fake_metric_is_not_dropped(self):
        for change in ('reference', 'nan', 'fake'):
            comparison = deepcopy(self.base)
            if change == 'reference': comparison['records'][0]['recovery']['reference_terms'] = []
            elif change == 'nan': comparison['records'][0]['recovery']['coefficient_comparison'][0]['relative_error'] = float('nan')
            else: comparison['records'][0]['worst_true_coefficient_relative_error'] = 0.
            # No audit context is needed to confirm fail-closed arithmetic.
            with self.subTest(change=change):
                result = evaluator.assess(comparison)
                self.assertFalse(result['completeness']['metrics_complete'])
                self.assertTrue(all(g['mean_worst_true_coefficient_error'] is None for g in result['groups']))
                json.dumps(result, allow_nan=False)

    def test_better_than_primary_controls_but_worse_than_random_fails(self):
        comparison = deepcopy(self.base)
        for row in comparison['records']:
            if row['policy'] == 'random': set_metrics(row, .07)
        result = evaluator.assess(bind(comparison))
        self.assertTrue(gate(result, 'primary_vs_fixed_rich')['passed'])
        self.assertFalse(gate(result, 'primary_vs_random')['passed'])
        self.assertFalse(result['all_gates_passed'])

    def test_wrong_structure_cannot_win_with_small_true_coefficient_error(self):
        comparison = deepcopy(self.base)
        for row in comparison['records']:
            if row['policy'] == 'rich_then_information' and row['budget'] == 6 and row['noise_label'] == 'noise3':
                set_metrics(row, .001, extra=True)
                break
        result = evaluator.assess(bind(comparison))
        self.assertTrue(gate(result, 'primary_vs_fixed_rich')['passed'])
        self.assertFalse(gate(result, 'structure_noise3_budget6_extra_term_total_vs_fixed_rich')['passed'])
        self.assertFalse(gate(result, 'structure_noise3_budget6_exact_support_count_vs_rich_then_variance')['passed'])

    def test_terminal_tail_uses_better_control_maximum(self):
        comparison = deepcopy(self.base)
        for row in comparison['records']:
            if row['budget'] == 9 and row['policy'] == 'rich_then_information' and row['spec_index'] == 120:
                set_metrics(row, .11)
        result = evaluator.assess(bind(comparison))
        self.assertTrue(gate(result, 'primary_vs_fixed_rich')['passed'])
        self.assertFalse(gate(result, 'terminal_tail_noise1')['passed'])
        self.assertAlmostEqual(gate(result, 'terminal_tail_noise1')['maximum_allowed'], .105+1e-12)

    def test_stress_mean_and_each_condition_forecast_have_independent_gates(self):
        comparison = deepcopy(self.base)
        for row in comparison['records']:
            if row['budget'] == 9 and row['policy'] == 'rich_then_information' and row['noise_label'] == 'noise3':
                set_metrics(row, .106)
        self.assertFalse(gate(evaluator.assess(bind(comparison)), 'stress_mean')['passed'])
        comparison = deepcopy(self.base)
        target = next(r for r in comparison['records'] if r['budget'] == 9 and r['policy'] == 'rich_then_information')
        set_metrics(target, .08, forecast=.05001)
        result = evaluator.assess(bind(comparison))
        self.assertLess(group(result)['mean_condition_forecast_error'], .05)
        self.assertFalse(gate(result, 'complete_forecast_noise1')['passed'])
        set_metrics(target, .08, forecast=.049999999999)
        self.assertTrue(gate(evaluator.assess(bind(comparison)), 'complete_forecast_noise1')['passed'])

    def test_forecast_failures_and_missing_slots_never_average_success_subset(self):
        comparison = deepcopy(self.base)
        target = next(r for r in comparison['records'] if r['budget'] == 9 and r['policy'] == 'rich_then_information')
        set_metrics(target, .08, failed_forecasts=1)
        result = evaluator.assess(bind(comparison))
        self.assertTrue(result['completeness']['metrics_complete'])
        self.assertIsNone(group(result)['mean_condition_forecast_error'])
        self.assertEqual(group(result)['failed_prediction_slots'], 1)
        target['forecasts'].pop()
        self.assertFalse(evaluator.assess(bind(comparison))['completeness']['metrics_complete'])

    def test_independent_audit_pairing_and_content_binding_are_required(self):
        for comparison in (self.base, bind(self.base, status='failed'), bind(self.base, paired=False),
                           bind(self.base, bad_hash=True)):
            result = evaluator.assess(comparison)
            self.assertFalse(result['all_gates_passed'])
        comparison = bind(self.base)
        comparison['records'][0]['status'] = 'changed after audit'
        self.assertFalse(gate(evaluator.assess(comparison), 'comparison_binding_passed')['passed'])

    def test_paired_bootstrap_is_family_stratified_deterministic_and_descriptive(self):
        comparison = deepcopy(self.base)
        for row in comparison['records']:
            if row['policy'] == 'rich_then_information':
                set_metrics(row, .04+.01*((row['spec_index']-120)//6)+.001*(row['spec_index']%6))
        first = evaluator.assess(bind(comparison)); second = evaluator.assess(bind(comparison))
        self.assertEqual(first['primary_paired_intervals'], second['primary_paired_intervals'])
        interval = next(r for r in first['primary_paired_intervals'] if r['comparator'] == 'fixed_rich')
        deltas = [r['coefficient_error_difference'] for r in first['paired_condition_deltas']
                  if r['comparator'] == 'fixed_rich' and r['noise_label'] == 'noise1' and r['budget'] == 9]
        rng = np.random.default_rng(970000)
        draw = np.concatenate([rng.choice([j for j in range(18) if j%6 == family], size=(2000,3), replace=True)
                               for family in range(6)], axis=1)
        expected = np.quantile(np.mean(np.asarray(deltas)[draw], axis=1), [.025,.975])
        np.testing.assert_array_equal(interval['percentile_interval'], expected)
        self.assertEqual(interval['tasks_per_family'], 3)
        self.assertFalse(interval['used_for_promotion'])
        self.assertFalse(interval['calibrated_probability_claimed'])

    def test_old_cost_target_uses_all_thirty_six_conditions_and_saved_budgets(self):
        comparison = deepcopy(self.base)
        for row in comparison['records']:
            start = 6 if row['policy'] == 'rich_then_information' else 8
            set_metrics(row, .05 if row['budget'] >= start else .2)
        result = evaluator.assess(bind(comparison)); costs = result['costs']
        self.assertEqual(costs['denominator_per_policy_budget'], 36)
        self.assertEqual(costs['first_budget_at_target']['rich_then_information'], 6)
        self.assertEqual(costs['first_budget_at_target']['fixed_rich'], 8)
        self.assertEqual(costs['candidate_fraction_saved_vs']['fixed_rich'], .25)
        self.assertFalse(costs['used_for_promotion'])
        self.assertFalse(costs['success_target_changed'])
        for row in comparison['records']:
            if row['policy'] == 'rich_then_information' and row['spec_index'] in (120,121,122,123):
                set_metrics(row, missing=True, failed_forecasts=3)
        costs = evaluator.assess(bind(comparison))['costs']
        self.assertIsNone(costs['first_budget_at_target']['rich_then_information'])
        self.assertIsNone(costs['candidate_fraction_saved_vs']['fixed_rich'])

    def test_cli_binds_exact_bytes_and_never_overwrites_input_or_existing_output(self):
        with tempfile.TemporaryDirectory(prefix='information-assess-manufactured-') as temporary:
            output = Path(temporary)
            freeze_bytes = json.dumps({'sources': {name: hashlib.sha256(path.read_bytes()).hexdigest()
                for name, path in evaluator.SOURCE_FILES.items()}}).encode()
            (output/'freeze_index.json').write_bytes(freeze_bytes)
            comparison = deepcopy(self.base)
            comparison['freeze_index_sha256'] = hashlib.sha256(freeze_bytes).hexdigest()
            comparison_bytes = json.dumps(comparison, indent=1).encode()
            (output/'comparison.json').write_bytes(comparison_bytes)
            audit = {'status':'passed','comparison_sha256':hashlib.sha256(comparison_bytes).hexdigest(),
                     'checks':[{'name':evaluator.PAIRING_CHECK,'status':'passed'}]}
            audit_bytes = json.dumps(audit).encode(); (output/'audit.json').write_bytes(audit_bytes)
            with patch('sys.stdout', new=io.StringIO()):
                self.assertEqual(evaluator.main(['--output', str(output)]), 0)
            saved = (output/'assessment.json').read_bytes(); result = json.loads(saved)
            self.assertTrue(result['all_gates_passed'])
            self.assertEqual((output/'comparison.json').read_bytes(), comparison_bytes)
            self.assertEqual((output/'audit.json').read_bytes(), audit_bytes)
            with self.assertRaises(FileExistsError), patch('sys.stdout', new=io.StringIO()):
                evaluator.main(['--output', str(output)])
            self.assertEqual((output/'assessment.json').read_bytes(), saved)
            self.assertFalse(list(output.glob('.assessment-*')))

    def test_assessment_source_binding_missing_freeze_sources_or_changed_criteria_fails(self):
        comparison = bind(self.base)
        self.assertTrue(gate(evaluator.assess(comparison), 'source_binding_passed')['passed'])
        for key in evaluator.SOURCE_FILES:
            changed = deepcopy(comparison)
            changed[evaluator.CONTEXT_KEY]['frozen_sources'].pop(key)
            self.assertFalse(gate(evaluator.assess(changed), 'source_binding_passed')['passed'])
        criteria = json.loads(evaluator.CRITERIA_FILE.read_text())
        criteria['forecast_limit'] += .01
        self.assertFalse(gate(evaluator.assess(comparison, criteria), 'source_binding_passed')['passed'])

    def test_cli_rejects_post_audit_changes_to_criteria_assessor_or_freeze(self):
        for changed_kind in ('criteria', 'assessor', 'freeze'):
            with self.subTest(changed_kind=changed_kind), tempfile.TemporaryDirectory(
                    prefix='information-assess-binding-manufactured-') as temporary:
                output = Path(temporary)
                source_dir = output/'source'; source_dir.mkdir()
                source_files = {}
                for name, path in evaluator.SOURCE_FILES.items():
                    copy = source_dir/Path(name).name
                    copy.write_bytes(path.read_bytes()); source_files[name] = copy
                freeze = {'sources': {name: hashlib.sha256(path.read_bytes()).hexdigest()
                                      for name, path in source_files.items()}}
                freeze_bytes = json.dumps(freeze).encode()
                (output/'freeze_index.json').write_bytes(freeze_bytes)
                comparison = deepcopy(self.base)
                comparison['freeze_index_sha256'] = hashlib.sha256(freeze_bytes).hexdigest()
                comparison_bytes = json.dumps(comparison).encode()
                (output/'comparison.json').write_bytes(comparison_bytes)
                audit = {'status': 'passed', 'comparison_sha256': hashlib.sha256(comparison_bytes).hexdigest(),
                         'checks': [{'name': evaluator.PAIRING_CHECK, 'status': 'passed'}]}
                (output/'audit.json').write_text(json.dumps(audit))
                if changed_kind == 'criteria':
                    path = source_files['information_study/decision_protocol.json']
                    altered = json.loads(path.read_text()); altered['forecast_limit'] = .2
                    path.write_text(json.dumps(altered))
                elif changed_kind == 'assessor':
                    path = source_files['information_study/assess.py']
                    path.write_bytes(path.read_bytes()+b'\n# manufactured later edit\n')
                else:
                    freeze['sources']['information_study/assess.py'] = 'b'*64
                    (output/'freeze_index.json').write_text(json.dumps(freeze))
                with patch.object(evaluator, 'SOURCE_FILES', source_files), patch('sys.stdout', new=io.StringIO()):
                    evaluator.main(['--output', str(output)])
                result = json.loads((output/'assessment.json').read_text())
                self.assertTrue(gate(result, 'audit_passed')['passed'])
                self.assertTrue(gate(result, 'comparison_binding_passed')['passed'])
                self.assertFalse(gate(result, 'source_binding_passed')['passed'])
                self.assertEqual(result['decision'], 'retain_baseline_do_not_merge')

    def test_cli_missing_audit_writes_failed_review_decision_without_fabricating_pass(self):
        with tempfile.TemporaryDirectory(prefix='information-assess-manufactured-') as temporary:
            output = Path(temporary); (output/'comparison.json').write_text(json.dumps(self.base))
            with patch('sys.stdout', new=io.StringIO()): evaluator.main(['--output', str(output)])
            result = json.loads((output/'assessment.json').read_text())
            self.assertEqual(result['decision'], 'retain_baseline_do_not_merge')
            self.assertFalse(gate(result, 'audit_passed')['passed'])


if __name__ == '__main__':
    unittest.main()
