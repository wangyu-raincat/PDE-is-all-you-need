"""Saved-receipt reporting tests: no fitting, field arrays, or formal data."""
from copy import deepcopy
import csv
import json
from pathlib import Path
import tempfile
import unittest

from scripts import render_coefficient_study as report


def fixture(family='allen_cahn', sampling='full', condition='clean', method='noise_stable'):
    identity = '--'.join((family, sampling, condition, method))
    error = {'prefix_baseline': .10, 'stable_selection': .08, 'noise_corrected': .07, 'noise_stable': .05}[method]
    coefficient = 1 + error
    stable = method in ('stable_selection', 'noise_stable')
    receipt = {'status': 'selected', 'fallback_used': False, 'fallback_reason': None,
               'baseline_candidate_id': 'first', 'candidate_diagnostics': {}} if stable else None
    terms = [{'name': 'u', 'coefficient': coefficient}]
    metadata = {'conditioning': 'observed_prefix', 'final_predictor': 'prefix_aligned_resolved_factor4',
                'initial_filter': 'prefix_propagate_average',
                'coefficient_selection': 'weak_residual_then_trajectory_stability' if stable else 'forecast_only',
                'coefficient_noise_correction': method in ('noise_corrected', 'noise_stable')}
    model = {'id': identity, 'method': method, 'selected': {'candidate_id': 'first', 'terms': terms},
             'candidates': [], 'sources': {}, 'input_hashes': {}, 'reliability_selection': receipt, **metadata}
    score = {'id': identity, 'family': family, 'sampling': sampling, 'condition': condition, 'method': method,
             'noise_level': .01 if condition.startswith('noise1_') else .001 if condition.startswith('noise0p1_') else 0,
             'noise_seed': int(condition.split('_s')[-1]) if '_s' in condition else 0,
             'equation': 'u_t = coefficient*u', 'terms': terms, 'reliability_selection': receipt,
             'recovery': {'exact_support': True, 'missing_terms': [], 'spurious_terms': [],
                          'coefficient_relative_l2': error,
                          'coefficient_comparison': [{'name': 'u', 'reference': 1., 'discovered': coefficient,
                                                      'relative_error': error, 'absolute_error': error}]},
             'forecasts': [{'status': 'evaluated', 'clean': {'relative_l2': .03},
                            'conditioning': 'observed_prefix', 'prefix_frames_used': report.PREFIX_FRAMES[sampling],
                            'forecast_duration': report.DURATIONS[family]} for _ in range(3)],
             'forecast_failures': 0, 'mean_clean_relative_l2': .03, **metadata}
    return score, model


def saved_final(directory):
    models = directory / 'models'
    models.mkdir()
    records, frozen = [], []
    for family in report.FAMILIES:
        for sampling in report.SAMPLINGS:
            for condition in report.CONDITIONS:
                for method in report.METHODS:
                    score, model = fixture(family, sampling, condition, method)
                    file = f"models/{score['id']}.json"
                    (directory / file).write_text(json.dumps(model))
                    score['frozen_model_sha256'] = report.digest(directory / file)
                    frozen.append({'id': score['id'], 'file': file, 'sha256': score['frozen_model_sha256']})
                    records.append(score)
    index = {'records': frozen, 'sources': {}, 'input_hashes': {}, 'all_120_frozen_before_final': True}
    (directory / 'freeze_index.json').write_text(json.dumps(index))
    checks = [{'name': 'source_quality', 'passed': True}]
    for score in records:
        if score['method'] == 'noise_stable':
            checks += [{'name': score['id'] + suffix, 'passed': True} for suffix in (
                ':exact_structure', ':each_true_term_accuracy', ':mean_prediction_at_most_5pct', ':vs_prefix_baseline:coefficient')]
    checks += [{'name': family + ':vs_prefix_baseline:worst_true_term_nonregression', 'passed': True} for family in report.FAMILIES]
    checks += [{'name': name, 'passed': True} for name in (
        'allen_cahn:worst_true_term_relative_reduction_at_least_20pct', 'all_families:maximum_true_term_error_not_increased')]
    result = {'records': records, 'quality': {family: {'status': 'passed'} for family in report.FAMILIES},
              'final_reference_failures': {family: [None] * 3 for family in report.FAMILIES},
              'freeze_index_sha256': report.digest(directory / 'freeze_index.json'), 'final_reference_sha256': {},
              'all_declared_models_reported': True, 'final_scores_used_for_tuning': False,
              'all_data_numerical_not_real_measurement': True,
              'gate': {'candidate': 'noise_stable', 'merged': False, 'passed': True,
                       'prediction_target_passed': True, 'checks': checks}}
    (directory / 'comparison.json').write_text(json.dumps(result))
    return result


class CoefficientReportingTests(unittest.TestCase):
    def test_all_four_arms_use_same_prefix_and_full_horizon(self):
        for method in report.METHODS:
            score, model = fixture(method=method, sampling='coarse')
            row = report.flat_row(score, model)
            self.assertEqual(row['prefix_frames_used'], 6)
            self.assertEqual(row['unchanged_future_duration'], .5)
            self.assertTrue(row['condition_mean_target_passed'])
        score['forecasts'][0]['forecast_duration'] *= .9
        with self.assertRaisesRegex(ValueError, 'horizon'):
            report.flat_row(score, model)

    def test_failed_trajectory_keeps_denominator_and_cannot_have_mean(self):
        score, model = fixture()
        score['forecasts'][0] = {'status': 'failed', 'reason': 'solver failure'}
        score['forecast_failures'] = 1
        with self.assertRaisesRegex(ValueError, 'null'):
            report.flat_row(score, model)
        score['mean_clean_relative_l2'] = None
        row = report.flat_row(score, model)
        self.assertFalse(row['condition_mean_target_passed'])
        self.assertIsNone(row['mean_final_prediction_l2_percent'])
        grouped = report.group_summary([row], [score])
        self.assertEqual(grouped['configurations'], 1)
        self.assertEqual(grouped['complete_prediction_conditions'], 0)
        self.assertFalse(grouped['all_condition_means_at_most5pct'])
        self.assertIsNone(grouped['max_condition_mean_prediction_l2'])

    def test_noise_receipts_deduplicate_variants_not_trajectories(self):
        _, model = fixture()
        noise = {'estimated_spatial_noise_sigma': .01, 'estimated_spatial_noise_variance': .0001,
                 'true_noise_used': False, 'reference_used': False}
        blocks = [{'noise_estimation': noise} for _ in range(3)]
        model['candidates'] = [{'diagnostics': {'block_diagnostics': deepcopy(blocks)}} for _ in range(6)]
        self.assertEqual(len(report.noise_receipts(model)), 3)
        model['candidates'][0]['diagnostics']['block_diagnostics'][0]['noise_estimation']['estimated_spatial_noise_sigma'] = .02
        with self.assertRaisesRegex(ValueError, 'disagree'):
            report.noise_receipts(model)

    def test_selector_fallback_is_not_hidden(self):
        score, model = fixture()
        receipt = {'status': 'fallback_to_prediction', 'fallback_used': True,
                   'fallback_reason': 'rank deficient', 'baseline_candidate_id': 'first',
                   'candidate_diagnostics': {'first': {'status': 'failed', 'reason': 'rank deficient'}}}
        score['reliability_selection'] = model['reliability_selection'] = receipt
        row = report.flat_row(score, model)
        self.assertTrue(row['selector_fallback_used'])
        self.assertEqual(row['selector_failed_diagnostics'], 1)
        self.assertFalse(row['selector_changed_candidate'])

    def test_extra_terms_preserve_absolute_error_and_true_vectors(self):
        score, model = fixture()
        score['recovery']['exact_support'] = False
        score['recovery']['spurious_terms'] = ['1']
        score['recovery']['coefficient_comparison'].append({'name': '1', 'reference': 0, 'discovered': .003,
                                                         'absolute_error': .003, 'relative_error': None})
        row = report.flat_row(score, model)
        self.assertEqual(json.loads(row['extra_term_absolute_errors']), {'1': .003})
        self.assertEqual(json.loads(row['true_reference_coefficients']), {'u': 1.})

    def test_missing_final_does_not_create_outputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            with self.assertRaisesRegex(ValueError, 'does not exist'):
                report.main(path, path / 'absent')
            self.assertEqual(list(path.iterdir()), [])

    def test_saved_json_render_hash_binding_lf_and_no_candidate_switch(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            original = saved_final(directory)
            summary = report.main(directory, directory / 'missing_retrospective')
            self.assertEqual(summary['gate_summary']['checks_passed'], 126)
            self.assertEqual(summary['candidate_frozen_before_final'], 'noise_stable')
            self.assertFalse(summary['retrospective_diagnostic']['available'])
            self.assertFalse(summary['array_files_loaded_by_renderer'])
            self.assertFalse(summary['final_winner_switch_allowed'])
            for name in ('summary.csv', 'coefficients.csv'):
                self.assertNotIn(b'\r', (directory / name).read_bytes())
                with (directory / name).open() as stream:
                    self.assertEqual(len(list(csv.DictReader(stream))), 120)
            for name, checksum in summary['artifact_hashes'].items():
                self.assertEqual(report.digest(directory / name), checksum)
            original['gate']['candidate'] = 'stable_selection'
            (directory / 'comparison.json').write_text(json.dumps(original))
            with self.assertRaisesRegex(ValueError, 'frozen candidate'):
                report.load_final(directory)
            original['gate']['candidate'] = 'noise_stable'
            (directory / 'comparison.json').write_text(json.dumps(original))
            first = next((directory / 'models').glob('*.json'))
            first.write_text(first.read_text() + ' ')
            with self.assertRaisesRegex(ValueError, 'model bytes'):
                report.load_final(directory)

    def test_gate_tampering_and_duplicate_conditions_fail(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            original = saved_final(directory)
            original['gate']['checks'][1]['passed'] = False
            (directory / 'comparison.json').write_text(json.dumps(original))
            with self.assertRaisesRegex(ValueError, 'gate disagrees'):
                report.load_final(directory)
            original['records'][0] = original['records'][1]
            (directory / 'comparison.json').write_text(json.dumps(original))
            with self.assertRaisesRegex(ValueError, '120 unique'):
                report.load_final(directory)


if __name__ == '__main__':
    unittest.main()
