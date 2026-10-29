"""Independent saved-result audit checks using manufactured records only.

No scientific generator, fitter, production gate or formal result is imported.
"""
import ast
from copy import deepcopy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


SCRIPT = Path(__file__).resolve().parents[1] / 'scripts' / 'audit_coefficient_study.py'
SPEC = importlib.util.spec_from_file_location('coefficient_saved_audit', SCRIPT)
audit_module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit_module)

FAMILIES = ('allen_cahn', 'low_viscosity_burgers', 'ks_unstable')
SAMPLINGS = ('full', 'coarse')
METHODS = ('prefix_baseline', 'stable_selection', 'noise_corrected', 'noise_stable')
NOISES = (('clean', 0., 0), ('noise0p1_s79', .001, 79), ('noise0p1_s109', .001, 109),
          ('noise1_s79', .01, 79), ('noise1_s109', .01, 109))


def recovery(error):
    return {
        'exact_support': True, 'missing_terms': [], 'spurious_terms': [],
        'coefficient_relative_l2': error,
        'coefficient_comparison': [{'name': 'u_xx', 'reference': .1,
                                    'discovered': .1*(1+error), 'absolute_error': .1*error,
                                    'relative_error': error}],
    }


def records():
    rows = []
    for family in FAMILIES:
        for sampling in SAMPLINGS:
            for condition, level, seed in NOISES:
                for method in METHODS:
                    error = .04 if method == 'noise_stable' else .08
                    rows.append({
                        'id': '--'.join((family, sampling, condition, method)),
                        'family': family, 'sampling': sampling, 'condition': condition,
                        'noise_level': level, 'noise_seed': seed, 'method': method,
                        'training_rms_for_corruption_only': 1., 'source_quality_passed': True,
                        'training_trajectories': 3, 'development_trajectories': 2,
                        'structural_selection_predictor': 'prefix_aligned_resolved_factor4',
                        'coefficient_selection_predictor': 'prefix_aligned_resolved_factor4',
                        'final_predictor': 'prefix_aligned_resolved_factor4',
                        'initial_filter': 'prefix_propagate_average',
                        'conditioning': 'observed_prefix',
                        'coefficient_noise_correction': method in ('noise_corrected', 'noise_stable'),
                        'coefficient_selection': ('weak_residual_then_trajectory_stability'
                                                  if method in ('stable_selection', 'noise_stable')
                                                  else 'forecast_only'),
                        'recovery': recovery(error), 'mean_clean_relative_l2': .049,
                        'forecast_failures': 0,
                        'forecasts': [{'status': 'evaluated', 'clean': {'relative_l2': value}}
                                      for value in (.02, .027, .1)],
                    })
    return rows


def candidate(rows):
    return next(row for row in rows if row['method'] == 'noise_stable')


def set_mean(row, value):
    row['mean_clean_relative_l2'] = value
    row['forecasts'] = [{'status': 'evaluated', 'clean': {'relative_l2': value}}
                        for _ in range(3)]


def saved_fixture(root):
    """Create only fixed little byte strings and JSON inside a temporary root."""
    root = Path(root)
    output = root / 'coefficient_study/results/manufactured'

    def put(relative, value):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(value if isinstance(value, bytes) else
                         (json.dumps(value, sort_keys=True, allow_nan=False)+'\n').encode())
        return hashlib.sha256(path.read_bytes()).hexdigest()

    source_paths = [f'coefficient_study/{name}' for name in (
        '__init__.py', 'data.py', 'engine.py', 'generators.py', 'noise.py', 'run.py', 'selection.py',
        'protocol.json', 'references.json')]
    source_paths += [
        'challenge_study/data.py', 'challenge_study/generators.py', 'l1_study/engine.py',
        'weak_study/engine.py', 'weak_study/data.py', 'weak_study/generators.py',
        'resolved_study/forecast.py', 'resolved_study/selection.py', 'fusion_study/engine.py',
        'five_study/engine.py', 'five_study/initial.py', 'five_study/forecast.py',
        'prefix_study/engine.py', 'prefix_study/forecast.py',
        'pde_discovery/generic_engine.py', 'pde_discovery/generic_experiment.py']
    protocol = {'variant_menu_status': 'frozen'}
    truth = [{'name': 'u_xx', 'coefficient': .1}]
    source_values = {'coefficient_study/protocol.json': protocol,
                     'coefficient_study/references.json': {'references': {f: truth for f in FAMILIES}}}
    sources = {path: put(path, source_values.get(path, ('manufactured source '+path).encode()))
               for path in source_paths}
    input_paths = ['coefficient_study/data/calibration.json'] + [
        f'coefficient_study/data/clean/{family}/{role}/{index}.npz'
        for family in FAMILIES for role, count in (('training', 3), ('development', 2))
        for index in range(count)]
    inputs = {path: put(path, ('manufactured input '+path).encode()) for path in input_paths}
    common = {'sources': sources, 'input_hashes': inputs,
              'quality': {family: {'status': 'passed'} for family in FAMILIES},
              'training_rms': {family: 1. for family in FAMILIES},
              'all_source_quality_passed': True, 'protocol': protocol}
    preparation_sha = put('coefficient_study/data/prepared.json', {
        **common, 'synthetic_final_generated': False})
    final_hashes = {}
    for family in FAMILIES:
        for index in range(3):
            relative = f'final_reference/{family}/{index}.npz'
            final_hashes[relative] = put(output.relative_to(root)/relative,
                                        ('manufactured final '+relative).encode())

    rows = records()
    manifest = []
    horizons = {'allen_cahn': .5, 'low_viscosity_burgers': 1.5, 'ks_unstable': 20.}
    spaces = {'allen_cahn': 256, 'low_viscosity_burgers': 128, 'ks_unstable': 128}
    for row in rows:
        # Within each shared raw/corrected pool, fallback selects the same sole
        # candidate as the corresponding prediction-only arm.
        error = .04 if row['method'] in ('noise_corrected', 'noise_stable') else .08
        row['recovery'] = recovery(error)
        terms = [{'name': 'u_xx', 'coefficient': .1*(1+error)}]
        row['terms'] = terms
        chosen = {'candidate_id': 'manufactured', 'terms': terms, 'support_size': 1,
                  'fit_success': True, 'eligible': True, 'development': {'rmse': .1}}
        stable = row['method'] in ('stable_selection', 'noise_stable')
        receipt = {'status': 'fallback_to_prediction', 'fallback_used': True,
                   'fallback_reason': 'manufactured diagnostic failure',
                   'baseline_candidate_id': 'manufactured', 'whole_trajectory_omissions': True,
                   'reference_used': False, 'final_used': False, 'prediction_recomputed': False,
                   'confidence_interval_claimed': False} if stable else None
        row['reliability_selection'] = deepcopy(receipt)
        model = {**row, 'sources': sources, 'input_hashes': inputs, 'selected': chosen,
                 'candidates': [chosen], 'reference_used_for_selection': False,
                 'clean_future_used_for_selection': False, 'final_loaded_during_selection': False,
                 'failure_reason': None, 'reliability_selection': receipt}
        relative = 'models/'+row['id']+'.json'
        model_sha = put(output.relative_to(root)/relative, model)
        row['frozen_model_sha256'] = model_sha
        manifest.append({'id': row['id'], 'file': relative, 'sha256': model_sha})
        prefix_count = 11 if row['sampling'] == 'full' else 6
        stride = 1 if row['sampling'] == 'full' else 2
        for forecast in row['forecasts']:
            forecast['clean'].update(excluded_initial_frame=True, scored_frames=100//stride,
                                     space_points=spaces[row['family']]//stride)
            forecast['noisy'] = deepcopy(forecast['clean'])
            forecast.update(conditioning='observed_prefix', prefix_frames_used=prefix_count,
                            forecast_duration=horizons[row['family']],
                            initial_denoise_diagnostics={
                                'conditioning': 'observed_prefix',
                                'future_observations_used': False, 'independent_estimates_claimed': False,
                                'equal_frame_weights': True, 'candidate_PDE_used_for_state_estimation': True,
                                'prefix_frames': prefix_count, 'per_frame_filter': [{} for _ in range(prefix_count)],
                                'prefix_duration': .1*horizons[row['family']],
                                'forecast_duration': horizons[row['family']]})
    freeze = {**common, 'records': manifest, 'preparation_sha256': preparation_sha,
              'all_120_frozen_before_final': True, 'synthetic_final_generated': False}
    freeze_sha = put(output.relative_to(root)/'freeze_index.json', freeze)
    # Numerical gate boundaries are independently tested above. This fixture
    # exercises only the saved-file graph, receipt checks and tamper detection.
    gate = {**audit_module.rederive_gate(rows), 'merged': False}
    comparison = {'records': rows, 'gate': gate, 'freeze_index_sha256': freeze_sha,
                  'final_reference_sha256': final_hashes,
                  'final_reference_failures': {family: [None]*3 for family in FAMILIES},
                  'final_scores_used_for_tuning': False, 'all_data_numerical_not_real_measurement': True,
                  'all_declared_models_reported': True, 'configurations_with_forecast_failure': 0,
                  'prediction_above_5pct_all_arms': 0}
    put(output.relative_to(root)/'comparison.json', comparison)
    return output


class IndependentCoefficientGateTests(unittest.TestCase):
    def test_rederives_exactly126_checks_and_condition_mean_target(self):
        rows = records()
        before = deepcopy(rows)
        gate = audit_module.rederive_gate(rows)
        self.assertTrue(gate['passed'])
        self.assertTrue(gate['prediction_target_passed'])
        self.assertTrue(gate['coefficient_improvement_passed'])
        self.assertEqual(len(gate['checks']), 126)
        self.assertEqual(len({check['name'] for check in gate['checks']}), 126)
        self.assertEqual(gate['candidate'], 'noise_stable')
        self.assertEqual(rows, before)
        # Each condition has a 10% trajectory, but its three-slot mean is 4.9%.
        self.assertTrue(all(row['forecasts'][-1]['clean']['relative_l2'] > .05 for row in rows))

    def test_missing_duplicate_and_unknown_configuration_fail_closed(self):
        rows = records()
        variants = [rows[:-1], [rows[0], rows[0], *rows[2:]]]
        unknown = deepcopy(rows)
        unknown[0]['id'] = 'unknown'
        variants.append(unknown)
        for invalid in variants:
            with self.subTest(ids=len(invalid)):
                self.assertFalse(audit_module.rederive_gate(invalid)['passed'])

    def test_zero_ac_baseline_cannot_claim_relative_improvement(self):
        rows = records()
        for row in rows:
            if row['family'] == 'allen_cahn':
                row['recovery'] = recovery(0.)
        gate = audit_module.rederive_gate(rows)
        self.assertTrue(gate['prediction_target_passed'])
        self.assertFalse(gate['coefficient_improvement_passed'])
        self.assertFalse(gate['passed'])

    def test_ac_gain_must_reach_twenty_percent(self):
        rows = records()
        for row in rows:
            if row['family'] == 'allen_cahn' and row['method'] == 'noise_stable':
                row['recovery'] = recovery(.065)
        gate = audit_module.rederive_gate(rows)
        self.assertTrue(gate['prediction_target_passed'])
        self.assertFalse(gate['coefficient_improvement_passed'])
        self.assertFalse(gate['passed'])

    def test_global_tail_increase_is_not_hidden_by_ac_gain(self):
        rows = records()
        for row in rows:
            if row['family'] == 'ks_unstable' and row['method'] == 'noise_stable':
                row['recovery'] = recovery(.081)
        gate = audit_module.rederive_gate(rows)
        self.assertFalse(gate['passed'])
        check = next(item for item in gate['checks']
                     if item['name'] == 'all_families:maximum_true_term_error_not_increased')
        self.assertFalse(check['passed'])

    def test_failed_trajectory_cannot_disappear_from_denominator(self):
        rows = records()
        row = candidate(rows)
        row['forecasts'][1] = {'status': 'failed', 'reason': 'manufactured failure'}
        row['forecast_failures'] = 1
        row['mean_clean_relative_l2'] = .01  # Illegitimate success-only mean.
        gate = audit_module.rederive_gate(rows)
        self.assertFalse(gate['prediction_target_passed'])
        self.assertFalse(gate['passed'])
        self.assertEqual(len(row['forecasts']), 3)

    def test_missing_slot_and_forged_mean_fail_prediction_target(self):
        rows = records()
        candidate(rows)['forecasts'].pop()
        self.assertFalse(audit_module.rederive_gate(rows)['prediction_target_passed'])
        rows = records()
        candidate(rows)['mean_clean_relative_l2'] = .001
        self.assertFalse(audit_module.rederive_gate(rows)['prediction_target_passed'])

    def test_nonfinite_or_negative_mean_never_passes(self):
        for value in (None, float('nan'), float('inf'), -.01):
            rows = records()
            set_mean(candidate(rows), value)
            with self.subTest(value=value):
                self.assertFalse(audit_module.rederive_gate(rows)['prediction_target_passed'])

    def test_five_percent_boundary_is_inclusive_without_rounding_down(self):
        for value, passed in ((.05, True), (.0500001, False)):
            rows = records()
            set_mean(candidate(rows), value)
            with self.subTest(value=value):
                self.assertEqual(audit_module.rederive_gate(rows)['prediction_target_passed'], passed)

    def test_final_ablation_does_not_replace_predeclared_candidate(self):
        rows = records()
        for row in rows:
            if row['method'] == 'noise_stable':
                set_mean(row, .06)
            elif row['method'] == 'stable_selection':
                set_mean(row, 1e-8)
                row['recovery'] = recovery(.001)
        gate = audit_module.rederive_gate(rows)
        self.assertEqual(gate['candidate'], 'noise_stable')
        self.assertFalse(gate['passed'])

    def test_source_failure_cannot_pass(self):
        gate = audit_module.rederive_gate(records(), source_quality=False)
        self.assertFalse(gate['passed'])
        self.assertFalse(next(item['passed'] for item in gate['checks']
                              if item['name'] == 'source_quality'))

    def test_good_prediction_cannot_override_missing_or_extra_terms(self):
        for missing, extra in ((['u_xx'], []), ([], ['u'])):
            rows = records()
            row = candidate(rows)
            set_mean(row, 1e-8)
            row['recovery'].update(exact_support=False, missing_terms=missing, spurious_terms=extra)
            gate = audit_module.rederive_gate(rows)
            self.assertTrue(gate['prediction_target_passed'])
            self.assertFalse(gate['passed'])


class SavedAuditBoundaryTests(unittest.TestCase):
    def test_missing_comparison_blocks_all_other_input_reads(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / 'results'
            comparison = output / 'comparison.json'
            reads = []
            original_open = Path.open

            def guarded_open(path, *args, **kwargs):
                reads.append(path)
                self.assertEqual(path, comparison, 'Audit accessed data before final comparison exists.')
                return original_open(path, *args, **kwargs)

            with patch.object(Path, 'open', guarded_open):
                result = audit_module.audit(root, output)
            self.assertFalse(result['passed'])
            self.assertTrue(result['errors'])
            self.assertTrue(all(path == comparison for path in reads))

    def test_incomplete_comparison_blocks_all_other_input_reads(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / 'results'
            output.mkdir()
            comparison = output / 'comparison.json'
            comparison.write_text('{"records": []}')
            reads = []
            original_open = Path.open

            def guarded_open(path, *args, **kwargs):
                reads.append(path)
                self.assertEqual(path, comparison, 'Audit accessed artifacts before a complete final report.')
                return original_open(path, *args, **kwargs)

            with patch.object(Path, 'open', guarded_open):
                result = audit_module.audit(root, output)
            self.assertFalse(result['passed'])
            self.assertTrue(result['errors'])
            self.assertEqual(reads, [comparison])

    def test_auditor_does_not_import_or_call_production_gate(self):
        tree = ast.parse(SCRIPT.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                self.assertFalse(any(alias.name == 'promotion_gate' for alias in node.names))
                self.assertNotEqual(node.module, 'coefficient_study.run')
            if isinstance(node, ast.Call):
                called = node.func
                self.assertFalse(isinstance(called, ast.Attribute) and called.attr == 'promotion_gate')
                self.assertFalse(isinstance(called, ast.Name) and called.id == 'promotion_gate')


class SavedAuditIntegrationTests(unittest.TestCase):
    def test_complete_manufactured_files_verify_counts_and_fallback_receipts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = saved_fixture(root)
            result = audit_module.audit(root, output)
            self.assertTrue(result['passed'], result['errors'])
            self.assertEqual(result['verified_counts'], {
                'source_hashes': 25, 'input_hashes': 16, 'final_hashes': 9,
                'model_hashes': 120, 'forecast_receipts': 360})
            self.assertTrue(result['gate']['passed'])
            self.assertFalse(result['read_numeric_arrays'])
            self.assertFalse(result['generated_or_fitted_anything'])
            self.assertFalse(result['production_gate_called'])
            for method in ('stable_selection', 'noise_stable'):
                summary = result['aggregates'][method]
                self.assertEqual(summary['reliability_statuses'], {'fallback_to_prediction': 30})
                self.assertEqual(len(summary['fallback_conditions']), 30)

    def test_model_input_final_or_source_byte_change_cannot_pass(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = saved_fixture(root)
            targets = [
                (output/'models'/('--'.join((FAMILIES[0], 'full', 'clean', METHODS[0]))+'.json'), 'model hash'),
                (root/'coefficient_study/data/clean/allen_cahn/training/0.npz', 'input hash'),
                (output/'final_reference/allen_cahn/0.npz', 'Final reference hash'),
                (root/'coefficient_study/generators.py', 'Scientific source hash')]
            for path, expected in targets:
                original = path.read_bytes()
                with self.subTest(target=path):
                    path.write_bytes(original+b' ')
                    result = audit_module.audit(root, output)
                    self.assertFalse(result['passed'])
                    self.assertTrue(any(expected in error for error in result['errors']), result['errors'])
                path.write_bytes(original)

    def test_scored_fallback_receipt_must_match_frozen_receipt(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = saved_fixture(root)
            path = output/'comparison.json'
            report = json.loads(path.read_text())
            row = next(row for row in report['records'] if row['method'] == 'noise_stable')
            row['reliability_selection']['fallback_reason'] = 'different unrecorded explanation'
            path.write_text(json.dumps(report))
            result = audit_module.audit(root, output)
            self.assertFalse(result['passed'])
            self.assertTrue(any('reliability receipt.fallback_reason' in error for error in result['errors']),
                            result['errors'])


class IndependentRecoveryTests(unittest.TestCase):
    def test_true_terms_extras_missing_terms_and_signed_relative_errors(self):
        truth = [{'name': 'u', 'coefficient': 2.}, {'name': 'u_xx', 'coefficient': .1},
                 {'name': 'u^3', 'coefficient': -1.}]
        found = [{'name': 'u', 'coefficient': 2.2}, {'name': 'u^3', 'coefficient': -.9},
                 {'name': 'u_x', 'coefficient': .3}]
        result = audit_module.rederive_recovery(found, truth)
        self.assertFalse(result['exact_support'])
        self.assertEqual(result['missing_terms'], ['u_xx'])
        self.assertEqual(result['spurious_terms'], ['u_x'])
        self.assertAlmostEqual(result['coefficient_relative_l2'], math.sqrt(.15)/math.sqrt(5.01))
        by_name = {row['name']: row for row in result['coefficient_comparison']}
        self.assertAlmostEqual(by_name['u']['relative_error'], .1)
        self.assertAlmostEqual(by_name['u^3']['relative_error'], .1)
        self.assertEqual(by_name['u_xx']['relative_error'], 1.)
        self.assertIsNone(by_name['u_x']['relative_error'])
        self.assertAlmostEqual(by_name['u_x']['absolute_error'], .3)

    def test_exact_recovery_ignores_explicit_zero_and_empty_fit_has_unit_error(self):
        truth = [{'name': 'u_xx', 'coefficient': -.1}]
        result = audit_module.rederive_recovery(truth+[{'name': 'u', 'coefficient': 0.}], truth)
        self.assertTrue(result['exact_support'])
        self.assertEqual(result['coefficient_relative_l2'], 0.)
        absent = audit_module.rederive_recovery([], truth)
        self.assertEqual(absent['missing_terms'], ['u_xx'])
        self.assertEqual(absent['coefficient_relative_l2'], 1.)

    def test_duplicate_unknown_nonfinite_coefficients_and_empty_truth_are_rejected(self):
        truth = [{'name': 'u_xx', 'coefficient': .1}]
        invalid = [truth+truth, [{'name': 'unknown', 'coefficient': .1}],
                   [{'name': 'u_xx', 'coefficient': float('nan')}],
                   [{'name': 'u_xx', 'coefficient': float('inf')}]]
        for found in invalid:
            with self.subTest(found=found), self.assertRaises(ValueError):
                audit_module.rederive_recovery(found, truth)
        with self.assertRaises(ValueError):
            audit_module.rederive_recovery(truth, [])


if __name__ == '__main__':
    unittest.main()
