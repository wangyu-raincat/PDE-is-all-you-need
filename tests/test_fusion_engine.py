"""Support preservation, independent variant weighting, and dev-only safeguards."""
from copy import deepcopy
import unittest
from unittest.mock import patch
import numpy as np

from fusion_study import engine as f


def candidate(identity, variant, coefficient, error=1., extra=None):
    terms = [{'name': 'u_xx', 'coefficient': coefficient}]
    if extra:
        terms.append({'name': extra, 'coefficient': -.01})
    return {'candidate_id': identity, 'weak_variant': variant, 'terms': terms,
            'support_size': len(terms), 'fit_success': True, 'eligible': True,
            'development': {'rmse': error}, 'alpha_values': [.01]}


def selection(candidates, selected=0):
    return {'selected': deepcopy(candidates[selected]) if selected is not None else None,
            'candidates': deepcopy(candidates), 'failure_reason': None}


def score_median(error=1., eligible=True):
    def score(candidates, developments):
        c = deepcopy(candidates[0])
        c['eligible'] = eligible
        if eligible:
            c['development'] = {'rmse': error}
        else:
            c['rejection_reason'] = 'forecast failed'
        return {'selected': c if eligible else None, 'candidates': [c],
                'failure_reason': None if eligible else 'No eligible candidate.'}
    return score


class FusionEngineTests(unittest.TestCase):
    def setUp(self):
        self.members = [candidate('compact-a', 'compact', .1, 1.01),
                        candidate('low-a', 'low', .2, 1.),
                        candidate('resolved-a', 'resolved', .9, 1.02)]
        self.original = selection(self.members)
        self.resolved = selection(self.members, 1)
        self.dev = [{'x': np.arange(16), 't': np.arange(16), 'u': np.zeros((16,16))}]

    def test_support_anchor_excludes_better_other_structure(self):
        other = candidate('wrong-support', 'compact', .15, .01, 'u_xxxx')
        self.resolved['candidates'].append(other)
        self.resolved['selected'] = other
        actual = f.select_from_scored(self.original, self.resolved, 'anchored_best')
        self.assertEqual(actual['selected']['candidate_id'], 'low-a')
        self.assertEqual(actual['fusion']['anchor_support'], ['u_xx'])
        self.assertEqual(len(actual['fusion']['excluded_candidates']), 1)

    def test_median_equal_variants_preserves_support(self):
        with patch.object(f.refined, 'select_candidates', side_effect=score_median()):
            actual = f.select_from_scored(self.original, self.resolved, 'fused_median', self.dev)
        self.assertEqual(actual['selected']['terms'], [{'name': 'u_xx', 'coefficient': .2}])
        self.assertEqual(actual['fusion']['median_status'], 'accepted')
        self.assertEqual(actual['fusion']['member_variants'], ['compact', 'low', 'resolved'])
        self.assertNotIn('fit_rmse', actual['selected'])
        self.assertTrue(actual['selected']['fit_objective_not_recomputed'])

    def test_duplicate_lambda_paths_and_variant_duplicates_cannot_reweight(self):
        copies = [dict(deepcopy(self.members[0]), candidate_id=f'compact-z{i}', alpha_values=[.01]*500)
                  for i in range(20)]
        self.resolved['candidates'].extend(copies)
        with patch.object(f.refined, 'select_candidates', side_effect=score_median()):
            actual = f.select_from_scored(self.original, self.resolved, 'fused_median', self.dev)
        self.assertEqual(actual['selected']['terms'][0]['coefficient'], .2)
        self.assertEqual(len(actual['fusion']['member_candidate_ids']), 3)
        self.assertEqual(actual['fusion']['member_candidate_ids'][0], 'compact-a')

    def test_far_variant_is_excluded_before_median(self):
        self.resolved['candidates'][2]['development']['rmse'] = 2.
        with patch.object(f.refined, 'select_candidates', side_effect=score_median()):
            actual = f.select_from_scored(self.original, self.resolved, 'fused_median', self.dev)
        self.assertAlmostEqual(actual['selected']['terms'][0]['coefficient'], .15)
        self.assertEqual(actual['fusion']['member_variants'], ['compact', 'low'])

    def test_variant_representative_uses_training_id_before_dev_filter(self):
        earlier = candidate('compact-0', 'compact', 9., 1.2)
        self.resolved['candidates'].append(earlier)
        with patch.object(f.refined, 'select_candidates', side_effect=score_median()):
            actual = f.select_from_scored(self.original, self.resolved, 'fused_median', self.dev)
        self.assertEqual(actual['fusion']['variant_representative_ids'][0], 'compact-0')
        self.assertEqual(actual['fusion']['member_variants'], ['low', 'resolved'])
        self.assertAlmostEqual(actual['selected']['terms'][0]['coefficient'], .55)

    def test_median_worse_on_development_retains_anchored_best(self):
        with patch.object(f.refined, 'select_candidates', side_effect=score_median(1.051)):
            actual = f.select_from_scored(self.original, self.resolved, 'fused_median', self.dev)
        self.assertEqual(actual['selected']['candidate_id'], 'low-a')
        self.assertEqual(actual['fusion']['median_status'], 'development_rejected')
        self.assertAlmostEqual(actual['fusion']['acceptance_threshold'], 1.05+1e-12)

    def test_forecast_failure_retains_anchored_best_with_evidence(self):
        with patch.object(f.refined, 'select_candidates', side_effect=score_median(eligible=False)):
            actual = f.select_from_scored(self.original, self.resolved, 'fused_median', self.dev)
        self.assertEqual(actual['selected']['candidate_id'], 'low-a')
        self.assertEqual(actual['fusion']['median_status'], 'forecast_rejected')
        self.assertEqual(actual['fusion']['median_reason'], 'forecast failed')

    def test_zero_median_does_not_delete_anchor_term(self):
        members = [candidate('a', 'compact', -.2), candidate('b', 'low', .2)]
        with patch.object(f.refined, 'select_candidates', side_effect=AssertionError('no forecast')):
            actual = f.select_from_scored(selection(members), selection(members), 'fused_median', self.dev)
        self.assertEqual(actual['fusion']['median_status'], 'invalid_coefficients')
        self.assertEqual(actual['selected']['terms'][0]['coefficient'], -.2)
        self.assertEqual(actual['fusion']['median_proposed_terms'], [{'name': 'u_xx', 'coefficient': 0.}])

    def test_one_variant_no_new_vector_or_forecast(self):
        self.resolved['candidates'] = self.resolved['candidates'][:1]
        with patch.object(f.refined, 'select_candidates', side_effect=AssertionError('no forecast')):
            actual = f.select_from_scored(self.original, self.resolved, 'fused_median', self.dev)
        self.assertEqual(actual['fusion']['median_status'], 'single_variant_no_fusion')
        self.assertEqual(actual['selected']['candidate_id'], 'compact-a')

    def test_no_anchor_or_no_refined_result_fails_closed(self):
        for original, resolved in [(dict(self.original, selected=None), self.resolved),
                                   (self.original, dict(self.resolved, selected=None))]:
            result = f.select_from_scored(original, resolved, 'anchored_best')
            self.assertIsNone(result['selected'])
            self.assertTrue(result['failure_reason'])

    def test_no_eligible_same_support_fails_closed(self):
        for c in self.resolved['candidates']:
            c['terms'].append({'name': 'u', 'coefficient': -.5})
        result = f.select_from_scored(self.original, self.resolved, 'anchored_best')
        self.assertIsNone(result['selected'])
        self.assertIn('anchored support', result['failure_reason'])

    def test_unknown_variant_and_nonfinite_score_are_excluded(self):
        self.resolved['candidates'][0]['weak_variant'] = 'undeclared'
        self.resolved['candidates'][1]['development']['rmse'] = float('nan')
        result = f.select_from_scored(self.original, self.resolved, 'anchored_best')
        self.assertEqual(result['selected']['candidate_id'], 'resolved-a')
        self.assertEqual(len(result['fusion']['excluded_candidates']), 2)

    def test_malformed_anchor_fails_without_deleting_terms(self):
        for value in (0., float('nan'), float('inf'), True):
            original = deepcopy(self.original)
            original['selected']['terms'][0]['coefficient'] = value
            result = f.select_from_scored(original, self.resolved, 'anchored_best')
            self.assertIsNone(result['selected'])

    def test_inputs_and_original_training_diagnostics_unchanged(self):
        before = deepcopy((self.original, self.resolved))
        dev_before = deepcopy(self.dev)
        with patch.object(f.refined, 'select_candidates', side_effect=score_median()):
            result = f.select_from_scored(self.original, self.resolved, 'fused_median', self.dev)
        result['selected']['terms'][0]['coefficient'] = 33
        self.assertEqual((self.original, self.resolved), before)
        np.testing.assert_array_equal(self.dev[0]['u'], dev_before[0]['u'])

    def test_median_requires_development_and_no_reference_interface(self):
        with self.assertRaises(ValueError):
            f.select_from_scored(self.original, self.resolved, 'fused_median')
        with self.assertRaises(TypeError):
            f.select_from_scored(self.original, self.resolved, 'anchored_best', reference=[])
        with self.assertRaises(TypeError):
            f.fit_observations([], [], true_noise=1.)

    def test_wrapper_generates_once_scores_twice_and_declares_stages(self):
        with patch.object(f, 'generate_candidates', return_value=(self.members, [])) as generate, \
             patch.object(f.sparse, 'select_candidates', return_value=deepcopy(self.original)) as original_score, \
             patch.object(f.refined, 'select_candidates', side_effect=[deepcopy(self.resolved),
                                                                       score_median()([self.members[0]], self.dev)]) as refined_score:
            result = f.fit_observations(self.dev, self.dev)
        generate.assert_called_once()
        original_score.assert_called_once()
        self.assertEqual(refined_score.call_count, 2)
        self.assertEqual(set(result), set(f.METHODS))
        self.assertEqual(result['weak_original']['final_predictor'], 'original_observed_grid')
        for method in ('anchored_best', 'fused_median'):
            self.assertEqual(result[method]['structural_selection_predictor'], 'original_observed_grid')
            self.assertEqual(result[method]['coefficient_selection_predictor'], 'resolved_factor4')
            self.assertEqual(result[method]['final_predictor'], 'resolved_factor4')

    def test_real_forecasters_screen_median_on_manufactured_noisy_heat_field(self):
        x = np.arange(32)*2*np.pi/32
        t = np.linspace(0, .2, 33)
        # Independent analytic heat component plus a deterministic measurement disturbance.
        u = np.exp(-.1*t[:, None])*np.cos(x)[None, :]
        u += .003*np.sin(20*t[:, None])*np.cos(9*x)[None, :]
        data = {'x': x, 't': t, 'u': u, 'boundary': 'periodic'}
        candidates = [candidate(v, v, c) for v, c in [('compact', .099), ('low', .1), ('resolved', .101)]]
        original = f.sparse.select_candidates(candidates, [data])
        resolved = f.refined.select_candidates(candidates, [data])
        actual = f.select_from_scored(original, resolved, 'fused_median', [data])
        self.assertEqual(actual['fusion']['median_status'], 'accepted')
        self.assertEqual(actual['selected']['terms'], [{'name': 'u_xx', 'coefficient': .1}])
        self.assertEqual(actual['selected']['development']['trajectory_count'], 1)
        self.assertLessEqual(actual['selected']['development']['rmse'], actual['fusion']['acceptance_threshold'])

    def test_candidate_generation_matches_historical_weak_fit_exactly(self):
        from weak_study.run import fit_observations as historical
        x = np.arange(32)*2*np.pi/32
        t = np.linspace(0, .2, 33)
        u = np.exp(-.1*t[:, None])*np.cos(x)[None, :]
        u += .2*np.exp(-.4*t[:, None])*np.sin(2*x)[None, :]
        data = {'x': x, 't': t, 'u': u, 'boundary': 'periodic'}
        actual, trials = f.generate_candidates([data])
        with patch('weak_study.run.engine.select_candidates',
                   side_effect=lambda candidates, developments: {'candidates': deepcopy(candidates)}):
            old = historical([data], [data], 'weak_adaptive')['candidates']
        self.assertTrue(actual)
        self.assertEqual(actual, old)
        self.assertEqual({r['variant'] for r in trials}, set(f.weak.VARIANTS))


if __name__ == '__main__':
    unittest.main()
