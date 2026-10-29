"""Observable reliability selection, estimator reconstruction, and fallback."""
from copy import deepcopy
import unittest

import numpy as np

from coefficient_study import selection


def field(index):
    x = np.linspace(0., 2*np.pi, 16, endpoint=False)
    t = np.linspace(0., 1., 17)
    return {'x': x, 't': t, 'u': np.full((len(t), len(x)), float(index)),
            'boundary': 'periodic', 'private_truth': 'must not reach weak builder',
            'prefix_u': np.zeros((3, len(x)))}


def candidate(variant, coefficient=1., prediction=1., name=None):
    return {'candidate_id': name or variant+'::manufactured', 'weak_variant': variant,
            'support_size': 2, 'terms': [{'name': 'u', 'coefficient': 2.},
                                       {'name': 'u_xx', 'coefficient': coefficient}],
            'fit_success': True, 'eligible': True,
            'development': {'rmse': prediction}, 'equation': 'manufactured'}


def builder(profiles=None, development_coefficient=1., seen=None):
    profiles = profiles or {'compact': [1., 1., 1.], 'low': [.7, 1., 1.3],
                            'resolved': [1., 1., 1.]}

    def build(observation, variant='compact'):
        if set(observation) != {'x', 't', 'u', 'boundary'}:
            raise AssertionError('Reference, history or unrelated metadata reached weak builder.')
        index = int(observation['u'][0, 0])
        if seen is not None:
            seen.append((index, variant))
        coefficient = profiles[variant][index] if index < 3 else development_coefficient
        X = np.zeros((8, 9))
        X[::2, 1] = 1.
        X[1::2, 5] = 1.
        y = 2.*X[:, 1] + coefficient*X[:, 5]
        return {'X': X, 'y': y}
    return build


class CoefficientSelectionTests(unittest.TestCase):
    def setUp(self):
        self.training = [field(i) for i in range(3)]
        self.developments = [field(10), field(11)]

    def select(self, candidates, build=None):
        return selection.select_candidates(candidates, self.training, self.developments,
                                           block_builder=build or builder())

    def test_predictively_near_stable_coefficients_can_win(self):
        candidates = [candidate('low', prediction=.99), candidate('compact', prediction=1.)]
        result = self.select(candidates)
        self.assertEqual(result['selected']['candidate_id'], 'compact::manufactured')
        receipt = result['reliability_selection']
        self.assertEqual(receipt['status'], 'selected')
        self.assertFalse(receipt['fallback_used'])
        self.assertFalse(receipt['prediction_recomputed'])
        a, b = (receipt['candidate_diagnostics'][v+'::manufactured'] for v in ('compact', 'low'))
        self.assertLess(a['worst_relative_omission_rms'], b['worst_relative_omission_rms'])
        self.assertEqual(len(a['leave_out_folds']), 3)
        self.assertEqual({f['omitted_training_trajectory'] for f in a['leave_out_folds']}, {0, 1, 2})

    def test_prediction_pool_never_relaxes_to_get_stability(self):
        result = self.select([candidate('low', prediction=1.), candidate('compact', prediction=1.051)])
        self.assertEqual(result['selected']['candidate_id'], 'low::manufactured')
        self.assertEqual(result['reliability_selection']['prediction_pool_ids'], ['low::manufactured'])

    def test_minimum_support_priority_precedes_weak_and_stability(self):
        low = candidate('low')
        bigger = candidate('compact', prediction=.99)
        bigger['terms'].append({'name': 'u^3', 'coefficient': .1}); bigger['support_size'] = 3
        result = self.select([low, bigger])
        self.assertEqual(result['selected']['candidate_id'], low['candidate_id'])
        self.assertNotIn(bigger['candidate_id'], result['reliability_selection']['candidate_diagnostics'])

    def test_same_common_validation_blocks_are_used_for_all_candidates(self):
        seen = []
        result = self.select([candidate('low'), candidate('compact')], builder(seen=seen))
        self.assertEqual(result['reliability_selection']['status'], 'selected')
        self.assertEqual([item for item in seen if item[0] >= 10], [(10, 'resolved'), (11, 'resolved')])
        self.assertEqual(len(seen), 11)  # 3 train trajectories per variant plus 2 common dev blocks.

    def test_common_weak_residual_gate_precedes_stability(self):
        profiles = {'compact': [2., 2., 2.], 'low': [.7, 1., 1.3], 'resolved': [1., 1., 1.]}
        result = self.select([candidate('compact', coefficient=2.), candidate('low')], builder(profiles))
        self.assertEqual(result['selected']['candidate_id'], 'low::manufactured')
        self.assertEqual(result['reliability_selection']['weak_pool_ids'], ['low::manufactured'])

    def test_development_weak_information_can_change_choice_without_changing_coefficients(self):
        profiles = {'compact': [2., 2., 2.], 'low': [.7, 1., 1.3], 'resolved': [1., 1., 1.]}
        candidates = [candidate('compact', coefficient=2.), candidate('low')]
        a = self.select(candidates, builder(profiles, development_coefficient=1.))
        b = self.select(candidates, builder(profiles, development_coefficient=2.))
        self.assertNotEqual(a['selected']['candidate_id'], b['selected']['candidate_id'])
        self.assertEqual([c['terms'] for c in a['candidates']], [c['terms'] for c in b['candidates']])

    def test_aliases_never_supply_multiple_median_votes(self):
        profiles = {'compact': [1., 1., 1.], 'low': [1.4, 2., 2.6], 'resolved': [1.4, 2., 2.6]}
        median = candidate(None, coefficient=1.5, name='support-median::1-5')
        median['diagnostics'] = {'distinct_vector_groups': [
            {'coefficients': [2., 1.], 'variants': ['compact']},
            {'coefficients': [2., 2.], 'variants': ['low', 'resolved']}]}
        result = self.select([median], builder(profiles, development_coefficient=1.5))
        self.assertEqual(result['reliability_selection']['status'], 'selected')
        diagnostic = result['selected']['coefficient_reliability']
        self.assertTrue(diagnostic['median_reconstructed'])
        self.assertEqual(diagnostic['participating_variants'], ['compact', 'low', 'resolved'])
        self.assertEqual(len(diagnostic['full_distinct_vector_groups']), 2)
        for fold in diagnostic['leave_out_folds']:
            groups = fold['distinct_vector_groups']
            self.assertEqual(len(groups), 2)
            self.assertEqual(groups[1]['variants'], ['low', 'resolved'])
            self.assertAlmostEqual(fold['coefficients'][1], np.mean([g['coefficients'][1] for g in groups]))

    def test_failed_required_diagnostic_retains_prediction_baseline_and_all_candidates(self):
        candidates = [candidate('low', prediction=.99), candidate('compact')]
        base = builder()
        def failure(observation, variant='compact'):
            if variant == 'compact':
                raise ValueError('manufactured bad weak block')
            return base(observation, variant)
        result = self.select(candidates, failure)
        self.assertEqual(result['selected']['candidate_id'], 'low::manufactured')
        self.assertEqual(len(result['candidates']), 2)
        self.assertTrue(all(c['eligible'] for c in result['candidates']))
        receipt = result['reliability_selection']
        self.assertEqual(receipt['status'], 'fallback_to_prediction')
        self.assertIn('manufactured bad weak block', receipt['fallback_reason'])
        self.assertEqual(receipt['candidate_diagnostics']['compact::manufactured']['status'], 'failed')

    def test_changed_training_estimator_is_not_silently_accepted(self):
        result = self.select([candidate('compact', coefficient=1.1)])
        self.assertEqual(result['reliability_selection']['status'], 'fallback_to_prediction')
        self.assertIn('do not match', result['reliability_selection']['fallback_reason'])

    def test_leave_out_rank_failure_is_explicit_fallback(self):
        base = builder()
        def rank_failure(observation, variant='compact'):
            block = base(observation, variant)
            if int(observation['u'][0, 0]) in (1, 2):
                block['X'][:, 5] = 0.
                block['y'] = 2.*block['X'][:, 1]
            return block
        result = self.select([candidate('compact')], rank_failure)
        self.assertEqual(result['reliability_selection']['status'], 'fallback_to_prediction')
        self.assertIn('no finite measurable', result['reliability_selection']['fallback_reason'])

    def test_too_few_training_trajectories_retains_baseline(self):
        result = selection.select_candidates([candidate('compact')], self.training[:2], self.developments,
                                             block_builder=builder())
        self.assertTrue(result['reliability_selection']['fallback_used'])
        self.assertIn('3..32', result['reliability_selection']['fallback_reason'])

    def test_no_predictive_candidate_does_not_invoke_block_builder(self):
        c = candidate('compact'); c['eligible'] = False
        def forbidden(*args, **kwargs):
            raise AssertionError('No weak data needed without prediction baseline.')
        result = self.select([c], forbidden)
        self.assertIsNone(result['selected'])
        self.assertEqual(result['reliability_selection']['status'], 'no_baseline_candidate')

    def test_input_candidates_and_observations_are_unchanged(self):
        candidates = [candidate('low'), candidate('compact')]
        before_candidates = deepcopy(candidates); before_data = deepcopy(self.training+self.developments)
        self.select(candidates)
        self.assertEqual(candidates, before_candidates)
        for before, after in zip(before_data, self.training+self.developments):
            for name in ('x', 't', 'u', 'prefix_u'):
                np.testing.assert_array_equal(before[name], after[name])

    def test_duplicate_candidate_ids_rejected(self):
        with self.assertRaises(ValueError):
            self.select([candidate('compact'), candidate('compact')])

    def test_true_or_final_data_are_not_public_parameters(self):
        with self.assertRaises(TypeError):
            selection.select_candidates([candidate('compact')], self.training, self.developments,
                                        reference_coefficients={'u_xx': 1.})
        with self.assertRaises(TypeError):
            selection.select_candidates([candidate('compact')], self.training, self.developments,
                                        final=self.developments)


if __name__ == '__main__':
    unittest.main()
