"""Manufactured information-design checks; no provider or formal reference data."""
from copy import deepcopy
import inspect
import json
import unittest
from unittest.mock import patch

import numpy as np

from active_study import actions
from active_study import acquisition as original
from information_study import acquisition as design


def observation(index=0, frames=101):
    x = np.arange(64)*2*np.pi/64
    t = np.linspace(.1, 1.1, frames)
    u = np.exp(-.3*t[:, None])*(np.sin(x+.23*index)[None, :]
                               + .2*np.cos(3*x-.17*index)[None, :])
    return {'x': x, 't': t, 'u': u, 'boundary': 'periodic'}


def selected(terms=None):
    return {'terms': terms or [{'name': 'u', 'coefficient': -.3}],
            'fit_success': True, 'eligible': True}


class InformationAcquisitionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pool = actions.make_action_pool(317)
        cls.observations = [observation(i) for i in range(3)]
        cls.committee = [selected(), selected([{'name': 'u_xx', 'coefficient': .3}])]

    def test_all_original_policies_delegate_without_changing_inputs_or_results(self):
        for policy in original.POLICIES:
            sentinel = {'old_implementation_result': object()}
            with self.subTest(policy=policy), patch.object(original, 'select_action', return_value=sentinel) as old:
                result = design.select_action(policy, self.committee, actions=self.pool,
                    observations=self.observations, seed=9, used_action_ids=['fixed-rich'],
                    cost_per_action=2, selected={'malformed_new_anchor_ignored_by_old_policy': True})
                self.assertIs(result, sentinel)
                old.assert_called_once_with(policy, self.committee, self.pool, self.observations,
                    seed=9, used_action_ids=['fixed-rich'], cost_per_action=2)

    def test_new_strategies_have_identical_rich_opening_without_using_anchor(self):
        baseline = original.select_action('fixed_rich', self.committee, self.pool, self.observations)
        for policy in ('rich_then_variance', 'rich_then_information'):
            with self.subTest(policy=policy), patch.object(design, 'observed_geometry', side_effect=AssertionError('must not fit')):
                result = design.select_action(policy, self.committee, self.pool, self.observations,
                                              selected={'invalid': True})
            self.assertEqual(result['selected_action'], baseline['selected_action'])
            self.assertEqual(result['fixed_action_order'], baseline['fixed_action_order'])
            self.assertEqual(result['scores'], baseline['scores'])
            self.assertTrue(result['first_action_matched_fixed_rich'])
            self.assertEqual(result['model_forecast_attempts'], 0)

    def test_variance_after_rich_uses_unchanged_ensemble_rule(self):
        baseline = original.select_action('ensemble_variance', self.committee, self.pool[:4],
                                          self.observations, used_action_ids=['fixed-rich'])
        result = design.select_action('rich_then_variance', self.committee, self.pool[:4],
                                      self.observations, used_action_ids=['fixed-rich'], selected=None)
        self.assertEqual(result['selected_action'], baseline['selected_action'])
        self.assertEqual(result['scores'], baseline['scores'])
        self.assertEqual(result['delegated_policy'], 'ensemble_variance')

    def test_relative_coordinate_formula_and_equal_trajectory_accumulation(self):
        observations = [observation(0, 101), observation(1, 65)]
        geometry = design.observed_geometry(observations, selected())
        direct = np.zeros((1, 1))
        q = geometry['diagnostics']['observed_target_scale']
        for obs in observations:
            filtered, _ = design.observed_engine.preprocess_observation(obs)
            block = design.weak.weak_block(filtered, 'resolved')
            X = block['X'][:, [design.sparse.TERM_NAMES.index('u')]]
            Z = X*.3/q
            direct += Z.T@Z/len(Z)
        np.testing.assert_allclose(geometry['gram'], direct, rtol=1e-13)
        doubled = design.observed_geometry(observations+observations, selected())
        np.testing.assert_allclose(doubled['gram'], 2*geometry['gram'], rtol=1e-13)
        self.assertEqual(geometry['diagnostics']['training_frames_per_trajectory'], [101, 65])
        self.assertFalse(geometry['diagnostics']['overlapping_rows_are_independent_evidence'])

    def test_inverse_trace_identity_matches_direct_inverse_and_handles_cancellation(self):
        G = np.array([[2., .2], [.2, 1.]])
        B = np.array([[.3, .1], [.1, .7]])
        ridge = 1e-8
        result = design.trace_reduction(G, B, ridge)
        expected = np.trace(np.linalg.inv(G+ridge*np.eye(2)))-np.trace(np.linalg.inv(G+B+ridge*np.eye(2)))
        self.assertAlmostEqual(result['trace_reduction'], expected, places=13)
        tiny = design.trace_reduction(np.eye(2), np.eye(2)*1e-18, 1e-8)
        self.assertGreater(tiny['trace_reduction'], 0)
        self.assertAlmostEqual(tiny['trace_reduction']/2e-18, 1., places=7)
        zero = design.trace_reduction(G, np.zeros_like(G), ridge)
        self.assertEqual(zero['trace_reduction'], 0.)
        rounded = design.trace_reduction(np.eye(2), np.diag([1., -1e-14]), ridge)
        self.assertGreater(rounded['trace_reduction'], 0.)
        with self.assertRaisesRegex(ValueError, 'positive semidefinite'):
            design.trace_reduction(G, np.diag([1., -.1]), ridge)
        anisotropic = np.diag([.01, 10.])
        weak_direction = design.trace_reduction(anisotropic, np.diag([1., 0.]), ridge)
        strong_direction = design.trace_reduction(anisotropic, np.diag([0., 1.]), ridge)
        self.assertGreater(weak_direction['trace_reduction'], 1000*strong_direction['trace_reduction'])

    def test_two_uniform_forward_calls_and_exact_linear_trajectory(self):
        grid = self.observations[0]['x']
        terms = selected()['terms']
        action = self.pool[0]
        with patch.object(original, 'candidate_forecast', wraps=original.candidate_forecast) as forward:
            prediction = design.predict_anchor(terms, grid, action)
        self.assertEqual(forward.call_count, 2)
        np.testing.assert_array_equal(forward.call_args_list[0].args[2], [0., .1])
        np.testing.assert_allclose(forward.call_args_list[1].args[2], np.linspace(.1, 1.1, 33))
        expected = np.exp(-.3*prediction['t'][:, None])*actions.action_field(action, grid)[None, :]
        np.testing.assert_allclose(prediction['u'], expected, atol=2e-14, rtol=2e-13)
        self.assertEqual(prediction['u'].shape, (33, 64))

    def test_candidate_action_information_uses_shared_observed_scales_and_ridge(self):
        pool = self.pool[:4]
        result = design.select_action('rich_then_information', self.committee, pool, self.observations,
                                      used_action_ids=['fixed-rich'], selected=selected())
        self.assertEqual(result['status'], 'selected')
        self.assertEqual(result['forecast_solver_calls_attempted'], 6)
        self.assertEqual(result['model_forecast_attempts'], 3)
        self.assertEqual(result['model_count_used_for_information'], 1)
        self.assertTrue(all(r['ridge'] == result['anchor_geometry']['ridge'] for r in result['scores']))
        self.assertTrue(all(r['predicted_weak_metadata']['training_frames'] == 33 for r in result['scores']))
        self.assertEqual(result['anchor_geometry']['training_frames_per_trajectory'], [101, 101, 101])
        self.assertNotIn(result['selected_action_id'], ['fixed-rich'])
        self.assertEqual(result['extra_measured_values_used'], 0)
        self.assertFalse(result['statistical_covariance_or_information_gain_claimed'])
        self.assertFalse(result['support_correctness_established'])

    def test_information_score_ignores_other_candidates_and_oracle_metadata(self):
        pool = self.pool[:3]
        kwargs = {'used_action_ids': ['fixed-rich'], 'selected': selected()}
        baseline = design.select_action('rich_then_information', [], pool, self.observations, **kwargs)
        altered = [dict(obs, true_noise=1e8, family='invented', clean_future=np.zeros((3, 2)),
                        reference_coefficients={'u': 10000.}) for obs in self.observations]
        anchor = dict(selected(), reference={'u': 12345.}, true_relative_error=0.)
        result = design.select_action('rich_then_information', self.committee*3, pool, altered,
                                      used_action_ids=['fixed-rich'], selected=anchor)
        self.assertEqual(result['selected_action_id'], baseline['selected_action_id'])
        self.assertEqual(result['scores'], baseline['scores'])
        self.assertEqual(result['anchor_geometry'], baseline['anchor_geometry'])
        with self.assertRaises(TypeError):
            design.select_action('rich_then_information', [], pool, self.observations, true_noise=.1)

    def test_missing_or_invalid_anchor_falls_back_to_unused_fixed_rich(self):
        for anchor in (None, {}, {'terms': [None]}, selected([{'name': 'u_xx', 'coefficient': -.1}]),
                       selected([{'name': 'u', 'coefficient': 0.}])):
            with self.subTest(anchor=anchor):
                result = design.select_action('rich_then_information', [], self.pool[:4], self.observations,
                                              used_action_ids=['fixed-rich'], selected=anchor)
            self.assertEqual(result['status'], 'fallback_fixed_rich')
            self.assertEqual(result['selected_action_id'], 'fixed-rich-1')
            self.assertTrue(result['anchor_failure'])
            self.assertEqual(result['model_forecast_attempts'], 0)
            self.assertTrue(all(r['score'] is None for r in result['scores']))

    def test_rank_deficient_anchor_support_is_not_certified_by_ridge(self):
        obs = observation()
        obs['u'] = np.exp(-.3*obs['t'][:, None])*np.sin(obs['x'])[None, :]
        anchor = selected([{'name': 'u', 'coefficient': -.2}, {'name': 'u_xx', 'coefficient': .1}])
        result = design.select_action('rich_then_information', [], self.pool[:4], [obs]*3,
                                      used_action_ids=['fixed-rich'], selected=anchor)
        self.assertEqual(result['status'], 'fallback_fixed_rich')
        self.assertIn('rank deficient', result['anchor_failure'])
        self.assertFalse(result['identification_certainty_claimed'])

    def test_small_nonzero_selected_coefficients_are_never_clipped_or_deleted(self):
        anchor = selected([{'name': 'u', 'coefficient': -.3}, {'name': 'u_x', 'coefficient': 1e-9}])
        geometry = design.observed_geometry(self.observations, anchor)
        self.assertEqual(geometry['support'], ('u', 'u_x'))
        self.assertEqual(geometry['terms'][1]['coefficient'], 1e-9)
        expected = 1e-9*geometry['scales'][1]/geometry['diagnostics']['observed_target_scale']
        self.assertEqual(geometry['relative_scale'][1], expected)

    def test_no_positive_information_is_exploration_not_certainty(self):
        with patch.object(design, 'trace_reduction', return_value={
                'trace_reduction': 0., 'inverse_trace_before': 1., 'inverse_trace_after': 1.}):
            result = design.select_action('rich_then_information', [], self.pool[:4], self.observations,
                                          used_action_ids=['fixed-rich'], selected=selected())
        self.assertEqual(result['status'], 'fallback_fixed_rich')
        self.assertEqual(result['selected_action_id'], 'fixed-rich-1')
        self.assertTrue(all(r['score'] == 0 for r in result['scores']))
        self.assertFalse(result['identification_certainty_claimed'])

    def test_prediction_failures_remain_explicit_and_fallback_retains_them(self):
        with patch.object(original, 'candidate_forecast', side_effect=RuntimeError('manufactured forward failure')):
            result = design.select_action('rich_then_information', [], self.pool[:4], self.observations,
                                          used_action_ids=['fixed-rich'], selected=selected())
        self.assertEqual(result['status'], 'fallback_fixed_rich')
        self.assertEqual(result['selected_action_id'], 'fixed-rich-1')
        self.assertEqual(result['model_forecast_failures'], 3)
        self.assertEqual(result['forecast_solver_calls_attempted'], 3)
        self.assertTrue(all(len(r['failed_models']) == 1 and r['score'] is None for r in result['scores']))

    def test_failed_action_is_excluded_without_changing_anchor_for_other_actions(self):
        pool = self.pool[:4]
        bad_initial = actions.action_field(pool[1], self.observations[0]['x'])
        forward = original.candidate_forecast
        def selective_failure(terms, x, t, initial):
            if np.array_equal(initial, bad_initial):
                raise RuntimeError('one action fails')
            return forward(terms, x, t, initial)
        with patch.object(original, 'candidate_forecast', side_effect=selective_failure):
            result = design.select_action('rich_then_information', self.committee, pool, self.observations,
                                          used_action_ids=['fixed-rich'], selected=selected())
        self.assertEqual(result['model_forecast_failures'], 1)
        self.assertNotEqual(result['selected_action_id'], pool[1]['id'])
        self.assertEqual(result['model_count_used_for_information'], 1)
        self.assertEqual(result['forecast_solver_calls_attempted'], 5)

    def test_cost_normalization_and_richness_ties_are_explicit(self):
        # Under a pure decay anchor all fixed-rich phase variants have identical
        # information; retain the declared fixed-rich order rather than sampling.
        first = design.select_action('rich_then_information', [], self.pool[:4], self.observations,
                                     used_action_ids=['fixed-rich'], selected=selected())
        self.assertEqual(first['selected_action_id'], 'fixed-rich-1')
        costs = {a['id']: 1. for a in self.pool[:4]}
        costs['fixed-rich-1'] = 10.
        second = design.select_action('rich_then_information', [], self.pool[:4], self.observations,
                                      used_action_ids=['fixed-rich'], selected=selected(), cost_per_action=costs)
        by = {r['action_id']: r for r in first['scores']}
        for row in second['scores']:
            self.assertAlmostEqual(row['score'], by[row['action_id']]['score']/costs[row['action_id']], places=12)
        self.assertEqual(second['selected_action_id'], 'fixed-rich-2')

    def test_no_mutation_json_safe_receipts_and_exhaustion(self):
        before = deepcopy(self.observations)
        anchor = selected()
        frozen_anchor = deepcopy(anchor)
        result = design.select_action('rich_then_information', [], list(reversed(self.pool[:4])), self.observations,
                                      used_action_ids=['fixed-rich'], selected=anchor)
        json.dumps(result, allow_nan=False)
        self.assertEqual(anchor, frozen_anchor)
        for old, new in zip(before, self.observations):
            for key in ('x', 't', 'u'):
                np.testing.assert_array_equal(old[key], new[key])
        exhausted = design.select_action('rich_then_information', [], self.pool[:2], self.observations,
                                        used_action_ids=[a['id'] for a in self.pool[:2]], selected=None)
        self.assertEqual(exhausted['status'], 'exhausted')
        self.assertIsNone(exhausted['selected_action'])
        imports = '\n'.join(n for n in inspect.getsource(design).splitlines()
                            if n.startswith(('from ', 'import ')))
        self.assertNotIn('provider', imports)


if __name__ == '__main__':
    unittest.main()
