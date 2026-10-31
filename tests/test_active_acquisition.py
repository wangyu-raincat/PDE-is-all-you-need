"""Manufactured checks for acquisition scores, guards and observation isolation."""
from copy import deepcopy
import json
import unittest
from unittest.mock import patch

import numpy as np

from active_study import actions
from active_study import acquisition as acq
from pde_discovery import generic_engine as base


def observed(seed=0, noise=0., frames=101):
    x = np.arange(64)*2*np.pi/64
    t = np.linspace(.1, 1.1, frames)
    u = np.exp(-t[:, None])*np.sin(x)[None, :]
    u += np.random.default_rng(seed).normal(0., noise, u.shape)
    return {'x': x, 't': t, 'u': u, 'boundary': 'periodic'}


def member(name, value, identity=None):
    return {'candidate_id': identity or name, 'terms': [{'name': name, 'coefficient': value}]}


class ActiveAcquisitionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pool = actions.make_action_pool(17)
        cls.observations = [observed(i) for i in range(3)]
        cls.committee = [member('u', -1., 'decay'), member('u_xx', 1., 'diffusion')]

    def test_linear_acceleration_matches_existing_solver_with_affine_forcing(self):
        x = self.observations[0]['x']
        t = np.asarray(acq.ACQUISITION_TIMES)
        initial = actions.action_field(actions.fixed_rich_action(), x)
        terms = [{'name': name, 'coefficient': coefficient} for name, coefficient in
                 (('1', .03), ('u', -.17), ('u_x', -.4), ('u_xx', .07), ('u_xxxx', -.001))]
        accelerated = acq.candidate_forecast(terms, x, t, initial)
        reference = base.forecast(terms, x, t, initial)
        np.testing.assert_allclose(accelerated, reference, rtol=2e-11, atol=2e-13)

    def test_nonlinear_candidate_uses_existing_forecast_without_future_input(self):
        x = self.observations[0]['x']
        t = np.asarray(acq.ACQUISITION_TIMES)
        initial = actions.action_field(actions.fixed_rich_action(), x)
        terms = [{'name': 'u_xx', 'coefficient': .08}, {'name': 'u*u_x', 'coefficient': -.5}]
        with patch.object(acq.solver, 'forecast', wraps=base.forecast) as forward:
            result = acq.candidate_forecast(terms, x, t, initial)
        self.assertEqual(forward.call_count, 1)
        self.assertEqual(len(forward.call_args.args), 4)
        self.assertEqual(result.shape, (3, 64))
        self.assertTrue(np.isfinite(result).all())

    def test_unbounded_forward_operators_and_growth_are_rejected(self):
        x = self.observations[0]['x']
        initial = np.sin(x)
        for terms in ([{'name': 'u_xx', 'coefficient': -.1}],
                      [{'name': 'u_xxxx', 'coefficient': .01}],
                      [{'name': 'u', 'coefficient': 1000.}]):
            with self.subTest(terms=terms), self.assertRaises((ValueError, FloatingPointError)):
                acq.candidate_forecast(terms, x, np.asarray(acq.ACQUISITION_TIMES), initial)

    def test_changing_mode_breaks_observational_alias_and_single_mode_does_not(self):
        pool = [a for a in self.pool if a['id'] in ('single-1-0', 'single-2-0')]
        for policy in ('ensemble_variance', 'structural_disagreement'):
            result = acq.select_action(policy, self.committee, pool, self.observations)
            by = {r['action_id']: r for r in result['scores']}
            self.assertEqual(result['selected_action_id'], 'single-2-0')
            self.assertEqual(by['single-1-0']['score'], 0.)
            self.assertGreater(by['single-2-0']['score'], 0.)
            self.assertFalse(result['prospective_observations_used'])
            self.assertEqual(result['actual_experiment_queries'], 0)
        only = acq.select_action('structural_disagreement', self.committee, pool[:1], self.observations)
        self.assertEqual(only['status'], 'fallback_coverage')
        self.assertFalse(only['identification_certainty_claimed'])

    def test_structures_receive_equal_mass_and_variance_is_not_path_frequency(self):
        predictions = np.array([np.full((2, 4), value) for value in (0., 2., 10.)])
        result = acq.variance_components(predictions, [('u',), ('u',), ('u_xx',)])
        self.assertAlmostEqual(result['between_structure_variance'], 20.25)
        self.assertAlmostEqual(result['within_structure_variance'], .5)
        self.assertAlmostEqual(result['ensemble_variance'], 56/3)
        self.assertEqual(result['structure_count'], 2)

    def test_exact_duplicate_vectors_do_not_gain_votes(self):
        pool = self.pool[:2]
        expected = acq.select_action('structural_disagreement', self.committee, pool, self.observations)
        duplicate = deepcopy(self.committee[0])
        duplicate['candidate_id'] = 'another-lambda-alias'
        result = acq.select_action('structural_disagreement', self.committee+[duplicate], pool, self.observations)
        self.assertEqual(result['distinct_coefficient_vectors'], 2)
        self.assertEqual(result['raw_committee_members'], 3)
        self.assertEqual(result['scores'], expected['scores'])
        self.assertEqual(result['selected_action_id'], expected['selected_action_id'])

    def test_failure_never_creates_a_score_from_a_smaller_committee(self):
        pool = self.pool[:2]
        grid = self.observations[0]['x']
        bad_initial = actions.action_field(pool[0], grid)
        original = acq.candidate_forecast

        def selective_failure(terms, x, t, initial):
            if terms[0]['name'] == 'u' and np.array_equal(initial, bad_initial):
                raise RuntimeError('manufactured candidate failure')
            return original(terms, x, t, initial)

        with patch.object(acq, 'candidate_forecast', side_effect=selective_failure):
            result = acq.select_action('structural_disagreement', self.committee, pool, self.observations)
        by = {r['action_id']: r for r in result['scores']}
        bad = by[pool[0]['id']]
        self.assertIsNone(bad['score'])
        self.assertEqual(bad['valid_model_count'], 1)
        self.assertEqual(bad['status'], 'ineligible_candidate_failure')
        self.assertEqual(result['model_forecast_failures'], 1)
        self.assertEqual(result['model_forecast_attempts'], 4)
        self.assertEqual(result['selected_action_id'], pool[1]['id'])

    def test_every_failed_candidate_is_retained_in_explicit_fallback(self):
        with patch.object(acq, 'candidate_forecast', side_effect=RuntimeError('manufactured failure')):
            result = acq.select_action('ensemble_variance', self.committee, self.pool[:2], self.observations)
        self.assertEqual(result['status'], 'fallback_coverage')
        self.assertEqual(result['model_forecast_failures'], 4)
        self.assertTrue(all(len(r['failed_models']) == 2 for r in result['scores']))
        self.assertTrue(all(r['score'] is None for r in result['scores']))
        self.assertFalse(result['identification_certainty_claimed'])

    def test_single_structure_and_empty_committee_never_imply_certainty(self):
        for committee in ([], [member('u_xx', .1), member('u_xx', .2)]):
            result = acq.select_action('structural_disagreement', committee, self.pool[:2], self.observations)
            self.assertTrue(result['fallback_used'])
            self.assertFalse(result['identification_certainty_claimed'])
            self.assertEqual(result['model_forecast_attempts'], 0)
        within = acq.select_action('ensemble_variance', [member('u_xx', .1), member('u_xx', .2)],
                                  self.pool[:2], self.observations)
        self.assertEqual(within['status'], 'selected')

    def test_rich_baseline_is_strong_predetermined_and_never_repeats(self):
        used = []
        for expected in ('fixed-rich', 'fixed-rich-1', 'fixed-rich-2', 'fixed-rich-3'):
            result = acq.select_action('fixed_rich', self.committee, self.pool, self.observations,
                                      used_action_ids=used)
            self.assertEqual(result['selected_action_id'], expected)
            self.assertEqual(result['model_forecast_attempts'], 0)
            used.append(expected)
        exhausted = acq.select_action('random', [], self.pool[:2], self.observations,
                                     used_action_ids=[a['id'] for a in self.pool[:2]])
        self.assertEqual(exhausted['status'], 'exhausted')
        self.assertIsNone(exhausted['selected_action'])

    def test_random_selection_is_seeded_order_independent_and_excludes_used_ids(self):
        used = [a['id'] for a in self.pool[:7]]
        first = acq.select_action('random', self.committee, self.pool, self.observations,
                                 seed=93, used_action_ids=used)
        second = acq.select_action('random', list(reversed(self.committee)), list(reversed(self.pool)),
                                  self.observations, seed=93, used_action_ids=list(reversed(used)))
        self.assertEqual(first['selected_action'], second['selected_action'])
        self.assertNotIn(first['selected_action_id'], used)
        self.assertEqual(first['available_actions'], 25)

    def test_noise_estimate_uses_observations_not_oracle_labels_and_exposes_assumption(self):
        training = [observed(i, .02, 1001) for i in range(3)]
        _, receipt = acq.estimate_observation_noise(training)
        self.assertAlmostEqual(receipt['estimated_spatial_variance'], .02**2, delta=.000025)
        self.assertFalse(receipt['true_noise_used'])
        clean = observed()
        x = clean['x']
        clean['u'] = clean['u'] + .02*np.cos(x[:, None]*np.arange(16, 32)[None, :]).sum(axis=1)
        _, physical = acq.estimate_observation_noise([clean])
        self.assertGreater(physical['estimated_spatial_variance'], .001)
        self.assertTrue(physical['physical_upper_band_signal_can_bias_estimate'])
        first = acq.select_action('structural_disagreement', self.committee, self.pool[:2], self.observations)
        metadata = [dict(d, family='invented', true_noise=10000., reference_coefficients={'u': 1000.})
                    for d in self.observations]
        second = acq.select_action('structural_disagreement', self.committee, self.pool[:2], metadata)
        self.assertEqual(first['scores'], second['scores'])
        self.assertEqual(first['noise_estimate'], second['noise_estimate'])
        with self.assertRaises(TypeError):
            acq.select_action('random', [], self.pool, self.observations, true_noise=.1)

    def test_cost_normalization_is_explicit_and_invalid_costs_are_rejected(self):
        pool = [a for a in self.pool if a['id'] in ('single-2-0', 'single-3-0')]
        first = acq.select_action('structural_disagreement', self.committee, pool, self.observations)
        second = acq.select_action('structural_disagreement', self.committee, pool, self.observations,
                                  cost_per_action=2.)
        for a, b in zip(first['scores'], second['scores']):
            self.assertEqual(a['score']/2, b['score'])
        for bad in (0., -1., np.nan, True, {pool[0]['id']: 1.}):
            with self.subTest(cost=bad), self.assertRaises(ValueError):
                acq.select_action('random', [], pool, self.observations, cost_per_action=bad)

    def test_malformed_input_is_rejected_without_mutation_and_receipts_are_json_safe(self):
        original_pool, original_models = deepcopy(self.pool[:2]), deepcopy(self.committee)
        result = acq.select_action('structural_disagreement', original_models, original_pool, self.observations)
        self.assertEqual(original_pool, self.pool[:2])
        self.assertEqual(original_models, self.committee)
        json.dumps(result, allow_nan=False)
        for models in ([member('u', np.nan)], [member('unknown', 1.)], [member('u', True)]):
            with self.assertRaises(ValueError):
                acq.select_action('random', models, self.pool[:2], self.observations)
        for invalid in ([self.pool[0], self.pool[0]], []):
            with self.assertRaises(ValueError):
                acq.select_action('random', [], invalid, self.observations)
        with self.assertRaises(ValueError):
            acq.select_action('random', [], self.pool[:2], self.observations, seed=True)


if __name__ == '__main__':
    unittest.main()
