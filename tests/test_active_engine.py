"""Manufactured-data checks for active structural uncertainty, not final evidence."""
from copy import deepcopy
import itertools
import json
import unittest
from unittest.mock import patch

import numpy as np
from threadpoolctl import threadpool_limits

from active_study import engine


def observation(phase=0., seed=0, noise=0., rich=False, diffusion=.12, velocity=0., amplitude=1.):
    x = np.arange(64)*2*np.pi/64
    t = np.linspace(.1, 1.1, 101)
    modes = ((1, 1., phase), (3, .35, -.4*phase), (5, .2, .3*phase)) if rich else ((1, 1., phase),)
    u = sum(amplitude*scale*np.exp(-diffusion*mode**2*t[:, None])
            *np.sin(mode*x[None, :]-velocity*mode*t[:, None]+angle)
            for mode, scale, angle in modes)
    u += noise*np.random.default_rng(seed).standard_normal(u.shape)
    return {'x': x, 't': t, 'u': u, 'boundary': 'periodic'}


def data(noise=0., rich=False, velocity=0.):
    training = [observation(i/3, i, noise, rich, velocity=velocity) for i in range(3)]
    development = [observation(.7+i/4, 10+i, noise, rich, velocity=velocity) for i in range(2)]
    return training, development


class ActiveEngineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.thread_limit = threadpool_limits(limits=1)

    @classmethod
    def tearDownClass(cls):
        cls.thread_limit.restore_original_limits()

    def test_single_mode_retains_decay_diffusion_and_fourth_order_alternatives(self):
        training, development = data()
        result = engine.discover(training, development, bootstrap_replicates=0)
        supports = {tuple(s) for s in result['candidate_supports']}
        self.assertTrue({('u',), ('u_xx',), ('u_xxxx',)} <= supports)
        self.assertEqual(result['status'], 'observational_ambiguity')
        self.assertFalse(result['diagnostics']['unique_physical_equation_established'])
        self.assertLess(result['diagnostics']['full_library_rank'], len(engine.TERM_NAMES))

    def test_noisy_single_mode_keeps_observational_alternatives(self):
        training, development = data(noise=.004)
        result = engine.discover(training, development, bootstrap_replicates=0)
        supports = {tuple(s) for s in result['candidate_supports']}
        self.assertTrue({('u',), ('u_xx',), ('u_xxxx',)} <= supports)
        self.assertEqual(result['status'], 'observational_ambiguity')
        for record in result['diagnostics']['preprocessing']['training']:
            self.assertEqual(record['retained_modes'], [0, 1])
            self.assertTrue(record['time_shared_mask'])
            self.assertFalse(record['true_noise_used'])
            self.assertAlmostEqual(record['removed_observed_energy_fraction'],
                                   record['relative_field_change']**2)
        json.dumps(result, allow_nan=False)

    def test_transport_single_mode_retains_more_than_first_four_structures(self):
        training, development = data(velocity=.4)
        result = engine.discover(training, development, bootstrap_replicates=0)
        supports = {frozenset(s) for s in result['candidate_supports']}
        alternatives = {frozenset((even, odd)) for even in ('u', 'u_xx', 'u_xxxx')
                        for odd in ('u_x', 'u_xxx')}
        self.assertTrue(alternatives <= supports)
        self.assertGreaterEqual(len(supports), 6)

    def test_a_rich_new_trajectory_resolves_the_manufactured_linear_alias(self):
        training, development = data()
        before = engine.discover(training, development, bootstrap_replicates=0)
        training.append(observation(.4, rich=True))
        after = engine.discover(training, development, bootstrap_replicates=0)
        self.assertGreater(len(before['candidate_supports']), 1)
        self.assertEqual(after['selected']['support'], ['u_xx'])
        self.assertEqual(after['candidate_supports'], [['u_xx']])
        self.assertAlmostEqual(after['selected']['terms'][0]['coefficient'], .12, delta=2e-6)

    def test_amplitude_only_addition_does_not_resolve_spatial_alias(self):
        training, development = data()
        training.append(observation(.4, amplitude=1.5))
        result = engine.discover(training, development, bootstrap_replicates=0)
        self.assertTrue({('u',), ('u_xx',), ('u_xxxx',)} <= {tuple(s) for s in result['candidate_supports']})
        self.assertEqual(result['status'], 'observational_ambiguity')

    def test_rich_noisy_observations_find_the_known_diffusion_term(self):
        training, development = data(noise=.004, rich=True)
        result = engine.discover(training, development, bootstrap_replicates=0)
        self.assertEqual(result['selected']['support'], ['u_xx'])
        self.assertAlmostEqual(result['selected']['terms'][0]['coefficient'], .12, delta=.002)
        self.assertFalse(result['diagnostics']['singleton_committee_proves_identifiability'])

    def test_oracle_metadata_and_prefixes_are_not_read_and_inputs_remain_unchanged(self):
        class ForbiddenArray:
            def __array__(self, *args, **kwargs):
                raise AssertionError('Forbidden metadata was accessed.')
        training, development = data(noise=.004)
        pristine = deepcopy(training+development)
        for item in training+development:
            item.update(prefix_u=ForbiddenArray(), prefix_t=ForbiddenArray(), exact_ut=ForbiddenArray(),
                        true_coefficients=ForbiddenArray(), family=ForbiddenArray(), true_noise=ForbiddenArray())
        original_block = engine.weak.weak_block
        inspected = []
        def inspect(item, variant):
            inspected.append(set(item))
            self.assertEqual(set(item), {'x', 't', 'u', 'boundary'})
            return original_block(item, variant=variant)
        with patch.object(engine.weak, 'weak_block', side_effect=inspect):
            result = engine.discover(training, development, bootstrap_replicates=0)
        self.assertEqual(len(inspected), 5)
        for old, new in zip(pristine, training+development):
            for key in ('x', 't', 'u'):
                np.testing.assert_array_equal(old[key], new[key])
        self.assertFalse(result['diagnostics']['final_data_used'])

    def test_bootstrap_reruns_adaptive_structure_selection_and_caches_exact_draws(self):
        training, development = data(noise=.004, rich=True)
        original = engine.sparse.fit_blocks
        input_signatures = []
        def tracked(blocks, method):
            self.assertEqual(method, 'adaptive')
            input_signatures.append(tuple(float(np.sum(block['y'])) for block in blocks))
            return original(blocks, method)
        with patch.object(engine.sparse, 'fit_blocks', side_effect=tracked):
            result = engine.discover(training, development, seed=4, bootstrap_replicates=8)
        uncertainty = result['uncertainty']
        self.assertEqual(len(uncertainty['records']), 8)
        self.assertGreater(len(input_signatures), 1)
        self.assertGreater(len(set(input_signatures)), 1)
        self.assertLessEqual(len(input_signatures), 9)
        self.assertEqual(len(input_signatures), result['diagnostics']['computation']['adaptive_l1_complete_refits'])
        self.assertTrue(any(record['cached_exact_multiplicity'] for record in uncertainty['records']))
        self.assertTrue(all(record['structure_selection_repeated'] for record in uncertainty['records']))
        self.assertEqual(uncertainty['number_of_available_multiplicity_vectors'], 10)
        for name in engine.TERM_NAMES:
            count = sum(name in record['selected_support'] for record in uncertainty['records'] if record['status'] == 'selected')
            self.assertEqual(uncertainty['raw_counts'][name], count)
            self.assertEqual(uncertainty['term_inclusion_frequencies'][name], count/8)
        self.assertFalse(uncertainty['calibrated'])
        self.assertFalse(uncertainty['probability_claimed'])

    def test_failed_resamples_do_not_disappear_from_frequency_denominator(self):
        training, development = data(noise=.004)
        original = engine.sparse.fit_blocks
        calls = 0
        def fail_second(blocks, method):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise ValueError('Manufactured failed complete refit.')
            return original(blocks, method)
        with patch.object(engine.sparse, 'fit_blocks', side_effect=fail_second):
            result = engine.discover(training, development, seed=4, bootstrap_replicates=8)
        uncertainty = result['uncertainty']
        self.assertGreater(uncertainty['failed_replicates'], 0)
        self.assertEqual(uncertainty['completed_replicates']+uncertainty['failed_replicates'], 8)
        self.assertTrue(all(value is None for value in uncertainty['term_inclusion_frequencies'].values()))
        for name, bounds in uncertainty['frequency_bounds_including_failures'].items():
            self.assertEqual(bounds[0], uncertainty['raw_counts'][name]/8)
            self.assertEqual(bounds[1], (uncertainty['raw_counts'][name]+uncertainty['failed_replicates'])/8)
        self.assertTrue(any('Manufactured' in item['reason'] for item in result['diagnostics']['candidate_generation_failures']))
        json.dumps(result, allow_nan=False)

    def test_committee_deduplicates_vectors_without_turning_paths_into_votes(self):
        training, development = data()
        result = engine.discover(training, development, seed=5, bootstrap_replicates=8)
        keys = [engine._vector_key(candidate) for candidate in result['committee']]
        self.assertEqual(len(keys), len(set(keys)))
        self.assertTrue(all(candidate['provenance'] for candidate in result['committee']))
        self.assertFalse(result['protocol']['path_frequency_weighting'])
        self.assertFalse(result['protocol']['cross_lambda_objective_comparison'])

    def test_development_observations_do_not_refit_training_coefficients(self):
        training, development = data(rich=True)
        result = engine.discover(training, development, bootstrap_replicates=0)
        other = [observation(.2+i/3, rich=True, diffusion=.07) for i in range(2)]
        changed = engine.discover(training, other, bootstrap_replicates=0)
        self.assertEqual([candidate['terms'] for candidate in result['candidates']],
                         [candidate['terms'] for candidate in changed['candidates']])
        self.assertNotEqual([candidate['development'] for candidate in result['candidates']],
                            [candidate['development'] for candidate in changed['candidates']])

    def test_unexcited_observations_are_explicitly_insufficient(self):
        training, development = data()
        for item in training+development:
            item['u'][:] = 1.
        result = engine.discover(training, development, seed=1, bootstrap_replicates=4)
        self.assertIsNone(result['selected'])
        self.assertEqual(result['committee'], [])
        self.assertEqual(result['status'], 'insufficient')
        self.assertEqual(result['uncertainty']['failed_replicates'], 4)
        self.assertFalse(result['diagnostics']['unique_physical_equation_established'])
        json.dumps(result, allow_nan=False)

    def test_alias_closure_is_complete_and_finite_without_first_k_truncation(self):
        pairs = list(itertools.combinations(range(9), 2))
        closed = engine._expand_supports([(0, 1)], pairs)
        self.assertEqual(set(closed), set(itertools.combinations(range(9), 2)))
        self.assertEqual(len(closed), 36)

    def test_forward_operator_guard_keeps_rejected_alias_evidence(self):
        candidate = {'candidate_id': 'bad', 'terms': [{'name': 'u_xxxx', 'coefficient': 1.}],
                     'fit_success': True, 'support_size': 1}
        x = np.zeros((2, 9)); y = np.ones(2)
        scored = engine._score_candidate(candidate, [(x, y)], 1.)
        self.assertFalse(scored['eligible'])
        self.assertIn('antidissipative', scored['rejection_reason'])
        self.assertEqual(scored['terms'], candidate['terms'])

    def test_broadband_physical_signal_exposes_filter_assumption(self):
        item = observation()
        x = item['x']
        field = sum(.02*np.sin(mode*x+.03*mode) for mode in range(16, 32))
        item['u'] = np.tile(field, (len(item['t']), 1))
        _, diagnostic = engine.preprocess_observation(item)
        self.assertGreater(diagnostic['estimated_coefficient_noise_rms'], 0.)
        self.assertGreater(diagnostic['relative_field_change'], .9)
        self.assertTrue(diagnostic['weak_physical_modes_may_be_removed'])

    def test_validation_rejects_unsupported_inputs_before_fitting(self):
        training, development = data()
        for kwargs in ({'seed': True}, {'seed': -1}, {'bootstrap_replicates': True},
                       {'bootstrap_replicates': -1}, {'bootstrap_replicates': 65}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                engine.discover(training, development, **kwargs)
        with self.assertRaises(ValueError):
            engine.discover(training[:2], development)
        with self.assertRaises(TypeError):
            engine.discover(training, development, true_noise=.004)
        bad = deepcopy(development)
        bad[0]['x'] += .1
        with self.assertRaises(ValueError):
            engine.discover(training, bad)
        bad = deepcopy(training)
        bad[0]['u'] = bad[0]['u'].astype(complex)
        with self.assertRaises(ValueError):
            engine.discover(bad, development)

    def test_fixed_seed_gives_reproducible_serializable_receipts(self):
        training, development = data(noise=.004)
        first = engine.discover(training, development, seed=13, bootstrap_replicates=8)
        second = engine.discover(training, development, seed=13, bootstrap_replicates=8)
        self.assertEqual(first, second)
        json.dumps(first, allow_nan=False)


if __name__ == '__main__':
    unittest.main()
