"""Independent protocol checks using fabricated observations, never hidden PDEs."""
from copy import deepcopy
import inspect
import json
import unittest
from unittest.mock import patch

import numpy as np

from active_study import actions, controller


def fabricated_observation(phase=0.):
    x = np.arange(64) * (2 * np.pi / 64)
    full_t = np.linspace(0., 1.1, 111)
    field = np.exp(-full_t[:, None]) * np.sin(x[None, :] + phase)
    return {'x': x, 't': full_t[10:].copy(), 'u': field[10:].copy(),
            'prefix_t': full_t[:11].copy(), 'prefix_u': field[:11].copy(),
            'boundary': 'periodic'}


def fabricated_model():
    return {'selected': {'terms': [{'name': 'u', 'coefficient': -1.}]},
            'committee': [], 'candidate_supports': [['u']],
            'uncertainty': {'calibrated': False},
            'diagnostics': {'status': 'candidate',
                            'unique_physical_equation_established': False}}


class ActiveControllerTests(unittest.TestCase):
    def setUp(self):
        self.training = [fabricated_observation(i / 7) for i in range(3)]
        self.development = [fabricated_observation(i / 5) for i in range(2)]
        self.pool = actions.make_action_pool(27)
        self.query_observation = fabricated_observation(.3)
        self.cost = 111 * 64

    def query(self, action, index):
        return {'observation': deepcopy(self.query_observation),
                'receipt': {'action_id': action['id'], 'index': index}}

    def run_episode(self, query=None, **kwargs):
        return controller.run_episode(self.training, self.development, self.pool,
                                      query or self.query, policy=kwargs.pop('policy', 'fixed_rich'),
                                      **kwargs)

    def test_all_initial_development_and_prefix_samples_are_charged_at_every_budget(self):
        with patch.object(controller.engine, 'discover', return_value=fabricated_model()) as discover:
            result = self.run_episode(seed=110)
        self.assertEqual([c['budget'] for c in result['checkpoints']], [5, 6, 7, 8, 9])
        self.assertEqual(result['total_experiment_attempts'], 9)
        self.assertEqual(result['initial_measurement_scalars'], 5 * self.cost)
        self.assertEqual(result['charged_measurement_scalars'], 9 * self.cost)
        self.assertEqual(len(result['queries']), 4)
        self.assertEqual(len({q['action']['id'] for q in result['queries']}), 4)
        for i, checkpoint in enumerate(result['checkpoints']):
            self.assertEqual(checkpoint['training_trajectories'], 3 + i)
            self.assertEqual(checkpoint['development_trajectories'], 2)
            self.assertEqual(checkpoint['charged_measurement_scalars'], (5 + i) * self.cost)
            self.assertEqual(checkpoint['available_measurement_scalars'], (5 + i) * self.cost)
            self.assertFalse(checkpoint['final_used_for_selection'])
        self.assertEqual([c.kwargs['seed'] for c in discover.call_args_list], list(range(110, 115)))
        json.dumps(result, allow_nan=False)

    def test_started_legal_failures_use_budget_without_free_replacement(self):
        def fail(action, index):
            raise OSError('manufactured measurement outage')

        with patch.object(controller.engine, 'discover', return_value=fabricated_model()):
            result = self.run_episode(fail)
        self.assertEqual(result['total_experiment_attempts'], 9)
        self.assertEqual(len(result['queries']), 4)
        self.assertEqual(result['charged_measurement_scalars'], 9 * self.cost)
        self.assertTrue(all(q['status'] == 'failed' for q in result['queries']))
        self.assertTrue(all(q['returned_measurement_scalars'] == 0 for q in result['queries']))
        self.assertTrue(all(c['training_trajectories'] == 3 for c in result['checkpoints']))
        self.assertTrue(all(c['available_measurement_scalars'] == 5 * self.cost
                            for c in result['checkpoints']))
        self.assertEqual([c['budget'] for c in result['checkpoints']], [5, 6, 7, 8, 9])

    def test_outside_and_changed_pool_actions_are_rejected_before_query(self):
        outside = deepcopy(self.pool[0]); outside['id'] = 'not-in-declared-pool'
        changed = deepcopy(self.pool[1]); changed['id'] = self.pool[0]['id']
        for illegal in (outside, changed, None):
            with self.subTest(action=illegal), \
                    patch.object(controller.engine, 'discover', return_value=fabricated_model()), \
                    patch.object(controller.acquisition, 'select_action', return_value={'selected_action': illegal}), \
                    patch.object(self, 'query', wraps=self.query) as query:
                result = self.run_episode(query)
            query.assert_not_called()
            self.assertEqual(result['status'], 'stopped_failure')
            self.assertEqual(result['total_experiment_attempts'], 5)
            self.assertEqual(len(result['checkpoints']), 1)
            self.assertIn('acquisition_failed', result['terminal_reason'])

    def test_duplicate_action_is_rejected_without_second_charge_or_provider_call(self):
        proposal = {'selected_action': self.pool[0]}
        with patch.object(controller.engine, 'discover', return_value=fabricated_model()), \
                patch.object(controller.acquisition, 'select_action', return_value=proposal), \
                patch.object(self, 'query', wraps=self.query) as query:
            result = self.run_episode(query)
        self.assertEqual(query.call_count, 1)
        self.assertEqual(result['total_experiment_attempts'], 6)
        self.assertEqual([c['budget'] for c in result['checkpoints']], [5, 6])
        self.assertEqual(result['status'], 'stopped_failure')

    def test_reference_metadata_and_clean_fields_never_reach_learning_or_policy_inputs(self):
        allowed = {'x', 't', 'u', 'boundary', 'prefix_t', 'prefix_u'}
        forbidden = {'family': 'secret', 'reference_coefficients': {'u': 999.},
                     'true_noise': 999., 'clean_future': np.full((101, 64), np.nan)}
        for record in self.training + self.development:
            record.update(forbidden)
        learned_lengths = []

        def discover(train, dev, **kwargs):
            self.assertTrue(all(set(d) == allowed for d in train + dev))
            learned_lengths.append(len(train))
            return fabricated_model()

        real_select = controller.acquisition.select_action

        def select(policy, committee, pool, observations, **kwargs):
            self.assertTrue(all(set(d) == allowed for d in observations))
            self.assertFalse(any(k in kwargs for k in forbidden))
            return real_select(policy, committee, pool, observations, **kwargs)

        def query(action, index):
            self.assertEqual(set(action), {'id', 'a', 'b'})
            return {'observation': dict(deepcopy(self.query_observation), **forbidden),
                    'clean': forbidden['clean_future'], 'spec': forbidden, 'receipt': {'index': index}}

        with patch.object(controller.engine, 'discover', side_effect=discover), \
                patch.object(controller.acquisition, 'select_action', side_effect=select):
            result = self.run_episode(query)
        self.assertEqual(learned_lengths, [3, 4, 5, 6, 7])
        self.assertFalse(result['final_used'])
        self.assertFalse(result['reference_coefficients_used'])
        parameters = inspect.signature(controller.run_episode).parameters
        self.assertFalse({'final', 'truth', 'reference', 'spec'} & set(parameters))

    def test_shared_initial_fit_is_copied_and_later_fits_use_common_round_seeds(self):
        initial = fabricated_model()
        with patch.object(controller.engine, 'discover', side_effect=lambda *a, **k: fabricated_model()) as discover:
            first = self.run_episode(seed=200, initial_model=initial)
            second = self.run_episode(seed=200, initial_model=initial, policy='random')
        self.assertEqual(discover.call_count, 8)
        self.assertEqual([c.kwargs['seed'] for c in discover.call_args_list], [201, 202, 203, 204] * 2)
        self.assertEqual(first['checkpoints'][0]['model'], second['checkpoints'][0]['model'])
        first['checkpoints'][0]['model']['diagnostics']['status'] = 'mutated-after-return'
        self.assertEqual(initial['diagnostics']['status'], 'candidate')
        self.assertEqual(second['checkpoints'][0]['model']['diagnostics']['status'], 'candidate')

    def test_changed_returned_geometry_is_a_charged_failed_experiment(self):
        bad = deepcopy(self.query_observation)
        bad['t'] = bad['t'] + .01
        bad['prefix_t'] = bad['prefix_t'] + .01

        def changed_geometry(action, index):
            return {'observation': bad, 'receipt': {}}

        with patch.object(controller.engine, 'discover', return_value=fabricated_model()):
            result = self.run_episode(changed_geometry, max_additions=1)
        self.assertEqual(result['total_experiment_attempts'], 6)
        self.assertEqual(result['queries'][0]['status'], 'failed')
        self.assertIn('geometry', result['queries'][0]['failure_reason'])
        self.assertEqual(result['checkpoints'][-1]['available_measurement_scalars'], 5 * self.cost)
        self.assertEqual(result['checkpoints'][-1]['charged_measurement_scalars'], 6 * self.cost)

    def test_initial_prefix_endpoint_disagreement_is_rejected_before_learning(self):
        self.training[0]['prefix_u'][-1, 0] += .1
        with patch.object(controller.engine, 'discover') as discover, \
                patch.object(self, 'query', wraps=self.query) as query, self.assertRaisesRegex(ValueError, 'endpoint'):
            self.run_episode(query)
        discover.assert_not_called()
        query.assert_not_called()

    def test_complex_observations_are_rejected_without_silently_discarding_imaginary_parts(self):
        for key in ('x', 't', 'u', 'prefix_t', 'prefix_u'):
            with self.subTest(field=key):
                value = fabricated_observation()
                value[key] = value[key].astype(complex) + .01j
                with self.assertRaisesRegex(ValueError, 'real'):
                    controller.observation_view(value)

    def test_discovery_failure_is_retained_and_does_not_stop_the_fixed_budget_curve(self):
        with patch.object(controller.engine, 'discover', side_effect=RuntimeError('manufactured fit failure')):
            result = self.run_episode()
        self.assertEqual(len(result['checkpoints']), 5)
        self.assertEqual(result['total_experiment_attempts'], 9)
        self.assertTrue(all(c['model']['selected'] is None for c in result['checkpoints']))
        self.assertTrue(all(c['model']['diagnostics']['status'] == 'insufficient' for c in result['checkpoints']))
        self.assertTrue(all('manufactured' in c['model']['diagnostics']['failure_reason']
                            for c in result['checkpoints']))

    def test_random_policy_reproduction_and_seeded_rounds(self):
        real_select = controller.acquisition.select_action
        calls = []

        def record(*args, **kwargs):
            calls.append((kwargs['seed'], kwargs['used_action_ids']))
            return real_select(*args, **kwargs)

        with patch.object(controller.engine, 'discover', return_value=fabricated_model()), \
                patch.object(controller.acquisition, 'select_action', side_effect=record):
            first = self.run_episode(seed=98, policy='random')
            second = self.run_episode(seed=98, policy='random')
        self.assertEqual([q['action'] for q in first['queries']], [q['action'] for q in second['queries']])
        self.assertEqual([seed for seed, _ in calls], [98, 99, 100, 101] * 2)
        self.assertEqual([len(used) for _, used in calls], [0, 1, 2, 3] * 2)

    def test_callback_cannot_mutate_the_action_record(self):
        def mutating_query(action, index):
            action['a'][0] = 1000.
            action['id'] = 'callback-mutated'
            return {'observation': deepcopy(self.query_observation), 'receipt': {}}

        with patch.object(controller.engine, 'discover', return_value=fabricated_model()):
            result = self.run_episode(mutating_query, max_additions=1)
        record = result['queries'][0]
        self.assertNotEqual(record['action']['id'], 'callback-mutated')
        self.assertEqual(record['action_sha256'], actions.action_digest(record['action']))
        self.assertEqual(record['status'], 'observed')


if __name__ == '__main__':
    unittest.main()
