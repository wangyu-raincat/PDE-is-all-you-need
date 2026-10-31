"""Manufactured protocol fixtures; no formal reference system is executed."""
from contextlib import ExitStack, redirect_stdout
from copy import deepcopy
import hashlib
import io
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from active_study import actions, controller, run
from tests.test_active_controller import fabricated_model, fabricated_observation


class ActivePipelineTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.root = Path(self.stack.enter_context(tempfile.TemporaryDirectory())).resolve()
        self.data = self.root/'active_study/data'
        self.output = self.root/'active_study/results/manufactured'
        for name, value in (('ROOT', self.root), ('STUDY', self.root/'active_study'), ('DATA', self.data)):
            self.stack.enter_context(patch.object(run, name, value))
        self.science = {'manufactured-source': 'immutable-fixture'}
        self.design = {'status': 'frozen', 'manufactured': True}
        self.stack.enter_context(patch.object(run, 'sources', return_value=self.science))
        self.stack.enter_context(patch.object(run, 'validate_design', return_value=self.design))
        self.stack.enter_context(patch.object(run.provider, 'get_task_spec',
            side_effect=lambda role, index: {'role': role, 'index': index, 'manufactured': True}))
        self.clean_oracle = self.stack.enter_context(patch.object(run.provider, 'generate_clean',
            side_effect=AssertionError('No hidden reference field may be generated in this test.')))
        self.stack.enter_context(patch.object(run.engine, 'discover', side_effect=lambda *a, **k: fabricated_model()))
        self.stack.enter_context(patch.object(run.provider, 'generate_observation', side_effect=self.query))
        self.data.mkdir(parents=True)
        run.write(self.data/'calibration.json', {'manufactured': True})
        tasks = []
        for index in run.INDICES:
            initial = actions.initial_actions(857000+index)
            pool = actions.make_action_pool(856000+index)
            spec = run.provider.get_task_spec('formal', index)
            for label, level in run.NOISES:
                identity = run.task_id(index, label)
                task = {'id': identity, 'spec_index': index, 'spec': spec, 'noise_label': label,
                        'noise_level': level, 'pool': pool, 'initial_actions': initial,
                        'observations': {'training': [], 'development': []}, 'failures': []}
                for role in ('training', 'development'):
                    for j, action in enumerate(initial[role]):
                        common = {'index': j, 'action_sha256': actions.action_digest(action),
                                  'noise_seed': run.noise_seed(spec, action)}
                        # Keep all declared conditions while only one has measured fixtures.
                        if index == 0 and label == 'clean':
                            obs = fabricated_observation(.1*j + (.7 if role == 'development' else 0.))
                            relative = f'active_study/data/initial/{identity}/{role}-{j}.npz'
                            run.save_observation(self.root/relative, obs)
                            task['observations'][role].append(dict(common, file=relative,
                                sha256=run.sha(self.root/relative), receipt=self.receipt(obs, action)))
                        else:
                            task['failures'].append(dict(common, role=role, reason='manufactured initial outage',
                                                         charged_experiments=1))
                tasks.append(task)
        self.prepared = {'sources': self.science, 'protocol': self.design, 'tasks': tasks,
                         'calibration_sha256': run.sha(self.data/'calibration.json'),
                         'final_generated': False, 'all_initial_conditions_retained': True,
                         'prepared_at_utc': '2026-01-01T00:00:00+00:00'}
        run.write(self.data/'prepared.json', self.prepared)

    @staticmethod
    def receipt(obs, action):
        field = np.concatenate((obs['prefix_u'][:-1], obs['u']))
        return {'experiments': 1, 'measurement_values': 111*64,
                'action_sha256': actions.action_digest(action),
                'observed_field_sha256': hashlib.sha256(field.tobytes()).hexdigest()}

    def query(self, spec, action, noise_seed, noise_level):
        obs = fabricated_observation(.3)
        return {'observation': obs, 'receipt': self.receipt(obs, action)}

    def freeze(self):
        with redirect_stdout(io.StringIO()):
            return run.freeze(self.output, workers=1)

    def test_complete_initial_success_failure_partition_is_verified_without_reference_access(self):
        verified = run.verify_preparation()
        self.assertEqual(len(verified['tasks']), 24)
        self.assertEqual(sum(len(t['failures']) for t in verified['tasks']), 115)
        self.clean_oracle.assert_not_called()

    def test_preparation_rejects_noise_pool_action_seed_and_missing_or_duplicate_allocations(self):
        mutations = [
            lambda p: p['tasks'][0].update(noise_level=.02),
            lambda p: p['tasks'][0].update(pool=actions.make_action_pool(1)),
            lambda p: p['tasks'][0].update(initial_actions=actions.initial_actions(1)),
            lambda p: p['tasks'][0]['observations']['training'][0].update(noise_seed=1),
            lambda p: p['tasks'][0]['observations']['training'].pop(),
            lambda p: p['tasks'][1]['failures'].append(deepcopy(p['tasks'][1]['failures'][0])),
            lambda p: p['tasks'][1]['failures'][0].update(action_sha256='changed'),
            lambda p: p['tasks'][0]['observations']['training'].reverse(),
            lambda p: p['tasks'][0]['observations']['training'][0].update(file='outside-design.npz'),
            lambda p: p['sources'].update(changed='source'),
        ]
        for mutate in mutations:
            with self.subTest(mutation=mutate):
                changed = deepcopy(self.prepared); mutate(changed)
                run.write(self.data/'prepared.json', changed)
                with self.assertRaises(ValueError): run.verify_preparation()
        run.write(self.data/'prepared.json', self.prepared)

    def test_refreshed_container_hash_cannot_hide_changed_observation_bytes(self):
        changed = deepcopy(self.prepared)
        record = changed['tasks'][0]['observations']['training'][0]
        path = self.root/record['file']; obs = run.load_observation(path)
        obs['u'][3, 7] += .1
        run.save_observation(path, obs); record['sha256'] = run.sha(path)
        run.write(self.data/'prepared.json', changed)
        with self.assertRaisesRegex(ValueError, 'receipt'):
            run.verify_preparation()

    def test_global_freeze_retains_every_failed_condition_and_all_declared_budget_slots(self):
        index = self.freeze()
        self.assertEqual(len(index['records']), 96)
        self.assertEqual(run.verify_freeze(self.output), index)
        episodes = [run.read(self.output/r['file']) for r in index['records']]
        self.assertEqual(sum(len(ep['budget_slots']) for ep in episodes), 480)
        failures = [ep for ep in episodes if ep['status'] == 'initial_source_failure']
        self.assertEqual(len(failures), 92)
        self.assertTrue(all(len(ep['budget_slots']) == 5 for ep in failures))
        self.assertTrue(all(slot['status'] == 'missing_checkpoint_failure' for ep in failures
                            for slot in ep['budget_slots']))
        self.clean_oracle.assert_not_called()

    def test_frozen_ledger_rejects_altered_inputs_cost_and_action_even_with_new_episode_hash(self):
        index = self.freeze()
        receipt = next(r for r in index['records'] if r['id'] == 'task-00--clean--fixed_rich')
        path = self.output/receipt['file']; original = run.read(path)
        mutations = [
            lambda ep: ep['checkpoints'][1]['training_hashes'].__setitem__(-1, 'changed'),
            lambda ep: ep['checkpoints'][1].update(available_measurement_scalars=0),
            lambda ep: ep['queries'][0].update(charged_experiments=0),
            lambda ep: ep['provider_calls'][0].update(action=deepcopy(ep['queries'][1]['action'])),
            lambda ep: ep['queries'][0].update(observation_sha256='changed'),
            lambda ep: ep['checkpoints'][0]['next_proposal'].update(selected_action=deepcopy(ep['queries'][1]['action'])),
            lambda ep: ep['budget_slots'].pop(),
        ]
        for mutate in mutations:
            with self.subTest(mutation=mutate):
                ep = deepcopy(original); mutate(ep); run.write(path, ep)
                changed_index = deepcopy(index)
                next(r for r in changed_index['records'] if r['id'] == receipt['id'])['sha256'] = run.sha(path)
                run.write(self.output/'freeze_index.json', changed_index)
                with self.assertRaises(ValueError): run.verify_freeze(self.output)
        run.write(path, original); run.write(self.output/'freeze_index.json', index)

    def test_global_release_is_not_reset_by_new_output_or_interrupted_attempt(self):
        index = self.freeze()
        first = run._register_final_release(self.output, index)
        self.assertTrue(first['is_first_final_release'])
        second_output = self.root/'another-output'
        run.write(second_output/'freeze_index.json', index)
        second = run._register_final_release(second_output, index)
        self.assertFalse(second['is_first_final_release'])
        self.assertEqual(first['sha256'], second['sha256'])
        marker = run.read(self.data/'final_release.json')
        self.assertTrue(marker['marker_created_before_final_generation'])
        self.assertTrue(marker['interrupted_first_release_remains_consumed'])
        self.clean_oracle.assert_not_called()

    def test_fresh_preparation_timestamp_keeps_same_fixed_tasks_retrospective(self):
        index = self.freeze()
        first = run._register_final_release(self.output, index)
        prepared = deepcopy(self.prepared)
        prepared['prepared_at_utc'] = '2026-02-01T00:00:00+00:00'
        run.write(self.data/'prepared.json', prepared)
        current = deepcopy(index); current['prepared_sha256'] = run.sha(self.data/'prepared.json')
        second_output = self.root/'fresh-clone-output'
        run.write(second_output/'freeze_index.json', current)
        second = run._register_final_release(second_output, current)
        self.assertFalse(second['is_first_final_release'])
        self.assertEqual(first['sha256'], second['sha256'])
        self.assertEqual(second['prepared_sha256'], current['prepared_sha256'])
        self.assertEqual(run.read(self.data/'final_release.json')['first_prepared_sha256'], index['prepared_sha256'])

    def test_final_release_precedes_any_reference_attempt_and_reproduction_is_labeled(self):
        self.freeze()

        def no_reference(spec, action):
            self.assertTrue((self.data/'final_release.json').is_file())
            raise ValueError('manufactured final-source outage')

        def score(task):
            receipt = task[1]
            return [{'id': receipt['id']+f'--budget-{budget}'} for budget in run.BUDGETS]

        with patch.object(run.provider, 'generate_clean', side_effect=no_reference) as oracle, \
                patch.object(run, '_score_episode', side_effect=score), redirect_stdout(io.StringIO()):
            first = run.evaluate(self.output, workers=1)
            second_output = self.root/'reproduction'
            for folder in ('episodes', 'queries'):
                shutil.copytree(self.output/folder, second_output/folder)
            shutil.copy2(self.output/'freeze_index.json', second_output/'freeze_index.json')
            second = run.evaluate(second_output, workers=1)
        self.assertEqual(oracle.call_count, 72)
        self.assertTrue(first['is_first_final_release'])
        self.assertFalse(second['is_first_final_release'])
        self.assertIn('retrospective', second['evaluation_label'])
        self.assertEqual(len(second['records']), 480)
        self.assertTrue(all(r['status'] == 'failed' for refs in second['final_reference_records'].values() for r in refs))


if __name__ == '__main__':
    unittest.main()
