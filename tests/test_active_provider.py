"""Action budgets and independent references, using reserved roles only."""
import ast
from copy import deepcopy
import inspect
import unittest
from unittest.mock import patch

import numpy as np

from active_study import actions, provider


class ActiveActionTests(unittest.TestCase):
    def test_common_pool_is_deterministic_balanced_and_equal_energy(self):
        pool = actions.make_action_pool(17)
        self.assertEqual(pool, actions.make_action_pool(17))
        self.assertNotEqual(pool, actions.make_action_pool(19))
        self.assertEqual(len(pool), 32)
        self.assertEqual(len({a['id'] for a in pool}), 32)
        self.assertEqual([a['id'] for a in pool[:4]], ['fixed-rich', 'fixed-rich-1', 'fixed-rich-2', 'fixed-rich-3'])
        self.assertEqual(sum(a['id'].startswith('single-') for a in pool), 12)
        self.assertEqual(sum(a['id'].startswith('mixture-') for a in pool), 16)
        x = np.arange(8192) * 2 * np.pi / 8192
        for action in pool:
            u = actions.action_field(action, x)
            self.assertAlmostEqual(float(np.sqrt(np.mean(u*u))), .4, places=12)
            self.assertAlmostEqual(float(np.mean(u)), 0., places=12)
            self.assertLessEqual(float(np.max(np.abs(u))), 1.2)
            self.assertEqual(actions.validate_action(action), action)

    def test_fixed_baseline_is_rich_and_does_not_depend_on_pool_seed(self):
        for i in range(4):
            action = actions.fixed_rich_action(i)
            amplitudes = np.hypot(action['a'], action['b'])
            self.assertTrue(np.all(amplitudes > 0))
            np.testing.assert_allclose(amplitudes, amplitudes[0], rtol=0, atol=1e-15)
            self.assertEqual(action, actions.make_action_pool(1)[i])
            self.assertEqual(action, actions.make_action_pool(2)[i])

    def test_initial_five_actions_have_weak_common_spectrum(self):
        first = actions.initial_actions(5)
        self.assertEqual(first, actions.initial_actions(5))
        self.assertEqual(len(first['training']), 3)
        self.assertEqual(len(first['development']), 2)
        items = first['training'] + first['development']
        self.assertEqual(len({actions.action_digest(a) for a in items}), 5)
        for action in items:
            np.testing.assert_array_equal(np.asarray(action['a'])[1:], np.zeros(5))
            np.testing.assert_array_equal(np.asarray(action['b'])[1:], np.zeros(5))
            self.assertAlmostEqual(float(np.hypot(action['a'][0], action['b'][0])), np.sqrt(2)*.4)

    def test_action_rejects_bad_shape_budget_oracle_fields_and_continuous_peaks(self):
        original = actions.fixed_rich_action()
        cases = []
        a = deepcopy(original); a['a'] = a['a'][:5]; cases.append(a)
        a = deepcopy(original); a['a'][0] = np.nan; cases.append(a)
        a = deepcopy(original); a['b'] = (np.asarray(a['b'])*2).tolist(); cases.append(a)
        a = deepcopy(original); a['family'] = 'diffusion'; cases.append(a)
        a = deepcopy(original); a['a'] = np.asarray(a['a'], complex); cases.append(a)
        # Equal RMS alone is insufficient: aligned cosine modes exceed 1.2.
        cases.append({'id':'bad-crest','a':[0.]*6,'b':[.4*np.sqrt(2/6)]*6})
        for action in cases:
            with self.assertRaises(ValueError): actions.validate_action(action)
        for seed in (-1, True, 1.5, 2**31):
            with self.assertRaises(ValueError): actions.make_action_pool(seed)
        for index in (-1, True, 4):
            with self.assertRaises(ValueError): actions.fixed_rich_action(index)

    def test_action_does_not_mutate_and_accepts_reference_grids(self):
        action = actions.fixed_rich_action(); before = deepcopy(action)
        for n in (16, 64, 512, 1024):
            field = actions.action_field(action, np.arange(n)*2*np.pi/n)
            self.assertEqual(field.shape, (n,))
            self.assertAlmostEqual(float(np.sqrt(np.mean(field*field))), .4, places=12)
        self.assertEqual(action, before)
        with self.assertRaises(ValueError): actions.action_field(action, np.array([1j]))


class ActiveProviderTests(unittest.TestCase):
    def test_task_specs_are_reproducible_private_and_role_separated(self):
        self.assertEqual(len(set(provider.ROLE_SEEDS.values())), 3)
        for i, family in enumerate(provider.FAMILIES):
            spec = provider.get_task_spec('development', i)
            self.assertEqual(spec, provider.get_task_spec('development', i))
            self.assertEqual(spec['family'], family)
            self.assertNotEqual(spec['parameter_seed'], provider.get_task_spec('calibration', i)['parameter_seed'])
            other = provider.get_task_spec('development', i+6)
            self.assertEqual(other['family'], family)
            self.assertNotEqual(other['coefficients'], spec['coefficients'])
            self.assertEqual(provider.reference_terms(spec), [{'name':k,'coefficient':v} for k,v in spec['coefficients'].items()])
        for role, index in [('unknown',0),('development',True),('development',-1),('development',6000)]:
            with self.assertRaises(ValueError): provider.get_task_spec(role,index)

    def test_independent_reference_has_no_discovery_or_forecast_import(self):
        names = []
        for node in ast.walk(ast.parse(inspect.getsource(provider))):
            if isinstance(node, ast.Import): names += [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom): names.append(node.module or '')
        self.assertFalse(any(n.startswith(('pde_discovery','l1_study','weak_study','resolved_study','prefix_study')) for n in names))
        self.assertIn('coefficient_study.generators', names)

    def test_exact_linear_evolution_matches_separate_modal_formula(self):
        action = actions.make_action_pool(31)[10]
        a, b = np.asarray(action['a']), np.asarray(action['b'])
        k = np.arange(1,7)
        for i in range(5):
            spec = provider.get_task_spec('calibration',i)
            data = provider.generate_clean(spec,action)
            c = spec['coefficients']
            decay = c.get('u',0.) - c.get('u_xx',0.)*k*k + c.get('u_xxxx',0.)*k**4
            phase = data['x'][None,:,None]*k[None,None,:] + c.get('u_x',0.)*data['t'][:,None,None]*k[None,None,:]
            truth = np.sum(np.exp(data['t'][:,None,None]*decay[None,None,:]) *
                           (a[None,None,:]*np.sin(phase)+b[None,None,:]*np.cos(phase)),axis=2)
            np.testing.assert_allclose(data['u'], truth, rtol=2e-13, atol=2e-14)
            self.assertEqual(data['u'].shape,(111,64))
            self.assertAlmostEqual(data['t'][-1],1.1)

    def test_observation_views_hide_truth_and_keep_complete_prefix_budget(self):
        spec = provider.get_task_spec('development',1)
        action = actions.fixed_rich_action()
        generated = provider.generate_observation(spec,action,57,.01)
        observed = generated['observation']; clean = generated['clean']; receipt = generated['receipt']
        self.assertEqual(set(observed), {'x','t','u','boundary','prefix_t','prefix_u'})
        self.assertEqual(observed['u'].shape,(101,64)); self.assertEqual(observed['prefix_u'].shape,(11,64))
        np.testing.assert_array_equal(observed['prefix_u'][-1], observed['u'][0])
        self.assertEqual(observed['prefix_t'][-1],observed['t'][0])
        self.assertAlmostEqual(observed['t'][-1]-observed['t'][0],1.)
        self.assertAlmostEqual(observed['prefix_t'][-1]-observed['prefix_t'][0],.1)
        self.assertEqual(receipt['measurement_values'],111*64)
        self.assertEqual(receipt['experiments'],1)
        self.assertTrue(receipt['history_endpoint_counted_once'])
        self.assertFalse(np.array_equal(clean['u'][10:], observed['u']))
        self.assertNotIn('family',receipt); self.assertNotIn('coefficients',receipt); self.assertNotIn('noise_level',receipt)

    def test_noise_uses_fixed_initial_rms_not_future_or_task_dependent_scale(self):
        spec = provider.get_task_spec('development',0); action=actions.fixed_rich_action()
        low = provider.generate_observation(spec,action,67,.001)
        high = provider.generate_observation(spec,action,67,.01)
        noise=np.random.default_rng(np.random.SeedSequence([provider.NOISE_SEED_BASE,67])).standard_normal((111,64))
        np.testing.assert_array_equal(high['observation']['u'],high['clean']['u'][10:]+.004*noise[10:])
        np.testing.assert_allclose(high['observation']['u']-high['clean']['u'][10:],
                                   10*(low['observation']['u']-low['clean']['u'][10:]),atol=2e-15)
        again = provider.generate_observation(spec,action,67,.01)
        self.assertEqual(high['receipt'],again['receipt'])
        np.testing.assert_array_equal(high['observation']['u'],again['observation']['u'])
        changed=provider.generate_observation(spec,action,68,.01)
        self.assertNotEqual(high['receipt']['observed_field_sha256'],changed['receipt']['observed_field_sha256'])

    def test_corrupted_spec_invalid_noise_or_observation_grid_are_rejected(self):
        spec=provider.get_task_spec('development',0);action=actions.fixed_rich_action()
        broken=deepcopy(spec);broken['coefficients']['u']*=2
        with self.assertRaises(ValueError):provider.generate_clean(broken,action)
        for noise in (True,np.nan,-.1,.1,'noise'):
            with self.assertRaises(ValueError):provider.generate_observation(spec,action,1,noise)
        for seed in (True,-1,1.5):
            with self.assertRaises(ValueError):provider.generate_observation(spec,action,seed,.01)
        clean=provider.generate_clean(spec,action)
        bad=deepcopy(clean);bad['t']=bad['t'][:-1]
        with self.assertRaises(ValueError):provider.observation_view(bad,clean['u'])

    def test_burgers_reference_uses_same_ic_under_refinement_and_converges(self):
        spec=provider.get_task_spec('calibration',5);action=actions.fixed_rich_action()
        base=provider.generate_clean(spec,action);fine=provider.generate_clean(spec,action,refined=True)
        np.testing.assert_array_equal(base['x'],fine['x']);np.testing.assert_array_equal(base['t'],fine['t'])
        # Independent FFT projection accumulates ordinary double roundoff.
        np.testing.assert_allclose(base['u'][0],actions.action_field(action,base['x']),rtol=0,atol=3e-15)
        difference=base['u'][1:]-fine['u'][1:]
        relative=float(np.linalg.norm(difference)/np.linalg.norm(fine['u'][1:]))
        self.assertLess(relative,provider.CALIBRATION_RELATIVE_LIMIT)
        self.assertLess(float(np.max(np.abs(difference))),provider.CALIBRATION_MAX_ABSOLUTE_LIMIT)

    def test_calibration_requests_only_reserved_role_and_keeps_source_failures(self):
        roles=[]
        def generate(spec,action,refined=False):
            roles.append(spec['role'])
            if spec['family']=='burgers':raise FloatingPointError('manufactured source failure')
            return {'x':np.arange(64),'t':np.arange(111),'u':np.ones((111,64))}
        with patch.object(provider,'generate_clean',side_effect=generate):result=provider.calibration()
        self.assertEqual(set(roles),{'calibration'})
        self.assertEqual(len(result['records']),12)
        self.assertEqual(sum(r['status']=='source_quality_failure' for r in result['records']),2)
        self.assertFalse(result['all_passed']);self.assertFalse(result['formal_data_generated'])


if __name__ == '__main__':
    unittest.main()
