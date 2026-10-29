"""Fresh-role, complete-horizon and paired-noise checks without formal fields."""
import ast
from copy import deepcopy
import hashlib
import inspect
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

from coefficient_study import generators as current, data
from prefix_study import generators as previous, data as previous_data


class CoefficientGeneratorTests(unittest.TestCase):
    def test_roles_have_new_disjoint_deterministic_seed_allocation(self):
        self.assertEqual(current.ROLE_SEEDS, {'calibration':168000, 'training':168100,
                                             'development':168200, 'final':168300})
        self.assertEqual(current.ROLE_COUNTS, previous.ROLE_COUNTS)
        old={(previous.ROLE_SEEDS[role]+i, previous.FAMILY_IDS[family])
             for family in previous.FAMILIES
             for role,count in previous.ROLE_COUNTS.items() for i in range(count)}
        new=[(current.ROLE_SEEDS[role]+i, current.FAMILY_IDS[family])
             for family in current.FAMILIES
             for role,count in current.ROLE_COUNTS.items() for i in range(count)]
        self.assertEqual(len(new),len(set(new)))
        self.assertFalse(set(new)&old)
        for family in current.FAMILIES:
            for role,count in current.ROLE_COUNTS.items():
                for i in range(count):
                    a=current._initial_parameters(family,role,i)
                    b=current._initial_parameters(family,role,i)
                    self.assertEqual(a[1],[current.ROLE_SEEDS[role]+i,current.FAMILY_IDS[family]])
                    np.testing.assert_array_equal(a[2],b[2])
                    np.testing.assert_array_equal(a[3],b[3])

    def test_physics_distribution_prefix_and_calibration_limits_unchanged(self):
        for name in ('_contour_coefficients','_integrate','_initial_parameters',
                     'generate','reference_terms','calibration'):
            self.assertEqual(inspect.getsource(getattr(current,name)),
                             inspect.getsource(getattr(previous,name)))
        changed={key for key in current.GENERATOR_PROTOCOL
                 if current.GENERATOR_PROTOCOL[key]!=previous.GENERATOR_PROTOCOL[key]}
        self.assertEqual(changed,{'role_seed_bases'})
        self.assertEqual(current.CALIBRATION_RELATIVE_LIMIT,1e-5)
        self.assertEqual(current.CALIBRATION_MAX_ABSOLUTE_LIMIT,1e-4)
        self.assertEqual(current.GENERATOR_PROTOCOL['observed_nt'],111)
        self.assertEqual(current.GENERATOR_PROTOCOL['prefix_intervals'],10)
        self.assertEqual(current.GENERATOR_PROTOCOL['forecast_intervals'],100)

    def test_calibration_refinement_preserves_ic_grid_and_future_duration(self):
        def mock_integrate(initial,period,times,linear,transport,cubic,max_dt):
            return np.broadcast_to(initial,(len(times),len(initial))).copy()
        horizons={'allen_cahn':.5,'low_viscosity_burgers':1.5,'ks_unstable':20.}
        with patch.object(current,'_integrate',side_effect=mock_integrate):
            for family,horizon in horizons.items():
                base=current.generate(family,'calibration',0)
                fine=current.generate(family,'calibration',0,refined=True)
                self.assertEqual(base['metadata']['seed_sequence'],[168000,current.FAMILY_IDS[family]])
                self.assertEqual(base['metadata']['seed_sequence'],fine['metadata']['seed_sequence'])
                self.assertEqual(fine['metadata']['reference_n'],2*base['metadata']['reference_n'])
                self.assertEqual(fine['metadata']['max_dt'],base['metadata']['max_dt']/2)
                for key in ('x','t','u'):np.testing.assert_array_equal(base[key],fine[key])
                self.assertFalse(base['metadata']['is_measured_data'])
                self.assertEqual(len(base['t']),111)
                self.assertAlmostEqual(base['t'][10]-base['t'][0],.1*horizon)
                self.assertAlmostEqual(base['t'][-1]-base['t'][10],horizon)

    def test_calibration_never_requests_formal_roles_and_keeps_failures(self):
        roles=[]
        def mock_generate(family,role,index,refined):
            roles.append(role)
            if family=='ks_unstable':raise FloatingPointError('Reserved calibration failure')
            return {'x':np.arange(16),'t':np.arange(16),'u':np.ones((16,16)),
                    'metadata':{'role':role,'refined':refined}}
        with patch.object(current,'generate',side_effect=mock_generate):record=current.calibration()
        self.assertEqual(set(roles),{'calibration'})
        self.assertEqual(len(record['records']),3)
        self.assertEqual(record['records'][-1]['status'],'source_quality_failure')
        self.assertFalse(record['all_passed'])
        self.assertFalse(record['formal_training_development_final_generated'])
        self.assertFalse(record['thresholds_changed_after_results'])

    def test_reference_has_no_discovery_dependencies(self):
        tree=ast.parse(inspect.getsource(current));names=[]
        for node in ast.walk(tree):
            if isinstance(node,ast.Import):names.extend(alias.name for alias in node.names)
            elif isinstance(node,ast.ImportFrom):names.append(node.module or '')
        self.assertFalse(any(name.startswith(('pde_discovery','l1_study','weak_study',
                                              'resolved_study','fusion_study.engine')) for name in names))

    @staticmethod
    def manufactured_data(horizon=.5):
        x=np.arange(32)*2*np.pi/32;t=np.linspace(0,1.1*horizon,111)
        return {'x':x,'t':t,'u':1+np.exp(-.1*t[:,None])*np.sin(x)[None,:],
                'boundary':'periodic'}

    def test_corruption_is_nested_with_shared_endpoint_and_complete_horizon(self):
        for family,horizon in [('allen_cahn',.5),('low_viscosity_burgers',1.5),('ks_unstable',20.)]:
            clean=self.manufactured_data(horizon)
            full=data.corrupt(clean,family,'training',0,'full',.01,79,1.)
            coarse=data.corrupt(clean,family,'training',0,'coarse',.01,79,1.)
            np.testing.assert_array_equal(coarse['u'],full['u'][::2,::2])
            np.testing.assert_array_equal(coarse['prefix_u'],full['prefix_u'][::2,::2])
            self.assertEqual(len(full['prefix_t']),11);self.assertEqual(len(coarse['prefix_t']),6)
            self.assertEqual(len(full['t']),101);self.assertEqual(len(coarse['t']),51)
            for d in (full,coarse):
                np.testing.assert_array_equal(d['prefix_u'][-1],d['u'][0])
                self.assertEqual(d['prefix_t'][-1],d['t'][0])
                self.assertAlmostEqual(d['t'][-1]-d['t'][0],horizon)
                self.assertAlmostEqual(d['prefix_t'][-1]-d['prefix_t'][0],.1*horizon)
            self.assertEqual(set(full),{'x','t','u','boundary','prefix_t','prefix_u'})
            np.testing.assert_array_equal(clean['u'],self.manufactured_data(horizon)['u'])

    def test_noise_seed_base_is_new_and_observations_are_role_isolated(self):
        clean=self.manufactured_data()
        first=data.corrupt(clean,'allen_cahn','training',0,'full',.01,79,1.)
        again=data.corrupt(clean,'allen_cahn','training',0,'full',.01,79,1.)
        rng=np.random.default_rng(np.random.SeedSequence([179000,0,0,0,79]))
        expected=clean['u']+.01*rng.standard_normal(clean['u'].shape)
        np.testing.assert_array_equal(first['u'],expected[10:])
        np.testing.assert_array_equal(first['prefix_u'],expected[:11])
        np.testing.assert_array_equal(first['u'],again['u'])
        for role,index,seed in [('development',0,79),('final',0,79),('training',1,79),('training',0,109)]:
            other=data.corrupt(clean,'allen_cahn',role,index,'full',.01,seed,1.)
            self.assertFalse(np.array_equal(first['u'],other['u']))
        for seed in previous_data.PROTOCOL['nonzero_noise_seeds']:
            with self.assertRaises(ValueError):data.corrupt(clean,'allen_cahn','training',0,'full',.01,seed,1.)

    def test_noise_levels_share_realization_and_training_scale_excludes_prefix(self):
        clean=self.manufactured_data();changed=deepcopy(clean);changed['u'][:10]+=100
        self.assertEqual(data.training_scale([clean]),data.training_scale([changed]))
        low=data.corrupt(clean,'allen_cahn','training',0,'full',.001,109,1.)
        high=data.corrupt(clean,'allen_cahn','training',0,'full',.01,109,1.)
        np.testing.assert_allclose(high['u']-clean['u'][10:],10*(low['u']-clean['u'][10:]),atol=3e-15)
        np.testing.assert_allclose(high['prefix_u']-clean['u'][:11],10*(low['prefix_u']-clean['u'][:11]),atol=3e-15)

    def test_saved_calibration_receipt_binds_current_generator_and_all_families(self):
        source=Path(current.__file__)
        receipt=json.loads((source.parent/'data/calibration.json').read_text())
        self.assertEqual(receipt['generator_sha256'],hashlib.sha256(source.read_bytes()).hexdigest())
        self.assertEqual(receipt['protocol'],json.loads(json.dumps(current.GENERATOR_PROTOCOL)))
        self.assertEqual({r['family'] for r in receipt['records']},set(current.FAMILIES))
        self.assertTrue(receipt['all_passed'])
        self.assertFalse(receipt['formal_training_development_final_generated'])
        for r in receipt['records']:
            self.assertEqual(r['role'],'calibration');self.assertEqual(r['status'],'passed')
            self.assertLessEqual(r['relative_l2'],1e-5)
            self.assertLessEqual(r['max_absolute_error'],1e-4)


if __name__=='__main__':unittest.main()
