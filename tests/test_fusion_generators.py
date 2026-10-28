"""Fresh-role and paired-noise checks without generating formal trajectories."""
import ast
import inspect
import unittest
from unittest.mock import patch

import numpy as np

from fusion_study import generators as current, data
from resolved_study import generators as previous, data as previous_data


class FusionGeneratorTests(unittest.TestCase):
    def test_roles_have_new_disjoint_deterministic_seed_allocation(self):
        self.assertEqual(current.ROLE_SEEDS, {'calibration': 108000, 'training': 108100,
                                            'development': 108200, 'final': 108300})
        self.assertEqual(current.ROLE_COUNTS, previous.ROLE_COUNTS)
        old = {(previous.ROLE_SEEDS[role]+i, previous.FAMILY_IDS[family])
               for family in previous.FAMILIES
               for role, count in previous.ROLE_COUNTS.items() for i in range(count)}
        new = [(current.ROLE_SEEDS[role]+i, current.FAMILY_IDS[family])
               for family in current.FAMILIES
               for role, count in current.ROLE_COUNTS.items() for i in range(count)]
        self.assertEqual(len(new), len(set(new)))
        self.assertFalse(set(new) & old)
        # Only random initial coefficients, never trajectory fields, are formed.
        for family in current.FAMILIES:
            for role, count in current.ROLE_COUNTS.items():
                for i in range(count):
                    a = current._initial_parameters(family, role, i)
                    b = current._initial_parameters(family, role, i)
                    self.assertEqual(a[1], [current.ROLE_SEEDS[role]+i, current.FAMILY_IDS[family]])
                    np.testing.assert_array_equal(a[2], b[2])
                    np.testing.assert_array_equal(a[3], b[3])

    def test_physics_distribution_and_calibration_limits_unchanged(self):
        for name in ('_contour_coefficients', '_integrate', '_initial_parameters',
                     'generate', 'reference_terms', 'calibration'):
            self.assertEqual(inspect.getsource(getattr(current, name)),
                             inspect.getsource(getattr(previous, name)))
        changed = {key for key in current.GENERATOR_PROTOCOL
                   if current.GENERATOR_PROTOCOL[key] != previous.GENERATOR_PROTOCOL[key]}
        self.assertEqual(changed, {'role_seed_bases'})
        self.assertEqual(current.CALIBRATION_RELATIVE_LIMIT, 1e-5)
        self.assertEqual(current.CALIBRATION_MAX_ABSOLUTE_LIMIT, 1e-4)

    def test_calibration_refinement_preserves_ic_and_observation_grid(self):
        def mock_integrate(initial, period, times, linear, transport, cubic, max_dt):
            return np.broadcast_to(initial, (len(times), len(initial))).copy()
        with patch.object(current, '_integrate', side_effect=mock_integrate):
            for family in current.FAMILIES:
                base = current.generate(family, 'calibration', 0)
                fine = current.generate(family, 'calibration', 0, refined=True)
                self.assertEqual(base['metadata']['seed_sequence'], [108000, current.FAMILY_IDS[family]])
                self.assertEqual(base['metadata']['seed_sequence'], fine['metadata']['seed_sequence'])
                self.assertEqual(fine['metadata']['reference_n'], 2*base['metadata']['reference_n'])
                self.assertEqual(fine['metadata']['max_dt'], base['metadata']['max_dt']/2)
                for key in ('x', 't', 'u'):
                    np.testing.assert_array_equal(base[key], fine[key])
                self.assertFalse(base['metadata']['is_measured_data'])

    def test_calibration_never_requests_formal_roles_and_keeps_failures(self):
        roles = []
        def mock_generate(family, role, index, refined):
            roles.append(role)
            if family == 'ks_unstable':
                raise FloatingPointError('Reserved calibration failure')
            return {'x': np.arange(16), 't': np.arange(16), 'u': np.ones((16, 16)),
                    'metadata': {'role': role, 'refined': refined}}
        with patch.object(current, 'generate', side_effect=mock_generate):
            record = current.calibration()
        self.assertEqual(set(roles), {'calibration'})
        self.assertEqual(len(record['records']), 3)
        self.assertEqual(record['records'][-1]['status'], 'source_quality_failure')
        self.assertFalse(record['all_passed'])
        self.assertFalse(record['formal_training_development_final_generated'])
        self.assertFalse(record['thresholds_changed_after_results'])

    def test_reference_has_no_discovery_dependencies(self):
        tree = ast.parse(inspect.getsource(current))
        names = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                names.append(node.module or '')
        self.assertFalse(any(name.startswith(('pde_discovery', 'l1_study', 'weak_study',
                                              'resolved_study', 'fusion_study.engine')) for name in names))

    @staticmethod
    def manufactured_data():
        x = np.arange(32)*2*np.pi/32
        t = np.linspace(0, 1, 32)
        return {'x': x, 't': t, 'u': np.ones((32, 32))+np.sin(x)[None, :],
                'boundary': 'periodic'}

    def test_corruption_is_deterministic_nested_and_role_isolated(self):
        clean = self.manufactured_data()
        full = data.corrupt(clean, 'allen_cahn', 'training', 0, 'full', .01, 61, 1.)
        again = data.corrupt(clean, 'allen_cahn', 'training', 0, 'full', .01, 61, 1.)
        coarse = data.corrupt(clean, 'allen_cahn', 'training', 0, 'coarse', .01, 61, 1.)
        dev = data.corrupt(clean, 'allen_cahn', 'development', 0, 'full', .01, 61, 1.)
        np.testing.assert_array_equal(full['u'], again['u'])
        np.testing.assert_array_equal(coarse['u'], full['u'][::2, ::2])
        self.assertFalse(np.array_equal(full['u'], dev['u']))
        self.assertEqual(set(full), {'x', 't', 'u', 'boundary'})
        np.testing.assert_array_equal(clean['u'], self.manufactured_data()['u'])

    def test_corruption_uses_new_seed_base_and_rejects_previous_noise_ids(self):
        clean = self.manufactured_data()
        output = data.corrupt(clean, 'allen_cahn', 'training', 0, 'full', .01, 61, 1.)
        rng = np.random.default_rng(np.random.SeedSequence([119000, 0, 0, 0, 61]))
        np.testing.assert_array_equal(output['u'], clean['u']+.01*rng.standard_normal(clean['u'].shape))
        for seed in (53, 89):
            with self.assertRaises(ValueError):
                data.corrupt(clean, 'allen_cahn', 'training', 0, 'full', .01, seed, 1.)
        second = data.corrupt(clean, 'allen_cahn', 'training', 0, 'full', .01, 97, 1.)
        self.assertFalse(np.array_equal(output['u'], second['u']))


if __name__ == '__main__':
    unittest.main()
