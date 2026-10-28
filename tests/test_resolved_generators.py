"""New-seed source identity checks; existing tests cover unchanged physics."""
import ast
import inspect
import unittest
from unittest.mock import patch

import numpy as np

from resolved_study import generators as current
from weak_study import generators as previous


class ResolvedGeneratorTests(unittest.TestCase):
    def test_new_seed_allocation_is_fixed_and_disjoint(self):
        self.assertEqual(current.ROLE_SEEDS, {'calibration': 88000, 'training': 88100,
                                            'development': 88200, 'final': 88300})
        self.assertEqual(current.ROLE_COUNTS, {'calibration': 1, 'training': 3,
                                             'development': 2, 'final': 3})
        old_keys = {(previous.ROLE_SEEDS[role]+i, previous.FAMILY_IDS[family])
                    for family in previous.FAMILIES
                    for role, count in previous.ROLE_COUNTS.items() for i in range(count)}
        new_keys = [(current.ROLE_SEEDS[role]+i, current.FAMILY_IDS[family])
                    for family in current.FAMILIES
                    for role, count in current.ROLE_COUNTS.items() for i in range(count)]
        self.assertEqual(len(new_keys), len(set(new_keys)))
        self.assertFalse(set(new_keys) & old_keys)
        for family in current.FAMILIES:
            # These requests construct random IC coefficients, not trajectory fields.
            for role, count in current.ROLE_COUNTS.items():
                for i in range(count):
                    first = current._initial_parameters(family, role, i)
                    second = current._initial_parameters(family, role, i)
                    self.assertEqual(first[1], [current.ROLE_SEEDS[role]+i, current.FAMILY_IDS[family]])
                    np.testing.assert_array_equal(first[2], second[2])
                    np.testing.assert_array_equal(first[3], second[3])

    def test_reference_operator_and_initial_distribution_unchanged(self):
        # Exact source parity transfers the existing independent physics tests;
        # new experiment code cannot silently replace its reference operator.
        for name in ('_contour_coefficients', '_integrate', '_initial_parameters',
                     'generate', 'reference_terms', 'calibration'):
            self.assertEqual(inspect.getsource(getattr(current, name)),
                             inspect.getsource(getattr(previous, name)))
        changed = {key for key in current.GENERATOR_PROTOCOL
                   if current.GENERATOR_PROTOCOL[key] != previous.GENERATOR_PROTOCOL[key]}
        self.assertEqual(changed, {'kind', 'role_seed_bases'})
        self.assertEqual(current.CALIBRATION_RELATIVE_LIMIT, 1e-5)
        self.assertEqual(current.CALIBRATION_MAX_ABSOLUTE_LIMIT, 1e-4)

    def test_calibration_metadata_identifies_new_seeds_and_exact_grid(self):
        calls = []
        def manufactured(initial, period, times, linear, transport, cubic, max_dt):
            calls.append((len(initial), len(times)))
            return np.broadcast_to(initial, (len(times), len(initial))).copy()
        # All generated outputs here are mock static fields; only calibration
        # role is requested, with no actual numerical source trajectory.
        with patch.object(current, '_integrate', side_effect=manufactured):
            for family in current.FAMILIES:
                base = current.generate(family, 'calibration', 0)
                fine = current.generate(family, 'calibration', 0, refined=True)
                self.assertEqual(base['metadata']['seed_sequence'], [88000, current.FAMILY_IDS[family]])
                self.assertEqual(base['metadata']['role'], 'calibration')
                self.assertEqual(base['metadata']['seed_sequence'], fine['metadata']['seed_sequence'])
                self.assertEqual(fine['metadata']['reference_n'], 2*base['metadata']['reference_n'])
                self.assertEqual(fine['metadata']['max_dt'], base['metadata']['max_dt']/2)
                np.testing.assert_array_equal(base['x'], fine['x'])
                np.testing.assert_array_equal(base['t'], fine['t'])
                np.testing.assert_array_equal(base['u'], fine['u'])
                self.assertEqual(base['metadata']['initial_observed_sha256'], fine['metadata']['initial_observed_sha256'])
                self.assertFalse(base['metadata']['is_measured_data'])
        self.assertEqual(len(calls), 6)

    def test_reference_has_no_discovery_dependency_or_label_feedback(self):
        tree = ast.parse(inspect.getsource(current))
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend(name.name for name in node.names)
            elif isinstance(node, ast.ImportFrom):
                imports.append(node.module or '')
        self.assertFalse(any(name.startswith(('pde_discovery', 'l1_study',
            'weak_study.engine', 'resolved_study.engine')) for name in imports))
        generate_tree = ast.parse(inspect.getsource(current.generate))
        direct_calls = [node.func.id for node in ast.walk(generate_tree)
                        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)]
        self.assertNotIn('reference_terms', direct_calls)


if __name__ == '__main__':
    unittest.main()
