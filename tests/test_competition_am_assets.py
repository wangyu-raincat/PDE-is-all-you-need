import json
import re
import unittest
import tempfile
import hashlib
from pathlib import Path
from unittest.mock import patch

import numpy as np

from competition_agent.preprocess import prepare_am_catalog
from competition_agent.scenario_am.catalog import ROOT, asset_id, build_catalog, sha
from competition_agent.scenario_am.heat_tools import heat_identify, heat_holdout_summary, heat_compare, _verify_text_hash


class ManufacturingAssetsTests(unittest.TestCase):
    def test_historical_hash_only_allows_newline_conversion(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'frozen.json'
            expected = hashlib.sha256(b'{\n"value": 1\n}\n').hexdigest()
            path.write_bytes(b'{\r\n"value": 1\r\n}\r\n')
            self.assertEqual(_verify_text_hash(path, expected)['match_mode'], 'crlf_to_lf_only')
            path.write_bytes(b'{\r\n"value": 2\r\n}\r\n')
            with self.assertRaises(ValueError):
                _verify_text_hash(path, expected)

    def test_catalog_complete_reproducible_and_manifest_hashes(self):
        catalog = build_catalog()
        saved = json.loads((ROOT / 'competition_agent/scenario_am/asset_catalog.json').read_text(encoding='utf-8'))
        self.assertEqual(catalog, saved)
        cards = {c['asset_id']: c for c in catalog['assets']}
        for dataset in ('additive_heat', 'mendeley_heat'):
            for path in (ROOT / 'pde_discovery/data' / dataset).rglob('*'):
                if path.is_file():
                    self.assertIn(asset_id(path), cards)
                    self.assertEqual(cards[asset_id(path)]['sha256'], sha(path))
        raw = [c for c in cards.values() if c['modality'] == 'csv']
        self.assertEqual(len(raw), 47)
        for c in raw:
            self.assertEqual(c['sha256'], c['manifest_sha256'])
            self.assertEqual(c['license'], 'CC BY 4.0')
            self.assertGreater(c['rows'], 0)
            self.assertEqual(c['cols'], 8)
        self.assertEqual(catalog['dormancy'], dict(used_for_modeling=26, never_used=0, evaluated_not_blind=21, unknown=0))

    def test_metadata_has_only_headers_and_manifests_with_exact_citations(self):
        result = prepare_am_catalog()
        self.assertFalse(result['numerical_training_input'])
        self.assertEqual(len(result['records']), 60)
        for record in result['records']:
            for e in record['evidence']:
                text = (ROOT / e['locator']['path']).read_text(encoding='utf-8')
                loc = e['locator']
                self.assertEqual(text[loc['start_char']:loc['end_char']], e['quote'])
                if record['classification'] == 'measurement_header':
                    self.assertEqual(e['quote'], text.splitlines()[0])
                    self.assertNotIn('preview_rows', record)

    def test_metadata_rejects_path_escape_and_stale_hash(self):
        catalog = build_catalog()
        card = next(c for c in catalog['assets'] if c['modality'] == 'csv')
        card['sha256'] = 'bad'
        with self.assertRaises(ValueError):
            prepare_am_catalog({'assets': [card]})
        card['path'] = '../escape.csv'
        with self.assertRaises(ValueError):
            prepare_am_catalog({'assets': [card]})

    def test_xn_real_fit_matches_frozen_and_never_loads_final(self):
        from pde_discovery.real_experiment import load_measurements
        seen = []
        def guarded(directory, entry, fmt):
            seen.append(entry['file'])
            self.assertTrue(entry['file'].endswith(('xn1.csv', 'xn2.csv')))
            return load_measurements(directory, entry, fmt)
        paths = list((ROOT / 'artifacts/additive_heat').rglob('*.json'))
        before = {p: sha(p) for p in paths}
        with patch('competition_agent.scenario_am.heat_tools.load_measurements', side_effect=guarded):
            result = heat_identify('xn')
        self.assertEqual(len(seen), 2)
        self.assertEqual(result['selected']['model_name'], 'D+h+c')
        self.assertTrue(result['frozen_consistency']['matches'])
        self.assertEqual({t['name'] for t in result['terms']}, {'1', 'theta', 'theta_xx'})
        coefficients = {t['name']: t['coefficient'] for t in result['terms']}
        np.testing.assert_allclose([coefficients['theta_xx'], -coefficients['theta'], coefficients['1']],
                                   [2.08886467e-6, .0150036559, .0422881945], rtol=1e-6)
        self.assertEqual(before, {p: sha(p) for p in paths})

    def test_all_twelve_scores_match_report_without_loading_measurements(self):
        with patch('pde_discovery.real_experiment.load_measurements', side_effect=AssertionError('No evaluation')):
            result = heat_holdout_summary('all')
        self.assertFalse(result['recomputed'])
        self.assertEqual(len(result['records']), 12)
        report = (ROOT / 'experiments/ADDITIVE_REPORT.md').read_text(encoding='utf-8')
        for record in result['records']:
            row = re.search(r'\| ' + record['case'] + r' \| ([\d.]+) \| ([\d.]+) \| ([\d.]+) \| ([\d.]+) \|', report)
            self.assertIsNotNone(row)
            v = record['validation']
            values = [100*v['relative_l2'], v['rmse_K'], v['baselines']['persistence']['rmse'], v['baselines']['linear_boundary']['rmse']]
            for i, (value, expected) in enumerate(zip(values, row.groups())):
                self.assertAlmostEqual(value, float(expected), delta=5.1e-5 if i == 0 else 5.1e-7)

    def test_reject_final_refitting_and_invalid_groups(self):
        for split in ('development', 'final', 'holdout', '../train'):
            with self.assertRaises(ValueError):
                heat_identify('xn', split)
        for group in ('../xn', 'all', 'ss', ''):
            with self.assertRaises(ValueError):
                heat_identify(group)

    def test_metal_scores_are_existing_refined_results(self):
        result = heat_holdout_summary('50ss')
        self.assertEqual(len(result['records']), 1)
        self.assertIn('artifacts/refined_heat/50ss/', result['records'][0]['source_path'])

    def test_compare_explicitly_has_no_confidence_interval(self):
        a = {'selected': {'parameters': {'D': 2e-6, 'h': .01, 'c': -.1}}, 'evidence': []}
        b = {'selected': {'parameters': {'D': 3e-6, 'h': .02, 'c': .1}}, 'evidence': []}
        with patch('competition_agent.scenario_am.heat_tools.heat_identify', side_effect=[a, b]) as identify:
            result = heat_compare('xn', 'xp')
        self.assertEqual(identify.call_count, 2)
        for group in ("xn", "xp"):
            repeats = result["repeatability"][group]
            self.assertEqual(repeats["D_h_correlation"]["n"], 5)
            self.assertEqual(len(repeats["fits"]), 5)
        self.assertIsNone(result['bootstrap_intervals'])
        self.assertIn('不能', result['comparison_scope'])
        self.assertAlmostEqual(result['parameter_comparison'][0]['difference_b_minus_a'], 1e-6)
        self.assertEqual(result['parameter_comparison'][0]['unit'], 'm^2/s')


if __name__ == '__main__':
    unittest.main()
