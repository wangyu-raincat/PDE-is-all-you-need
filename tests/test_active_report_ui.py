"""Read-only reporting and replay tests, using invented receipts only."""
from copy import deepcopy
import http.client
import json
from pathlib import Path
import shutil
import tempfile
import threading
import unittest

from active_study.app import make_server
from scripts import render_active_study as report


def fixture(task_id='synthetic-fixture-0', policy='fixed_rich', budget=5):
    terms = [{'name': 'u_xx', 'coefficient': .1}]
    score = {'task_id': task_id, 'spec_index': 0 if task_id.endswith('0') else 1,
        'family': 'diffusion', 'noise_level': 0., 'policy': policy, 'budget': budget,
        'status': 'evaluated', 'terms': terms, 'mean_clean_relative_l2': .02,
        'recovery': {'exact_support': True, 'missing_terms': [], 'spurious_terms': [],
            'coefficient_relative_l2': .01,
            'coefficient_comparison': [{'name': 'u_xx', 'reference': .1, 'discovered': .1,
                                        'relative_error': .01, 'absolute_error': .001}]},
        'forecasts': [{'status': 'evaluated', 'clean': {'relative_l2': .02},
                      'conditioning': 'observed_prefix', 'prefix_frames_used': 11,
                      'forecast_duration': 1.} for _ in range(3)],
        'candidate_supports': [['u_xx']], 'set_contains_true_support': True, 'set_size': 1,
        'frequencies': {'u_xx': 1., 'u': 0.}, 'term_brier_positive': 0., 'term_brier_negative': 0.,
        'charged_measurement_scalars': budget*7104,
        'model_file': f'episodes/{task_id}--{policy}.json', 'checkpoint_index': budget-5}
    checkpoint = {'budget': budget, 'charged_measurement_scalars': budget*7104,
        'model': {'selected': {'terms': terms, 'equation': 'u_t = 0.1 u_xx'},
            'candidate_supports': [['u_xx']], 'status': 'candidate',
            'uncertainty': {'term_inclusion_frequencies': {'u_xx': 1., 'u': 0.},
                'completed_replicates': 8, 'requested_replicates': 8, 'failed_replicates': 0}},
        'next_proposal': None if budget == 9 else {'selected_action_id': 'rich-0', 'status': 'selected',
            'reason': 'Fixture proposal', 'scores': [{'action_id': 'rich-0', 'score': 2., 'status': 'scored'}]},
        'final_used_for_selection': False}
    return score, checkpoint


def complete_fixture(directory):
    records = []
    (directory/'episodes').mkdir()
    for identity in ('synthetic-fixture-0', 'synthetic-fixture-1'):
        for policy in report.POLICIES:
            checkpoints, scores = [], []
            for budget in report.BUDGETS:
                score, checkpoint = fixture(identity, policy, budget)
                scores.append(score); checkpoints.append(checkpoint)
            episode = {'policy': policy, 'checkpoints': checkpoints, 'queries': [],
                       'final_used': False, 'reference_coefficients_used': False}
            path = directory/scores[0]['model_file']
            path.write_text(json.dumps(episode))
            for score in scores:
                score['frozen_episode_sha256'] = report.digest(path)
            records.extend(scores)
    comparison = {'records': records, 'expected_conditions': 2,
                  'evaluation_label': 'Synthetic test fixture, not a scientific result.',
                  'audit': {'status': 'fixture', 'passed': False}}
    (directory/'comparison.json').write_text(json.dumps(comparison))
    return comparison


class ActiveReportTests(unittest.TestCase):
    def test_joint_success_is_not_only_prediction_accuracy(self):
        record, _ = fixture()
        record['recovery']['coefficient_comparison'][0]['relative_error'] = .11
        row = report.flat_row(record)
        self.assertFalse(row['success'])
        self.assertTrue(row['exact_support'])

    def test_failures_preserve_three_forecasts_and_denominator(self):
        record, _ = fixture()
        record['forecasts'][2] = {'status': 'failed', 'reason': 'fixture solver failure'}
        record['mean_clean_relative_l2'] = None
        summary, rows = report.summarize({'records': [record]}, expected_conditions=2)
        group = summary['groups'][0]
        self.assertEqual(group['planned_conditions'], 2)
        self.assertEqual(group['success_count'], 0)
        self.assertEqual(group['forecast_failure_slots'], 1)
        self.assertEqual(group['missing_conditions'], 1)
        self.assertIsNone(rows[0]['mean_clean_relative_l2'])
        self.assertTrue(all(v is None for v in summary['total_budget_savings_vs_comparator'].values()))
        record['mean_clean_relative_l2'] = .02
        with self.assertRaisesRegex(ValueError, 'Incomplete'):
            report.flat_row(record)

    def test_absent_checkpoint_is_retained_without_invented_cost(self):
        record, _ = fixture()
        record.update(status='missing_budget_checkpoint', checkpoint_index=None,
                      charged_measurement_scalars=None, mean_clean_relative_l2=None,
                      forecasts=[{'status': 'failed'} for _ in range(3)])
        self.assertIsNone(report.flat_row(record)['charged_measurement_scalars'])

    def test_duplicate_pair_and_nonfinite_metrics_rejected(self):
        record, _ = fixture()
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            report.summarize({'records': [record, deepcopy(record)]})
        record['frequencies']['u_xx'] = float('nan')
        with self.assertRaisesRegex(ValueError, 'finite'):
            report.flat_row(record)

    def test_prefix_and_coefficient_drift_rejected(self):
        record, _ = fixture()
        record['forecasts'][0]['forecast_duration'] = .1
        with self.assertRaisesRegex(ValueError, 'full horizon'):
            report.flat_row(record)
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary); comparison = complete_fixture(directory)
            comparison['records'][0]['terms'] = []
            with self.assertRaisesRegex(ValueError, 'coefficients'):
                report.load_episode(directory, 'synthetic-fixture-0', 'fixed_rich', comparison)

    def test_candidate_set_size_and_unavailable_brier_are_honest(self):
        record, _ = fixture()
        record.update(candidate_supports=[['u']], set_contains_true_support=False,
                      term_brier_positive=None, frequencies={'u': None, 'u_xx': None})
        summary, rows = report.summarize({'records': [record]}, expected_conditions=1)
        self.assertTrue(rows[0]['wrong_singleton_set'])
        self.assertEqual(summary['groups'][0]['balanced_term_brier_available_conditions'], 0)
        self.assertIsNone(summary['groups'][0]['balanced_term_brier_mean'])

    def test_complete_budget_comparison_render_and_source_immutability(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary); complete_fixture(directory)
            before = report.digest(directory/'comparison.json')
            summary = report.render(directory)
            self.assertEqual(summary['result_state'], 'complete')
            self.assertEqual(summary['first_observed_budget_at_target']['structural_disagreement'], 5)
            self.assertEqual(summary['total_budget_savings_vs_comparator']['fixed_rich'], 0)
            self.assertEqual(report.digest(directory/'comparison.json'), before)
            for filename in ('REPORT.md', 'summary.json', 'budget_curves.png', 'comparison.csv'):
                self.assertGreater((directory/filename).stat().st_size, 20)
            self.assertIn('null', (directory/'summary.json').read_text())
            self.assertFalse(summary['audit']['passed'])

    def test_unattained_target_has_null_savings(self):
        with tempfile.TemporaryDirectory() as temporary:
            comparison = complete_fixture(Path(temporary))
            for record in comparison['records']:
                record['recovery']['coefficient_comparison'][0]['relative_error'] = .2
            summary, _ = report.summarize(comparison)
            self.assertEqual(summary['result_state'], 'complete')
            self.assertTrue(all(value is None for value in summary['first_observed_budget_at_target'].values()))
            self.assertTrue(all(value is None for value in summary['total_budget_savings_vs_comparator'].values()))

    def test_missing_and_path_escape_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(FileNotFoundError):
                report.load_results(temporary)
            with self.assertRaisesRegex(ValueError, 'inside'):
                report.safe_child(temporary, '../../secret.json')


class ActiveReplayTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.directory = Path(self.temporary.name)
        self.comparison = complete_fixture(self.directory)
        self.server = make_server(self.directory, port=0)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self):
        self.server.shutdown(); self.server.server_close(); self.thread.join(timeout=3)
        self.temporary.cleanup()

    def request(self, path, method='GET'):
        connection = http.client.HTTPConnection('127.0.0.1', self.server.server_port, timeout=5)
        connection.request(method, path)
        response = connection.getresponse(); status = response.status; body = response.read()
        connection.close()
        return status, body

    def test_read_only_routes_and_failure_responses(self):
        self.assertEqual(self.request('/')[0], 200)
        self.assertEqual(self.request('/api/results')[0], 200)
        self.assertEqual(self.request('/api/episode?task_id=synthetic-fixture-0&policy=fixed_rich')[0], 200)
        self.assertEqual(self.request('/api/episode?task_id=unknown&policy=fixed_rich')[0], 404)
        self.assertEqual(self.request('/api/episode?task_id=x&task_id=y&policy=fixed_rich')[0], 400)
        self.assertEqual(self.request('/download/../../comparison.json')[0], 404)
        for method in ('POST', 'PUT', 'PATCH', 'DELETE'):
            self.assertEqual(self.request('/api/results', method)[0], 405)
        (self.directory/'comparison.json').unlink()
        self.assertEqual(self.request('/api/results')[0], 503)

    def test_modified_episode_is_rejected(self):
        path = self.directory/'episodes/synthetic-fixture-0--fixed_rich.json'
        episode = json.loads(path.read_text()); episode['queries'] = [{'unexpected': True}]
        path.write_text(json.dumps(episode))
        status, body = self.request('/api/episode?task_id=synthetic-fixture-0&policy=fixed_rich')
        self.assertEqual(status, 422)
        self.assertIn(b'frozen digest', body)

    @unittest.skipUnless(shutil.which('chromium'), 'Browser integration requires a local Chromium.')
    def test_browser_switching_and_error_clear_stale_results(self):
        try:
            from playwright.sync_api import sync_playwright
        except ImportError:
            self.skipTest('Browser integration requires Playwright.')
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(executable_path=shutil.which('chromium'), headless=True,
                                                 args=['--no-sandbox'])
            try:
                page = browser.new_page()
                page.goto(f'http://127.0.0.1:{self.server.server_port}')
                page.locator('#detail').wait_for(state='visible')
                self.assertIn('未校准频率', page.locator('#bootstrap').inner_text())
                page.select_option('#budget', '9')
                page.locator('#proposal').filter(has_text='预算终点').wait_for()
                self.assertEqual(page.locator('#actions tr').count(), 0)
                page.route('**/api/episode?*', lambda route: route.fulfill(status=503,
                    content_type='application/json', body=json.dumps({'error': 'Fixture unavailable episode'})))
                page.select_option('#policy', 'random')
                page.locator('#error').wait_for(state='visible')
                self.assertTrue(page.locator('#detail').is_hidden())
                self.assertEqual(page.locator('#equation').inner_text(), '')
                self.assertEqual(page.locator('#comparison tr').count(), 0)
                self.assertIn('unavailable', page.locator('#error').inner_text())
            finally:
                browser.close()


if __name__ == '__main__':
    unittest.main()
