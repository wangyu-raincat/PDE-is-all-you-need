"""Read-only external-result endpoints must not expose stale successful plots."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from fastapi import HTTPException

from pde_discovery import app as web


class ExternalResultApiTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.output = Path(self.temporary.name)
        self.patcher = patch.object(web, "EXTERNAL_OUTPUT", self.output)
        self.patcher.start()
        self.addCleanup(self.patcher.stop)

    def test_invalid_case_and_missing_result(self):
        for case, status in [("../brass", 422), ("aluminum", 404)]:
            with self.subTest(case=case), self.assertRaises(HTTPException) as context:
                web.external_experiment_report(case)
            self.assertEqual(context.exception.status_code, status)

    def test_stale_plot_is_hidden_after_failure(self):
        folder = self.output / "aluminum"
        folder.mkdir()
        (folder / "validation.png").write_bytes(b"STALE FIXTURE")
        for report in [{}, {"plot_available": False, "validation": {"status": "evaluated"}},
                       {"plot_available": True, "validation": {"status": "failed"}}]:
            (folder / "report.json").write_text(json.dumps(report))
            with self.subTest(report=report), self.assertRaises(HTTPException) as context:
                web.external_experiment_plot("aluminum")
            self.assertEqual(context.exception.status_code, 404)

    def test_current_plot_and_report_are_accessible(self):
        folder = self.output / "brass"
        folder.mkdir()
        report = {"case": "brass", "plot_available": True, "validation": {"status": "evaluated"}}
        (folder / "report.json").write_text(json.dumps(report))
        (folder / "validation.png").write_bytes(b"MANUFACTURED FILE RESPONSE FIXTURE")
        self.assertEqual(json.loads(web.external_experiment_report("brass").body), report)
        self.assertEqual(Path(web.external_experiment_plot("brass").path), folder / "validation.png")

    def test_summary_missing_then_available(self):
        with self.assertRaises(HTTPException) as context:
            web.external_benchmark_report()
        self.assertEqual(context.exception.status_code, 404)
        summary = {"records": [{"case": "aluminum"}, {"case": "brass"}]}
        (self.output / "benchmark.json").write_text(json.dumps(summary))
        self.assertEqual(json.loads(web.external_benchmark_report().body), summary)
