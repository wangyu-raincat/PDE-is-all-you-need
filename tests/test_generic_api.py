"""Read-only cross-equation result paths reject unknown cases and stale plots."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from fastapi import HTTPException
from pde_discovery import app as web


class GenericResultApiTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.output = Path(temp.name)
        mock = patch.object(web, "GENERIC_OUTPUT", self.output)
        mock.start()
        self.addCleanup(mock.stop)

    def test_invalid_case_and_missing_result(self):
        for case, code in [("../kdv", 422), ("burgers", 404)]:
            with self.subTest(case=case), self.assertRaises(HTTPException) as error:
                web.generic_experiment_report(case)
            self.assertEqual(error.exception.status_code, code)

    def test_failed_report_cannot_expose_a_stale_plot(self):
        folder = self.output / "ks"
        folder.mkdir()
        (folder / "validation.png").write_bytes(b"STALE FIXTURE")
        (folder / "report.json").write_text(json.dumps({"plot_available": False, "validation": {"status": "failed"}}))
        with self.assertRaises(HTTPException) as error:
            web.generic_experiment_plot("ks")
        self.assertEqual(error.exception.status_code, 404)

    def test_current_report_and_plot(self):
        folder = self.output / "kdv"
        folder.mkdir()
        report = {"case": "kdv", "plot_available": True, "equation_recovery": {"exact_support": False}}
        (folder / "report.json").write_text(json.dumps(report))
        (folder / "validation.png").write_bytes(b"MANUFACTURED FILE RESPONSE")
        self.assertEqual(json.loads(web.generic_experiment_report("kdv").body), report)
        self.assertEqual(Path(web.generic_experiment_plot("kdv").path), folder / "validation.png")

    def test_summary_missing_and_available(self):
        with self.assertRaises(HTTPException) as error:
            web.generic_benchmark_report()
        self.assertEqual(error.exception.status_code, 404)
        report = {"records": [{"case": c} for c in web.GENERIC_CASES]}
        (self.output / "benchmark.json").write_text(json.dumps(report))
        self.assertEqual(json.loads(web.generic_benchmark_report().body), report)
