"""Isolation tests using only temporary files and manufactured/mock arrays."""
from contextlib import ExitStack
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from l1_study import run


def manufactured_data(future_shift=0.):
    x = np.linspace(0, 2*np.pi, 16, endpoint=False)
    t = np.linspace(0, .1, 16)
    u = np.tile(.2+np.sin(x), (len(t), 1))
    u[1:] += future_shift
    return {"x": x, "t": t, "u": u, "boundary": "periodic",
            "exact_features": np.zeros((*u.shape, 9)), "exact_ut": np.ones_like(u)}


def selected_fixture():
    return {"equation": "u_t = 0.1 u_xx", "terms": [{"name": "u_xx", "coefficient": .1}],
            "support_size": 1, "fit_success": True, "candidate_id": "fixture"}


def model_fixture(suite, case, mode, count, method, *_args):
    selected = selected_fixture()
    return {"id": run.model_id(suite, case, mode, count, method), "suite": suite,
            "case": case, "derivatives": mode, "training_trajectories": count,
            "method": method, "selected": selected, "candidates": [selected],
            "near_optimal_supports": [["u_xx"]], "failure_reason": None,
            "reference_used_for_selection": False, "final_loaded_during_selection": False,
            "retrospective": suite == "public"}


def all_model_fixtures():
    models = [model_fixture("public", c, "integrated", 1, m)
              for c in run.PUBLIC for m in run.METHODS]
    models.extend(model_fixture("analytic", c, d, n, m)
                  for c in run.fixtures.FAMILIES for d in run.MODES
                  for n in (1, 3) for m in run.METHODS)
    assert len(models) == 63
    return models


class StudyPipelineIsolationTests(unittest.TestCase):
    def test_prepare_generates_only_training_development_and_does_not_parse_reference(self):
        generated = []

        def generate(family, role, index):
            self.assertIn(role, ("training", "development"))
            generated.append((family, role, index))
            return manufactured_data()

        with tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
            data = Path(tmp)/"data"
            stack.enter_context(patch.object(run, "DATA", data))
            stack.enter_context(patch.object(run, "sources", return_value={"reference_bytes": "digest_only"}))
            stack.enter_context(patch.object(run, "data_hashes", return_value={}))
            stack.enter_context(patch.object(run.fixtures, "generate", side_effect=generate))
            refs = stack.enter_context(patch.object(run.fixtures, "reference_terms", side_effect=AssertionError("reference read")))
            read = stack.enter_context(patch.object(run, "read", side_effect=AssertionError("JSON parsed")))
            record = run.prepare()
            self.assertEqual(len(generated), 15)
            self.assertFalse(record["final_analytic_fields_generated"])
            self.assertEqual(len(list(data.glob("*/*/*.npz"))), 15)
            self.assertFalse(list(data.glob("*/final/*")))
            refs.assert_not_called()
            read.assert_not_called()

    def test_freeze_reads_only_train_dev_and_writes_index_after_all_63_models(self):
        loads, writes = [], []
        block = {"X": np.ones((12, 9)), "y": np.ones(12)}
        real_write = run.write

        def loader(path, exact=False):
            self.assertNotIn("final", Path(path).parts)
            self.assertNotEqual(Path(path).name, "final.npz")
            loads.append(str(path))
            return manufactured_data()

        def guarded_read(path):
            if Path(path).name == "frozen_model.json":
                return {"selected": selected_fixture()}
            if Path(path).name == "protocol.json":
                return {"unit_test_fixture": True}
            raise AssertionError(f"Unexpected read during freeze: {path}")

        def record_write(path, value):
            if Path(path).name == "freeze_index.json":
                self.assertEqual(sum(p.parent.name == "models" for p in writes), 63)
                self.assertEqual(len(value["records"]), 63)
            writes.append(Path(path))
            real_write(path, value)

        with tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
            folder = Path(tmp)
            data = folder/"data"
            data.mkdir()
            (data/"prepared.json").write_text("{}")
            stack.enter_context(patch.object(run, "DATA", data))
            stack.enter_context(patch.object(run, "verify_preparation", return_value={"data_sha256": {}}))
            stack.enter_context(patch.object(run, "sources", return_value={}))
            stack.enter_context(patch.object(run, "load_array", side_effect=loader))
            stack.enter_context(patch.object(run, "read", side_effect=guarded_read))
            stack.enter_context(patch.object(run, "write", side_effect=record_write))
            stack.enter_context(patch.object(run, "fit_record", side_effect=model_fixture))
            stack.enter_context(patch.object(run.engine, "integrated_block", return_value=block))
            stack.enter_context(patch.object(run.engine, "pointwise_block", return_value=block))
            generate = stack.enter_context(patch.object(run.fixtures, "generate", side_effect=AssertionError("generated role during freeze")))
            refs = stack.enter_context(patch.object(run.fixtures, "reference_terms", side_effect=AssertionError("scoring reference during freeze")))
            index = run.freeze(folder/"output")
            self.assertEqual(writes[-1].name, "freeze_index.json")
            self.assertEqual(len(index["records"]), 63)
            self.assertEqual(len({r["id"] for r in index["records"]}), 63)
            self.assertTrue(index["all_models_frozen_before_final"])
            self.assertFalse(index["analytic_final_fields_generated"])
            self.assertEqual(len(loads), 21)
            generate.assert_not_called()
            refs.assert_not_called()

    def test_evaluate_rejects_invalid_freeze_before_loading_or_generating_any_final(self):
        with tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
            stack.enter_context(patch.object(run, "verify_freeze", side_effect=ValueError("incomplete freeze")))
            read = stack.enter_context(patch.object(run, "read", side_effect=AssertionError("reference/model parsed early")))
            load = stack.enter_context(patch.object(run, "load_array", side_effect=AssertionError("final loaded early")))
            generate = stack.enter_context(patch.object(run.fixtures, "generate", side_effect=AssertionError("final generated early")))
            with self.assertRaisesRegex(ValueError, "incomplete freeze"):
                run.evaluate(Path(tmp)/"output")
            read.assert_not_called()
            load.assert_not_called()
            generate.assert_not_called()

    def test_complete_freeze_guards_reject_model_data_source_and_receipt_tampering(self):
        with tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
            folder = Path(tmp)
            data, output = folder/"data", folder/"output"
            data.mkdir()
            receipt = data/"prepared.json"
            receipt.write_text('{"fixture":true}')
            records = []
            for model in all_model_fixtures():
                path = output/"models"/f"{model['id']}.json"
                run.write(path, model)
                records.append({"id": model["id"], "file": str(path.relative_to(output)), "sha256": run.sha(path)})
            source = {"code.py": "original"}
            hashes = {"training.npz": "training_hash", "final.npz": "byte_hash_only"}
            index = {"records": records, "sources": source, "data_sha256": hashes,
                     "preparation_sha256": run.sha(receipt)}
            run.write(output/"freeze_index.json", index)
            stack.enter_context(patch.object(run, "DATA", data))
            source_mock = stack.enter_context(patch.object(run, "sources", return_value=source))
            data_mock = stack.enter_context(patch.object(run, "data_hashes", return_value=hashes))
            self.assertEqual(len(run.verify_freeze(output)["records"]), 63)
            target = output/records[0]["file"]
            original = target.read_bytes()
            target.write_bytes(original+b" ")
            with self.assertRaisesRegex(ValueError, "Frozen model bytes changed"):
                run.verify_freeze(output)
            target.write_bytes(original)
            source_mock.return_value = {"code.py": "changed"}
            with self.assertRaisesRegex(ValueError, "changed code"):
                run.verify_freeze(output)
            source_mock.return_value = source
            data_mock.return_value = {"training.npz": "changed"}
            with self.assertRaisesRegex(ValueError, "Prepared data"):
                run.verify_freeze(output)
            data_mock.return_value = hashes
            receipt.write_text("changed receipt")
            with self.assertRaisesRegex(ValueError, "Prepared data"):
                run.verify_freeze(output)
            receipt.write_text('{"fixture":true}')
            run.write(output/"freeze_index.json", {**index, "records": records[:-1]})
            with self.assertRaisesRegex(ValueError, "Incomplete freeze"):
                run.verify_freeze(output)
            duplicated = [records[0], records[0], *records[2:]]
            unknown = [{**records[0], "id": "undeclared-model"}, *records[1:]]
            for changed in (duplicated, unknown):
                with self.subTest(identity_case=changed[0]["id"]):
                    run.write(output/"freeze_index.json", {**index, "records": changed})
                    with self.assertRaisesRegex(ValueError, "Incomplete freeze"):
                        run.verify_freeze(output)
            wrong_path = [{**records[0], "file": "models/wrong-name.json"}, *records[1:]]
            run.write(output/"freeze_index.json", {**index, "records": wrong_path})
            with self.assertRaisesRegex(ValueError, "path must match"):
                run.verify_freeze(output)
            original_model = json.loads(target.read_text())
            run.write(target, {**original_model, "method": "adaptive"})
            wrong_metadata = [{**records[0], "sha256": run.sha(target)}, *records[1:]]
            run.write(output/"freeze_index.json", {**index, "records": wrong_metadata})
            with self.assertRaisesRegex(ValueError, "metadata does not match"):
                run.verify_freeze(output)

    def test_single_and_joint_row_budget_is_equal_with_unique_deterministic_indices(self):
        blocks = [{"X": np.tile(np.arange(n)[:, None]+1000*i, (1, 9)),
                   "y": np.arange(n, dtype=float)} for i, n in enumerate((103, 101, 109))]
        single, joint = run.budget_blocks(blocks, 1), run.budget_blocks(blocks, 3)
        self.assertEqual(sum(len(b["y"]) for b in single), 99)
        self.assertEqual(sum(len(b["y"]) for b in joint), 99)
        for b in [*single, *joint]:
            self.assertEqual(len(np.unique(b["y"])), len(b["y"]))
            self.assertEqual(b["metadata"]["common_total_budget"], 99)
        for a, b in zip(joint, run.budget_blocks(blocks, 3)):
            np.testing.assert_array_equal(a["X"], b["X"])

    def test_final_observations_change_scores_but_not_frozen_selections(self):
        models = all_model_fixtures()
        model_by_name = {m["id"]+".json": m for m in models}
        original_selections = {m["id"]: deepcopy(m["selected"]) for m in models}
        summaries = []
        for shift in (0., .2):
            verified = []

            def verify(_output):
                verified.append(True)
                return {"records": [{"id": m["id"], "file": "models/"+m["id"]+".json", "sha256": "fixed"}
                                    for m in models]}

            def reader(path):
                self.assertTrue(verified, "Final/reference read preceded complete verification")
                name = Path(path).name
                if name == "generic_reference_equations.json":
                    return {"references": {c: {"terms": selected_fixture()["terms"]} for c in run.PUBLIC}}
                return deepcopy(model_by_name[name])

            def loader(_path, exact=False):
                self.assertTrue(verified)
                return manufactured_data(shift)

            def generate(_family, role, _index):
                self.assertTrue(verified)
                self.assertEqual(role, "final")
                return manufactured_data(shift)

            with tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
                output = Path(tmp)/"output"
                output.mkdir()
                (output/"freeze_index.json").write_text("fixture-index")
                stack.enter_context(patch.object(run, "verify_freeze", side_effect=verify))
                stack.enter_context(patch.object(run, "read", side_effect=reader))
                stack.enter_context(patch.object(run, "load_array", side_effect=loader))
                stack.enter_context(patch.object(run.fixtures, "generate", side_effect=generate))
                stack.enter_context(patch.object(run.fixtures, "reference_terms", return_value=selected_fixture()["terms"]))
                stack.enter_context(patch.object(run.baseline, "forecast", side_effect=lambda _terms, _x, t, initial: np.tile(initial, (len(t), 1))))
                fit = stack.enter_context(patch.object(run.engine, "fit_blocks", side_effect=AssertionError("fit during final")))
                select = stack.enter_context(patch.object(run.engine, "select_candidates", side_effect=AssertionError("select during final")))
                summaries.append(run.evaluate(output))
                fit.assert_not_called()
                select.assert_not_called()
        self.assertEqual(len(summaries[0]["records"]), 63)
        for before, after in zip(summaries[0]["records"], summaries[1]["records"]):
            self.assertEqual(before["terms"], after["terms"])
            self.assertEqual(before["equation"], after["equation"])
            self.assertEqual(before["frozen_model_sha256"], after["frozen_model_sha256"])
            self.assertGreater(after["mean_final_relative_l2"], before["mean_final_relative_l2"])
        self.assertEqual(original_selections, {m["id"]: m["selected"] for m in models})

    @staticmethod
    def gate_records(analytic_gain=False):
        records = []
        for model in all_model_fixtures():
            exact = not (model["suite"] == "public" and model["method"] == "plain")
            if (analytic_gain and model["id"] ==
                    run.model_id("analytic", "advdiff", "integrated", 1, "plain")):
                exact = False
            records.append({**model, "recovery": {"exact_support": exact, "missing_terms": [],
                            "spurious_terms": [] if exact else ["u_x"], "coefficient_relative_l2": .001},
                            "forecast_failures": 0, "mean_final_relative_l2": .0001})
        return records

    def test_no_non_oracle_gain_means_keep_baseline_even_if_adaptive_is_perfect(self):
        records = self.gate_records(False)
        self.assertEqual(len(records), 63)
        self.assertTrue(run.complete_records(records))
        self.assertTrue(all(r["recovery"]["exact_support"] for r in records if r["method"] == "adaptive"))
        gate = run.promotion_gate(records)
        self.assertEqual(gate["candidate"], "irl1")
        self.assertFalse(gate["numerical_gate_passed"])
        self.assertEqual(gate["decision"], "keep_baseline_do_not_merge")
        self.assertFalse(gate["merged"])

    def test_success_to_failure_regression_blocks_promotion(self):
        records = self.gate_records(True)
        old = next(r for r in records if r["suite"] == "public" and r["method"] == "plain")
        old["recovery"]["exact_support"] = True
        old["recovery"]["spurious_terms"] = []
        new = next(r for r in records if r["case"] == old["case"] and r["method"] == "irl1")
        new["recovery"]["exact_support"] = False
        new["recovery"]["spurious_terms"] = ["u^3"]
        gate = run.promotion_gate(records)
        self.assertFalse(gate["numerical_gate_passed"])
        self.assertTrue(any(not c["passed"] and c["name"].endswith(":no_structure_loss")
                            for c in gate["checks"]))
        self.assertEqual(gate["decision"], "keep_baseline_do_not_merge")

    def test_numerical_gate_pass_still_does_not_merge_without_regression_review(self):
        gate = run.promotion_gate(self.gate_records(True))
        self.assertTrue(gate["numerical_gate_passed"])
        self.assertEqual(gate["decision"], "eligible_for_review_only")
        self.assertTrue(gate["regression_tests_required_separately"])
        self.assertFalse(gate["merged"])

    def test_gate_rejects_missing_duplicate_and_undeclared_model_identities(self):
        valid = self.gate_records(True)
        self.assertTrue(run.complete_records(valid))
        malformed = {
            "missing": valid[:-1],
            "duplicate_at_same_total_count": [valid[0], valid[0], *valid[2:]],
            "unknown_at_same_total_count": [{**valid[0], "id": "undeclared-model"}, *valid[1:]],
        }
        for label, records in malformed.items():
            with self.subTest(label=label):
                self.assertFalse(run.complete_records(records))
                gate = run.promotion_gate(records)
                self.assertFalse(gate["numerical_gate_passed"])
                self.assertFalse(gate["merged"])
                self.assertEqual(gate["candidate"], "irl1")
                self.assertEqual(gate["decision"], "keep_baseline_do_not_merge")
                self.assertEqual(gate["checks"], [{"name": "all_63_unique_declared_models", "passed": False}])


if __name__ == "__main__":
    unittest.main()
