"""Resolved four-arm isolation checks with temporary files and manufactured arrays only."""
from contextlib import ExitStack, redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from resolved_study import run


def field(shift=0.):
    x = np.linspace(0., 2*np.pi, 32, endpoint=False)
    t = np.linspace(0., .2, 32)
    u = np.tile(.6 + .15*np.sin(x), (len(t), 1))
    u[1:] += shift
    return {"x": x, "t": t, "u": u, "boundary": "periodic"}


def selected():
    return {"equation": "u_t = 0.1 u_xx", "terms": [{"name": "u_xx", "coefficient": .1}],
            "support_size": 1, "fit_success": True, "candidate_id": "manufactured"}


def preparation():
    return {"sources": {"code.py": "source_hash"}, "input_hashes": {"input.npz": "byte_hash"},
            "quality": {f: {"status": "passed"} for f in run.FAMILIES},
            "all_source_quality_passed": True, "training_rms": {f: 1. for f in run.FAMILIES}}


def design():
    canonical=lambda value: json.loads(json.dumps(value,allow_nan=False))
    return {'variant_menu_status':'frozen',
            'weak_variants':canonical(run.weak_engine.VARIANTS),
            'weak_engine_protocol':canonical(run.weak_engine.PROTOCOL),
            'forecast_protocol':canonical(run.resolved_forecast.PROTOCOL),
            'roles':run.data.ROLE_COUNTS|{'calibration':1},
            'families':list(run.FAMILIES),'methods':list(run.METHODS)}


def calibration_receipt():
    return {'generator_sha256':'generator_hash',
            'protocol':json.loads(json.dumps(run.generators.GENERATOR_PROTOCOL)),
            'formal_training_development_final_generated':False,
            'thresholds_changed_after_results':False,'all_passed':True,
            'records':[{'family':family,'role':'calibration','status':'passed',
                        'relative_l2':1e-7,'max_absolute_error':1e-6,
                        'relative_limit':1e-5,'max_absolute_limit':1e-4}
                       for family in run.FAMILIES]}


def models():
    result = []
    for family in run.FAMILIES:
        for sampling in run.SAMPLINGS:
            for condition, level, seed in run.NOISES:
                for method in run.METHODS:
                    result.append({"id": run.model_id(family, sampling, condition, method),
                        "family": family, "sampling": sampling, "condition": condition,
                        "noise_level": level, "noise_seed": seed, "method": method,
                        "training_rms_for_corruption_only": 1., "source_quality_passed": True,
                        "selection_predictor":"resolved_factor4" if method.endswith("_resolved") else "original_observed_grid",
                        "training_trajectories": run.role_count(family, "training"),
                        "development_trajectories": run.role_count(family, "development"),
                        "selected": selected(), "candidates": [selected()],
                        "failure_reason": None, "near_optimal_supports": [["u_xx"]],
                        "sources": preparation()["sources"], "input_hashes": preparation()["input_hashes"]})
    return result


class ResolvedPipelineTests(unittest.TestCase):
    def test_four_arms_share_fitting_inputs_and_dispatch_declared_selectors(self):
        observed=[field(),field(.1)];developments=[field(.2),field(.3)]
        blocks=[];choices=[]
        def block(d,variant=None):
            self.assertIn(id(d),[id(v) for v in observed])
            self.assertEqual(set(d),{'x','t','u','boundary'})
            blocks.append(('integrated' if variant is None else variant,id(d)))
            return {'X':np.ones((8,9)),'y':np.ones(8)}
        def choose(which,candidates,dev):
            self.assertIs(dev,developments)
            choices.append(which)
            return {'selected':candidates[0],'candidates':candidates,'failure_reason':None}
        with ExitStack() as stack:
            stack.enter_context(patch.object(run.weak_engine,'VARIANTS',('a','b','c')))
            stack.enter_context(patch.object(run.engine,'integrated_block',side_effect=block))
            stack.enter_context(patch.object(run.weak_engine,'weak_block',side_effect=block))
            fit=stack.enter_context(patch.object(run.engine,'fit_blocks',side_effect=lambda b,m: [selected()]))
            stack.enter_context(patch.object(run.engine,'select_candidates',side_effect=lambda c,d: choose('original',c,d)))
            stack.enter_context(patch.object(run.selection,'select_candidates',side_effect=lambda c,d: choose('resolved',c,d)))
            for method in run.METHODS:
                result=run.fit_observations(observed,developments,method)
                self.assertEqual(result['selection_predictor'],
                    'resolved_factor4' if method.endswith('_resolved') else 'original_observed_grid')
            self.assertEqual(choices,['original','original','resolved','resolved'])
            self.assertEqual(len(blocks),16)
            self.assertEqual(sum(k=='integrated' for k,_ in blocks),4)
            self.assertEqual([c.args[1] for c in fit.call_args_list],['adaptive']*8)

    def test_resolved_selector_uses_only_initial_frame_and_preserves_near_best_rule(self):
        observations=[field(),field(.15)]
        # Give the second trajectory different timestamps, to identify it in the
        # manufactured forecast without passing its future into that interface.
        observations[1]['t']=observations[1]['t']+.5
        candidates=[]
        for identifier,error,size in [('best',.01,2),('sparse-near',.0104,1),('too-far',.011,1)]:
            candidate=selected();candidate.update(candidate_id=identifier,support_size=size,
                terms=[{'name':'u_xx','coefficient':error}])
            candidates.append(candidate)
        saved=deepcopy(candidates)
        def forecast(terms,x,t,initial):
            source=observations[int(t[0]>.2)]
            np.testing.assert_array_equal(initial,source['u'][0])
            np.testing.assert_array_equal(x,source['x'])
            prediction=source['u'].copy()
            prediction[1:]+=terms[0]['coefficient']
            return prediction
        with patch.object(run.selection,'forecast',side_effect=forecast) as refined, \
             patch.object(run.predictor,'forecast',side_effect=AssertionError('old predictor in refined selector')):
            result=run.selection.select_candidates(candidates,observations)
        self.assertEqual(result['selected']['candidate_id'],'sparse-near')
        self.assertEqual(refined.call_count,6)
        self.assertEqual(candidates,saved)
        self.assertTrue(all(c['eligible'] for c in result['candidates']))

    def test_noise_roles_and_seeds_are_disjoint_and_coarse_is_nested(self):
        clean=field()
        kwargs=('allen_cahn','training',0)
        full=run.data.corrupt(clean,*kwargs,'full',.001,53,1.)
        coarse=run.data.corrupt(clean,*kwargs,'coarse',.001,53,1.)
        self.assertEqual(set(full),{'x','t','u','boundary'})
        np.testing.assert_array_equal(coarse['u'],full['u'][::2,::2])
        np.testing.assert_array_equal(coarse['x'],full['x'][::2])
        np.testing.assert_array_equal(coarse['t'],full['t'][::2])
        again=run.data.corrupt(clean,*kwargs,'full',.001,53,1.)
        np.testing.assert_array_equal(again['u'],full['u'])
        self.assertFalse(np.array_equal(full['u'][0],clean['u'][0]))
        for role,index,seed in [('development',0,53),('final',0,53),('training',1,53),('training',0,89)]:
            other=run.data.corrupt(clean,'allen_cahn',role,index,'full',.001,seed,1.)
            self.assertFalse(np.array_equal(other['u'],full['u']))
        strong=run.data.corrupt(clean,*kwargs,'full',.01,53,1.)
        np.testing.assert_allclose(strong['u']-clean['u'],10*(full['u']-clean['u']),atol=2e-15)
        np.testing.assert_array_equal(clean['u'],field()['u'])

    def test_all_weak_variants_use_only_observations_and_failures_are_retained(self):
        observed=[field(),field(.1)];dev=[field(.2)]
        calls=[]
        def block(d,variant):
            self.assertIn(id(d),[id(x) for x in observed])
            self.assertEqual(set(d),{'x','t','u','boundary'})
            calls.append(variant)
            if variant=='broken':raise ValueError('manufactured variant failure')
            return {'X':np.ones((8,9)),'y':np.ones(8)}
        def choose(candidates,developments):
            self.assertIs(developments,dev)
            self.assertEqual({c['weak_variant'] for c in candidates},{'a','b'})
            self.assertEqual(len({c['candidate_id'] for c in candidates}),2)
            return {'selected':candidates[0],'candidates':candidates,'failure_reason':None}
        with ExitStack() as stack:
            stack.enter_context(patch.object(run.weak_engine,'VARIANTS',('a','broken','b')))
            stack.enter_context(patch.object(run.weak_engine,'weak_block',side_effect=block))
            fit=stack.enter_context(patch.object(run.engine,'fit_blocks',side_effect=lambda blocks,method: [selected()]))
            stack.enter_context(patch.object(run.selection,'select_candidates',side_effect=choose))
            gen=stack.enter_context(patch.object(run.generators,'generate',side_effect=AssertionError('new fields in fit')))
            truth=stack.enter_context(patch.object(run.generators,'reference_terms',side_effect=AssertionError('truth in fit')))
            result=run.fit_observations(observed,dev,'weak_resolved')
            self.assertEqual(calls,['a','a','broken','b','b'])
            self.assertEqual([c.args[1] for c in fit.call_args_list],['adaptive','adaptive'])
            self.assertEqual(len(result['variant_trials']),3)
            self.assertIn('manufactured',result['variant_trials'][1]['failure_reason'])
            gen.assert_not_called();truth.assert_not_called()
            with self.assertRaises(TypeError):
                run.fit_observations(observed,dev,'weak_resolved',family='allen_cahn')
            with self.assertRaises(TypeError):
                run.fit_observations(observed,dev,'weak_resolved',noise_sigma=.01)

    def test_prepare_reuses_calibration_and_only_generates_train_development(self):
        generated = []
        calibration = calibration_receipt()

        def reader(path):
            if Path(path).name == "calibration.json": return calibration
            if Path(path).name == "protocol.json": return design()
            raise AssertionError("Reference or unrelated file read during preparation")

        def generate(family, role, index):
            self.assertIn(role, ("training", "development"))
            generated.append((family, role, index))
            return field()

        with tempfile.TemporaryDirectory() as tmp, ExitStack() as stack, redirect_stdout(io.StringIO()):
            stack.enter_context(patch.object(run, "DATA", Path(tmp)/"data"))
            stack.enter_context(patch.object(run, "read", side_effect=reader))
            stack.enter_context(patch.object(run, "sha", return_value="generator_hash"))
            stack.enter_context(patch.object(run, "sources", return_value={}))
            stack.enter_context(patch.object(run, "input_hashes", return_value={}))
            stack.enter_context(patch.object(run.generators, "generate", side_effect=generate))
            writer = stack.enter_context(patch.object(run.data, "write_clean"))
            loader = stack.enter_context(patch.object(run.data, "read_clean", side_effect=lambda _root, _f, role, _i: field() if role == "training" else (_ for _ in ()).throw(AssertionError("non-training scale"))))
            recalibrate = stack.enter_context(patch.object(run.generators, "calibration", side_effect=AssertionError("unneeded reference rerun")))
            reference = stack.enter_context(patch.object(run.generators, "reference_terms", side_effect=AssertionError("truth during preparation")))
            receipt = run.prepare()
            self.assertEqual(len(generated), 15)
            self.assertEqual(writer.call_count, 15)
            self.assertEqual(loader.call_count, 9)
            self.assertEqual(set(receipt["training_rms"]), set(run.FAMILIES))
            self.assertTrue(receipt["all_families_independently_generated"])
            self.assertFalse(receipt["synthetic_final_generated"])
            self.assertTrue(receipt["all_source_quality_passed"])
            recalibrate.assert_not_called()
            reference.assert_not_called()

    def test_calibration_hash_mismatch_blocks_all_source_actions(self):
        with tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
            stack.enter_context(patch.object(run, "DATA", Path(tmp)/"data"))
            stack.enter_context(patch.object(run, "read", side_effect=lambda path: design() if Path(path).name=='protocol.json' else calibration_receipt()))
            stack.enter_context(patch.object(run, "sha", return_value="new"))
            generate = stack.enter_context(patch.object(run.generators, "generate"))
            with self.assertRaisesRegex(ValueError, "identity/protocol"):
                run.prepare()
            generate.assert_not_called()

    def test_pending_or_changed_design_blocks_all_formal_generation(self):
        baseline=design()
        changes=[('variant_menu_status','pending'),('weak_variants',{}),
                 ('weak_engine_protocol',{'changed':True}),
                 ('forecast_protocol',{'changed':True}),
                 ('families',[run.FAMILIES[0]]*len(run.FAMILIES)),
                 ('roles',{'training':2,'development':2,'final':3,'calibration':1}),
                 ('methods',['plain','weak_resolved'])]
        with tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
            stack.enter_context(patch.object(run,'DATA',Path(tmp)/'data'))
            stack.enter_context(patch.object(run,'sha',return_value='generator_hash'))
            generate=stack.enter_context(patch.object(run.generators,'generate',side_effect=AssertionError('formal role before design freeze')))
            for key,value in changes:
                with self.subTest(key=key):
                    changed=deepcopy(baseline);changed[key]=value
                    with patch.object(run,'read',side_effect=lambda path: changed if Path(path).name=='protocol.json' else calibration_receipt()):
                        with self.assertRaisesRegex(ValueError,'must be frozen'):
                            run.prepare()
            generate.assert_not_called()

    def test_calibration_rejects_false_passes_but_accepts_explicit_failures(self):
        valid=calibration_receipt();protocol=design();invalid=[]
        duplicate=deepcopy(valid);duplicate['records'][1]=deepcopy(duplicate['records'][0]);invalid.append(duplicate)
        for key,value in [('role','training'),('status','unknown'),('relative_l2',1e-4),
                          ('relative_l2',float('nan')),('relative_l2',float('inf')),
                          ('relative_l2',True),('relative_l2',-1e-9),
                          ('max_absolute_error',float('nan')),('max_absolute_error',1e-3),
                          ('relative_limit',1e-4),('max_absolute_limit',1e-3)]:
            changed=deepcopy(valid);changed['records'][0][key]=value;invalid.append(changed)
        for key,value in [('protocol',{}),('formal_training_development_final_generated',True),
                          ('thresholds_changed_after_results',True),('all_passed',False),
                          ('all_passed',1)]:
            changed=deepcopy(valid);changed[key]=value;invalid.append(changed)
        with patch.object(run,'sha',return_value='generator_hash'):
            run.validate_design_and_calibration(protocol,valid)
            for index,receipt in enumerate(invalid):
                with self.subTest(invalid_receipt=index):
                    with self.assertRaises(ValueError):
                        run.validate_design_and_calibration(protocol,receipt)
            failed=deepcopy(valid)
            failed['records'][0]={'family':run.FAMILIES[0],'role':'calibration',
                                  'status':'source_quality_failure','failure_reason':'manufactured'}
            failed['all_passed']=False
            run.validate_design_and_calibration(protocol,failed)
            failed['all_passed']=True
            with self.assertRaisesRegex(ValueError,'aggregate'):
                run.validate_design_and_calibration(protocol,failed)

    def test_freeze_exposes_no_clean_or_sigma_to_model_and_writes_global_index_last(self):
        prep = preparation()
        loads, writes = [], []
        real_write = run.write

        def load(_root, family, role, index):
            self.assertIn(role, ("training", "development"))
            loads.append((family, role, index))
            d = field()
            d["private_clean_marker"] = True
            return d

        def corrupt(clean, family, role, index, sampling, level, seed, scale):
            self.assertIn(role, ("training", "development"))
            self.assertGreater(scale, 0.)
            # Deliberately distinct fields ensure selector is given observations.
            return {"x": clean["x"].copy(), "t": clean["t"].copy(),
                    "u": clean["u"].copy()+.25, "boundary": "periodic"}

        def block(d):
            self.assertEqual(set(d), {"x", "t", "u", "boundary"})
            np.testing.assert_allclose(d["u"], field()["u"]+.25)
            return {"X": np.ones((20, 9)), "y": np.ones(20)}

        def select(candidates, dev):
            for d in dev:
                self.assertEqual(set(d), {"x", "t", "u", "boundary"})
                np.testing.assert_allclose(d["u"], field()["u"]+.25)
            return {"selected": selected(), "candidates": candidates,
                    "near_optimal_supports": [["u_xx"]], "failure_reason": None}

        def writer(path, value):
            if Path(path).name == "freeze_index.json":
                self.assertEqual(sum(p.parent.name == "models" for p in writes), 120)
                self.assertTrue(run.complete(value["records"]))
            writes.append(Path(path))
            real_write(path, value)

        with tempfile.TemporaryDirectory() as tmp, ExitStack() as stack, redirect_stdout(io.StringIO()):
            root = Path(tmp)
            data_root = root/"data"; data_root.mkdir(); (data_root/"prepared.json").write_text("{}")
            stack.enter_context(patch.object(run, "DATA", data_root))
            stack.enter_context(patch.object(run, "verify_preparation", return_value=prep))
            stack.enter_context(patch.object(run, "sources", return_value=prep["sources"]))
            stack.enter_context(patch.object(run, "read", side_effect=lambda p: {} if Path(p).name == "protocol.json" else (_ for _ in ()).throw(AssertionError("unexpected read"))))
            stack.enter_context(patch.object(run, "write", side_effect=writer))
            stack.enter_context(patch.object(run.data, "read_clean", side_effect=load))
            stack.enter_context(patch.object(run.data, "corrupt", side_effect=corrupt))
            stack.enter_context(patch.object(run.engine, "integrated_block", side_effect=block))
            stack.enter_context(patch.object(run.weak_engine, "weak_block", side_effect=lambda d, variant: block(d)))
            stack.enter_context(patch.object(run.engine, "fit_blocks", side_effect=lambda *a: [selected()]))
            stack.enter_context(patch.object(run.engine, "select_candidates", side_effect=select))
            stack.enter_context(patch.object(run.selection, "select_candidates", side_effect=select))
            generate = stack.enter_context(patch.object(run.generators, "generate", side_effect=AssertionError("no generation in freeze")))
            index = run.freeze(root/"out", workers=1)
            self.assertEqual(len(loads), 15)
            self.assertEqual(writes[-1].name, "freeze_index.json")
            self.assertEqual(len(index["records"]), 120)
            self.assertFalse(index["synthetic_final_generated"])
            generate.assert_not_called()

    def test_source_failure_retains_all_forty_family_models_without_fit_or_load(self):
        prep = preparation(); prep["quality"]["allen_cahn"] = {"status": "source_quality_failure"}
        prep["training_rms"]["allen_cahn"] = None
        with tempfile.TemporaryDirectory() as tmp, ExitStack() as stack, redirect_stdout(io.StringIO()):
            stack.enter_context(patch.object(run, "verify_preparation", return_value=prep))
            load = stack.enter_context(patch.object(run.data, "read_clean", side_effect=AssertionError("failed source read")))
            fit = stack.enter_context(patch.object(run.engine, "fit_blocks", side_effect=AssertionError("failed source fit")))
            records = run._freeze_family("allen_cahn", Path(tmp))
            self.assertEqual(len(records), 40)
            for r in records:
                m = json.loads((Path(tmp)/r["file"]).read_text())
                self.assertIsNone(m["selected"])
                self.assertIn("Source-quality", m["failure_reason"])
            load.assert_not_called(); fit.assert_not_called()

    def test_verify_rejects_incomplete_identity_tampering_and_worker_dependency_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
            root = Path(tmp); out = root/"out"; data_root = root/"data"; data_root.mkdir()
            receipt = data_root/"prepared.json"
            prep = preparation(); run.write(receipt, prep)
            receipt_original = receipt.read_bytes(); records = []
            for model in models():
                path = out/"models"/f"{model['id']}.json"; run.write(path, model)
                records.append({"id": model["id"], "file": str(path.relative_to(out)), "sha256": run.sha(path)})
            index = {"records": records, **prep, "preparation_sha256": run.sha(receipt)}
            run.write(out/"freeze_index.json", index)
            stack.enter_context(patch.object(run, "DATA", data_root))
            source = stack.enter_context(patch.object(run, "sources", return_value=prep["sources"]))
            hashes = stack.enter_context(patch.object(run, "input_hashes", return_value=prep["input_hashes"]))
            self.assertEqual(len(run.verify_freeze(out)["records"]), 120)
            for invalid in (records[:-1], [records[0], records[0], *records[2:]],
                            [{**records[0], "id": "unknown"}, *records[1:]]):
                run.write(out/"freeze_index.json", {**index, "records": invalid})
                with self.assertRaises(ValueError): run.verify_freeze(out)
            run.write(out/"freeze_index.json", index)
            source.return_value = {"code.py": "changed"}
            with self.assertRaises(ValueError): run.verify_freeze(out)
            source.return_value = prep["sources"]
            hashes.return_value = {"input.npz": "changed"}
            with self.assertRaises(ValueError): run.verify_freeze(out)
            hashes.return_value = prep["input_hashes"]
            receipt.write_text('{"changed":true}')
            with self.assertRaises(ValueError): run.verify_freeze(out)
            receipt.write_bytes(receipt_original)
            target = out/records[0]["file"]; original = target.read_bytes()
            target.write_bytes(original+b" ")
            with self.assertRaises(ValueError): run.verify_freeze(out)
            target.write_bytes(original)
            for key, value in (("method", "adaptive"), ("selection_predictor","mismatched"), ("sources", {}), ("input_hashes", {}), ("noise_level", .01), ("noise_level", False), ("noise_seed", 53), ("noise_seed", False), ("training_rms_for_corruption_only", 2.), ("training_rms_for_corruption_only", True), ("training_trajectories", 1), ("source_quality_passed", False)):
                modified = json.loads(original); modified[key] = value; run.write(target, modified)
                changed = [{**records[0], "sha256": run.sha(target)}, *records[1:]]
                run.write(out/"freeze_index.json", {**index, "records": changed})
                with self.assertRaises(ValueError): run.verify_freeze(out)
            target.write_bytes(original)
            # Even a whole family of consistently rehashed models plus matching
            # index cannot change the scale anchored in the original receipt.
            changed_index = deepcopy(index)
            changed_index["training_rms"]["allen_cahn"] = 2.
            for record in changed_index["records"]:
                path = out/record["file"]
                model = run.read(path)
                if model["family"] == "allen_cahn":
                    model["training_rms_for_corruption_only"] = 2.
                    run.write(path, model)
                    record["sha256"] = run.sha(path)
            run.write(out/"freeze_index.json", changed_index)
            with self.assertRaisesRegex(ValueError, "original preparation: training_rms"):
                run.verify_freeze(out)
            wrong_path = [{**records[0], "file": "models/wrong.json"}, *records[1:]]
            run.write(out/"freeze_index.json", {**index, "records": wrong_path})
            with self.assertRaises(ValueError): run.verify_freeze(out)

    def test_invalid_global_freeze_blocks_final_and_reference_reads(self):
        with tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
            stack.enter_context(patch.object(run, "verify_freeze", side_effect=ValueError("incomplete")))
            reader = stack.enter_context(patch.object(run, "read", side_effect=AssertionError("reference before freeze")))
            loader = stack.enter_context(patch.object(run.data, "read_clean", side_effect=AssertionError("final before freeze")))
            gen = stack.enter_context(patch.object(run.generators, "generate", side_effect=AssertionError("final before freeze")))
            with self.assertRaisesRegex(ValueError, "incomplete"): run.evaluate(Path(tmp)/"out")
            reader.assert_not_called(); loader.assert_not_called(); gen.assert_not_called()

    def _evaluate_manufactured(self, shift=0., failed_family=None, final_failure=None):
        model_list = models()
        if failed_family:
            for m in model_list:
                if m["family"] == failed_family:
                    m.update(selected=None, candidates=[], failure_reason="Source-quality failure",
                             source_quality_passed=False, training_rms_for_corruption_only=None,
                             training_trajectories=0, development_trajectories=0)
        by_name = {m["id"]+".json": m for m in model_list}
        verified, last_observation, noisy_initial_checks = [], [], []
        prep = preparation()
        if failed_family:
            prep["quality"][failed_family] = {"status": "source_quality_failure"}
            prep["all_source_quality_passed"] = False
            prep["training_rms"][failed_family] = None
        idx = {**prep, "records": [{"id": m["id"], "file": "models/"+m["id"]+".json", "sha256": "frozen"} for m in model_list]}

        def verify(_out):
            verified.append(True); return idx

        def reader(path):
            self.assertTrue(verified)
            if Path(path).name == "references.json":
                return {"references": {f: selected()["terms"] for f in run.FAMILIES}}
            return deepcopy(by_name[Path(path).name])

        def load(_root, family, role, index):
            self.assertTrue(verified); self.assertEqual(role, "final")
            return field(shift)

        def generate(family, role, index):
            self.assertTrue(verified); self.assertEqual(role, "final")
            if (family, index) == final_failure:
                raise RuntimeError("manufactured final reference failure")
            return field(shift)

        def observe(clean, family, role, index, sampling, level, seed, scale):
            self.assertTrue(verified); self.assertEqual(role, "final")
            observed = run.data.corrupt(clean, family, role, index, sampling, level, seed, scale)
            last_observation[:] = [(observed, clean, level, sampling)]
            return observed

        def predict(terms, x, t, initial):
            observed, clean, level, sampling = last_observation[0]
            np.testing.assert_array_equal(initial, observed["u"][0])
            stride = 1 if sampling == "full" else 2
            if level:
                self.assertFalse(np.array_equal(initial, clean["u"][0, ::stride]))
                noisy_initial_checks.append(True)
            return np.tile(initial, (len(t), 1))

        with tempfile.TemporaryDirectory() as tmp, ExitStack() as stack, redirect_stdout(io.StringIO()):
            out = Path(tmp)/"out"; out.mkdir(); (out/"freeze_index.json").write_text("manufactured")
            stack.enter_context(patch.object(run, "verify_freeze", side_effect=verify))
            stack.enter_context(patch.object(run, "read", side_effect=reader))
            stack.enter_context(patch.object(run.data, "read_clean", side_effect=load))
            stack.enter_context(patch.object(run.generators, "generate", side_effect=generate))
            stack.enter_context(patch.object(run, "observation", side_effect=observe))
            original_forecast=stack.enter_context(patch.object(run.predictor, "forecast", side_effect=predict))
            resolved_forecast=stack.enter_context(patch.object(run.resolved_forecast, "forecast", side_effect=predict))
            fit = stack.enter_context(patch.object(run.engine, "fit_blocks", side_effect=AssertionError("refit during final")))
            choose = stack.enter_context(patch.object(run.engine, "select_candidates", side_effect=AssertionError("selection during final")))
            report = run.evaluate(out)
            fit.assert_not_called(); choose.assert_not_called()
            expected_calls=120 if failed_family else 180
            if final_failure:expected_calls-=20
            self.assertEqual(original_forecast.call_count,expected_calls)
            self.assertEqual(resolved_forecast.call_count,expected_calls)
            self.assertEqual(sum(len(r['forecasts']) for r in report['records']),360)
        self.assertGreater(len(noisy_initial_checks), 0)
        return report

    def test_future_clean_changes_scores_not_frozen_models_and_uses_noisy_initial(self):
        first = self._evaluate_manufactured(0.)
        changed = self._evaluate_manufactured(.8)
        self.assertEqual(len(first["records"]), 120)
        for a, b in zip(first["records"], changed["records"]):
            self.assertEqual(a["terms"], b["terms"])
            self.assertEqual(a["equation"], b["equation"])
            self.assertEqual(a["frozen_model_sha256"], b["frozen_model_sha256"])
            self.assertGreater(b["mean_clean_relative_l2"], a["mean_clean_relative_l2"])
            for m in a["forecasts"]: self.assertTrue(m["initial_is_noisy"])

    def test_evaluation_retains_source_failure_family_in_all_120_records(self):
        report = self._evaluate_manufactured(failed_family="allen_cahn")
        self.assertTrue(report["all_declared_models_reported"])
        failed = [r for r in report["records"] if r["family"] == "allen_cahn"]
        self.assertEqual(len(failed), 40)
        self.assertTrue(all(not r["selected"] and r["forecast_failures"] for r in failed))
        self.assertFalse(report["gate"]["passed"])

    def test_one_failed_final_reference_keeps_three_trajectory_slots_and_all_models(self):
        report = self._evaluate_manufactured(final_failure=("ks_unstable", 1))
        self.assertTrue(report["all_declared_models_reported"])
        self.assertIsNone(report["final_reference_failures"]["ks_unstable"][0])
        self.assertIn("manufactured", report["final_reference_failures"]["ks_unstable"][1])
        records = [r for r in report["records"] if r["family"] == "ks_unstable"]
        self.assertEqual(len(records), 40)
        for r in records:
            self.assertEqual(len(r["forecasts"]), 3)
            self.assertEqual(r["forecasts"][0]["status"], "evaluated")
            self.assertEqual(r["forecasts"][1]["status"], "failed")
            self.assertEqual(r["forecast_failures"], 1)
            self.assertIsNone(r["mean_clean_relative_l2"])
        self.assertFalse(report["gate"]["passed"])

    @staticmethod
    def gate_rows():
        rows = []
        reference = [{"name": "u_xx", "coefficient": .0001},
                     {"name": "u", "coefficient": 5.}, {"name": "u^3", "coefficient": -5.}]
        for m in models():
            terms = deepcopy(reference)
            if m["method"] == "adaptive_resolved" and m["condition"] != "clean":
                terms.append({"name": "u_x", "coefficient": .00001})
            rows.append({**m, "recovery": run.reference_metrics(terms, reference),
                         "mean_clean_relative_l2": .001, "forecast_failures": 0})
        return rows

    def test_gate_predeclares_weak_resolved_and_rejects_small_missing_diffusion(self):
        rows = self.gate_rows()
        gate = run.promotion_gate(rows)
        self.assertTrue(gate["passed"])
        self.assertFalse(gate["merged"])
        self.assertEqual(gate["candidate"], "weak_resolved")
        self.assertEqual(gate["comparators"], ["adaptive_resolved","weak_original"])
        self.assertEqual(gate["decision"], "eligible_for_review_only")
        target = next(r for r in rows if r["family"] == "allen_cahn" and
                      r["condition"] == "clean" and r["method"] == "weak_resolved")
        refs = target["recovery"]["reference_terms"]
        target["recovery"] = run.reference_metrics([t for t in refs if t["name"] != "u_xx"], refs)
        self.assertLess(target["recovery"]["coefficient_relative_l2"], 2e-5)
        gate = run.promotion_gate(rows)
        self.assertFalse(gate["passed"])
        self.assertTrue(any(c["name"].endswith(":each_true_term_accuracy") and not c["passed"] for c in gate["checks"]))
        self.assertEqual(gate["decision"], "keep_baseline_do_not_merge")

    def test_each_true_coefficient_limits_apply_at_every_noise_level(self):
        for condition,level,seed in run.NOISES:
            limit=.25 if level==.01 else .10
            for relative,passes in ((limit-1e-6,True),(limit+1e-6,False)):
                rows=self.gate_rows()
                target=next(r for r in rows if r['family']=='allen_cahn' and r['sampling']=='full'
                            and r['condition']==condition and r['method']=='weak_resolved')
                refs=target['recovery']['reference_terms'];terms=deepcopy(refs)
                next(t for t in terms if t['name']=='u_xx')['coefficient']*=1+relative
                target['recovery']=run.reference_metrics(terms,refs)
                # The vector error is tiny; the scientific check must inspect D itself.
                self.assertLess(target['recovery']['coefficient_relative_l2'],1e-5)
                gate=run.promotion_gate(rows)
                check=next(c for c in gate['checks'] if c['name']==target['id']+':each_true_term_accuracy')
                self.assertEqual(check['passed'],passes,(condition,relative))
                if not passes:self.assertFalse(gate['passed'])
            rows=self.gate_rows()
            target=next(r for r in rows if r['family']=='allen_cahn' and r['sampling']=='full'
                        and r['condition']==condition and r['method']=='weak_resolved')
            refs=target['recovery']['reference_terms']
            target['recovery']=run.reference_metrics([t for t in refs if t['name']!='u_xx'],refs)
            row=next(c for c in target['recovery']['coefficient_comparison'] if c['name']=='u_xx')
            self.assertEqual(row['relative_error'],1.)
            self.assertFalse(run.promotion_gate(rows)['passed'])

    def test_coefficient_gain_path_requires_two_families_and_substantial_absolute_gain(self):
        def rows_with_errors(old_error,new_error,families=2):
            rows=self.gate_rows()
            # All supports are identical: only the predeclared coefficient path
            # can pass, and ordinary/original controls remain unchanged.
            for row in rows:
                refs=row['recovery']['reference_terms'];terms=deepcopy(refs)
                if row['family'] in run.FAMILIES[:families]:
                    error={'adaptive_resolved':old_error,'weak_resolved':new_error}.get(row['method'],0.)
                    next(t for t in terms if t['name']=='u_xx')['coefficient']*=1+error
                row['recovery']=run.reference_metrics(terms,refs)
            return rows
        gate=run.promotion_gate(rows_with_errors(.08,.04))
        self.assertTrue(gate['passed'])
        self.assertFalse(gate['structure_gain_path'])
        self.assertTrue(gate['coefficient_gain_path'])
        self.assertEqual(gate['substantial_coefficient_gain_families'],list(run.FAMILIES[:2]))
        for old,new,families in [(.008,.004,2),(.02,.0125,2),(.06,.047,2),(.08,.04,1)]:
            with self.subTest(old=old,new=new,families=families):
                gate=run.promotion_gate(rows_with_errors(old,new,families))
                self.assertFalse(gate['passed'])
                self.assertFalse(gate['structure_gain_path'])
                self.assertFalse(gate['coefficient_gain_path'])

    def test_forecast_nonregression_is_required_against_both_comparators(self):
        for strict,loose in [('adaptive_resolved','weak_original'),('weak_original','adaptive_resolved')]:
            rows=self.gate_rows()
            matching=lambda method:next(r for r in rows if r['family']=='allen_cahn'
                and r['sampling']=='full' and r['condition']=='clean' and r['method']==method)
            matching('weak_resolved')['mean_clean_relative_l2']=.06
            matching(strict)['mean_clean_relative_l2']=.001
            matching(loose)['mean_clean_relative_l2']=.1
            gate=run.promotion_gate(rows)
            prefix=matching('weak_resolved')['id']+':vs_'
            checks={c['name']:c['passed'] for c in gate['checks']}
            self.assertFalse(checks[prefix+strict+':prediction'])
            self.assertTrue(checks[prefix+loose+':prediction'])
            self.assertFalse(gate['passed'])

    def test_partial_freeze_or_final_artifacts_are_not_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
            out=Path(tmp)/'out';(out/'models').mkdir(parents=True)
            artifact=out/'models'/'partial.json';artifact.write_text('{"retained":true}')
            verify=stack.enter_context(patch.object(run,'verify_preparation',side_effect=AssertionError('partial restart')))
            with self.assertRaises(RuntimeError):run.freeze(out)
            self.assertEqual(artifact.read_text(),'{"retained":true}')
            verify.assert_not_called()
            final_out=Path(tmp)/'final';(final_out/'final_reference').mkdir(parents=True)
            partial=final_out/'final_reference'/'partial.npz';partial.write_bytes(b'retained partial evidence')
            verify_final=stack.enter_context(patch.object(run,'verify_freeze',side_effect=AssertionError('partial final restart')))
            with self.assertRaises(RuntimeError):run.evaluate(final_out)
            self.assertEqual(partial.read_bytes(),b'retained partial evidence')
            verify_final.assert_not_called()

    def test_gate_fails_closed_for_bad_identity_source_quality_and_absent_gain(self):
        rows = self.gate_rows()
        for invalid in (rows[:-1], [rows[0], rows[0], *rows[2:]],
                        [{**rows[0], "id": "unknown"}, *rows[1:]]):
            self.assertFalse(run.promotion_gate(invalid)["passed"])
        self.assertFalse(run.promotion_gate(rows, source_quality=False)["passed"])
        for key, value in (("noise_seed", 53), ("noise_seed", False),
                           ("noise_level", .01), ("noise_level", False),
                           ("training_rms_for_corruption_only", 2.),
                           ("training_rms_for_corruption_only", True),
                           ("source_quality_passed", False), ("training_trajectories", 1)):
            changed = deepcopy(rows)
            changed[0][key] = value
            self.assertFalse(run.promotion_gate(changed)["passed"], key)
        for r in rows:
            if r["method"] == "weak_resolved":
                p = next(x for x in rows if x["id"] == run.model_id(r["family"], r["sampling"], r["condition"], "adaptive_resolved"))
                r["recovery"] = deepcopy(p["recovery"])
        # Other arms cannot replace the predeclared primary comparison.
        gate = run.promotion_gate(rows)
        self.assertFalse(gate["passed"])
        self.assertEqual(gate["candidate"], "weak_resolved")

    def test_saved_preparation_freeze_and_scores_are_not_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
            root = Path(tmp); data_root = root/"data"; data_root.mkdir()
            out = root/"out"; out.mkdir()
            (data_root/"prepared.json").write_text("{}")
            (out/"freeze_index.json").write_text("{}")
            (out/"comparison.json").write_text("{}")
            stack.enter_context(patch.object(run, "DATA", data_root))
            gen = stack.enter_context(patch.object(run.generators, "generate", side_effect=AssertionError("existing results overwritten")))
            with self.assertRaises(RuntimeError): run.prepare()
            with self.assertRaises(RuntimeError): run.freeze(out)
            with self.assertRaises(RuntimeError): run.evaluate(out)
            gen.assert_not_called()


if __name__ == "__main__":
    unittest.main()
