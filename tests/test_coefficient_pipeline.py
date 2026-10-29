"""Coefficient-study isolation/gates using temporary fixtures, never formal data."""
from contextlib import ExitStack, redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from coefficient_study import run

TERMS=[{'name':'u_xx','coefficient':.1}]


def field(family='allen_cahn',shift=0.):
    x=np.linspace(0.,2*np.pi,32,endpoint=False)
    t=np.linspace(0.,1.1*run.TARGET_PROTOCOL['forecast_duration'][family],111)
    u=.2+np.exp(-.1*t[:,None])*np.sin(x)[None,:]
    u[11:]+=shift
    return {'x':x,'t':t,'u':u,'boundary':'periodic'}


def selected():
    return {'equation':'u_t = 0.1 u_xx','terms':deepcopy(TERMS),
            'support_size':1,'fit_success':True,'candidate_id':'manufactured'}


def preparation():
    return {'sources':{'code.py':'source_hash'},'input_hashes':{'input.npz':'byte_hash'},
        'quality':{f:{'status':'passed'} for f in run.FAMILIES},
        'all_source_quality_passed':True,'training_rms':{f:1. for f in run.FAMILIES},'protocol':{}}


def design():
    canonical=lambda value:json.loads(json.dumps(value,allow_nan=False))
    return {'variant_menu_status':'frozen','weak_variants':canonical(run.weak_engine.VARIANTS),
        'weak_engine_protocol':canonical(run.weak_engine.PROTOCOL),
        'forecast_protocol':canonical(run.prefix_forecast.PROTOCOL),
        'coefficient_engine_protocol':canonical(run.engine.PROTOCOL),
        'coefficient_selection_protocol':canonical(run.selection.PROTOCOL),
        'noise_protocol':canonical(run.noise.PROTOCOL),
        'initial_protocol':canonical(run.initial.PROTOCOL),
        'target_protocol':canonical(run.TARGET_PROTOCOL),
        'coefficient_protocol':canonical(run.COEFFICIENT_PROTOCOL),
        'data_protocol':canonical(run.data.PROTOCOL),
        'generator_protocol':canonical(run.generators.GENERATOR_PROTOCOL),
        'roles':run.data.ROLE_COUNTS|{'calibration':1},
        'families':list(run.FAMILIES),'methods':list(run.METHODS),
        'sampling':list(run.SAMPLINGS),
        'noise_conditions':[{'label':label,'level':level,'seed':seed} for label,level,seed in run.NOISES],
        'declared_configurations':120}


def calibration():
    return {'generator_sha256':'generator_hash',
        'protocol':json.loads(json.dumps(run.generators.GENERATOR_PROTOCOL)),
        'formal_training_development_final_generated':False,'thresholds_changed_after_results':False,
        'all_passed':True,'records':[{'family':f,'role':'calibration','status':'passed',
            'relative_l2':1e-7,'max_absolute_error':1e-6,'relative_limit':1e-5,'max_absolute_limit':1e-4}
            for f in run.FAMILIES]}


def models():
    result=[]
    for family in run.FAMILIES:
        for sampling in run.SAMPLINGS:
            for condition,level,seed in run.NOISES:
                for method in run.METHODS:
                    result.append({'id':run.model_id(family,sampling,condition,method),
                        'family':family,'sampling':sampling,'condition':condition,
                        'noise_level':level,'noise_seed':seed,'method':method,
                        'training_rms_for_corruption_only':1.,'source_quality_passed':True,
                        **run.predictor_metadata(method),'training_trajectories':3,'development_trajectories':2,
                        'selected':selected(),'candidates':[selected()],'failure_reason':None,
                        'near_optimal_supports':[['u_xx']],
                        'sources':preparation()['sources'],'input_hashes':preparation()['input_hashes'],
                        'reliability_selection':{'status':'fallback_to_prediction','fallback_used':True,
                            'fallback_reason':'manufactured diagnostic failure'},'median_augmentation':None})
    return result


def gate_rows():
    rows=[]
    for m in models():
        error=.04 if m['method']=='noise_stable' else .08
        found=[{'name':'u_xx','coefficient':.1*(1+error)}]
        rows.append({**m,'recovery':run.reference_metrics(found,TERMS),
            'mean_clean_relative_l2':.049,'forecast_failures':0,
            'forecasts':[{'status':'evaluated','clean':{'relative_l2':v}} for v in (.02,.027,.1)]})
    return rows


def set_mean(row,value):
    row['mean_clean_relative_l2']=value
    row['forecasts']=[{'status':'evaluated','clean':{'relative_l2':value}} for _ in range(3)]


class CoefficientPreparationTests(unittest.TestCase):
    def test_protocol_sampling_noise_generator_and_configuration_count_are_bound(self):
        protocol=design();receipt=calibration()
        with patch.object(run,'sha',return_value='generator_hash'):
            run.validate_design_and_calibration(protocol,receipt)
            for key,value in (('sampling',['full']),('noise_conditions',[]),
                              ('generator_protocol',{}),('declared_configurations',119),
                              ('declared_configurations',120.0)):
                changed=deepcopy(protocol);changed[key]=value
                with self.subTest(key=key,value=value),self.assertRaisesRegex(ValueError,'must be frozen'):
                    run.validate_design_and_calibration(changed,receipt)

    def test_design_freeze_binds_all_new_algorithms_and_target(self):
        protocol=design();receipt=calibration()
        with patch.object(run,'sha',return_value='generator_hash'):
            run.validate_design_and_calibration(protocol,receipt)
            for key in ('variant_menu_status','coefficient_engine_protocol','coefficient_selection_protocol',
                        'noise_protocol','forecast_protocol','coefficient_protocol','target_protocol','data_protocol'):
                bad=deepcopy(protocol);bad[key]='changed'
                with self.subTest(key=key),self.assertRaisesRegex(ValueError,'must be frozen'):
                    run.validate_design_and_calibration(bad,receipt)
            wrong=deepcopy(receipt);wrong['generator_sha256']='stale'
            with self.assertRaisesRegex(ValueError,'identity/protocol'):
                run.validate_design_and_calibration(protocol,wrong)

    def test_prepare_only_generates_fifteen_training_development_fields(self):
        generated=[]
        def reader(p):
            if Path(p).name=='protocol.json':return design()
            if Path(p).name=='calibration.json':return calibration()
            raise AssertionError('unexpected source read')
        def generate(family,role,index):
            self.assertIn(role,('training','development'));generated.append((family,role,index));return field(family)
        with tempfile.TemporaryDirectory() as tmp,ExitStack() as stack,redirect_stdout(io.StringIO()):
            stack.enter_context(patch.object(run,'DATA',Path(tmp)/'data'))
            stack.enter_context(patch.object(run,'read',side_effect=reader))
            stack.enter_context(patch.object(run,'sha',return_value='generator_hash'))
            stack.enter_context(patch.object(run,'sources',return_value={}))
            stack.enter_context(patch.object(run,'input_hashes',return_value={}))
            stack.enter_context(patch.object(run.generators,'generate',side_effect=generate))
            result=run.prepare()
            self.assertEqual(len(generated),15);self.assertEqual(len(list(run.DATA.rglob('*.npz'))),15)
            self.assertTrue(all('final' not in p.parts for p in run.DATA.rglob('*.npz')))
            self.assertFalse(result['synthetic_final_generated'])
            with self.assertRaises(RuntimeError):run.prepare()

    def test_draft_stops_before_any_formal_generation(self):
        protocol=design();protocol['variant_menu_status']='draft'
        with tempfile.TemporaryDirectory() as tmp,patch.object(run,'DATA',Path(tmp)),patch.object(run,'read',side_effect=lambda p:protocol if Path(p).name=='protocol.json' else calibration()),patch.object(run.generators,'generate') as gen:
            with self.assertRaises(ValueError):run.prepare()
            gen.assert_not_called()


class CoefficientFreezeTests(unittest.TestCase):
    def test_all_models_are_written_before_global_freeze_without_final_access(self):
        prep=preparation();writes=[];loaded=[];real_write=run.write
        def loader(root,family,role,index):
            self.assertIn(role,('training','development'));loaded.append((family,role,index));return field(family)
        def fit(training,development):
            self.assertEqual(len(training),3);self.assertEqual(len(development),2)
            for d in training+development:self.assertEqual(set(d),{'x','t','u','boundary','prefix_t','prefix_u'})
            return {m:{'selected':selected(),'candidates':[selected()],'near_optimal_supports':[['u_xx']],
                'failure_reason':None,**run.predictor_metadata(m)} for m in run.METHODS}
        def writer(path,value):
            if Path(path).name=='freeze_index.json':self.assertEqual(sum(p.parent.name=='models' for p in writes),120)
            writes.append(Path(path));real_write(path,value)
        with tempfile.TemporaryDirectory() as tmp,ExitStack() as stack,redirect_stdout(io.StringIO()):
            root=Path(tmp);cache=root/'data';cache.mkdir();(cache/'prepared.json').write_text('{}')
            stack.enter_context(patch.object(run,'DATA',cache))
            stack.enter_context(patch.object(run,'verify_preparation',return_value=prep))
            stack.enter_context(patch.object(run,'sources',return_value=prep['sources']))
            stack.enter_context(patch.object(run,'read',return_value={}))
            stack.enter_context(patch.object(run,'write',side_effect=writer))
            stack.enter_context(patch.object(run.data,'read_clean',side_effect=loader))
            fitter=stack.enter_context(patch.object(run.engine,'fit_observations',side_effect=fit))
            gen=stack.enter_context(patch.object(run.generators,'generate',side_effect=AssertionError('generation during freeze')))
            index=run.freeze(root/'out')
            self.assertEqual(fitter.call_count,30);self.assertEqual(len(loaded),15)
            self.assertTrue(run.complete(index['records']));self.assertTrue(index['all_120_frozen_before_final'])
            self.assertEqual(writes[-1].name,'freeze_index.json');gen.assert_not_called()
            with self.assertRaises(RuntimeError):run.freeze(root/'out')

    def test_source_failure_retains_forty_models_without_fit_or_data_load(self):
        prep=preparation();prep['quality']['allen_cahn']={'status':'source_quality_failure'};prep['training_rms']['allen_cahn']=None
        with tempfile.TemporaryDirectory() as tmp,patch.object(run,'verify_preparation',return_value=prep),patch.object(run.data,'read_clean',side_effect=AssertionError('failed source loaded')),patch.object(run.engine,'fit_observations',side_effect=AssertionError('failed source fitted')),redirect_stdout(io.StringIO()):
            records=run._freeze_family('allen_cahn',Path(tmp))
            self.assertEqual(len(records),40)
            for r in records:self.assertIsNone(run.read(Path(tmp)/r['file'])['selected'])

    def test_frozen_sources_inputs_models_and_identity_cannot_change(self):
        with tempfile.TemporaryDirectory() as tmp,ExitStack() as stack:
            root=Path(tmp);cache=root/'data';out=root/'out';cache.mkdir()
            prep=preparation();run.write(cache/'prepared.json',prep);receipt=(cache/'prepared.json').read_bytes()
            records=[]
            for m in models():
                p=out/'models'/f"{m['id']}.json";run.write(p,m)
                records.append({'id':m['id'],'file':str(p.relative_to(out)),'sha256':run.sha(p)})
            index={'records':records,**prep,'preparation_sha256':run.sha(cache/'prepared.json'),
                   'all_120_frozen_before_final':True,'synthetic_final_generated':False}
            run.write(out/'freeze_index.json',index)
            stack.enter_context(patch.object(run,'DATA',cache))
            source=stack.enter_context(patch.object(run,'sources',return_value=prep['sources']))
            inputs=stack.enter_context(patch.object(run,'input_hashes',return_value=prep['input_hashes']))
            self.assertEqual(len(run.verify_freeze(out)['records']),120)
            for key,value in (('all_120_frozen_before_final',False),('all_120_frozen_before_final',1),
                              ('synthetic_final_generated',True),('protocol',{'changed':True})):
                run.write(out/'freeze_index.json',{**index,key:value})
                with self.subTest(key=key,value=value),self.assertRaises(ValueError):run.verify_freeze(out)
            run.write(out/'freeze_index.json',index)
            for invalid in (records[:-1],[records[0],records[0],*records[2:]]):
                run.write(out/'freeze_index.json',{**index,'records':invalid})
                with self.assertRaises(ValueError):run.verify_freeze(out)
            run.write(out/'freeze_index.json',index)
            source.return_value={'code.py':'changed'}
            with self.assertRaises(ValueError):run.verify_freeze(out)
            source.return_value=prep['sources'];inputs.return_value={'input.npz':'changed'}
            with self.assertRaises(ValueError):run.verify_freeze(out)
            inputs.return_value=prep['input_hashes'];(cache/'prepared.json').write_text('{}')
            with self.assertRaises(ValueError):run.verify_freeze(out)
            (cache/'prepared.json').write_bytes(receipt)
            target=out/records[0]['file'];original=target.read_bytes();target.write_bytes(original+b' ')
            with self.assertRaises(ValueError):run.verify_freeze(out)
            for key,value in (('coefficient_noise_correction',True),('coefficient_noise_correction',0),
                ('coefficient_selection','changed'),('conditioning','single_frame'),('final_predictor','changed'),
                ('training_rms_for_corruption_only',2.),('noise_seed',79),('sources',{}),('input_hashes',{})):
                changed=json.loads(original);changed[key]=value;run.write(target,changed)
                run.write(out/'freeze_index.json',{**index,'records':[{**records[0],'sha256':run.sha(target)},*records[1:]]})
                with self.subTest(key=key,value=value),self.assertRaises(ValueError):run.verify_freeze(out)
            target.write_bytes(original);run.write(out/'freeze_index.json',index)
            self.assertEqual(len(run.verify_freeze(out)['records']),120)

    def test_invalid_freeze_blocks_final_generation_and_reference_read(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(run,'verify_freeze',side_effect=ValueError('bad freeze')),patch.object(run,'read',side_effect=AssertionError('reference access')) as reader,patch.object(run.generators,'generate',side_effect=AssertionError('final access')) as generator:
            with self.assertRaisesRegex(ValueError,'bad freeze'):run.evaluate(Path(tmp))
            reader.assert_not_called();generator.assert_not_called()

    def test_engine_must_return_all_four_declared_arms_with_exact_metadata(self):
        results={m:{'selected':selected(),'candidates':[selected()],**run.predictor_metadata(m)} for m in run.METHODS}
        train=[field()];dev=[field()]
        with patch.object(run.engine,'fit_observations',return_value=deepcopy(results)) as fitter:
            self.assertEqual(set(run.fit_observations(train,dev)),set(run.METHODS));fitter.assert_called_once_with(train,dev)
        bad=deepcopy(results);del bad['noise_stable']
        with patch.object(run.engine,'fit_observations',return_value=bad):
            with self.assertRaises(ValueError):run.fit_observations(train,dev)
        bad=deepcopy(results);bad['noise_stable']['coefficient_selection']='forecast_only'
        with patch.object(run.engine,'fit_observations',return_value=bad):
            with self.assertRaises(ValueError):run.fit_observations(train,dev)


class CoefficientScoringTests(unittest.TestCase):
    def test_future_truth_cannot_enter_predictor_or_change_frozen_coefficients(self):
        model=next(m for m in models() if m['method']=='noise_stable');original=deepcopy(model)
        a=field();b=field(shift=2.);seen=[]
        def predict(terms,x,t,pt,pu):
            seen.append((deepcopy(terms),pt.copy(),pu.copy()));return a['u'][10:].copy(),{'future_observations_used':False}
        with patch.object(run.prefix_forecast,'forecast_with_diagnostics',side_effect=predict):
            r1=run._score_record((model,'hash',[a]*3,[None]*3,TERMS))
            r2=run._score_record((model,'hash',[b]*3,[None]*3,TERMS))
        self.assertEqual(model,original);self.assertNotEqual(r1['mean_clean_relative_l2'],r2['mean_clean_relative_l2'])
        self.assertEqual(r1['terms'],r2['terms']);self.assertEqual(r1['forecast_failures'],0)
        self.assertEqual(r1['reliability_selection'],original['reliability_selection'])
        self.assertTrue(r1['reliability_selection']['fallback_used'])
        self.assertEqual(r1['reliability_selection']['fallback_reason'],'manufactured diagnostic failure')
        for x,y in zip(seen[:3],seen[3:]):
            self.assertEqual(x[0],y[0]);np.testing.assert_array_equal(x[1],y[1]);np.testing.assert_array_equal(x[2],y[2])

    def test_every_arm_has_same_prefix_and_full_future_duration(self):
        for model in models()[::5]:
            f=field(model['family']);stride=1 if model['sampling']=='full' else 2
            with patch.object(run.prefix_forecast,'forecast_with_diagnostics',return_value=(f['u'][10::stride,::stride],{})):
                r=run._score_record((model,'hash',[f]*3,[None]*3,TERMS))
            for item in r['forecasts']:
                self.assertEqual(item['conditioning'],'observed_prefix')
                self.assertEqual(item['prefix_frames_used'],11 if stride==1 else 6)
                self.assertAlmostEqual(item['forecast_duration'],run.TARGET_PROTOCOL['forecast_duration'][model['family']])

    def test_shortened_horizon_is_failure_before_prediction(self):
        model=models()[0];short=field();short['t']*=.5
        with patch.object(run.prefix_forecast,'forecast_with_diagnostics',side_effect=AssertionError('short forecast')):
            r=run._score_record((model,'hash',[short]*3,[None]*3,TERMS))
        self.assertEqual(r['forecast_failures'],3);self.assertIsNone(r['mean_clean_relative_l2'])
        self.assertTrue(all('horizon' in f['reason'] for f in r['forecasts']))

    def test_failure_keeps_all_three_slots_and_nulls_mean(self):
        model=models()[0];f=field()
        with patch.object(run.prefix_forecast,'forecast_with_diagnostics',return_value=(f['u'][10:],{})):
            r=run._score_record((model,'hash',[f,None,f],[None,'manufactured',None],TERMS))
        self.assertEqual(len(r['forecasts']),3);self.assertEqual(r['forecast_failures'],1)
        self.assertIsNone(r['mean_clean_relative_l2'])
        self.assertEqual([x['status'] for x in r['forecasts']],['evaluated','failed','evaluated'])
        with self.assertRaises(ValueError):run._score_record((model,'hash',[f]*2,[None]*2,TERMS))

    def test_serial_and_process_scoring_agree(self):
        model_list=[next(m for m in models() if m['method']==method and m['family']=='allen_cahn' and m['sampling']=='coarse') for method in run.METHODS]
        tasks=[(m,'hash',[field()]*3,[None]*3,TERMS) for m in model_list]
        self.assertEqual(list(run._score_tasks(tasks,1)),list(run._score_tasks(tasks,2)))

    def test_evaluate_retains_all120_and_never_fits_or_selects(self):
        model_list=models();by={m['id']+'.json':m for m in model_list};verified=[];gen_calls=[]
        prep=preparation();index={**prep,'records':[{'id':m['id'],'file':'models/'+m['id']+'.json','sha256':'frozen'} for m in model_list]}
        def verify(_):verified.append(True);return index
        def reader(path):
            self.assertTrue(verified)
            if Path(path).name=='references.json':return {'references':{f:TERMS for f in run.FAMILIES}}
            return deepcopy(by[Path(path).name])
        def generate(f,role,i):
            self.assertTrue(verified);self.assertEqual(role,'final');gen_calls.append((f,role,i))
            if (f,i)==('ks_unstable',1):raise RuntimeError('manufactured source failure')
            return field(f)
        def predict(terms,x,t,pt,pu):return np.tile(pu[-1],(len(t),1)),{}
        with tempfile.TemporaryDirectory() as tmp,ExitStack() as stack,redirect_stdout(io.StringIO()):
            out=Path(tmp);(out/'freeze_index.json').write_text('{}')
            stack.enter_context(patch.object(run,'verify_freeze',side_effect=verify))
            stack.enter_context(patch.object(run,'read',side_effect=reader))
            stack.enter_context(patch.object(run.generators,'generate',side_effect=generate))
            stack.enter_context(patch.object(run.prefix_forecast,'forecast_with_diagnostics',side_effect=predict))
            fitter=stack.enter_context(patch.object(run.engine,'fit_observations',side_effect=AssertionError('refit')))
            result=run.evaluate(out)
            self.assertEqual(len(gen_calls),9);self.assertEqual(len(result['records']),120)
            self.assertEqual(sum(len(r['forecasts']) for r in result['records']),360)
            self.assertEqual(len(result['final_reference_sha256']),8);fitter.assert_not_called()
            affected=[r for r in result['records'] if r['family']=='ks_unstable']
            self.assertEqual(len(affected),40);self.assertTrue(all(r['forecast_failures']==1 and r['mean_clean_relative_l2'] is None for r in affected))
            self.assertFalse(result['gate']['passed'])
            with self.assertRaises(RuntimeError):run.evaluate(out)

    def test_freeze_index_changed_during_scoring_blocks_result_publication(self):
        model_list=models();by={m['id']+'.json':m for m in model_list}
        index={**preparation(),'records':[{'id':m['id'],'file':'models/'+m['id']+'.json',
            'sha256':'frozen'} for m in model_list]}
        def reader(path):
            if Path(path).name=='references.json':return {'references':{f:TERMS for f in run.FAMILIES}}
            return deepcopy(by[Path(path).name])
        with tempfile.TemporaryDirectory() as tmp,ExitStack() as stack,redirect_stdout(io.StringIO()):
            out=Path(tmp);(out/'freeze_index.json').write_text('{}')
            def scores(tasks,workers):
                (out/'freeze_index.json').write_text('{"changed_during_scoring": true}')
                yield from gate_rows()
            stack.enter_context(patch.object(run,'verify_freeze',return_value=index))
            stack.enter_context(patch.object(run,'read',side_effect=reader))
            stack.enter_context(patch.object(run.generators,'generate',side_effect=lambda f,r,i:field(f)))
            stack.enter_context(patch.object(run,'_score_tasks',side_effect=scores))
            with self.assertRaisesRegex(ValueError,'changed during final scoring'):run.evaluate(out)
            self.assertFalse((out/'comparison.json').exists())
            self.assertEqual(len(list((out/'final_reference').rglob('*.npz'))),9)


class CoefficientGateTests(unittest.TestCase):
    def test_exact126_checks_and_mean_not_individual_target(self):
        gate=run.promotion_gate(gate_rows())
        self.assertTrue(gate['passed']);self.assertTrue(gate['coefficient_improvement_passed'])
        self.assertEqual(len(gate['checks']),126);self.assertEqual(gate['target_conditions_passed'],30)
        self.assertEqual(gate['candidate'],'noise_stable');self.assertFalse(gate['merged'])

    def test_prediction_boundary_missing_slots_and_false_aggregates(self):
        rows=gate_rows();target=next(r for r in rows if r['method']=='noise_stable')
        for value,expected in ((.05,True),(.050001,False),(None,False),(float('nan'),False),(-1,False)):
            set_mean(target,value);self.assertEqual(run.promotion_gate(rows)['prediction_target_passed'],expected)
        set_mean(target,.049);target['forecasts'].pop()
        self.assertFalse(run.promotion_gate(rows)['prediction_target_passed'])
        set_mean(target,.049);target['mean_clean_relative_l2']=.001
        self.assertFalse(run.promotion_gate(rows)['prediction_target_passed'])

    def test_ac_gain_requires_twenty_percent_and_not_equal_performance(self):
        rows=gate_rows()
        for r in rows:
            if r['family']=='allen_cahn' and r['method']=='noise_stable':
                r['recovery']=run.reference_metrics([{'name':'u_xx','coefficient':.107}],TERMS)
        result=run.promotion_gate(rows)
        self.assertTrue(result['prediction_target_passed']);self.assertFalse(result['coefficient_improvement_passed'])
        self.assertFalse(next(c['passed'] for c in result['checks'] if c['name'].startswith('allen_cahn:worst_true_term_relative')))

    def test_zero_baseline_cannot_claim_relative_coefficient_improvement(self):
        rows=gate_rows()
        for r in rows:
            if r['family']=='allen_cahn':r['recovery']=run.reference_metrics(TERMS,TERMS)
        self.assertFalse(run.promotion_gate(rows)['coefficient_improvement_passed'])

    def test_global_coefficient_tail_cannot_worsen_despite_ac_gain(self):
        rows=gate_rows()
        for r in rows:
            if r['family']=='ks_unstable' and r['method']=='noise_stable':
                r['recovery']=run.reference_metrics([{'name':'u_xx','coefficient':.109}],TERMS)
        result=run.promotion_gate(rows)
        self.assertFalse(next(c['passed'] for c in result['checks'] if c['name']=='all_families:maximum_true_term_error_not_increased'))
        self.assertFalse(result['passed'])

    def test_small_prediction_cannot_override_structure_or_accuracy(self):
        rows=gate_rows();target=next(r for r in rows if r['method']=='noise_stable')
        set_mean(target,1e-8)
        for terms in (TERMS+[{'name':'u','coefficient':1e-12}],[],[{'name':'u_xx','coefficient':.2}]):
            target['recovery']=run.reference_metrics(terms,TERMS)
            result=run.promotion_gate(rows)
            self.assertTrue(result['prediction_target_passed']);self.assertFalse(result['passed'])

    def test_no_final_ablation_selection_and_incomplete_ids_fail_closed(self):
        rows=gate_rows()
        for r in rows:
            if r['method']=='noise_stable':set_mean(r,.06)
            elif r['method']=='stable_selection':set_mean(r,.0001)
        result=run.promotion_gate(rows)
        self.assertEqual(result['candidate'],'noise_stable');self.assertFalse(result['passed'])
        for bad in (rows[:-1],[rows[0],rows[0],*rows[2:]]):self.assertFalse(run.promotion_gate(bad)['passed'])


if __name__=='__main__':unittest.main()
