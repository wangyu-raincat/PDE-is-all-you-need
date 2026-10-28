"""History/future isolation, equal-horizon scoring, and prefix state estimation."""
from copy import deepcopy
from contextlib import ExitStack,redirect_stdout
import io,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from prefix_study import data,forecast,engine,run,generators

TERMS=[{'name':'u_xx','coefficient':.1}]

def clean():
    x=np.linspace(0,2*np.pi,32,endpoint=False);t=np.linspace(0,.55,111)
    return {'x':x,'t':t,'u':.2+np.exp(-.1*t[:,None])*np.sin(x)[None,:],'boundary':'periodic'}

def candidate():
    return {'candidate_id':'compact-test','weak_variant':'compact','terms':deepcopy(TERMS),
        'support_size':1,'fit_success':True,'equation':'u_t=0.1u_xx'}

def model(method='prefix_median',sampling='full'):
    r={'id':run.model_id('allen_cahn',sampling,'clean',method),'family':'allen_cahn',
       'sampling':sampling,'condition':'clean','noise_level':0.,'noise_seed':0,'method':method,
       'source_quality_passed':True,'training_rms_for_corruption_only':1.,
       'training_trajectories':3,'development_trajectories':2,
       'selected':candidate(),'candidates':[candidate()],'failure_reason':None}
    r.update(run.predictor_metadata(method));return r

class PrefixForecastTests(unittest.TestCase):
    def test_noise_free_heat_history_gives_correct_cutoff_and_future(self):
        obs=data.corrupt(clean(),'allen_cahn','development',0,'full',0.,0,1.)
        pred,diag=forecast.forecast_with_diagnostics(TERMS,obs['x'],obs['t'],obs['prefix_t'],obs['prefix_u'])
        np.testing.assert_allclose(pred,obs['u'],atol=3e-10,rtol=3e-10)
        self.assertEqual(diag['prefix_frames'],11)
        self.assertAlmostEqual(diag['forecast_duration'],.5)
        self.assertFalse(diag['future_observations_used'])

    def test_equal_weights_reduce_canceling_historical_dc_errors(self):
        x=np.linspace(0,2*np.pi,32,endpoint=False);pt=np.array([0.,.1,.2]);t=np.linspace(.2,.7,32)
        true=np.ones(32);history=np.array([true+.03,true-.03,true])
        state,diag=forecast.estimate_state(TERMS,x,t,pt,history)
        np.testing.assert_allclose(state,true,atol=1e-13)
        self.assertTrue(diag['equal_frame_weights'])

    def test_rejects_history_after_cutoff_and_bad_shapes(self):
        obs=data.corrupt(clean(),'allen_cahn','training',0,'coarse',0.,0,1.)
        for pt,pu in [(obs['prefix_t']+.001,obs['prefix_u']),
                      (obs['prefix_t'],obs['prefix_u'][:-1]),
                      (obs['prefix_t'][::-1],obs['prefix_u']),
                      (obs['prefix_t'][:1],obs['prefix_u'][:1])]:
            with self.assertRaises(ValueError):forecast.forecast(TERMS,obs['x'],obs['t'],pt,pu)

    def test_observed_arrays_are_not_mutated(self):
        obs=data.corrupt(clean(),'allen_cahn','training',0,'coarse',.01,71,1.)
        before=deepcopy(obs)
        forecast.forecast(TERMS,obs['x'],obs['t'],obs['prefix_t'],obs['prefix_u'])
        for key in ('x','t','u','prefix_t','prefix_u'):np.testing.assert_array_equal(obs[key],before[key])

    def test_predictor_has_no_future_or_noise_or_truth_argument(self):
        obs=data.corrupt(clean(),'allen_cahn','training',0,'full',0.,0,1.)
        with self.assertRaises(TypeError):forecast.forecast(TERMS,obs['x'],obs['t'],obs['prefix_t'],obs['prefix_u'],future=obs['u'])
        with self.assertRaises(TypeError):forecast.forecast(TERMS,obs['x'],obs['t'],obs['prefix_t'],obs['prefix_u'],noise_sigma=.1)


class PrefixDataAndEngineTests(unittest.TestCase):
    def test_shared_endpoint_full_horizon_and_nested_sampling(self):
        full=data.corrupt(clean(),'allen_cahn','final',0,'full',.01,71,1.)
        coarse=data.corrupt(clean(),'allen_cahn','final',0,'coarse',.01,71,1.)
        self.assertEqual(full['u'].shape,(101,32));self.assertEqual(coarse['u'].shape,(51,16))
        self.assertEqual(len(full['prefix_t']),11);self.assertEqual(len(coarse['prefix_t']),6)
        for d in (full,coarse):
            np.testing.assert_array_equal(d['u'][0],d['prefix_u'][-1])
            self.assertEqual(d['t'][0],d['prefix_t'][-1]);self.assertAlmostEqual(d['t'][-1]-d['t'][0],.5)
        np.testing.assert_array_equal(coarse['u'],full['u'][::2,::2])
        np.testing.assert_array_equal(coarse['prefix_u'],full['prefix_u'][::2,::2])

    def test_training_scale_does_not_use_historical_prefix(self):
        base=clean();other=deepcopy(base);other['u'][:10]+=1000
        self.assertEqual(data.training_scale([base]),data.training_scale([other]))
        changed=deepcopy(base);changed['u'][10:]*=2
        self.assertAlmostEqual(data.training_scale([changed]),2*data.training_scale([base]))

    def test_old_or_shortened_observation_grid_rejected(self):
        old=clean();old['u']=old['u'][:101];old['t']=old['t'][:101]
        with self.assertRaises(ValueError):data.corrupt(old,'allen_cahn','training',0,'full',0.,0,1.)

    def test_generator_preserves_future_durations_and_new_roles(self):
        self.assertEqual(generators.ROLE_SEEDS,{'calibration':148000,'training':148100,'development':148200,'final':148300})
        p=generators.GENERATOR_PROTOCOL
        self.assertEqual(p['observed_nt'],111);self.assertEqual(p['prefix_intervals'],10)
        for f in run.FAMILIES:self.assertEqual(p['families'][f]['end_time'],run.TARGET_PROTOCOL['forecast_duration'][f])

    def test_future_development_changes_score_not_state_estimator_input(self):
        obs=data.corrupt(clean(),'allen_cahn','development',0,'full',0.,0,1.);changed=deepcopy(obs)
        changed['u'][1:]+=2
        seen=[]
        def fake(terms,x,t,pt,pu):
            seen.append((pt.copy(),pu.copy()));return obs['u'].copy(),{'prefix_frames':len(pt)}
        with patch.object(engine,'forecast_with_diagnostics',side_effect=fake):
            a=engine.select_candidates([candidate()],[obs]);b=engine.select_candidates([candidate()],[changed])
        np.testing.assert_array_equal(seen[0][0],seen[1][0]);np.testing.assert_array_equal(seen[0][1],seen[1][1])
        self.assertNotEqual(a['selected']['development']['rmse'],b['selected']['development']['rmse'])

    def test_conflicting_shared_endpoint_rejected(self):
        obs=data.corrupt(clean(),'allen_cahn','development',0,'full',0.,0,1.);obs['u'][0]+=1
        with self.assertRaises(ValueError):engine.select_candidates([candidate()],[obs])

    def test_fitting_never_receives_prefix_values(self):
        obs=data.corrupt(clean(),'allen_cahn','training',0,'full',0.,0,1.)
        def gen(inputs):
            self.assertEqual(set(inputs[0]),{'x','t','u','boundary'});np.testing.assert_array_equal(inputs[0]['u'],obs['u'])
            return [candidate()],[]
        scored={'selected':candidate(),'candidates':[dict(candidate(),eligible=True,development={'rmse':1.})],'failure_reason':None}
        with patch.object(engine,'generate_candidates',side_effect=gen),patch.object(engine.raw,'select_candidates',return_value=deepcopy(scored)),patch.object(engine.single,'select_candidates',return_value=deepcopy(scored)),patch.object(engine,'select_candidates',return_value=deepcopy(scored)):
            results=engine.fit_observations([obs],[obs])
        self.assertEqual(set(results),set(run.METHODS))
        self.assertEqual(results['prefix_median']['conditioning'],'observed_prefix')


class PrefixPipelineTests(unittest.TestCase):
    def test_prepare_generates_only_fifteen_training_development_fields(self):
        protocol=run.read(run.STUDY/'protocol.json');protocol['variant_menu_status']='frozen'
        calibration=run.read(run.DATA/'calibration.json');generated=[]
        def generate(family,role,index):
            self.assertIn(role,('training','development'));generated.append((family,role,index));return clean()
        real_read=run.read
        with tempfile.TemporaryDirectory() as tmp,ExitStack() as stack,redirect_stdout(io.StringIO()):
            stack.enter_context(patch.object(run,'DATA',Path(tmp)/'data'))
            stack.enter_context(patch.object(run,'read',side_effect=lambda p:protocol if Path(p).name=='protocol.json' else calibration))
            stack.enter_context(patch.object(run,'sources',return_value={}))
            stack.enter_context(patch.object(run,'input_hashes',return_value={}))
            stack.enter_context(patch.object(run.generators,'generate',side_effect=generate))
            receipt=run.prepare()
            self.assertEqual(len(generated),15);self.assertFalse(receipt['synthetic_final_generated'])
            self.assertEqual(len(list(run.DATA.rglob('*.npz'))),15)
            self.assertTrue(all('final' not in p.parts for p in run.DATA.rglob('*.npz')))

    def test_freeze_writes_all_120_before_global_index_without_final_access(self):
        prep={'sources':{},'input_hashes':{},'quality':{f:{'status':'passed'} for f in run.FAMILIES},
            'training_rms':{f:1. for f in run.FAMILIES},'all_source_quality_passed':True}
        writes=[];original_write=run.write;loaded=[]
        def loader(root,family,role,index):
            self.assertIn(role,('training','development'));loaded.append((family,role,index));return clean()
        def fit(train,dev):
            self.assertEqual(len(train),3);self.assertEqual(len(dev),2)
            for d in train+dev:self.assertEqual(set(d),{'x','t','u','boundary','prefix_t','prefix_u'})
            return {m:{'selected':candidate(),'candidates':[candidate()],'near_optimal_supports':[['u_xx']],
                'failure_reason':None,**run.predictor_metadata(m)} for m in run.METHODS}
        def writer(path,value):
            if Path(path).name=='freeze_index.json':self.assertEqual(sum(p.parent.name=='models' for p in writes),120)
            writes.append(Path(path));original_write(path,value)
        with tempfile.TemporaryDirectory() as tmp,ExitStack() as stack,redirect_stdout(io.StringIO()):
            root=Path(tmp);data_root=root/'data';data_root.mkdir();(data_root/'prepared.json').write_text('{}')
            stack.enter_context(patch.object(run,'DATA',data_root))
            stack.enter_context(patch.object(run,'verify_preparation',return_value=prep))
            stack.enter_context(patch.object(run,'sources',return_value={}))
            stack.enter_context(patch.object(run,'write',side_effect=writer))
            stack.enter_context(patch.object(run.data,'read_clean',side_effect=loader))
            fitter=stack.enter_context(patch.object(run.five_engine,'fit_observations',side_effect=fit))
            gen=stack.enter_context(patch.object(run.generators,'generate',side_effect=AssertionError('final access')))
            index=run.freeze(root/'out')
            self.assertEqual(fitter.call_count,30);self.assertEqual(len(loaded),15)
            self.assertTrue(run.complete(index['records']));self.assertEqual(writes[-1].name,'freeze_index.json')
            gen.assert_not_called()

    @staticmethod
    def gate_rows():
        rows=[]
        for family in run.FAMILIES:
            for sampling in run.SAMPLINGS:
                for name,level,seed in run.NOISES:
                    for method in run.METHODS:
                        m=model(method,sampling);m.update(family=family,condition=name,noise_level=level,noise_seed=seed,
                            id=run.model_id(family,sampling,name,method),recovery=run.reference_metrics(TERMS,TERMS),
                            mean_clean_relative_l2=.049,forecast_failures=0,
                            forecasts=[{'status':'evaluated','clean':{'relative_l2':v}} for v in (.02,.027,.1)])
                        rows.append(m)
        return rows

    def test_five_percent_is_mean_not_individual_and_boundary_is_inclusive(self):
        rows=self.gate_rows();gate=run.promotion_gate(rows)
        self.assertTrue(gate['passed']);self.assertEqual(len(gate['checks']),157)
        target=next(r for r in rows if r['method']=='prefix_median')
        for value,expected in ((.05,True),(.050001,False),(None,False),(float('nan'),False)):
            target['mean_clean_relative_l2']=value
            self.assertEqual(run.promotion_gate(rows)['prediction_target_passed'],expected)

    def test_small_prediction_error_cannot_override_equation_guard(self):
        rows=self.gate_rows();target=next(r for r in rows if r['method']=='prefix_median')
        target['recovery']=run.reference_metrics(TERMS+[{'name':'u','coefficient':1e-12}],TERMS)
        gate=run.promotion_gate(rows)
        self.assertTrue(gate['prediction_target_passed']);self.assertFalse(gate['passed'])
        target['recovery']=run.reference_metrics([{'name':'u_xx','coefficient':.2}],TERMS)
        self.assertFalse(run.promotion_gate(rows)['passed'])

    def test_horizon_shortening_fails_before_forecast(self):
        short=clean();short['t']*=.5
        with patch.object(run.prefix_forecast,'forecast_with_diagnostics',side_effect=AssertionError('short horizon forecast')):
            result=run._score_record((model(),'sha',[short]*3,[None]*3,TERMS))
        self.assertEqual(result['forecast_failures'],3)
        self.assertTrue(all('horizon' in m['reason'] for m in result['forecasts']))

    def test_final_future_mutation_cannot_reach_prefix_forecaster(self):
        base=clean();changed=deepcopy(base);changed['u'][11:]+=3;seen=[]
        predicted=base['u'][10:]
        def fake(terms,x,t,pt,pu):
            seen.append((t.copy(),pt.copy(),pu.copy()));return predicted.copy(),{'future_observations_used':False}
        with patch.object(run.prefix_forecast,'forecast_with_diagnostics',side_effect=fake):
            a=run._score_record((model(),'sha',[base]*3,[None]*3,TERMS))
            b=run._score_record((model(),'sha',[changed]*3,[None]*3,TERMS))
        self.assertEqual(a['forecast_failures'],0);self.assertEqual(b['forecast_failures'],0)
        for x,y in zip(seen[:3],seen[3:]):
            for xx,yy in zip(x,y):np.testing.assert_array_equal(xx,yy)
        self.assertNotEqual(a['mean_clean_relative_l2'],b['mean_clean_relative_l2'])
        self.assertTrue(all(m['forecast_duration']==.5 and m['prefix_frames_used']==11 for m in a['forecasts']))

    def test_serial_parallel_scores_identical(self):
        tasks=[(model(m),'sha',[clean()]*3,[None]*3,TERMS) for m in run.METHODS]
        self.assertEqual(list(run._score_tasks(tasks,1)),list(run._score_tasks(tasks,2)))

    def test_stage_metadata_cannot_hide_prefix_conditioning(self):
        m=model();run.validate_configuration(m);m['conditioning']='last_frame_only'
        with self.assertRaises(ValueError):run.validate_configuration(m)

    def test_invalid_global_freeze_blocks_final_generation(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(run,'verify_freeze',side_effect=ValueError('bad global freeze')),patch.object(run.generators,'generate',side_effect=AssertionError('final leaked')):
            with self.assertRaises(ValueError):run.evaluate(Path(tmp))


if __name__=='__main__':unittest.main()
