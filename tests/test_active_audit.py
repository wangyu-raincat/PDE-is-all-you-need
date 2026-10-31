"""Manufactured audit fixtures only; never run or generate formal tasks."""
import ast
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1]/'scripts/audit_active_study.py'
SPEC = importlib.util.spec_from_file_location('independent_active_audit',SCRIPT)
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def model_terms(**values):
    return [{'name':name,'coefficient':value} for name,value in values.items()]


def lawful_action(index):
    phase = index*.41
    return {'id':f'manufactured-{index}',
            'a':[float(.4*np.sqrt(2)*np.cos(phase))]+[0.]*5,
            'b':[float(.4*np.sqrt(2)*np.sin(phase))]+[0.]*5}


def observation(index=0):
    x=np.arange(64)*2*np.pi/64;t=np.linspace(0,1.1,111)
    u=.4*np.sin(x[None,:]+.1*index)*np.exp(-.2*t[:,None])
    return {'x':x,'t':t[10:],'u':u[10:],'prefix_t':t[:11],'prefix_u':u[:11]}


def receipt(obs,action):
    full=np.concatenate((obs['prefix_u'][:-1],obs['u']))
    return {'experiments':1,'measurement_values':audit.UNIT_COST,'history_endpoint_counted_once':True,
            'action_sha256':audit.action_digest(action),'observed_field_sha256':hashlib.sha256(full.tobytes()).hexdigest()}


def ledger_fixture():
    initial=[observation(i) for i in range(5)]
    action_pool=[lawful_action(i) for i in range(4)]
    condition='task-00--clean';identity=condition+'--random'
    task={'id':condition,'spec_index':0,'noise_label':'clean','noise_level':0.,'failures':[],
          'pool':action_pool,'observations':{'training':[{'file':f'initial-{i}'} for i in range(3)],
                                           'development':[{'file':f'initial-{i}'} for i in range(3,5)]}}
    fields={f'initial-{i}':value for i,value in enumerate(initial)}
    train=[audit.array_digest(o) for o in initial[:3]];dev=[audit.array_digest(o) for o in initial[3:]]
    queries=[];calls=[];checkpoints=[];returned=0
    for i in range(5):
        checkpoint={'budget':5+i,'charged_measurement_scalars':(5+i)*audit.UNIT_COST,
                    'available_measurement_scalars':5*audit.UNIT_COST+returned,'training_hashes':train.copy(),
                    'development_hashes':dev.copy(),'training_trajectories':len(train),'development_trajectories':2,
                    'final_used_for_selection':False,'model':{},'next_proposal':None}
        checkpoints.append(checkpoint)
        if i==4:continue
        chosen=action_pool[i];checkpoint['next_proposal']={'selected_action':chosen}
        q={'query_index':i,'action':chosen,'action_sha256':audit.action_digest(chosen),
           'charged_experiments':1,'charged_measurement_scalars':audit.UNIT_COST,
           'returned_measurement_scalars':0,'observation_sha256':None}
        call={'query_index':i,'action':chosen,'started':True}
        if i==1:
            q.update(status='failed',failure_reason='manufactured measurement failure')
            call.update(status='failed',reason='manufactured measurement failure')
        else:
            obs=observation(i+5);path=f'queries/{identity}/{i}.npz';fields[path]=obs
            h=audit.array_digest(obs)
            q.update(status='observed',returned_measurement_scalars=audit.UNIT_COST,observation_sha256=h,
                     provider_receipt=receipt(obs,chosen))
            call.update(status='observed',file=path)
            train.append(h);returned+=audit.UNIT_COST
        queries.append(q);calls.append(call)
    ep={'id':identity,'task_id':condition,'spec_index':0,'policy':'random','noise_label':'clean','noise_level':0.,
        'final_used':False,'reference_coefficients_used':False,'retrospective_development':False,
        'total_experiment_attempts':9,'charged_measurement_scalars':9*audit.UNIT_COST,'seed':858000,
        'maximum_additions':4,'initial_measurement_scalars':5*audit.UNIT_COST,
        'per_experiment_measurement_scalars':audit.UNIT_COST,'queries':queries,'provider_calls':calls,
        'checkpoints':checkpoints,'status':'completed_budget',
        'budget_slots':[{'budget':b,'checkpoint_index':b-5,'status':'frozen_checkpoint'} for b in audit.BUDGETS]}
    return ep,task,lambda kind,record:fields[record['file']]


def forecast_row():
    diagnostic={'future_observations_used':False,'equal_frame_weights':True,'independent_estimates_claimed':False,
                'prefix_frames':11,'prefix_duration':.1,'forecast_duration':1.}
    forecasts=[]
    for j,value in enumerate((.01,.02,.03)):
        forecasts.append({'index':j,'status':'evaluated','forecast_duration':1.,'prefix_frames_used':11,
                          'conditioning':'observed_prefix','initial_state_diagnostics':diagnostic,
                          'clean':{'rmse':value,'mae':value,'max_absolute_error':value,'relative_l2':value,
                                   'excluded_initial_frame':True,'scored_frames':100,'space_points':64}})
    return {'forecasts':forecasts,'forecast_failures':0,'mean_clean_relative_l2':.02}


class ActiveIndependentAuditTests(unittest.TestCase):
    def test_complete_slot_identity_counts_and_duplicates(self):
        self.assertEqual(len(audit.expected_conditions()),24)
        self.assertEqual(len(audit.expected_episodes()),96)
        self.assertEqual(len(audit.expected_rows()),480)
        rows=[{'id':identity} for identity in sorted(audit.expected_rows())]
        audit.check_unique(rows,audit.expected_rows(),'manufactured')
        rows[-1]=rows[0]
        with self.assertRaises(audit.AuditError):audit.check_unique(rows,audit.expected_rows(),'duplicate')

    def test_coefficient_and_structure_metrics_are_independently_rederived(self):
        result=audit.recovery(model_terms(u=-.01,u_xx=.08),model_terms(u_xx=.1))
        self.assertFalse(result['exact_support']);self.assertEqual(result['spurious_terms'],['u'])
        self.assertEqual(result['missing_terms'],[])
        self.assertAlmostEqual(result['coefficient_relative_l2'],np.sqrt(.02**2+.01**2)/.1)
        self.assertAlmostEqual(result['coefficient_comparison'][1]['relative_error'],.2)
        empty=audit.recovery([],model_terms(u_xx=.1))
        self.assertEqual(empty['missing_terms'],['u_xx']);self.assertEqual(empty['coefficient_relative_l2'],1.)
        with self.assertRaises(audit.AuditError):audit.coefficient_map(model_terms(u=.1)+model_terms(u=.2))

    def test_bootstrap_failures_preserve_requested_denominator(self):
        records=[{'replicate':i,'whole_trajectory_resampling':True,'structure_selection_repeated':True,
                  'status':'selected','selected_support':['u_xx']} for i in range(4)]
        uncertainty={'requested_replicates':4,'records':records,'completed_replicates':4,'failed_replicates':0,
                     'raw_counts':{n:4 if n=='u_xx' else 0 for n in audit.TERMS},
                     'term_inclusion_frequencies':{n:1. if n=='u_xx' else 0. for n in audit.TERMS},
                     'calibrated':False,'probability_claimed':False,'reference_used':False,'final_used':False}
        model={'selected':{},'uncertainty':uncertainty}
        self.assertEqual(audit.inclusion_frequencies(model)['u_xx'],1.)
        records[3].update(status='failed',selected_support=None)
        uncertainty.update(completed_replicates=3,failed_replicates=1,
                           raw_counts={n:3 if n=='u_xx' else 0 for n in audit.TERMS},
                           term_inclusion_frequencies={n:None for n in audit.TERMS})
        self.assertTrue(all(v is None for v in audit.inclusion_frequencies(model).values()))
        uncertainty['term_inclusion_frequencies']['u_xx']=1.
        with self.assertRaises(audit.AuditError):audit.inclusion_frequencies(model)

    def test_support_coverage_and_positive_negative_brier(self):
        model={'committee':[{'terms':model_terms(u_xx=.08)},{'terms':model_terms(u=-.1)}],
               'candidate_supports':[['u'],['u_xx']]}
        supports=audit.support_set(model)
        frequencies={n:0. for n in audit.TERMS};frequencies.update(u_xx=.75,u=.2)
        metric=audit.uncertainty_metrics(supports,frequencies,model_terms(u_xx=.1))
        self.assertTrue(metric['set_contains_true_support']);self.assertFalse(metric['singleton_wrong'])
        self.assertAlmostEqual(metric['term_brier_positive'],.0625)
        self.assertAlmostEqual(metric['term_brier_negative'],.04/8)
        frequencies['u_xx']=None
        metric=audit.uncertainty_metrics(supports,frequencies,model_terms(u_xx=.1))
        self.assertIsNone(metric['term_brier_positive']);self.assertIsNone(metric['term_brier_negative'])
        model['candidate_supports']=[['u']]
        with self.assertRaises(audit.AuditError):audit.support_set(model)

    def test_failed_forecasts_cannot_be_averaged_away(self):
        row=forecast_row();refs=[{'status':'generated'}]*3
        self.assertAlmostEqual(audit.forecast_summary(row,refs),.02)
        row['forecasts'][1]={'index':1,'status':'failed','reason':'manufactured solver failure'}
        row.update(forecast_failures=1,mean_clean_relative_l2=None)
        self.assertIsNone(audit.forecast_summary(row,refs))
        row['mean_clean_relative_l2']=.02
        with self.assertRaises(audit.AuditError):audit.forecast_summary(row,refs)
        row['mean_clean_relative_l2']=None;row['forecasts'].pop()
        with self.assertRaises(audit.AuditError):audit.forecast_summary(row,refs)

    def test_paid_lineage_charges_failed_query_and_excludes_it_from_training(self):
        ep,task,loader=ledger_fixture()
        result=audit.verify_episode(ep,task,loader)
        self.assertEqual(result,{'queries':4,'failed_queries':1,'missing_checkpoints':0})
        self.assertEqual(ep['checkpoints'][-1]['charged_measurement_scalars'],63936)
        self.assertEqual(ep['checkpoints'][-1]['available_measurement_scalars'],56832)
        for change in ('refund','unselected','future_training','missing_slot'):
            bad=deepcopy(ep)
            if change=='refund':bad['queries'][1]['charged_measurement_scalars']=0
            elif change=='unselected':bad['provider_calls'][0]['action']=lawful_action(10)
            elif change=='future_training':bad['checkpoints'][0]['training_hashes'].append('future')
            else:bad['budget_slots'].pop()
            with self.subTest(change=change),self.assertRaises(audit.AuditError):audit.verify_episode(bad,task,loader)

    def test_initial_source_failure_keeps_all_five_missing_budget_slots(self):
        ep,task,loader=ledger_fixture()
        task['failures']=[{'role':'training','index':1,'reason':'manufactured','charged_experiments':1}]
        ep.update(status='initial_source_failure',initial_failures=task['failures'],checkpoints=[],queries=[],provider_calls=[],
                  total_experiment_attempts=5,charged_measurement_scalars=35520,
                  budget_slots=[{'budget':b,'checkpoint_index':None,'status':'missing_checkpoint_failure'} for b in audit.BUDGETS])
        result=audit.verify_episode(ep,task,loader)
        self.assertEqual(result['missing_checkpoints'],5)

    def test_prediction_replay_sample_is_four_worst_rows_and_at_most_twelve_forecasts(self):
        rows=[]
        for policy in audit.POLICIES:
            for i in range(24):rows.append({'id':f'{policy}-{i:02d}','policy':policy,'budget':9,'mean_clean_relative_l2':i/100})
        chosen=audit.worst_final_budget_rows(rows)
        self.assertEqual(len(chosen)*3,12)
        self.assertTrue(all(r['id'].endswith('23') for r in chosen))
        rows[0]['mean_clean_relative_l2']=None
        self.assertEqual(audit.worst_final_budget_rows(rows)[0]['id'],rows[0]['id'])

    def test_independent_forecast_metrics_exclude_only_shared_start(self):
        truth=np.ones((101,64));prediction=truth.copy();prediction[0]=100;prediction[1:]+=.02
        result=audit.independent_forecast_metrics(prediction,truth)
        self.assertAlmostEqual(result['relative_l2'],.02)
        self.assertEqual(result['scored_frames'],100)

    def test_observation_geometry_shared_endpoint_and_path_escape(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'example.npz';obs=observation()
            np.savez_compressed(path,**obs)
            loaded=audit.load_observation(path)
            self.assertEqual(audit.array_digest(loaded),audit.array_digest(obs))
            obs['prefix_u']=obs['prefix_u'].copy();obs['prefix_u'][-1,0]+=.01
            np.savez_compressed(path,**obs)
            with self.assertRaises(audit.AuditError):audit.load_observation(path)
            with self.assertRaises(audit.AuditError):audit.safe_path(Path(directory),'../escape')

    def test_final_release_receipt_distinguishes_reproduction(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);output=root/'result';output.mkdir()
            data=root/'active_study/data';data.mkdir(parents=True)
            prepared={'prepared_at_utc':'2026-01-01T00:00:00+00:00'}
            frozen={'sources':{},'frozen_at_utc':'2026-01-01T00:01:00+00:00'}
            (data/'prepared.json').write_text(json.dumps(prepared));(output/'freeze_index.json').write_text(json.dumps(frozen))
            design={'indices':list(range(12)),'noise_conditions':[['clean',0.],['noise1',.01]],'provider_role':'formal',
                    'initial_action_seed_base':857000,'pool_seed_base':856000,'trajectories_per_task':3,
                    'final_action_seed_base':859000,'final_action_pool_slice':[16,19]}
            design_hash=hashlib.sha256(json.dumps(design,sort_keys=True,separators=(',',':')).encode()).hexdigest()
            release={'first_prepared_sha256':audit.digest(data/'prepared.json'),'sources':{},
                     'first_freeze_index_sha256':audit.digest(output/'freeze_index.json'),'first_output':str(output),
                     'released_at_utc':'2026-01-01T00:02:00+00:00','marker_created_before_final_generation':True,
                     'interrupted_first_release_remains_consumed':True,'task_design':design,'task_design_sha256':design_hash}
            (data/'final_release.json').write_text(json.dumps(release))
            binding={k:release[k] for k in ('released_at_utc','first_output','first_freeze_index_sha256')}
            binding.update(file='active_study/data/final_release.json',sha256=audit.digest(data/'final_release.json'),is_first_final_release=True,
                           prepared_sha256=audit.digest(data/'prepared.json'),first_prepared_sha256=release['first_prepared_sha256'],
                           task_design_sha256=design_hash)
            report={'is_first_final_release':True,'final_release':binding,'evaluation_label':'first frozen same-family synthetic validation'}
            self.assertTrue(audit.check_release(root,output,prepared,frozen,report)['is_first_final_release'])
            report['is_first_final_release']=False;binding['is_first_final_release']=False
            with self.assertRaises(audit.AuditError):audit.check_release(root,output,prepared,frozen,report)
            report['evaluation_label']='reproduction on previously released final tasks; retrospective, not unseen validation'
            self.assertFalse(audit.check_release(root,output,prepared,frozen,report)['is_first_final_release'])
            # A fresh preparation after first release retains task identity but
            # receives a new timestamp and manifest hash; it is reproduction.
            prepared['prepared_at_utc']='2026-01-02T00:00:00+00:00'
            frozen['frozen_at_utc']='2026-01-02T00:01:00+00:00'
            (data/'prepared.json').write_text(json.dumps(prepared));(output/'freeze_index.json').write_text(json.dumps(frozen))
            binding['prepared_sha256']=audit.digest(data/'prepared.json')
            self.assertNotEqual(binding['prepared_sha256'],binding['first_prepared_sha256'])
            self.assertFalse(audit.check_release(root,output,prepared,frozen,report)['is_first_final_release'])

    def test_missing_completed_result_never_reads_arrays_or_changes_inputs(self):
        with tempfile.TemporaryDirectory() as directory,patch.object(audit.np,'load',side_effect=AssertionError('No arrays before completed result')):
            root=Path(directory);result=audit.audit(root,root/'absent')
            self.assertEqual(result['status'],'failed');self.assertEqual(result['prediction_calls'],0)
            output=root/'partial';output.mkdir();path=output/'comparison.json';path.write_text('{"records": []}')
            before=path.read_bytes();result=audit.audit(root,output)
            self.assertEqual(result['status'],'failed');self.assertEqual(path.read_bytes(),before)

    def test_complete_manufactured_failure_ledger_passes_without_generating_fields(self):
        """A complete honest failure report is auditable, never a success claim."""
        with tempfile.TemporaryDirectory() as directory,patch.object(audit.np,'load',side_effect=AssertionError('All manufactured sources failed')):
            root=Path(directory);output=root/'result';output.mkdir();data=root/'active_study/data';data.mkdir(parents=True)
            def write(path,value):
                path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))
            for relative in audit.SOURCE_FILES:
                path=root/relative;path.parent.mkdir(parents=True,exist_ok=True);path.write_text('"Manufactured audit-only source fixture."\n')
            (root/'active_study/controller.py').write_text('def run_episode():\n    action = acquisition.select_action()\n    ledger.append(action)\n    query(action)\n')
            ranges={'decay':{'u':[-.4,-.15]},'diffusion':{'u_xx':[.05,.15]},
                    'hyperdiffusion':{'u_xxxx':[-.005,-.001]},
                    'advection_diffusion':{'u_x':[-.7,-.3],'u_xx':[.05,.15]},
                    'reaction_diffusion':{'u':[-.4,-.15],'u_xx':[.05,.15]},
                    'burgers':{'u_xx':[.04,.08],'u*u_x':[-1.2,-.6]}}
            provider={'coefficient_ranges':ranges,'role_seed_bases':{'formal':853000},
                      'calibration_relative_l2_limit':1e-5,'calibration_max_absolute_limit':1e-4}
            protocol={'status':'frozen','provider':provider,'actions':{'manufactured_fixture':True},
                      'study':{'manufactured_fixture':True},'calibration_sources':{}}
            write(root/'active_study/protocol.json',protocol)
            calibration={'all_passed':True,'formal_data_generated':False,'protocol':provider,
                         'action_protocol':protocol['actions'],'records':[{'family':f,'action_id':a,'status':'passed',
                         'relative_l2':0.,'max_absolute_error':0.} for f in audit.FAMILIES for a in ('fixed-rich','single-6-0')]}
            write(data/'calibration.json',calibration)
            sources={r:audit.digest(root/r) for r in audit.SOURCE_FILES}
            tasks=[];pool=[lawful_action(i) for i in range(32)]
            initial={'training':[lawful_action(i) for i in range(3)],'development':[lawful_action(i) for i in range(3,5)]}
            for i in range(12):
                family=audit.FAMILIES[i%6]
                spec={'role':'formal','index':i,'family':family,'parameter_seed':[853000,i],
                      'coefficients':{n:sum(limits)/2 for n,limits in ranges[family].items()}}
                for label,level in audit.NOISES.items():
                    failures=[]
                    for role,actions in initial.items():
                        for j,action in enumerate(actions):
                            h=audit.action_digest(action);seed=int(hashlib.sha256(f'formal:{i}:{h}'.encode()).hexdigest()[:8],16)%2**31
                            failures.append({'role':role,'index':j,'reason':'Manufactured source failure',
                                             'charged_experiments':1,'action_sha256':h,'noise_seed':seed})
                    tasks.append({'id':audit.task_id(i,label),'spec_index':i,'spec':spec,'noise_label':label,'noise_level':level,
                                  'pool':pool,'initial_actions':initial,'observations':{'training':[],'development':[]},'failures':failures})
            prepared={'tasks':tasks,'sources':sources,'protocol':protocol,'final_generated':False,
                      'calibration_sha256':audit.digest(data/'calibration.json'),'prepared_at_utc':'2026-01-01T00:00:00+00:00'}
            write(data/'prepared.json',prepared)
            receipts=[];rows=[]
            for task in tasks:
                refs=[{'name':n,'coefficient':v} for n,v in task['spec']['coefficients'].items()]
                for policy in audit.POLICIES:
                    identity=task['id']+'--'+policy;relative=f'episodes/{identity}.json'
                    episode={'id':identity,'task_id':task['id'],'spec_index':task['spec_index'],'policy':policy,
                             'noise_label':task['noise_label'],'noise_level':task['noise_level'],'source_hashes':sources,
                             'final_used':False,'reference_coefficients_used':False,'retrospective_development':False,
                             'total_experiment_attempts':5,'charged_measurement_scalars':35520,'status':'initial_source_failure',
                             'initial_failures':task['failures'],'queries':[],'provider_calls':[],'checkpoints':[],
                             'budget_slots':[{'budget':b,'checkpoint_index':None,'status':'missing_checkpoint_failure'} for b in audit.BUDGETS]}
                    write(output/relative,episode);h=audit.digest(output/relative)
                    receipts.append({'id':identity,'task_id':task['id'],'file':relative,'sha256':h})
                    for budget in audit.BUDGETS:
                        rows.append({'id':identity+f'--budget-{budget}','task_id':task['id'],'policy':policy,'budget':budget,
                                     'spec_index':task['spec_index'],'noise_label':task['noise_label'],'noise_level':task['noise_level'],
                                     'family':task['spec']['family'],'terms':[],'equation':None,'checkpoint_index':None,
                                     'model_file':relative,'frozen_episode_sha256':h,'status':'missing_budget_checkpoint',
                                     'charged_measurement_scalars':None,'available_measurement_scalars':None,'query_failure_count_so_far':0,
                                     'recovery':audit.recovery([],refs),**audit.uncertainty_metrics([],{},refs),
                                     'forecasts':[{'index':j,'status':'failed','reason':'Manufactured missing frozen model'} for j in range(3)],
                                     'forecast_failures':3,'mean_clean_relative_l2':None,'offline_success':False})
            frozen={'records':receipts,'sources':sources,'prepared_sha256':audit.digest(data/'prepared.json'),
                    'all_episodes_frozen_before_final':True,'final_generated':False,'expected_conditions':24,
                    'expected_budgets':list(audit.BUDGETS),'expected_budget_records':480,
                    'missing_budget_checkpoints_are_failed_slots':True,'frozen_at_utc':'2026-01-01T00:01:00+00:00'}
            write(output/'freeze_index.json',frozen)
            design={'indices':list(range(12)),'noise_conditions':[['clean',0.],['noise1',.01]],'provider_role':'formal',
                    'initial_action_seed_base':857000,'pool_seed_base':856000,'trajectories_per_task':3,
                    'final_action_seed_base':859000,'final_action_pool_slice':[16,19]}
            dh=hashlib.sha256(json.dumps(design,sort_keys=True,separators=(',',':')).encode()).hexdigest()
            release={'sources':sources,'task_design':design,'task_design_sha256':dh,
                     'first_prepared_sha256':audit.digest(data/'prepared.json'),
                     'first_freeze_index_sha256':audit.digest(output/'freeze_index.json'),'first_output':str(output),
                     'released_at_utc':'2026-01-01T00:02:00+00:00','marker_created_before_final_generation':True,
                     'interrupted_first_release_remains_consumed':True}
            write(data/'final_release.json',release)
            binding={k:release[k] for k in ('first_prepared_sha256','task_design_sha256','first_freeze_index_sha256','first_output','released_at_utc')}
            binding.update(file='active_study/data/final_release.json',sha256=audit.digest(data/'final_release.json'),
                           prepared_sha256=audit.digest(data/'prepared.json'),is_first_final_release=True)
            comparison={'records':rows,'expected_conditions':24,'expected_budgets':list(audit.BUDGETS),'expected_records':480,
                        'independent_parameter_tasks':12,'policies':list(audit.POLICIES),'all_declared_records_present':True,
                        'final_used_for_tuning':False,'all_data_synthetic':True,'frequency_calibrated':False,
                        'reference_used_for_online_stopping':False,'final_validation_experiments_per_condition_policy':3,
                        'final_validation_measurement_scalars_per_condition_policy':3*audit.UNIT_COST,
                        'freeze_index_sha256':audit.digest(output/'freeze_index.json'),'policy':protocol['study'],
                        'final_reference_records':{str(i):[{'status':'failed','reason':'Manufactured source failure','action':lawful_action(j)}
                                                          for j in range(3)] for i in range(12)},
                        'is_first_final_release':True,'final_release':binding,'evaluation_label':'first frozen same-family synthetic validation'}
            write(output/'comparison.json',comparison);before=(output/'comparison.json').read_bytes()
            result=audit.audit(root,output)
            failures=[c for c in result['checks'] if c['status']=='failed']
            self.assertEqual(result['status'],'passed',failures)
            self.assertEqual(result['summary']['independently_rederived_rows'],480)
            self.assertEqual(result['summary']['all_forecast_slots'],1440)
            self.assertEqual(result['prediction_calls'],0)
            self.assertEqual(len(result['forecast_rechecks']),12)
            self.assertEqual(result['summary']['final_sources']['failed'],36)
            self.assertEqual((output/'comparison.json').read_bytes(),before)

    def test_auditor_has_no_data_generator_or_runner_import(self):
        tree=ast.parse(SCRIPT.read_text());names=[]
        for node in ast.walk(tree):
            if isinstance(node,ast.Import):names += [a.name for a in node.names]
            elif isinstance(node,ast.ImportFrom):names.append(node.module or '')
        self.assertFalse(any(n.startswith(('active_study','coefficient_study.generators')) for n in names))
        self.assertIn('prefix_study.forecast',names)
        order=audit.check_code_order(SCRIPT.parents[1])
        self.assertLess(order['selection_line'],order['charge_line']);self.assertLess(order['charge_line'],order['query_line'])


if __name__=='__main__':
    unittest.main()
