"""Prefix-conditioned five-percent prediction with an unchanged future horizon."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from threadpoolctl import threadpool_limits
from pde_discovery import generic_engine as predictor
from pde_discovery.generic_experiment import forecast_metrics, reference_metrics
from . import data, generators
from weak_study import engine as weak_engine
from resolved_study import forecast as resolved_forecast
from . import engine as five_engine
from five_study import forecast as denoised_forecast
from five_study import initial
from . import forecast as prefix_forecast

ROOT=Path(__file__).resolve().parents[1]
STUDY=ROOT/'prefix_study'
DATA=STUDY/'data'
OUTPUT=STUDY/'results'/'first_run'
FAMILIES=('allen_cahn','low_viscosity_burgers','ks_unstable')
METHODS=('weak_resolved','denoised_median','prefix_weak','prefix_median')
SAMPLINGS=('full','coarse')
NOISES=(('clean',0.,0),('noise0p1_s71',.001,71),('noise0p1_s103',.001,103),
        ('noise1_s71',.01,71),('noise1_s103',.01,103))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')


def sources():
    files=list(STUDY.glob('*.py'))+[STUDY/'protocol.json',STUDY/'references.json',
        ROOT/'challenge_study/data.py',ROOT/'challenge_study/generators.py',ROOT/'l1_study/engine.py',
        ROOT/'weak_study/engine.py',ROOT/'weak_study/data.py',ROOT/'weak_study/generators.py',
        ROOT/'resolved_study/forecast.py',ROOT/'resolved_study/selection.py',ROOT/'fusion_study/engine.py',
        ROOT/'five_study/engine.py',ROOT/'five_study/initial.py',ROOT/'five_study/forecast.py',
        ROOT/'pde_discovery/generic_engine.py',ROOT/'pde_discovery/generic_experiment.py']
    return {str(p.relative_to(ROOT)):sha(p) for p in sorted(files)}


def input_hashes():
    files=list((DATA/'clean').glob('*/*/*.npz'))+list((DATA/'raw').glob('*.mat'))
    files+=[DATA/'calibration.json']
    return {str(p.relative_to(ROOT)):sha(p) for p in sorted(files)}


def model_id(family,sampling,condition,method):
    return '--'.join((family,sampling,condition,method))


def expected_ids():
    return {model_id(f,s,n,m) for f in FAMILIES for s in SAMPLINGS for n,_,_ in NOISES for m in METHODS}


def complete(records):
    ids=[r.get('id') for r in records]
    return len(ids)==len(set(ids))==120 and set(ids)==expected_ids()


def role_count(family,role):
    return data.ROLE_COUNTS[role]


TARGET_PROTOCOL = {
    'candidate': 'prefix_median',
    'metric': 'maximum across condition means of final-trajectory relative L2; exclude initial frame',
    'relative_l2_limit': .05,
    'condition_count': 30,
    'trajectories_per_condition': 3,
    'individual_trajectory_limit': None,
    'all_conditions_required': True,
    'failure_action': 'target not achieved; retain failed condition',
    'conditioning': 'observed prefix before forecast start; 11 full or 6 coarse frames',
    'forecast_duration': {'allen_cahn':.5,'low_viscosity_burgers':1.5,'ks_unstable':20.},
    'history_fraction_of_forecast_horizon': .1,
}


def predictor_metadata(method):
    return five_engine.stage_metadata(method)


def validate_configuration(record,quality=None,scales=None):
    f,s,n,m=(record[k] for k in ('family','sampling','condition','method'))
    conditions={name:(level,seed) for name,level,seed in NOISES}
    if f not in FAMILIES or s not in SAMPLINGS or n not in conditions or m not in METHODS:
        raise ValueError('Undeclared configuration.')
    if any(record.get(k)!=v for k,v in predictor_metadata(m).items()):
        raise ValueError('Stage-specific structural/coefficient/final predictor metadata differs from declared algorithm.')
    if (isinstance(record['noise_level'],bool) or type(record['noise_seed']) is not int
            or record['id']!=model_id(f,s,n,m) or (record['noise_level'],record['noise_seed'])!=conditions[n]):
        raise ValueError('Noise metadata does not match the declared condition.')
    passed=record['source_quality_passed']
    if type(passed) is not bool or (quality is not None and passed!=(quality[f]['status']=='passed')):
        raise ValueError('Source quality metadata differs from preparation.')
    for role in ('training','development'):
        count=record[role+'_trajectories']
        if type(count) is not int or count!=(role_count(f,role) if passed else 0):
            raise ValueError('Role count does not match family and source status.')
    scale=record['training_rms_for_corruption_only']
    if (passed and (isinstance(scale,bool) or not isinstance(scale,(float,int)) or not np.isfinite(scale) or scale<=0)) or (not passed and scale is not None):
        raise ValueError('Invalid training noise scale.')
    if scales is not None and scale!=scales[f]:
        raise ValueError('Model noise scale differs from the training-frozen scale.')
    return True


def validate_design_and_calibration(protocol, calibration):
    """Fail closed before generating any formal role on stale/pending design."""
    canonical=lambda value:json.loads(json.dumps(value,allow_nan=False))
    if (protocol.get('variant_menu_status')!='frozen'
            or protocol.get('weak_variants')!=canonical(weak_engine.VARIANTS)
            or protocol.get('weak_engine_protocol')!=canonical(weak_engine.PROTOCOL)
            or protocol.get('forecast_protocol')!=canonical(resolved_forecast.PROTOCOL)
            or protocol.get('prefix_engine_protocol')!=canonical(five_engine.PROTOCOL)
            or protocol.get('denoised_forecast_protocol')!=canonical(denoised_forecast.PROTOCOL)
            or protocol.get('initial_protocol')!=canonical(initial.PROTOCOL)
            or protocol.get('prefix_forecast_protocol')!=canonical(prefix_forecast.PROTOCOL)
            or protocol.get('target_protocol')!=canonical(TARGET_PROTOCOL)
            or protocol.get('data_protocol')!=canonical(data.PROTOCOL)
            or protocol.get('roles')!=data.ROLE_COUNTS|{'calibration':1}
            or protocol.get('families')!=list(FAMILIES)
            or protocol.get('methods')!=list(METHODS)):
        raise ValueError('Method menu and study design must be frozen before preparation.')
    if (calibration.get('generator_sha256')!=sha(STUDY/'generators.py')
            or calibration.get('protocol')!=canonical(generators.GENERATOR_PROTOCOL)
            or calibration.get('formal_training_development_final_generated') is not False
            or calibration.get('thresholds_changed_after_results') is not False):
        raise ValueError('Independent calibration identity/protocol mismatch.')
    records=calibration.get('records',[])
    if len(records)!=len(FAMILIES) or {r.get('family') for r in records}!=set(FAMILIES):
        raise ValueError('Calibration must include each declared family exactly once.')
    for r in records:
        if r.get('role')!='calibration' or r.get('status') not in ('passed','source_quality_failure'):
            raise ValueError('Invalid calibration role/status.')
        if r['status']=='passed':
            for value,limit_key,limit in [('relative_l2','relative_limit',1e-5),('max_absolute_error','max_absolute_limit',1e-4)]:
                actual=r.get(value)
                if (isinstance(actual,bool) or not isinstance(actual,(int,float))
                        or not np.isfinite(actual) or not 0<=actual<=limit or r.get(limit_key)!=limit):
                    raise ValueError('Claimed source-quality pass contradicts fixed numerical limits.')
    if calibration.get('all_passed') is not all(r['status']=='passed' for r in records):
        raise ValueError('Calibration aggregate differs from family records.')


def prepare():
    if (DATA/'prepared.json').exists():
        raise RuntimeError('Preparation exists; preserve original study records.')
    calibration=read(DATA/'calibration.json')
    validate_design_and_calibration(read(STUDY/'protocol.json'),calibration)
    quality={}
    for family in generators.FAMILIES:
        record=next(r for r in calibration['records'] if r['family']==family)
        passed=record['status']=='passed'
        quality[family]={'status':'passed' if passed else 'source_quality_failure',
                         'calibration':record}
        if not passed:continue
        try:
            for role in ('training','development'):
                for i in range(role_count(family,role)):
                    clean=generators.generate(family,role,i)
                    data.write_clean(DATA,family,role,i,clean)
        except (OSError,ValueError,RuntimeError,FloatingPointError) as exc:
            quality[family]={'status':'source_quality_failure','reason':str(exc),'calibration':record}
    scales={f:data.training_scale([data.read_clean(DATA,f,'training',i)
        for i in range(role_count(f,'training'))]) if quality[f]['status']=='passed' else None for f in FAMILIES}
    receipt={'sources':sources(),'input_hashes':input_hashes(),'quality':quality,'training_rms':scales,
        'all_source_quality_passed':all(v['status']=='passed' for v in quality.values()),
        'synthetic_final_generated':False,'all_families_independently_generated':True,
        'generator_protocol':generators.GENERATOR_PROTOCOL,'protocol':read(STUDY/'protocol.json')}
    write(DATA/'prepared.json',receipt)
    print('Prepared new sources and training/development; synthetic final not generated.',flush=True)
    return receipt


def verify_preparation():
    receipt=read(DATA/'prepared.json')
    if receipt['sources']!=sources() or receipt['input_hashes']!=input_hashes():
        raise ValueError('Code, protocol or prepared input bytes changed.')
    return receipt


def observation(clean,family,role,index,sampling,level,seed,scale):
    return data.corrupt(clean,family,role,index,sampling,level,seed,scale)


def fit_observations(observed, developments):
    """Generate candidates once, then apply four declared selectors to observations."""
    results=five_engine.fit_observations(observed,developments)
    if set(results)!=set(METHODS):raise ValueError('Five-percent engine must retain all four arms.')
    for method,result in results.items():
        if any(result.get(k)!=v for k,v in predictor_metadata(method).items()):
            raise ValueError('Engine predictor metadata differs from the frozen study design.')
    return results


def _freeze_family(family,output):
    """Worker writes only its family files; no worker may read final roles."""
    output=Path(output)
    with threadpool_limits(limits=1):
        prep=verify_preparation(); quality=prep['quality'][family]
        train=dev=[];scale=None
        if quality['status']=='passed':
            train=[data.read_clean(DATA,family,'training',i) for i in range(role_count(family,'training'))]
            dev=[data.read_clean(DATA,family,'development',i) for i in range(role_count(family,'development'))]
            scale=prep['training_rms'][family]
        records=[]
        for sampling in SAMPLINGS:
            for condition,level,seed in NOISES:
                observed=developments=None;error=None
                if quality['status']=='passed':
                    try:
                        observed=[observation(d,family,'training',i,sampling,level,seed,scale) for i,d in enumerate(train)]
                        developments=[observation(d,family,'development',i,sampling,level,seed,scale) for i,d in enumerate(dev)]
                    except (ValueError,RuntimeError,FloatingPointError,np.linalg.LinAlgError) as exc:error=str(exc)
                else:error='Source-quality failure; do not replace this declared family.'
                results={}
                if error is None:
                    try:results=fit_observations(observed,developments)
                    except (ValueError,RuntimeError,FloatingPointError,np.linalg.LinAlgError) as exc:error=str(exc)
                for method in METHODS:
                    result=results.get(method,{'selected':None,'candidates':[],
                        'near_optimal_supports':[],'failure_reason':error,**predictor_metadata(method)})
                    identity=model_id(family,sampling,condition,method)
                    model={'id':identity,'family':family,'sampling':sampling,'condition':condition,
                        'noise_level':level,'noise_seed':seed,'method':method,'training_rms_for_corruption_only':scale,
                        'training_trajectories':len(train),'development_trajectories':len(dev),
                        'source_quality_passed':quality['status']=='passed',
                        'sources':prep['sources'],'input_hashes':prep['input_hashes'],
                        'reference_used_for_selection':False,'clean_future_used_for_selection':False,
                        **predictor_metadata(method),
                        'final_loaded_during_selection':False,**result}
                    validate_configuration(model,prep['quality'],prep['training_rms'])
                    path=output/'models'/f'{identity}.json';write(path,model)
                    records.append({'id':identity,'file':str(path.relative_to(output)),'sha256':sha(path)})
                    print('Frozen '+identity+': '+(model['selected']['equation'] if model['selected'] else str(model['failure_reason'])),flush=True)
        return records


def freeze(output=OUTPUT,workers=1):
    output=Path(output)
    if (output/'freeze_index.json').exists():raise RuntimeError('Use a new output directory; existing freeze is immutable.')
    if list((output/'models').glob('*.json')):raise RuntimeError('Partial model output exists; use a fresh directory.')
    prep=verify_preparation()
    if workers==1:records=[r for f in FAMILIES for r in _freeze_family(f,output)]
    else:
        with ProcessPoolExecutor(max_workers=min(3,workers)) as pool:
            futures=[pool.submit(_freeze_family,f,str(output)) for f in FAMILIES]
            records=[r for future in futures for r in future.result()]
    if not complete(records):raise ValueError('Require all120unique declared models before final.')
    index={'records':records,'sources':sources(),'input_hashes':prep['input_hashes'],
        'preparation_sha256':sha(DATA/'prepared.json'),'quality':prep['quality'],'training_rms':prep['training_rms'],
        'all_source_quality_passed':prep['all_source_quality_passed'],'protocol':read(STUDY/'protocol.json'),
        'all_120_frozen_before_final':True,'synthetic_final_generated':False}
    write(output/'freeze_index.json',index)
    return index


def verify_freeze(output):
    output=Path(output);index=read(output/'freeze_index.json')
    if not complete(index['records']) or index['sources']!=sources():raise ValueError('Incomplete or changed frozen study.')
    if index['input_hashes']!=input_hashes() or index['preparation_sha256']!=sha(DATA/'prepared.json'):
        raise ValueError('Prepared input changed after freeze.')
    preparation=read(DATA/'prepared.json')
    for key in ('sources','input_hashes','quality','training_rms','all_source_quality_passed'):
        if index[key]!=preparation[key]:raise ValueError('Frozen metadata differs from original preparation: '+key)
    for r in index['records']:
        if r['file']!=f"models/{r['id']}.json" or r['sha256']!=sha(output/r['file']):
            raise ValueError('Frozen model path/digest mismatch.')
        m=read(output/r['file'])
        validate_configuration(m,index['quality'],index['training_rms'])
        if m['id']!=r['id'] or model_id(m['family'],m['sampling'],m['condition'],m['method'])!=r['id']:
            raise ValueError('Frozen identity metadata mismatch.')
        if m['sources']!=index['sources'] or m['input_hashes']!=index['input_hashes']:
            raise ValueError('Worker freeze dependencies differ.')
    return index


def _finite_nonnegative(value):
    return (not isinstance(value, bool) and isinstance(value, (float,int))
            and np.isfinite(value) and value >= 0)


def family_tail_checks(records):
    """Protect coefficient tails of all families against both frozen baselines."""
    maxima={};checks=[]
    for family in FAMILIES:
        maxima[family]={}
        for method in METHODS:
            group=[r for r in records if r['family']==family and r['method']==method]
            error_rows=[[c['relative_error'] for c in r['recovery']['coefficient_comparison']
                         if c['reference']!=0] for r in group]
            errors=[value for row in error_rows for value in row]
            if len(group)!=10 or any(not row for row in error_rows) or any(not _finite_nonnegative(v) for v in errors):
                raise ValueError('Incomplete or nonfinite family true-coefficient metrics.')
            maxima[family][method]=max(errors)
        for comparator in ('weak_resolved','denoised_median'):
            checks.append({'name':f'{family}:vs_{comparator}:worst_true_term_nonregression',
                'passed':bool(maxima[family]['prefix_median']<=max(1.25*maxima[family][comparator],.01))})
    return maxima,checks


def promotion_gate(records,source_quality=True):
    """Require the user target plus PDE-identification and regression guards.

    The target is every condition's mean over three final trajectories, not the
    maximum individual trajectory error. A low prediction error cannot excuse
    extra/missing PDE terms or poor coefficient recovery.
    """
    failure={'candidate':'prefix_median','comparators':['weak_resolved','denoised_median'],
        'passed':False,'prediction_target_passed':False,'merged':False,'decision':'keep_baseline'}
    if not complete(records):return {**failure,'reason':'Incomplete/duplicate/unknown configurations.'}
    try:
        scales={f:next(r['training_rms_for_corruption_only'] for r in records if r['family']==f) for f in FAMILIES}
        for r in records:validate_configuration(r,scales=scales)
        if source_quality and not all(r['source_quality_passed'] for r in records):
            raise ValueError('Gate source status contradicts model records.')
        maxima,tail_checks=family_tail_checks(records)
    except (KeyError,ValueError,TypeError,StopIteration) as exc:return {**failure,'reason':str(exc)}
    by={r['id']:r for r in records};candidate=[r for r in records if r['method']=='prefix_median']
    checks=[{'name':'source_quality','passed':bool(source_quality)}];target_checks=[]
    for r in candidate:
        a=r['recovery'];prefix=r['id']
        structure=bool(a.get('exact_support') is True and not a.get('spurious_terms') and not a.get('missing_terms'))
        checks.append({'name':prefix+':exact_structure','passed':structure})
        true_rows=[x for x in a['coefficient_comparison'] if x['reference']!=0]
        limit=.25 if r['noise_level']==.01 else .10
        checks.append({'name':prefix+':each_true_term_accuracy',
            'passed':bool(true_rows) and all(_finite_nonnegative(x['relative_error']) and x['relative_error']<=limit for x in true_rows)})
        value=r['mean_clean_relative_l2']
        target=bool(r.get('forecast_failures')==0 and _finite_nonnegative(value) and value<=TARGET_PROTOCOL['relative_l2_limit'])
        target_checks.append(target)
        checks.append({'name':prefix+':mean_prediction_at_most_5pct','passed':target})
        for comparator in ('weak_resolved','denoised_median'):
            old=by[model_id(r['family'],r['sampling'],r['condition'],comparator)]['recovery']
            av,bv=a['coefficient_relative_l2'],old['coefficient_relative_l2']
            checks.append({'name':prefix+':vs_'+comparator+':coefficient',
                'passed':bool(_finite_nonnegative(av) and _finite_nonnegative(bv) and av<=max(1.25*bv,.01))})
    checks.extend(tail_checks)
    passed=all(c['passed'] for c in checks)
    return {'candidate':'prefix_median','comparators':['weak_resolved','denoised_median'],
        'passed':passed,'prediction_target_passed':all(target_checks),'merged':False,
        'decision':'eligible_for_review_only' if passed else 'keep_baseline_do_not_merge',
        'target_protocol':TARGET_PROTOCOL,'family_max_true_error':maxima,
        'target_conditions_passed':sum(target_checks),'target_conditions_total':len(target_checks),
        'checks':checks,'regression_required_separately':True}


def _score_record(task):
    """Score one immutable model; no source generation, fitting or selection.

    The same helper is used by serial and process workers. Only the parent
    creates final arrays, after validating the complete global model freeze.
    """
    model,frozen_sha,clean_list,errors,reference=task
    with threadpool_limits(limits=1):
        selected=model['selected'];terms=selected['terms'] if selected else []
        recovery=reference_metrics(terms,reference);metrics=[]
        for i,clean in enumerate(clean_list):
            try:
                if clean is None:raise ValueError('Final reference failure: '+str(errors[i]))
                if selected is None:raise ValueError('No eligible selected model.')
                noisy=observation(clean,model['family'],'final',i,model['sampling'],model['noise_level'],model['noise_seed'],model['training_rms_for_corruption_only'])
                expected_duration=TARGET_PROTOCOL['forecast_duration'][model['family']]
                if (not np.isclose(noisy['t'][-1]-noisy['t'][0],expected_duration,rtol=0,atol=1e-10)
                        or not np.isclose(noisy['prefix_t'][-1]-noisy['prefix_t'][0],.1*expected_duration,rtol=0,atol=1e-10)):
                    raise ValueError('Declared full future horizon and separate historical prefix must be preserved.')
                stride=1 if model['sampling']=='full' else 2
                truth=np.asarray(clean['u'])[data.PREFIX_INTERVALS::stride,::stride]
                filter_diagnostics=None
                if model['final_predictor']=='prefix_aligned_resolved_factor4':
                    prediction,filter_diagnostics=prefix_forecast.forecast_with_diagnostics(terms,noisy['x'],noisy['t'],noisy['prefix_t'],noisy['prefix_u'])
                elif model['final_predictor']=='denoised_resolved_factor4':
                    prediction,filter_diagnostics=denoised_forecast.forecast_with_diagnostics(terms,noisy['x'],noisy['t'],noisy['u'][0])
                else:
                    prediction=resolved_forecast.forecast(terms,noisy['x'],noisy['t'],noisy['u'][0])
                metrics.append({'status':'evaluated','clean':forecast_metrics(prediction,truth),
                    'noisy':forecast_metrics(prediction,noisy['u']),
                    'initial_is_noisy_input':True,'initial_preprocessing':model['initial_filter'],
                    'initial_denoise_diagnostics':filter_diagnostics,
                    'conditioning':model['conditioning'],
                    'forecast_duration':float(noisy['t'][-1]-noisy['t'][0]),
                    'prefix_frames_used':len(noisy['prefix_t']) if model['conditioning']=='observed_prefix' else 1})
            except (ValueError,RuntimeError,FloatingPointError,np.linalg.LinAlgError) as exc:
                metrics.append({'status':'failed','reason':str(exc)})
        failures=sum(m['status']!='evaluated' for m in metrics)
        result={k:model[k] for k in ('id','family','sampling','condition','noise_level','noise_seed','method','structural_selection_predictor','coefficient_selection_predictor','final_predictor','initial_filter','conditioning','training_trajectories','development_trajectories','source_quality_passed','training_rms_for_corruption_only')}
        result.update({'terms':terms,'equation':selected['equation'] if selected else None,
            'selected':selected is not None,'candidate_count':len(model['candidates']),'failure_reason':model['failure_reason'],
            'recovery':recovery,'forecasts':metrics,'forecast_failures':failures,
            'mean_clean_relative_l2':float(np.mean([m['clean']['relative_l2'] for m in metrics])) if not failures else None,
            'mean_noisy_relative_l2':float(np.mean([m['noisy']['relative_l2'] for m in metrics])) if not failures else None,
            'frozen_model_sha256':frozen_sha})
        return result


def _score_tasks(tasks,workers=1):
    """Yield scores in frozen model order, independent of completion order."""
    if type(workers) is not int or workers<1:
        raise ValueError('workers must be a positive integer.')
    if workers==1:
        for task in tasks:yield _score_record(task)
    else:
        with ProcessPoolExecutor(max_workers=min(3,workers)) as pool:
            yield from pool.map(_score_record,tasks,chunksize=1)


def evaluate(output=OUTPUT,workers=1):
    if type(workers) is not int or workers<1:raise ValueError('workers must be a positive integer.')
    output=Path(output)
    if (output/'comparison.json').exists():raise RuntimeError('Final scored output exists; reruns are reproduction, not new holdout.')
    if list((output/'final_reference').rglob('*.npz')):
        raise RuntimeError('Partial final references exist; preserve evidence and use a new reproduction run.')
    index=verify_freeze(output)
    references=read(STUDY/'references.json')['references']
    finals={};final_hashes={};final_errors={}
    for family in FAMILIES:
        finals[family]=[];final_errors[family]=[]
        for i in range(role_count(family,'final')):
            try:
                if index['quality'][family]['status']!='passed':raise ValueError('Source-quality failure.')
                clean=generators.generate(family,'final',i)
                path=output/'final_reference'/family/f'{i}.npz';path.parent.mkdir(parents=True,exist_ok=True)
                np.savez_compressed(path,**{k:clean[k] for k in ('x','t','u')})
                final_hashes[str(path.relative_to(output))]=sha(path)
                finals[family].append(clean);final_errors[family].append(None)
            except (OSError,ValueError,RuntimeError,FloatingPointError,np.linalg.LinAlgError) as exc:
                finals[family].append(None);final_errors[family].append(str(exc))
    tasks=[]
    for r in index['records']:
        model=read(output/r['file']);family=model['family']
        tasks.append((model,r['sha256'],finals[family],final_errors[family],references[family]))
    records=[]
    for result in _score_tasks(tasks,workers):
        records.append(result);recovery=result['recovery']
        print(f"Scored {len(records):03}/120 {result['id']}: exact={recovery['exact_support']} missing={len(recovery['missing_terms'])} extra={len(recovery['spurious_terms'])}",flush=True)
    final_quality=not any(error is not None for errors in final_errors.values() for error in errors)
    report={'records':records,'quality':index['quality'],'final_reference_failures':final_errors,
        'gate':promotion_gate(records,index['all_source_quality_passed'] and final_quality),
        'freeze_index_sha256':sha(output/'freeze_index.json'),'final_reference_sha256':final_hashes,
        'all_declared_models_reported':complete(records),'final_scores_used_for_tuning':False,
        'all_data_numerical_not_real_measurement':True,'future_runs_are_reproduction_or_retrospective':True,
        'prediction_above_5pct_all_arms':sum(r['mean_clean_relative_l2'] is not None and r['mean_clean_relative_l2']>.05 for r in records),
        'prediction_target_protocol':TARGET_PROTOCOL,
        'configurations_with_forecast_failure':sum(r['forecast_failures']>0 for r in records)}
    write(output/'comparison.json',report)
    return report


def main():
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('phase',choices=['prepare','freeze','evaluate'])
    parser.add_argument('--output',type=Path,default=OUTPUT);parser.add_argument('--workers',type=int,choices=[1,2,3],default=1)
    args=parser.parse_args()
    with threadpool_limits(limits=1):
        if args.phase=='prepare':prepare()
        elif args.phase=='freeze':freeze(args.output,args.workers)
        else:evaluate(args.output,args.workers)


if __name__=='__main__':main()
