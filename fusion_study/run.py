"""Support-anchored fusion: four-arm isolated immutable workflow."""
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
from . import engine as fusion_engine

ROOT=Path(__file__).resolve().parents[1]
STUDY=ROOT/'fusion_study'
DATA=STUDY/'data'
OUTPUT=STUDY/'results'/'first_run'
FAMILIES=('allen_cahn','low_viscosity_burgers','ks_unstable')
METHODS=('weak_original','weak_resolved','anchored_best','fused_median')
SAMPLINGS=('full','coarse')
NOISES=(('clean',0.,0),('noise0p1_s61',.001,61),('noise0p1_s97',.001,97),
        ('noise1_s61',.01,61),('noise1_s97',.01,97))


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
        ROOT/'resolved_study/forecast.py',ROOT/'resolved_study/selection.py',
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


def predictor_metadata(method):
    if method not in METHODS: raise ValueError('Undeclared method.')
    original='original_observed_grid'; refined='resolved_factor4'
    return {'structural_selection_predictor':refined if method=='weak_resolved' else original,
            'coefficient_selection_predictor':original if method=='weak_original' else refined,
            'final_predictor':original if method=='weak_original' else refined}


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
            or protocol.get('fusion_engine_protocol')!=canonical(fusion_engine.PROTOCOL)
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
    results=fusion_engine.fit_observations(observed,developments)
    if set(results)!=set(METHODS):raise ValueError('Fusion engine must retain all four arms.')
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


def clear_gain(records):
    """Pre-data engineering gain: improve a family's worst coefficient error.

    Maxima cover all ten paired sampling/noise conditions per family. These are
    engineering thresholds, not confidence bounds or significance tests.
    """
    failure={'passed':False,'gain_families':[],'family_max_true_error':{},
             'tail_checks':[],'reason':'Incomplete or invalid configuration metrics.'}
    if not complete(records):return failure
    maxima={}
    try:
        for family in FAMILIES:
            maxima[family]={}
            for method in METHODS:
                group=[r for r in records if r['family']==family and r['method']==method]
                if len(group)!=10:return failure
                values=[]
                for row in group:
                    errors=[c['relative_error'] for c in row['recovery']['coefficient_comparison']
                            if c['reference']!=0]
                    if not errors or any(isinstance(v,bool) or not isinstance(v,(float,int))
                                         or not np.isfinite(v) or v<0 for v in errors):return failure
                    values.append(max(errors))
                maxima[family][method]=max(values)
    except (KeyError,TypeError,ValueError):return failure
    gains=[];tail_checks=[]
    for family,values in maxima.items():
        candidate=values['fused_median']
        if all(candidate<=.75*values[old] and values[old]-candidate>=.05
               for old in ('weak_original','weak_resolved')):gains.append(family)
        for old in ('weak_original','weak_resolved'):
            tail_checks.append({'name':f'{family}:vs_{old}:worst_true_term_nonregression',
                'passed':bool(candidate<=max(1.25*values[old],.01))})
    ablation={family:{'anchored_best':values['anchored_best'],'fused_median':values['fused_median'],
                     'absolute_reduction':values['anchored_best']-values['fused_median']}
              for family,values in maxima.items()}
    return {'passed':bool(gains),'gain_families':gains,'family_max_true_error':maxima,
            'tail_checks':tail_checks,'anchored_best_ablation':ablation,
            'rule':'at least one family maximum true-term relative error reduced >=25% and >=.05 absolute against BOTH weak baselines',
            'engineering_threshold_not_significance_test':True}


def promotion_gate(records,source_quality=True):
    failure={'candidate':'fused_median','comparators':['weak_original','weak_resolved'],
        'passed':False,'merged':False,'decision':'keep_baseline'}
    if not complete(records):return {**failure,'reason':'Incomplete/duplicate/unknown configurations.'}
    try:
        scales={f:next(r['training_rms_for_corruption_only'] for r in records if r['family']==f) for f in FAMILIES}
        for r in records:validate_configuration(r,scales=scales)
        if source_quality and not all(r['source_quality_passed'] for r in records):
            raise ValueError('Gate source status contradicts model records.')
    except (KeyError,ValueError,TypeError,StopIteration) as exc:return {**failure,'reason':str(exc)}
    by={r['id']:r for r in records};candidate=[r for r in records if r['method']=='fused_median']
    checks=[('source_quality',bool(source_quality))];gains=[]
    for r in candidate:
        a=r['recovery']
        for comparator in ('weak_original','weak_resolved'):
            b=by[model_id(r['family'],r['sampling'],r['condition'],comparator)];old=b['recovery']
            prefix=r['id']+':vs_'+comparator
            checks.append((prefix+':coefficient',a['coefficient_relative_l2']<=max(1.25*old['coefficient_relative_l2'],.01)))
            av,bv=r['mean_clean_relative_l2'],b['mean_clean_relative_l2']
            checks.append((prefix+':prediction',av is not None and bv is not None and av<=max(1.25*bv,.05)))
            if r['condition']=='clean':checks.append((prefix+':no_clean_structure_loss',not old['exact_support'] or a['exact_support']))
            if comparator=='weak_resolved' and a['exact_support'] and not old['exact_support']:gains.append(r['family'])
        limit=.25 if r['noise_level']==.01 else .10
        true_rows=[x for x in a['coefficient_comparison'] if x['reference']!=0]
        checks.append((r['id']+':each_true_term_accuracy',bool(true_rows) and all(x['relative_error']<=limit for x in true_rows)))
    total=lambda group,key:sum(len(r['recovery'][key]) for r in group)
    for method in ('weak_original','weak_resolved'):
        group=[r for r in records if r['method']==method]
        checks.extend([(f'vs_{method}:no_net_exact_loss',sum(r['recovery']['exact_support'] for r in candidate)>=sum(r['recovery']['exact_support'] for r in group)),
            (f'vs_{method}:no_more_extra_total',total(candidate,'spurious_terms')<=total(group,'spurious_terms')),
            (f'vs_{method}:no_more_missing_total',total(candidate,'missing_terms')<=total(group,'missing_terms')),
            (f'vs_{method}:no_more_forecast_failures',sum(r['forecast_failures'] for r in candidate)<=sum(r['forecast_failures'] for r in group))])
    gain=clear_gain(records)
    checks.extend((item['name'],item['passed']) for item in gain['tail_checks'])
    checks.append(('prespecified_clear_gain',gain['passed']))
    passed=all(v for _,v in checks)
    return {'candidate':'fused_median','comparators':['weak_original','weak_resolved'],
        'passed':passed,'merged':False,'decision':'eligible_for_review_only' if passed else 'keep_baseline_do_not_merge',
        'clear_gain':gain,
        'checks':[{'name':k,'passed':bool(v)} for k,v in checks],'regression_required_separately':True}


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
                stride=1 if model['sampling']=='full' else 2
                truth=np.asarray(clean['u'])[::stride,::stride]
                forecast_function=predictor.forecast if model['final_predictor']=='original_observed_grid' else resolved_forecast.forecast
                prediction=forecast_function(terms,noisy['x'],noisy['t'],noisy['u'][0])
                metrics.append({'status':'evaluated','clean':forecast_metrics(prediction,truth),
                    'noisy':forecast_metrics(prediction,noisy['u']),'initial_is_noisy':True})
            except (ValueError,RuntimeError,FloatingPointError,np.linalg.LinAlgError) as exc:
                metrics.append({'status':'failed','reason':str(exc)})
        failures=sum(m['status']!='evaluated' for m in metrics)
        result={k:model[k] for k in ('id','family','sampling','condition','noise_level','noise_seed','method','structural_selection_predictor','coefficient_selection_predictor','final_predictor','training_trajectories','development_trajectories','source_quality_passed','training_rms_for_corruption_only')}
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
        'prediction_above_10pct_diagnostic_only':sum(r['mean_clean_relative_l2'] is not None and r['mean_clean_relative_l2']>.10 for r in records),
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
