"""Fresh-initial-condition weak PDE improvement; isolated immutable workflow."""
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
from l1_study import engine
from . import data, generators
from . import engine as weak_engine

ROOT=Path(__file__).resolve().parents[1]
STUDY=ROOT/'weak_study'
DATA=STUDY/'data'
OUTPUT=STUDY/'results'/'first_run'
FAMILIES=('allen_cahn','low_viscosity_burgers','ks_unstable')
METHODS=('plain','adaptive','weak_adaptive')
SAMPLINGS=('full','coarse')
NOISES=(('clean',0.,0),('noise0p1_s43',.001,43),('noise0p1_s71',.001,71),
        ('noise1_s43',.01,43),('noise1_s71',.01,71))


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
    return len(ids)==len(set(ids))==90 and set(ids)==expected_ids()


def role_count(family,role):
    return data.ROLE_COUNTS[role]


def validate_configuration(record,quality=None,scales=None):
    f,s,n,m=(record[k] for k in ('family','sampling','condition','method'))
    conditions={name:(level,seed) for name,level,seed in NOISES}
    if f not in FAMILIES or s not in SAMPLINGS or n not in conditions or m not in METHODS:
        raise ValueError('Undeclared configuration.')
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


def fit_observations(observed, developments, method):
    """Fit only observations; no family/noise label/reference accepted here."""
    if method not in METHODS: raise ValueError('Undeclared method.')
    if method != 'weak_adaptive':
        blocks=[engine.integrated_block(d) for d in observed]
        return engine.select_candidates(engine.fit_blocks(blocks,method),developments)
    candidates=[]; variants=[]
    for variant in weak_engine.VARIANTS:
        try:
            blocks=[weak_engine.weak_block(d,variant) for d in observed]
            fitted=engine.fit_blocks(blocks,'adaptive')
            for candidate in fitted:
                candidate['candidate_id']=str(variant)+'::'+candidate['candidate_id']
                candidate['weak_variant']=variant
                candidate['method']='weak_adaptive'
            candidates.extend(fitted)
            variants.append({'variant':variant,'candidate_count':len(fitted),'failure_reason':None})
        except (ValueError,RuntimeError,FloatingPointError,np.linalg.LinAlgError) as exc:
            variants.append({'variant':variant,'candidate_count':0,'failure_reason':str(exc)})
    result=engine.select_candidates(candidates,developments)
    result['variant_trials']=variants
    result['variant_selected_using']='development observations only, same selector as baseline'
    return result


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
                for method in METHODS:
                    result={'selected':None,'candidates':[],'near_optimal_supports':[], 'failure_reason':error}
                    if error is None:
                        try:result=fit_observations(observed,developments,method)
                        except (ValueError,RuntimeError,FloatingPointError,np.linalg.LinAlgError) as exc:result['failure_reason']=str(exc)
                    identity=model_id(family,sampling,condition,method)
                    model={'id':identity,'family':family,'sampling':sampling,'condition':condition,
                        'noise_level':level,'noise_seed':seed,'method':method,'training_rms_for_corruption_only':scale,
                        'training_trajectories':len(train),'development_trajectories':len(dev),
                        'source_quality_passed':quality['status']=='passed',
                        'sources':prep['sources'],'input_hashes':prep['input_hashes'],
                        'reference_used_for_selection':False,'clean_future_used_for_selection':False,
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
    if not complete(records):raise ValueError('Require all90unique declared models before final.')
    index={'records':records,'sources':sources(),'input_hashes':prep['input_hashes'],
        'preparation_sha256':sha(DATA/'prepared.json'),'quality':prep['quality'],'training_rms':prep['training_rms'],
        'all_source_quality_passed':prep['all_source_quality_passed'],'protocol':read(STUDY/'protocol.json'),
        'all_90_frozen_before_final':True,'synthetic_final_generated':False}
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


def promotion_gate(records,source_quality=True):
    failure={'candidate':'weak_adaptive','comparator':'adaptive','passed':False,'merged':False,'decision':'keep_baseline'}
    if not complete(records):return {**failure,'reason':'Incomplete/duplicate/unknown configurations.'}
    try:
        scales={f:next(r['training_rms_for_corruption_only'] for r in records if r['family']==f) for f in FAMILIES}
        for r in records:validate_configuration(r,scales=scales)
        if source_quality and not all(r['source_quality_passed'] for r in records):
            raise ValueError('Gate source status contradicts model records.')
    except (KeyError,ValueError,TypeError,StopIteration) as exc:
        return {**failure,'reason':str(exc)}
    by={r['id']:r for r in records};candidate=[r for r in records if r['method']=='weak_adaptive']
    comparator=[r for r in records if r['method']=='adaptive'];checks=[('source_quality',bool(source_quality))];gains=[]
    for r in candidate:
        b=by[model_id(r['family'],r['sampling'],r['condition'],'adaptive')]
        a,old=r['recovery'],b['recovery']
        if a['exact_support'] and not old['exact_support']:gains.append(r['family'])
        checks.append((r['id']+':coefficient',a['coefficient_relative_l2']<=max(1.25*old['coefficient_relative_l2'],.01)))
        av,bv=r['mean_clean_relative_l2'],b['mean_clean_relative_l2']
        checks.append((r['id']+':prediction',av is not None and bv is not None and av<=max(1.25*bv,.05)))
        if r['condition']=='clean':
            checks.append((r['id']+':no_clean_structure_loss',not old['exact_support'] or a['exact_support']))
        limit=.25 if r['noise_level']==.01 else .10
        true_rows=[x for x in a['coefficient_comparison'] if x['reference']!=0]
        checks.append((r['id']+':each_true_term_accuracy',bool(true_rows) and all(x['relative_error']<=limit for x in true_rows)))
    total=lambda group,key:sum(len(r['recovery'][key]) for r in group)
    checks += [('at_least_2_new_exact_conditions',len(gains)>=2),('gains_across_2_families',len(set(gains))>=2),
        ('no_more_extra_total',total(candidate,'spurious_terms')<=total(comparator,'spurious_terms')),
        ('no_more_missing_total',total(candidate,'missing_terms')<=total(comparator,'missing_terms')),
        ('no_more_forecast_failures',sum(r['forecast_failures'] for r in candidate)<=sum(r['forecast_failures'] for r in comparator))]
    passed=all(v for _,v in checks)
    return {'candidate':'weak_adaptive','comparator':'adaptive','passed':passed,'merged':False,
        'decision':'eligible_for_review_only' if passed else 'keep_baseline_do_not_merge',
        'new_exact_conditions':len(gains),'gain_families':sorted(set(gains)),
        'checks':[{'name':k,'passed':bool(v)} for k,v in checks],'regression_required_separately':True}


def evaluate(output=OUTPUT):
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
    records=[]
    for r in index['records']:
        model=read(output/r['file']);selected=model['selected'];terms=selected['terms'] if selected else []
        recovery=reference_metrics(terms,references[model['family']]);metrics=[]
        clean_list=finals[model['family']]
        for i,clean in enumerate(clean_list):
            try:
                if clean is None:raise ValueError('Final reference failure: '+str(final_errors[model['family']][i]))
                if selected is None:raise ValueError('No eligible selected model.')
                noisy=observation(clean,model['family'],'final',i,model['sampling'],model['noise_level'],model['noise_seed'],model['training_rms_for_corruption_only'])
                stride=1 if model['sampling']=='full' else 2
                truth=np.asarray(clean['u'])[::stride,::stride]
                prediction=predictor.forecast(terms,noisy['x'],noisy['t'],noisy['u'][0])
                metrics.append({'status':'evaluated','clean':forecast_metrics(prediction,truth),
                    'noisy':forecast_metrics(prediction,noisy['u']),'initial_is_noisy':True})
            except (ValueError,RuntimeError,FloatingPointError,np.linalg.LinAlgError) as exc:
                metrics.append({'status':'failed','reason':str(exc)})
        failures=sum(m['status']!='evaluated' for m in metrics)
        result={k:model[k] for k in ('id','family','sampling','condition','noise_level','noise_seed','method','training_trajectories','development_trajectories','source_quality_passed','training_rms_for_corruption_only')}
        result.update({'terms':terms,'equation':selected['equation'] if selected else None,
            'selected':selected is not None,'candidate_count':len(model['candidates']),'failure_reason':model['failure_reason'],
            'recovery':recovery,'forecasts':metrics,'forecast_failures':failures,
            'mean_clean_relative_l2':float(np.mean([m['clean']['relative_l2'] for m in metrics])) if not failures else None,
            'mean_noisy_relative_l2':float(np.mean([m['noisy']['relative_l2'] for m in metrics])) if not failures else None,
            'frozen_model_sha256':r['sha256']})
        records.append(result)
        print(f"Scored {len(records):02}/90 {r['id']}: exact={recovery['exact_support']} missing={len(recovery['missing_terms'])} extra={len(recovery['spurious_terms'])}",flush=True)
    final_quality=not any(error is not None for errors in final_errors.values() for error in errors)
    report={'records':records,'quality':index['quality'],'final_reference_failures':final_errors,
        'gate':promotion_gate(records,index['all_source_quality_passed'] and final_quality),
        'freeze_index_sha256':sha(output/'freeze_index.json'),'final_reference_sha256':final_hashes,
        'all_declared_models_reported':complete(records),'final_scores_used_for_tuning':False,
        'all_data_numerical_not_real_measurement':True,'future_runs_are_reproduction_or_retrospective':True}
    write(output/'comparison.json',report)
    return report


def main():
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('phase',choices=['prepare','freeze','evaluate'])
    parser.add_argument('--output',type=Path,default=OUTPUT);parser.add_argument('--workers',type=int,choices=[1,2,3],default=1)
    args=parser.parse_args()
    with threadpool_limits(limits=1):
        if args.phase=='prepare':prepare()
        elif args.phase=='freeze':freeze(args.output,args.workers)
        else:evaluate(args.output)


if __name__=='__main__':main()
