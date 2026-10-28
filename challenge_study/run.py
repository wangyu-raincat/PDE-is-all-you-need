"""Predeclared noisy/coarse PDE stress test; no baseline algorithm changes."""
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

ROOT=Path(__file__).resolve().parents[1]
STUDY=ROOT/'challenge_study'
DATA=STUDY/'data'
OUTPUT=STUDY/'results'/'first_run'
FAMILIES=('allen_cahn','low_viscosity_burgers','ks_unstable')
METHODS=('plain','adaptive','irl1')
SAMPLINGS=('full','coarse')
NOISES=(('clean',0.,0),('noise0p1_s17',.001,17),('noise0p1_s29',.001,29),
        ('noise1_s17',.01,17),('noise1_s29',.01,29))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')


def sources():
    files=list(STUDY.glob('*.py'))+[STUDY/'protocol.json',STUDY/'references.json',
        STUDY/'source/source_manifest.json',ROOT/'l1_study/engine.py',
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
    return 1 if family=='allen_cahn' else (3 if role=='training' else 2)


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


def prepare():
    if (DATA/'prepared.json').exists():
        raise RuntimeError('Preparation exists; preserve original study records.')
    calibration=read(DATA/'calibration.json')
    if calibration['generator_sha256']!=sha(STUDY/'generators.py'):
        raise ValueError('Independent source-quality receipt does not match generator code.')
    quality={}
    try:
        data.download_ac(DATA)
        adapter=data.prepare_ac(DATA)
        quality['allen_cahn']={'status':'passed','kind':'author public numerical benchmark',
            'receipt':adapter,'upstream_solver_convergence_independently_certified':False}
    except (OSError,ValueError,RuntimeError) as exc:
        quality['allen_cahn']={'status':'source_quality_failure','reason':str(exc)}
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
        'synthetic_final_generated':False,'AC_full_MAT_decode_disclosed':True,
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
                blocks=developments=None;error=None
                if quality['status']=='passed':
                    try:
                        observed=[observation(d,family,'training',i,sampling,level,seed,scale) for i,d in enumerate(train)]
                        developments=[observation(d,family,'development',i,sampling,level,seed,scale) for i,d in enumerate(dev)]
                        blocks=[engine.integrated_block(d) for d in observed]
                    except (ValueError,RuntimeError,FloatingPointError,np.linalg.LinAlgError) as exc:error=str(exc)
                else:error='Source-quality failure; do not replace this declared family.'
                for method in METHODS:
                    result={'selected':None,'candidates':[],'near_optimal_supports':[], 'failure_reason':error}
                    if error is None:
                        try:result=engine.select_candidates(engine.fit_blocks(blocks,method),developments)
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
    if not complete(records):return {'candidate':'adaptive','passed':False,'merged':False,'decision':'keep_baseline','reason':'Incomplete/duplicate/unknown configurations.'}
    try:
        scales={f:next(r['training_rms_for_corruption_only'] for r in records if r['family']==f) for f in FAMILIES}
        for r in records:validate_configuration(r,scales=scales)
        if source_quality and not all(r['source_quality_passed'] for r in records):
            raise ValueError('Gate source status contradicts model records.')
    except (KeyError,ValueError,TypeError,StopIteration) as exc:
        return {'candidate':'adaptive','passed':False,'merged':False,'decision':'keep_baseline','reason':str(exc)}
    by={r['id']:r for r in records};candidate=[r for r in records if r['method']=='adaptive']
    plain=[r for r in records if r['method']=='plain']; checks=[('source_quality',bool(source_quality))]
    gains=[]
    for r in candidate:
        b=by[model_id(r['family'],r['sampling'],r['condition'],'plain')]
        a,old=r['recovery'],b['recovery']
        if a['exact_support'] and not old['exact_support']:gains.append(r['family'])
        checks.append((r['id']+':coefficient',a['coefficient_relative_l2']<=max(1.25*old['coefficient_relative_l2'],.01)))
        av,bv=r['mean_clean_relative_l2'],b['mean_clean_relative_l2']
        checks.append((r['id']+':prediction',av is not None and bv is not None and av<=max(1.25*bv,.05)))
        if r['condition']=='clean':
            checks.append((r['id']+':no_clean_structure_loss',not old['exact_support'] or a['exact_support']))
            true_rows=[x for x in a['coefficient_comparison'] if x['reference']!=0]
            checks.append((r['id']+':clean_each_true_term_10pct',all(x['relative_error']<=.10 for x in true_rows)))
    checks.append(('at_least_2_new_exact_conditions',len(gains)>=2))
    checks.append(('gains_across_2_families',len(set(gains))>=2))
    total=lambda group,key:sum(len(r['recovery'][key]) for r in group)
    checks.append(('extra_terms_reduce_25pct',total(candidate,'spurious_terms')<=.75*total(plain,'spurious_terms')))
    checks.append(('no_more_missing_total',total(candidate,'missing_terms')<=total(plain,'missing_terms')))
    checks.append(('no_more_forecast_failures',sum(r['forecast_failures'] for r in candidate)<=sum(r['forecast_failures'] for r in plain)))
    passed=all(v for _,v in checks)
    return {'candidate':'adaptive','passed':passed,'merged':False,'decision':'eligible_for_review_only' if passed else 'keep_baseline_do_not_merge',
        'new_exact_conditions':len(gains),'gain_families':sorted(set(gains)),
        'checks':[{'name':k,'passed':bool(v)} for k,v in checks],'regression_required_separately':True}


def evaluate(output=OUTPUT):
    output=Path(output)
    if (output/'comparison.json').exists():raise RuntimeError('Final scored output exists; reruns are reproduction, not new holdout.')
    index=verify_freeze(output)
    references=read(STUDY/'references.json')['references']
    finals={};final_hashes={};final_errors={}
    for family in FAMILIES:
        finals[family]=[];final_errors[family]=[]
        for i in range(role_count(family,'final')):
            try:
                if index['quality'][family]['status']!='passed':raise ValueError('Source-quality failure.')
                clean=data.read_clean(DATA,family,'final',i) if family=='allen_cahn' else generators.generate(family,'final',i)
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
