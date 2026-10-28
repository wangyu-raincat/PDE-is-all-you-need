"""Old-data retrospective comparison for observation-only initial denoising.

All 60 new decisions are frozen using old development observations before
old final fields are opened. Two complete cached controls are retained.
Nothing here constitutes a new holdout or a promotion decision.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from five_study import engine
from five_study.forecast import forecast_with_diagnostics
from fusion_study import data as old_data
from pde_discovery.generic_experiment import forecast_metrics, reference_metrics

METHODS=('weak_resolved','fused_median','denoised_weak','denoised_median')
NEW_METHODS=('denoised_weak','denoised_median')


def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')


def source_hashes():
    names=['scripts/run_five_retrospective.py','five_study/engine.py','five_study/initial.py',
           'five_study/forecast.py','fusion_study/engine.py','weak_study/engine.py',
           'l1_study/engine.py','resolved_study/forecast.py','resolved_study/selection.py',
           'fusion_study/data.py','challenge_study/data.py',
           'pde_discovery/generic_engine.py','pde_discovery/generic_experiment.py']
    return {name:sha(ROOT/name) for name in names}


def _freeze_condition(job):
    source,output,family,sampling,condition=job
    source,output=Path(source),Path(output)
    prefix='--'.join((family,sampling,condition))
    path=source/'fusion_study/results/first_run/models'/f'{prefix}--weak_resolved.json'
    old=read(path)
    with threadpool_limits(limits=1):
        developments=[old_data.corrupt(old_data.read_clean(source/'fusion_study/data',family,'development',i),
            family,'development',i,sampling,old['noise_level'],old['noise_seed'],
            old['training_rms_for_corruption_only']) for i in range(2)]
        candidates=old['candidates']
        augmented,evidence=engine.generate_support_medians(candidates)
        # Shared candidate forecasts; each arm still selects within its own pool.
        # The near-best threshold is recomputed separately by choose_scored.
        augmented_result=engine.select_candidates(augmented,developments)
        original_ids={c['candidate_id'] for c in candidates}
        results={'denoised_weak':engine.choose_scored([
                    c for c in augmented_result['candidates'] if c['candidate_id'] in original_ids]),
                 'denoised_median':augmented_result}
        records=[]
        for method in NEW_METHODS:
            model={k:old[k] for k in ('family','sampling','condition','noise_level','noise_seed',
                'training_rms_for_corruption_only','source_quality_passed',
                'training_trajectories','development_trajectories')}
            model.update(results[method])
            model.update(engine.stage_metadata(method))
            model.update(id=prefix+'--'+method,method=method,retrospective=True,
                source_model_sha256=sha(path),source_model_file=str(path),
                median_generation=evidence if method=='denoised_median' else None,
                no_training_refit=True,old_final_loaded_for_selection=False)
            target=output/'models'/f"{model['id']}.json";write(target,model)
            records.append({'id':model['id'],'file':str(target.relative_to(output)),'sha256':sha(target)})
    return records


def _score_model(job):
    source,output,row,references,final_hashes=job
    source,output=Path(source),Path(output);old=source/'fusion_study/results/first_run'
    path=output/row['file']
    if sha(path)!=row['sha256']:raise ValueError('New retrospective frozen model changed.')
    model=read(path);selected=model['selected'];terms=selected['terms'] if selected else []
    metrics=[]
    with threadpool_limits(limits=1):
        for i in range(3):
            try:
                if selected is None:raise ValueError('No eligible selected model.')
                final_path=old/'final_reference'/model['family']/f'{i}.npz'
                if sha(final_path)!=final_hashes[str(final_path.relative_to(old))]:
                    raise ValueError('Old final field digest mismatch.')
                with np.load(final_path,allow_pickle=False) as z:
                    clean={k:z[k].copy() for k in ('x','t','u')}
                noisy=old_data.corrupt(clean,model['family'],'final',i,model['sampling'],
                    model['noise_level'],model['noise_seed'],model['training_rms_for_corruption_only'])
                stride=1 if model['sampling']=='full' else 2
                prediction,diagnostics=forecast_with_diagnostics(terms,noisy['x'],noisy['t'],noisy['u'][0])
                metrics.append({'status':'evaluated','clean':forecast_metrics(prediction,clean['u'][::stride,::stride]),
                    'noisy':forecast_metrics(prediction,noisy['u']),
                    'initial_observation_is_noisy':True,'initial_preprocessing':diagnostics})
            except (ValueError,RuntimeError,FloatingPointError,np.linalg.LinAlgError) as exc:
                metrics.append({'status':'failed','reason':str(exc)})
    failures=sum(m['status']!='evaluated' for m in metrics)
    result={k:model[k] for k in ('id','family','sampling','condition','noise_level','noise_seed','method')}
    result.update(terms=terms,equation=selected['equation'] if selected else None,
        recovery=reference_metrics(terms,references[model['family']]),forecasts=metrics,
        forecast_failures=failures,retrospective=True,selected=selected is not None,
        candidate_count=len(model['candidates']),selected_candidate_id=selected['candidate_id'] if selected else None,
        mean_clean_relative_l2=float(np.mean([m['clean']['relative_l2'] for m in metrics])) if not failures else None,
        mean_noisy_relative_l2=float(np.mean([m['noisy']['relative_l2'] for m in metrics])) if not failures else None,
        frozen_model_sha256=row['sha256'])
    return result


def summarize(records):
    summary={}
    for method in METHODS:
        rows=[r for r in records if r['method']==method]
        if len(rows)!=30:raise ValueError('Each declared arm must retain all30conditions.')
        errors=[max(c['relative_error'] for c in r['recovery']['coefficient_comparison'] if c['reference']!=0) for r in rows]
        predictions=[r['mean_clean_relative_l2'] for r in rows if r['mean_clean_relative_l2'] is not None]
        summary[method]={'conditions':30,'exact':sum(r['recovery']['exact_support'] for r in rows),
            'extras':sum(len(r['recovery']['spurious_terms']) for r in rows),
            'missing':sum(len(r['recovery']['missing_terms']) for r in rows),
            'coefficient_vector_median':float(np.median([r['recovery']['coefficient_relative_l2'] for r in rows])),
            'worst_true_error':max(errors),
            'true_accuracy_failures':sum(e>(.25 if r['noise_level']==.01 else .10) for r,e in zip(rows,errors)),
            'prediction_median':float(np.median(predictions)) if predictions else None,
            'prediction_max':max(predictions) if predictions else None,
            'prediction_above5pct':sum(v>.05 for v in predictions),
            'prediction_missing':30-len(predictions),
            'forecast_failures':sum(r['forecast_failures'] for r in rows),
            'diagnostic_all_30_prediction_means_below5pct':len(predictions)==30 and max(predictions)<=.05,
            'diagnostic_complete_structure':all(r['recovery']['exact_support'] for r in rows)}
    return summary


def main(source,output,workers=3):
    if workers not in (1,2,3):raise ValueError('Use1..3workers.')
    source,output=Path(source),Path(output)
    if output.exists():raise RuntimeError('Preserve previous records; choose a fresh output directory.')
    startup=source_hashes();old=source/'fusion_study/results/first_run'
    index=read(old/'freeze_index.json');previous=read(old/'comparison.json')
    for row in index['records']:
        if sha(old/row['file'])!=row['sha256']:raise ValueError('Old model digest mismatch.')
    for key in ('sources','input_hashes'):
        for path,digest in index[key].items():
            if sha(source/path)!=digest:raise ValueError('Old source/input digest mismatch: '+path)
    controls=[r for r in previous['records'] if r['method'] in METHODS[:2]]
    if len(controls)!=60:raise ValueError('Require60complete cachedcontrols.')
    conditions=sorted({(r['family'],r['sampling'],r['condition']) for r in controls})
    if len(conditions)!=30:raise ValueError('Require30uniqueconditions.')
    jobs=[(str(source),str(output),*c) for c in conditions]
    frozen=[]
    if workers==1:
        iterator=map(_freeze_condition,jobs)
        for records in iterator:
            frozen.extend(records);print(f'Retrospective frozen {len(frozen):02}/60',flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for records in pool.map(_freeze_condition,jobs):
                frozen.extend(records);print(f'Retrospective frozen {len(frozen):02}/60',flush=True)
    if len(frozen)!=60 or len({r['id'] for r in frozen})!=60:raise ValueError('Incompleteglobalfreeze.')
    if startup!=source_hashes():raise ValueError('Source changed while freezing; preserve partial records.')
    write(output/'freeze_index.json',{'records':frozen,'sources':startup,
        'engine_protocol':engine.PROTOCOL,'old_comparison_sha256':sha(old/'comparison.json'),
        'old_freeze_sha256':sha(old/'freeze_index.json'),'retrospective':True,
        'all60_frozen_before_old_final_load':True,'new_formal_fields_generated':False})
    # Reference labels and old final fields are used only after global freeze.
    references=read(source/'fusion_study/references.json')['references']
    score_jobs=[(str(source),str(output),r,references,previous['final_reference_sha256']) for r in frozen]
    records=list(controls)
    if workers==1:
        for result in map(_score_model,score_jobs):
            records.append(result);print(f'Retrospective scored {len(records)-60:02}/60',flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for result in pool.map(_score_model,score_jobs):
                records.append(result);print(f'Retrospective scored {len(records)-60:02}/60',flush=True)
    if startup!=source_hashes():raise ValueError('Source changed during final retrospective scoring.')
    summary=summarize(records)
    write(output/'comparison.json',{'records':records,'summary':summary,'sources':startup,
        'retrospective':True,'no_promotion_decision':True,'all_data_previously_viewed':True,
        'cached_control_configurations':60,'new_composite_configurations':60,
        'new_composite_rollouts':180,'underlying_old_final_initial_conditions':9,
        'new_final_fields_generated':False,'freeze_index_sha256':sha(output/'freeze_index.json')})
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--source',type=Path,default=Path('/workspace/pde-discovery-fused-weak'))
    parser.add_argument('--output',type=Path,default=ROOT/'five_study/development/retrospective')
    parser.add_argument('--workers',type=int,choices=(1,2,3),default=3)
    args=parser.parse_args()
    main(args.source,args.output,args.workers)
