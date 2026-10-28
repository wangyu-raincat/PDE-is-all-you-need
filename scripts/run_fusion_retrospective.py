"""Retrospective fusion of previously viewed, immutable resolved-study models.

No refitting, new IC generation, or truth-based model selection occurs here.
Freeze all 60 composite decisions using old development observations before
loading old final fields; report two cached control arms and two new arms.
"""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fusion_study.engine import select_from_scored, PROTOCOL
from resolved_study import data as old_data
from resolved_study.forecast import forecast
from pde_discovery.generic_experiment import forecast_metrics, reference_metrics


def read(path): return json.loads(Path(path).read_text())
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n')


def main(source, output):
    source, output = Path(source), Path(output)
    startup_sources = {str(p.relative_to(ROOT)): sha(p) for p in
                       (ROOT/'fusion_study/engine.py', Path(__file__).resolve())}
    def verify_startup_sources():
        if any(sha(ROOT/path) != digest for path, digest in startup_sources.items()):
            raise ValueError('Retrospective source bytes changed while running.')
    if output.exists(): raise RuntimeError('Preserve existing retrospective records; use a fresh directory.')
    old = source/'resolved_study/results/first_run'
    index = read(old/'freeze_index.json')
    previous = read(old/'comparison.json')
    for row in index['records']:
        if sha(old/row['file']) != row['sha256']: raise ValueError('Old frozen model changed.')
    for path, digest in index['sources'].items():
        if sha(source/path) != digest: raise ValueError('Old frozen code changed: '+path)
    for path, digest in index['input_hashes'].items():
        if sha(source/path) != digest: raise ValueError('Old training/development changed: '+path)
    controls = [r for r in previous['records'] if r['method'] in ('weak_original','weak_resolved')]
    if len(controls) != 60: raise ValueError('Require all 60 old controls.')
    models = []
    for anchor_row in [r for r in controls if r['method']=='weak_original']:
        family, sampling, condition = [anchor_row[k] for k in ('family','sampling','condition')]
        prefix = '--'.join((family, sampling, condition))
        original = read(old/'models'/f'{prefix}--weak_original.json')
        resolved = read(old/'models'/f'{prefix}--weak_resolved.json')
        dev = [old_data.corrupt(old_data.read_clean(source/'resolved_study/data', family,'development',i),
               family,'development',i,sampling,original['noise_level'],original['noise_seed'],
               original['training_rms_for_corruption_only']) for i in range(2)]
        for method in ('anchored_best','fused_median'):
            result = select_from_scored(original, resolved, method, dev)
            model = {k:original[k] for k in ('family','sampling','condition','noise_level','noise_seed',
                'training_rms_for_corruption_only','source_quality_passed','training_trajectories','development_trajectories')}
            model.update(result)
            model.update(id=prefix+'--'+method, method=method, retrospective=True,
                anchor_model_sha256=sha(old/'models'/f'{prefix}--weak_original.json'),
                resolved_model_sha256=sha(old/'models'/f'{prefix}--weak_resolved.json'))
            path=output/'models'/f"{model['id']}.json"
            write(path,model); models.append({'file':str(path.relative_to(output)),'sha256':sha(path)})
            print('Retrospective frozen '+model['id'],flush=True)
    verify_startup_sources()
    write(output/'freeze_index.json',{'records':models,'protocol':PROTOCOL,
        'startup_source_hashes':startup_sources,
        'engine_sha256':sha(ROOT/'fusion_study/engine.py'),
        'old_comparison_sha256':sha(old/'comparison.json'),'retrospective':True,
        'new_final_fields_generated':False,'old_final_fields_loaded_during_fusion':False})
    references=read(source/'resolved_study/references.json')['references']
    records=list(controls)
    for row in models:
        if sha(output/row['file'])!=row['sha256']: raise ValueError('Retrospective freeze changed.')
        model=read(output/row['file']); selected=model['selected']; terms=selected['terms'] if selected else []
        metrics=[]
        for i in range(3):
            try:
                if selected is None: raise ValueError('No eligible selected model.')
                path=old/'final_reference'/model['family']/f'{i}.npz'
                if sha(path)!=previous['final_reference_sha256'][str(path.relative_to(old))]:
                    raise ValueError('Old final field digest mismatch.')
                with np.load(path,allow_pickle=False) as z: clean={k:z[k].copy() for k in ('x','t','u')}
                noisy=old_data.corrupt(clean,model['family'],'final',i,model['sampling'],
                    model['noise_level'],model['noise_seed'],model['training_rms_for_corruption_only'])
                stride=1 if model['sampling']=='full' else 2
                predicted=forecast(terms,noisy['x'],noisy['t'],noisy['u'][0])
                metrics.append({'status':'evaluated','clean':forecast_metrics(predicted,clean['u'][::stride,::stride]),
                    'noisy':forecast_metrics(predicted,noisy['u']),'initial_is_noisy':True})
            except (ValueError,RuntimeError,FloatingPointError,np.linalg.LinAlgError) as exc:
                metrics.append({'status':'failed','reason':str(exc)})
        failures=sum(m['status']!='evaluated' for m in metrics)
        record={k:model[k] for k in ('id','family','sampling','condition','noise_level','noise_seed','method')}
        record.update(terms=terms,recovery=reference_metrics(terms,references[model['family']]),
            forecasts=metrics,forecast_failures=failures,
            mean_clean_relative_l2=float(np.mean([m['clean']['relative_l2'] for m in metrics])) if not failures else None,
            fusion=model['fusion'],retrospective=True,frozen_model_sha256=row['sha256'])
        records.append(record);print('Retrospective scored '+record['id'],flush=True)
    summary={}
    for method in ('weak_original','weak_resolved','anchored_best','fused_median'):
        rows=[r for r in records if r['method']==method]
        true_errors=[max(c['relative_error'] for c in r['recovery']['coefficient_comparison'] if c['reference']!=0) for r in rows]
        preds=[r['mean_clean_relative_l2'] for r in rows if r['mean_clean_relative_l2'] is not None]
        summary[method]={'conditions':len(rows),'exact':sum(r['recovery']['exact_support'] for r in rows),
            'extras':sum(len(r['recovery']['spurious_terms']) for r in rows),
            'missing':sum(len(r['recovery']['missing_terms']) for r in rows),
            'coefficient_vector_median':float(np.median([r['recovery']['coefficient_relative_l2'] for r in rows])),
            'worst_true_error':max(true_errors),'true_accuracy_failures':sum(e>(.25 if r['noise_level']==.01 else .10) for r,e in zip(rows,true_errors)),
            'prediction_median':float(np.median(preds)) if preds else None,
            'prediction_max':max(preds) if preds else None,'forecast_failures':sum(r['forecast_failures'] for r in rows)}
    verify_startup_sources()
    write(output/'comparison.json',{'records':records,'summary':summary,'retrospective':True,
        'cached_control_configurations':60,'new_composite_configurations':60,'new_composite_rollouts':180,
        'all_data_previously_viewed':True,'no_promotion_decision':True,'no_new_final_validation':True})
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--source',type=Path,default=Path('/workspace/pde-discovery-resolved-weak'))
    p.add_argument('--output',type=Path,default=ROOT/'fusion_study/development/retrospective')
    args=p.parse_args()
    with threadpool_limits(limits=1):main(args.source,args.output)
