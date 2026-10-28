"""Compare the final weak menu on previously viewed challenge data only."""
from pathlib import Path
import argparse
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from threadpoolctl import threadpool_limits
from challenge_study import data as old_data
from challenge_study.run import FAMILIES, SAMPLINGS, NOISES, role_count
from weak_study.run import fit_observations, read, write, sha, sources
from pde_discovery.generic_experiment import reference_metrics, forecast_metrics
from pde_discovery.generic_engine import forecast


def family_run(family, old_root, output):
    old_root,output=Path(old_root),Path(output)
    with threadpool_limits(limits=1):
        train=[old_data.read_clean(old_root/'challenge_study/data',family,'training',i) for i in range(role_count(family,'training'))]
        dev=[old_data.read_clean(old_root/'challenge_study/data',family,'development',i) for i in range(role_count(family,'development'))]
        scale=old_data.training_scale(train)
        references=read(old_root/'challenge_study/references.json')['references']
        rows=[]
        for sampling in SAMPLINGS:
            for condition,level,seed in NOISES:
                observed=[old_data.corrupt(d,family,'training',i,sampling,level,seed,scale) for i,d in enumerate(train)]
                developments=[old_data.corrupt(d,family,'development',i,sampling,level,seed,scale) for i,d in enumerate(dev)]
                try: result=fit_observations(observed,developments,'weak_adaptive')
                except (ValueError,RuntimeError,FloatingPointError,np.linalg.LinAlgError) as exc:
                    result={'selected':None,'candidates':[],'failure_reason':str(exc)}
                identity='--'.join((family,sampling,condition,'weak_adaptive'))
                write(output/'models'/f'{identity}.json',result)
                selected=result['selected'];terms=selected['terms'] if selected else [];scores=[]
                for i in range(role_count(family,'final')):
                    try:
                        with np.load(old_root/'challenge_study/results/first_run/final_reference'/family/f'{i}.npz',allow_pickle=False) as z:
                            clean={k:z[k].copy() for k in ('x','t','u')}
                        clean['boundary']='periodic'
                        noisy=old_data.corrupt(clean,family,'final',i,sampling,level,seed,scale)
                        if selected is None: raise ValueError('No eligible selected model.')
                        pred=forecast(terms,noisy['x'],noisy['t'],noisy['u'][0])
                        stride=1 if sampling=='full' else 2
                        scores.append({'status':'evaluated','clean':forecast_metrics(pred,clean['u'][::stride,::stride]),
                                       'noisy':forecast_metrics(pred,noisy['u'])})
                    except (OSError,ValueError,RuntimeError,FloatingPointError,np.linalg.LinAlgError) as exc:
                        scores.append({'status':'failed','reason':str(exc)})
                failures=sum(m['status']!='evaluated' for m in scores)
                row={'id':identity,'family':family,'sampling':sampling,'condition':condition,
                    'noise_level':level,'noise_seed':seed,'method':'weak_adaptive','terms':terms,
                    'equation':selected['equation'] if selected else None,
                    'selected_variant':selected.get('weak_variant') if selected else None,
                    'recovery':reference_metrics(terms,references[family]),'forecasts':scores,
                    'forecast_failures':failures,'mean_clean_relative_l2':None if failures else float(np.mean([m['clean']['relative_l2'] for m in scores])),
                    'frozen_model_sha256':sha(output/'models'/f'{identity}.json')}
                rows.append(row)
                print(identity+': '+str(row['equation']),flush=True)
        return rows


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--previous',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise RuntimeError('Retrospective output already exists; preserve it.')
    args.output.mkdir(parents=True)
    original=args.previous/'challenge_study/results/first_run/comparison.json'
    previous=read(original)
    with ProcessPoolExecutor(max_workers=3) as pool:
        futures=[pool.submit(family_run,f,str(args.previous),str(args.output)) for f in FAMILIES]
        rows=[row for future in futures for row in future.result()]
    assert len(rows)==len({r['id'] for r in rows})==30
    comparison=[r for r in previous['records'] if r['method'] in ('plain','adaptive')]+rows
    write(args.output/'comparison.json',{'study_kind':'retrospective development on already viewed data',
        'not_new_holdout':True,'records':comparison,'old_comparison_sha256':sha(original),
        'current_sources':sources(),'old_counts_reused_not_refitted':60,'new_weak_counts':30,
        'all_90_reported':len(comparison)==90,'promotion_decision_not_based_on_this_report':True})


if __name__=='__main__':main()
