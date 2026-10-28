"""Reproduce each method/family worst recorded condition without refitting.

This is a post-evaluation integrity check on already viewed final data, not new
holdout evidence. Prediction is rerun from the exact frozen model and history.
Relative L2 is recalculated directly with NumPy, without forecast_metrics.
"""
from pathlib import Path
import argparse
import json
import sys
import numpy as np
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from prefix_study import run,forecast,data
from five_study import forecast as single
from resolved_study import forecast as raw


def recheck(output):
    output=Path(output);run.verify_freeze(output)
    report=run.read(output/'comparison.json');rows=[]
    if any(r['forecast_failures'] for r in report['records']):
        raise ValueError('Tail reproduction requires a complete scored run; preserve failures separately.')
    with threadpool_limits(limits=1):
        for family in run.FAMILIES:
            for method in run.METHODS:
                result=max((r for r in report['records'] if r['family']==family and r['method']==method),
                    key=lambda r:r['mean_clean_relative_l2'])
                path=output/'models'/f"{result['id']}.json"
                if run.sha(path)!=result['frozen_model_sha256']:raise ValueError('Frozen model changed.')
                model=run.read(path);terms=model['selected']['terms'];errors=[]
                for i in range(3):
                    final=output/'final_reference'/family/f'{i}.npz'
                    relative=str(final.relative_to(output))
                    if run.sha(final)!=report['final_reference_sha256'][relative]:raise ValueError('Final bytes changed.')
                    with np.load(final,allow_pickle=False) as z:clean={k:z[k].copy() for k in ('x','t','u')}
                    obs=data.corrupt(clean,family,'final',i,model['sampling'],model['noise_level'],model['noise_seed'],model['training_rms_for_corruption_only'])
                    if method.startswith('prefix_'):
                        pred=forecast.forecast(terms,obs['x'],obs['t'],obs['prefix_t'],obs['prefix_u'])
                    elif method=='denoised_median':pred=single.forecast(terms,obs['x'],obs['t'],obs['u'][0])
                    else:pred=raw.forecast(terms,obs['x'],obs['t'],obs['u'][0])
                    stride=1 if model['sampling']=='full' else 2
                    truth=clean['u'][data.PREFIX_INTERVALS::stride,::stride]
                    error=float(np.linalg.norm(pred[1:]-truth[1:])/np.linalg.norm(truth[1:]))
                    saved=result['forecasts'][i]['clean']['relative_l2']
                    if not np.isclose(error,saved,rtol=2e-11,atol=2e-13):raise ValueError('Reproduced relative L2 differs.')
                    errors.append(error)
                mean=float(np.mean(errors))
                if not np.isclose(mean,result['mean_clean_relative_l2'],rtol=2e-11,atol=2e-13):raise ValueError('Mean differs.')
                row={'id':result['id'],'per_trajectory_relative_l2':errors,'mean_relative_l2':mean,
                     'saved_mean':result['mean_clean_relative_l2'],'difference':mean-result['mean_clean_relative_l2']}
                rows.append(row);print(result['id'],mean,flush=True)
    run.verify_freeze(output)
    receipt={'passed':True,'post_evaluation_reproduction_not_new_holdout':True,'models_rechecked':len(rows),
        'forecasts_recomputed':3*len(rows),'metric_recalculated_with_numpy':True,'no_fit_or_selection':True,
        'selection_of_checks':'worst saved condition for each family/method; verification only',
        'comparison_sha256':run.sha(output/'comparison.json'),'records':rows}
    run.write(output/'tail_reproduction.json',receipt)
    return receipt


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('--output',type=Path,default=run.OUTPUT)
    recheck(p.parse_args().output)
