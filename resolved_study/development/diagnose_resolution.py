"""Bounded retrospective controlled diagnosis; old viewed final data only."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import json,hashlib,time
import numpy as np
from threadpoolctl import threadpool_limits
from resolved_study.forecast import forecast, PROTOCOL
from weak_study.data import corrupt
from pde_discovery.generic_experiment import forecast_metrics

OLD=Path('/workspace/pde-discovery-weak-improvement/weak_study')
OUT=Path(__file__).resolve().parent/'resolution_diagnosis.json'
EXACT=[{'name':n,'coefficient':-1.} for n in ('u_xx','u_xxxx','u*u_x')]


def main():
    if OUT.exists(): raise RuntimeError('Diagnostic result exists; preserve it.')
    records=[];start=time.perf_counter()
    with threadpool_limits(limits=1):
        for condition,level,seed in [('clean',0.,0),('noise1_s71',.01,71)]:
            models={m:json.loads((OLD/'results/first_run/models'/f'ks_unstable--coarse--{condition}--{m}.json').read_text()) for m in ('adaptive','weak_adaptive')}
            for i in range(3):
                path=OLD/'results/first_run/final_reference/ks_unstable'/f'{i}.npz'
                with np.load(path,allow_pickle=False) as z: clean={k:z[k].copy() for k in ('x','t','u')}
                clean['boundary']='periodic'
                observed=corrupt(clean,'ks_unstable','final',i,'coarse',level,seed,models['adaptive']['training_rms_for_corruption_only'])
                truth=clean['u'][::2,::2]
                initial_hash=hashlib.sha256(observed['u'][0].tobytes()).hexdigest()
                for source in ('reference_diagnostic_only','adaptive','weak_adaptive'):
                    terms=EXACT if source=='reference_diagnostic_only' else models[source]['selected']['terms']
                    previous=None
                    for factor in (1,2,4):
                        row={'condition':condition,'trajectory':i,'coefficient_source':source,'factor':factor,
                             'time_refinement':1,'initial_sha256':initial_hash,'reference_terms_used':source=='reference_diagnostic_only'}
                        try:
                            pred=forecast(terms,observed['x'],observed['t'],observed['u'][0],factor=factor)
                            row.update(status='evaluated',metrics=forecast_metrics(pred,truth),
                                relative_change_from_previous_factor=None if previous is None else float(np.linalg.norm(pred[1:]-previous[1:])/np.linalg.norm(pred[1:])))
                            previous=pred
                        except (ValueError,RuntimeError,FloatingPointError) as exc:row.update(status='failed',reason=str(exc))
                        records.append(row)
                        print(condition,i,source,factor,row.get('metrics',{}).get('relative_l2',row.get('reason')),flush=True)
                if condition=='clean':
                    dense_t=np.linspace(observed['t'][0],observed['t'][-1],2*(len(observed['t'])-1)+1)
                    for factor in (1,4):
                        pred=forecast(EXACT,observed['x'],dense_t,observed['u'][0],factor=factor)[::2]
                        records.append({'condition':condition,'trajectory':i,'coefficient_source':'reference_diagnostic_only',
                            'factor':factor,'time_refinement':2,'initial_sha256':initial_hash,'reference_terms_used':True,
                            'status':'evaluated','metrics':forecast_metrics(pred,truth)})
    result={'kind':'retrospective controlled diagnosis on previously viewed data','not_new_holdout':True,
            'oracle_terms_diagnostic_only':True,'conditions_predeclared_in_script':['clean','noise1_s71'],
            'records':records,'forecast_protocol':PROTOCOL,'elapsed_seconds':time.perf_counter()-start,
            'no_model_refit':True,'no_new_formal_roles_generated':True}
    OUT.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')


if __name__=='__main__':main()
