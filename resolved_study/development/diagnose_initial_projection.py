"""Separate old-grid initial projection from later internal-grid resolution."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import json,hashlib
import numpy as np
from threadpoolctl import threadpool_limits
from resolved_study.forecast import forecast
from weak_study.data import corrupt
from pde_discovery.generic_experiment import forecast_metrics
from resolved_study.development.diagnose_resolution import OLD, EXACT

OUT=Path(__file__).resolve().parent/'initial_projection_diagnosis.json'


def main():
    if OUT.exists():raise RuntimeError('Preserve existing diagnostic evidence.')
    records=[]
    with threadpool_limits(limits=1):
        for condition,level,seed in [('clean',0.,0),('noise1_s71',.01,71)]:
            model=json.loads((OLD/'results/first_run/models'/f'ks_unstable--coarse--{condition}--weak_adaptive.json').read_text())
            for i in range(3):
                with np.load(OLD/'results/first_run/final_reference/ks_unstable'/f'{i}.npz',allow_pickle=False) as z:
                    clean={k:z[k].copy() for k in ('x','t','u')}
                clean['boundary']='periodic'
                observed=corrupt(clean,'ks_unstable','final',i,'coarse',level,seed,model['training_rms_for_corruption_only'])
                raw=observed['u'][0];n=len(raw);modes=np.rint(np.fft.fftfreq(n)*n)
                projected=np.fft.ifft(np.fft.fft(raw)*(np.abs(modes)<=n//3)).real
                for source,terms in [('reference_diagnostic_only',EXACT),('weak_adaptive',model['selected']['terms'])]:
                    pred=forecast(terms,observed['x'],observed['t'],projected,factor=4)
                    row={'condition':condition,'trajectory':i,'coefficient_source':source,
                        'factor':4,'initial_input':'old_observation_grid_two_thirds_projection',
                        'reference_terms_used':source=='reference_diagnostic_only',
                        'raw_initial_sha256':hashlib.sha256(raw.tobytes()).hexdigest(),
                        'projected_initial_sha256':hashlib.sha256(projected.tobytes()).hexdigest(),
                        'initial_projection_relative_l2':float(np.linalg.norm(projected-raw)/np.linalg.norm(raw)),
                        'metrics':forecast_metrics(pred,clean['u'][::2,::2])}
                    records.append(row)
                    print(condition,i,source,row['initial_projection_relative_l2'],row['metrics']['relative_l2'],flush=True)
    OUT.write_text(json.dumps({'kind':'retrospective initial-projection control','not_new_holdout':True,
        'no_new_formal_roles_generated':True,'records':records},indent=2)+'\n')


if __name__=='__main__':main()
