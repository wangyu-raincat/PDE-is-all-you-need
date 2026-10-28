"""Bounded retrospective initial-uncertainty propagation, not a new holdout."""
from pathlib import Path
import sys,json,time
import numpy as np
from threadpoolctl import threadpool_limits
ROOT=Path('/workspace/pde-discovery-five-percent');sys.path.insert(0,str(ROOT))
from resolved_study.forecast import forecast
from five_study.initial import denoise_initial
from fusion_study import data
from pde_discovery.generic_experiment import forecast_metrics

def predict(terms,x,t,observed,sigma_floor,pairs=8):
    central,diag=denoise_initial(x,observed)
    n=len(x);sigma=max(diag['estimated_spatial_noise_sigma'],sigma_floor)
    mask=np.zeros(n//2+1,bool);mask[0]=True
    mask[diag['retained_positive_modes']]=True
    # Same estimated scale also governs mode selection; no named modes.
    z=np.fft.rfft(observed)/n
    mask=np.abs(z)>=sigma*np.sqrt(2*np.log(n)/n);mask[0]=True
    central=np.fft.irfft(z*mask,n=n)*n
    center_forecast=forecast(terms,x,t,central)
    rng=np.random.default_rng(np.random.SeedSequence([4301,n]))
    total=np.zeros_like(center_forecast);half=None
    for i in range(pairs):
        eta=rng.standard_normal(n)*sigma
        perturbation=np.fft.irfft(np.fft.rfft(eta)*mask,n=n)
        for sign in (-1,1):total+=forecast(terms,x,t,central+sign*perturbation)
        if i+1==pairs//2:half=total/(2*(i+1))
    averaged=total/(2*pairs)
    return center_forecast,averaged,{'pairs':pairs,'sigma_estimate':sigma,
        'retained_modes':np.flatnonzero(mask).tolist(),
        'half_vs_full_relative_l2':float(np.linalg.norm(averaged[1:]-half[1:])/np.linalg.norm(averaged[1:])),
        'conditional_mean_not_single_PDE_solution':True,'new_measurements':False}

def main():
    started=time.monotonic();source=Path('/workspace/pde-discovery-fused-weak')
    out=source/'fusion_study/results/first_run'
    model=json.loads((out/'models/ks_unstable--coarse--noise1_s97--weak_resolved.json').read_text())
    scale=model['training_rms_for_corruption_only'];pool=[]
    for i in range(3):
        clean=data.read_clean(source/'fusion_study/data','ks_unstable','training',i)
        observed=data.corrupt(clean,'ks_unstable','training',i,'coarse',.01,97,scale)
        u=observed['u'][0];n=len(u)
        pool.extend(np.abs(np.fft.rfft(u)[n//4:n//2])/np.sqrt(n))
    floor=float(np.median(pool)/np.sqrt(np.log(2)))
    records=[]
    for i in range(3):
        with np.load(out/'final_reference/ks_unstable'/f'{i}.npz',allow_pickle=False) as z:
            clean={k:z[k].copy() for k in ('x','t','u')}
        obs=data.corrupt(clean,'ks_unstable','final',i,'coarse',.01,97,scale)
        center,pred,diag=predict(model['selected']['terms'],obs['x'],obs['t'],obs['u'][0],floor)
        row={'old_final_index':i,'center':forecast_metrics(center,clean['u'][::2,::2]),
             'ensemble':forecast_metrics(pred,clean['u'][::2,::2]),'diagnostics':diag}
        records.append(row);print(json.dumps(row),flush=True)
    report={'retrospective':True,'single_bounded_probe':True,'pairs_fixed_before_scores':8,
            'new_formal_fields':False,'reference_equation_used':False,'training_noise_floor':floor,
            'records':records,'means':{key:float(np.mean([r[key]['relative_l2'] for r in records]))
                 for key in ('center','ensemble')},'seconds':time.monotonic()-started}
    target=Path(__file__).with_name('result.json');target.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report['means']),flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=1):main()
