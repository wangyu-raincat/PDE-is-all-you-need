from pathlib import Path
import sys,json,numpy as np
from threadpoolctl import threadpool_limits
sys.path.insert(0,'/workspace/pde-discovery-fused-weak')
from fusion_study import data
from resolved_study.forecast import forecast
from pde_discovery.generic_experiment import forecast_metrics
root=Path('/workspace/pde-discovery-fused-weak');prep=json.loads((root/'fusion_study/data/prepared.json').read_text());scale=prep['training_rms']['ks_unstable']
terms=[{'name':'u_xx','coefficient':-1.},{'name':'u_xxxx','coefficient':-1.},{'name':'u*u_x','coefficient':-1.}]
def denoise(u):
 n=len(u);z=np.fft.rfft(u)/n;k=np.arange(len(z)); hi=z[(k>=n//4)&(k<n//2)]
 tau=float(np.median(np.abs(hi))/np.sqrt(np.log(2)))
 threshold=tau*np.sqrt(2*np.log(n))
 mask=np.abs(z)>=threshold;mask[0]=True
 filtered=np.fft.irfft(z*mask,n=n)*n
 return filtered,{'tau_estimate':tau,'threshold':threshold,'retained_positive_modes':np.flatnonzero(mask)[1:].tolist(),'initial_relative_change':float(np.linalg.norm(filtered-u)/np.linalg.norm(u))}
rows=[]
with threadpool_limits(limits=1):
 for i in range(2):
  clean=data.read_clean(root/'fusion_study/data','ks_unstable','development',i)
  for sampling in ('full','coarse'):
   stride=1 if sampling=='full' else 2
   for seed in (61,97):
    obs=data.corrupt(clean,'ks_unstable','development',i,sampling,.01,seed,scale)
    filtered,meta=denoise(obs['u'][0])
    row={'diagnostic':'known_reference_coefficients_only_old_dev','family':'ks_unstable','development_index':i,'sampling':sampling,'noise_seed':seed,'denoise':meta}
    for name,initial in [('raw',obs['u'][0]),('hard',filtered)]:
     predicted=forecast(terms,obs['x'],obs['t'],initial)
     row[name]={'prediction_relative_l2':forecast_metrics(predicted,clean['u'][::stride,::stride])['relative_l2'],'initial_relative_l2':float(np.linalg.norm(initial-clean['u'][0,::stride])/np.linalg.norm(clean['u'][0,::stride]))}
    rows.append(row);print(json.dumps(row),flush=True)
    Path('/workspace/five-percent-initial-diagnostic.json').write_text(json.dumps({'retrospective_diagnostic_only':True,'reference_coefficients_not_runtime_method':True,'observations':'old development first frames only fordenoising; old clean future usedsolelyfor scoring','records':rows},indent=2)+'\n')
