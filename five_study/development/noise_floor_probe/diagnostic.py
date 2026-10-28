from pathlib import Path
import sys,json,numpy as np
from threadpoolctl import threadpool_limits
sys.path.insert(0,'/workspace/pde-discovery-five-percent')
from fusion_study import data
from resolved_study.forecast import forecast
from pde_discovery.generic_experiment import forecast_metrics
source=Path('/workspace/pde-discovery-fused-weak');out=source/'fusion_study/results/first_run'
model=json.loads((out/'models/ks_unstable--coarse--noise1_s97--weak_resolved.json').read_text())
reference=json.loads((source/'fusion_study/references.json').read_text())['references']['ks_unstable']
scale=model['training_rms_for_corruption_only']
training=[]
for i in range(3):
 clean=data.read_clean(source/'fusion_study/data','ks_unstable','training',i)
 training.append(data.corrupt(clean,'ks_unstable','training',i,'coarse',.01,97,scale))
# The estimator sees observed training initial frames only, not clean or scale.
pooled=[];individual=[]
for obs in training:
 u=obs['u'][0];n=len(u);z=np.fft.rfft(u)/n
 samples=np.abs(z[n//4:n//2])*np.sqrt(n)
 pooled.extend(samples);individual.append(float(np.median(samples)/np.sqrt(np.log(2))))
floor=float(np.median(pooled)/np.sqrt(np.log(2)))
records=[]
with threadpool_limits(limits=1):
 for i in range(3):
  with np.load(out/'final_reference/ks_unstable'/f'{i}.npz',allow_pickle=False) as z:clean={k:z[k].copy() for k in ('x','t','u')}
  obs=data.corrupt(clean,'ks_unstable','final',i,'coarse',.01,97,scale)
  u=obs['u'][0];n=len(u);spectrum=np.fft.rfft(u)/n
  frame=float(np.median(np.abs(spectrum[n//4:n//2]))*np.sqrt(n/np.log(2)))
  used=max(frame,floor);threshold=used*np.sqrt(2*np.log(n)/n)
  mask=np.abs(spectrum)>=threshold;mask[0]=True
  filtered=np.fft.irfft(spectrum*mask,n=n)*n
  row={'old_final_index':i,'training_floor':floor,'frame_sigma':frame,'used_sigma':used,'threshold':threshold,'retained_positive_modes':np.flatnonzero(mask)[1:].tolist(),'initial_relative_change':float(np.linalg.norm(filtered-u)/np.linalg.norm(u))}
  for name,terms in [('frozen_fitted',model['selected']['terms']),('reference_diagnostic_only',reference)]:
   pred=forecast(terms,obs['x'],obs['t'],filtered)
   row[name]=forecast_metrics(pred,clean['u'][::2,::2])['relative_l2']
  records.append(row);print(json.dumps(row),flush=True)
report={'retrospective_diagnostic_only':True,'no_new_formal_fields':True,'condition':'ks_unstable--coarse--noise1_s97',
 'training_floor_estimator':'median pooled abs(rFFT(u0)/N)*sqrt(N), floorN/4<=k<floorN/2, divided by sqrt(log2)',
 'training_trajectory_sigma_estimates':individual,'pooled_training_initial_samples':len(pooled),'training_sigma_floor':floor,
 'runtime_rule':'max(single-initial-frame estimate, training-estimated floor); identical universal sqrt(2logN/N) threshold; DC preserved; no known mode selection',
 'records':records,'means':{k:float(np.mean([r[k] for r in records])) for k in ('frozen_fitted','reference_diagnostic_only')}}
Path('/workspace/five-percent-training-floor-diagnostic.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report['means']),flush=True)
