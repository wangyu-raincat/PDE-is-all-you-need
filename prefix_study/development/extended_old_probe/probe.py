from pathlib import Path
import sys,json,time
import numpy as np
from threadpoolctl import threadpool_limits
ROOT=Path('/workspace/pde-discovery-prefix-five');sys.path.insert(0,str(ROOT))
from fusion_study import generators as old, data
from prefix_study.forecast import forecast_with_diagnostics
from five_study.forecast import forecast as single_forecast
from pde_discovery.generic_experiment import forecast_metrics
source=Path('/workspace/pde-discovery-fused-weak/fusion_study/results/first_run')
model=json.loads((source/'models/ks_unstable--coarse--noise1_s97--weak_resolved.json').read_text())
scale=model['training_rms_for_corruption_only'];rows=[];start=time.monotonic()
with threadpool_limits(limits=1):
 for i in range(3):
  spec,seed,amplitudes,phases=old._initial_parameters('ks_unstable','final',i)
  n=spec['reference_n'];L=spec['period'];x=spec['x_left']+np.arange(n)*L/n
  phase=2*np.pi*(x-spec['x_left'])[:,None]/L*np.asarray(spec['modes'])[None,:]+phases
  initial=.1+np.sum(amplitudes[None,:]*np.sin(phase),axis=1)
  t=np.linspace(0,22,111)
  field=old._integrate(initial,L,t,{2:-1.,4:-1.},-1.,0.,spec['max_dt'])
  stride=n//spec['observed_nx'];clean={'x':x[::stride],'t':t,'u':field[:,::stride]}
  obs=data.corrupt(clean,'ks_unstable','final',i,'coarse',.01,97,scale)
  pt=obs['t'][:6];pu=obs['u'][:6];ft=obs['t'][5:]
  last=single_forecast(model['selected']['terms'],obs['x'],ft,pu[-1])
  prefixed,diag=forecast_with_diagnostics(model['selected']['terms'],obs['x'],ft,pt,pu)
  truth=clean['u'][10::2,::2]
  row={'old_initial_index':i,'old_seed':seed,'prefix_start':float(pt[0]),'forecast_start':float(ft[0]),'forecast_end':float(ft[-1]),'full_forecast_duration':float(ft[-1]-ft[0]),'last_frame':forecast_metrics(last,truth),'prefix':forecast_metrics(prefixed,truth),'diagnostics':diag}
  rows.append(row);print(json.dumps({k:v for k,v in row.items() if k!='diagnostics'}),flush=True)
 result={'development_only':True,'old_viewed_initial_conditions_extended_to_22':True,'new_untouched_validation':False,'fixed_prefix_duration':2.,'full_future_duration':20.,'noise_level':.01,'noise_seed':97,'reference_coefficients_used_by_predictor':False,'rows':rows,'means':{k:float(np.mean([r[k]['relative_l2'] for r in rows])) for k in ('last_frame','prefix')},'seconds':time.monotonic()-start}
 Path(__file__).with_name('result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result['means']),flush=True)
