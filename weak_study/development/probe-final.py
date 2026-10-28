import json,time
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from challenge_study import data
from l1_study import engine as old
from weak_study.engine import weak_block,VARIANTS
from pde_discovery.generic_experiment import reference_metrics
root=Path('/workspace/pde-discovery-l1-challenge/challenge_study/data')
prep=json.loads((root/'prepared.json').read_text())
refs=json.loads(Path('challenge_study/references.json').read_text())['references']
rows=[]
with threadpool_limits(limits=1):
 for family,sampling,level,seed in [('allen_cahn','full',.001,17),('allen_cahn','coarse',.01,17),('low_viscosity_burgers','coarse',0.,0),('ks_unstable','coarse',0.,0)]:
  counts=(1,1) if family=='allen_cahn' else (3,2)
  train=[data.corrupt(data.read_clean(root,family,'training',i),family,'training',i,sampling,level,seed,prep['training_rms'][family]) for i in range(counts[0])]
  dev=[data.corrupt(data.read_clean(root,family,'development',i),family,'development',i,sampling,level,seed,prep['training_rms'][family]) for i in range(counts[1])]
  for variant in VARIANTS:
   blocks=[weak_block(d,variant) for d in train]
   candidates=old.fit_blocks(blocks,'adaptive'); result=old.select_candidates(candidates,dev)
   selected=result['selected']; terms=[] if selected is None else selected['terms'];metrics=reference_metrics(terms,refs[family])
   r=dict(family=family,sampling=sampling,noise=level,variant=variant,equation=selected['equation'] if selected else None,recovery=metrics,dev=selected['development']['rmse'] if selected else None)
   rows.append(r);print(json.dumps(r),flush=True)
Path('weak_study/development/probe-v3-results.json').write_text(json.dumps(rows,indent=2)+'\n')
