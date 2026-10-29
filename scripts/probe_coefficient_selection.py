"""Retrospective coefficient diagnostics; no final field loading or new fields.

Cached baseline candidates were frozen in the previous prefix study. New arms
use only its training/development observations. Reference coefficients are read
only after all requested decisions have been saved and hashed.
"""
from pathlib import Path
import argparse,json,hashlib,sys
from copy import deepcopy
import numpy as np
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from coefficient_study import engine,noise,selection
from prefix_study import data,engine as prefix
from weak_study import engine as weak
from five_study.engine import generate_support_medians
from pde_discovery.generic_experiment import reference_metrics


def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False)+'\n')


def run(baseline_root,output,full=False):
 output=Path(output);assert not (output/'freeze.json').exists(),'Preserve previous probe.'
 base=Path(baseline_root)/'prefix_study';records=[];input_hashes={};source_paths=[Path(__file__)]+[ROOT/n for n in ('coefficient_study/engine.py','coefficient_study/noise.py','coefficient_study/selection.py','prefix_study/engine.py','prefix_study/forecast.py','prefix_study/data.py','weak_study/engine.py','l1_study/engine.py','five_study/engine.py','five_study/initial.py','five_study/forecast.py','resolved_study/forecast.py','resolved_study/selection.py','pde_discovery/generic_engine.py','pde_discovery/generic_experiment.py','fusion_study/engine.py')]
 sources={str(p.relative_to(ROOT)):sha(p) for p in source_paths}
 models=sorted((base/'results/first_run/models').glob('*--prefix_median.json'))
 if not full:models=[p for p in models if p.name.startswith('allen_cahn--coarse--noise1_')]
 with threadpool_limits(limits=1):
  for path in models:
   old=read(path);family=old['family'];observed={}
   for role,count in [('training',3),('development',2)]:
    observed[role]=[]
    for i in range(count):
     clean=data.read_clean(base/'data',family,role,i)
     file=base/'data/clean'/family/role/f'{i}.npz';input_hashes[str(file.relative_to(baseline_root))]=sha(file)
     observed[role].append(data.corrupt(clean,family,role,i,old['sampling'],old['noise_level'],old['noise_seed'],old['training_rms_for_corruption_only']))
   train=[{k:d[k] for k in ('x','t','u','boundary')} for d in observed['training']];dev=observed['development']
   corrected,trials=engine.corrected_candidates(train);augmented,evidence=generate_support_medians(corrected)
   corrected_score=prefix.select_candidates(augmented,dev)
   arms={'prefix_baseline':{k:deepcopy(old[k]) for k in ('selected','candidates','failure_reason','near_optimal_supports')},
    'stable_selection':selection.select_candidates(old['candidates'],train,dev,block_builder=weak.weak_block),
    'noise_corrected':corrected_score,
    'noise_stable':selection.select_candidates(corrected_score['candidates'],train,dev,block_builder=noise.weak_block)}
   for method,fit in arms.items():
    ident=old['id'].replace('--prefix_median','--'+method);dest=output/'models'/f'{ident}.json'
    write(dest,{'id':ident,'family':family,'sampling':old['sampling'],'condition':old['condition'],'method':method,
      'baseline_model_sha256':sha(path),'retrospective':True,'final_fields_loaded':False,**fit})
    records.append({'id':ident,'file':str(dest.relative_to(output)),'sha256':sha(dest)})
   print('Frozen retrospective',old['id'],flush=True)
  assert sources=={str(p.relative_to(ROOT)):sha(p) for p in source_paths},'Source changed during probe'
  write(output/'freeze.json',{'records':records,'sources':sources,'input_hashes':input_hashes,
    'retrospective':True,'final_fields_loaded':False,'formal_fields_generated':False,'reference_read_after_this_freeze':True})
  references=read(base/'references.json')['references'];scores=[]
  for item in records:
   assert sha(output/item['file'])==item['sha256'];m=read(output/item['file']);selected=m['selected']
   recovery=reference_metrics(selected['terms'] if selected else [],references[m['family']])
   scores.append({k:m[k] for k in ('id','family','sampling','condition','method')}|{'selected_id':selected['candidate_id'] if selected else None,
      'development_rmse':selected['development']['rmse'] if selected else None,'terms':selected['terms'] if selected else [],
      'recovery':recovery,'reliability_selection':m.get('reliability_selection'),'failure_reason':m.get('failure_reason')})
  write(output/'comparison.json',{'records':scores,'retrospective':True,'final_fields_loaded':False,'formal_fields_generated':False,'freeze_sha256':sha(output/'freeze.json')})
  for m in engine.METHODS:
   group=[s for s in scores if s['method']==m];print(m,'exact',sum(r['recovery']['exact_support'] for r in group),'/',len(group),
     'worst_true_pct',100*max(v['relative_error'] for r in group for v in r['recovery']['coefficient_comparison'] if v['reference']!=0),flush=True)


if __name__=='__main__':
 p=argparse.ArgumentParser(__doc__);p.add_argument('--baseline-root',type=Path,default=ROOT)
 p.add_argument('--output',type=Path,default=ROOT/'coefficient_study/development/ac_probe');p.add_argument('--full',action='store_true')
 a=p.parse_args();run(a.baseline_root,a.output,a.full)
