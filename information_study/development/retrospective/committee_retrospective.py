"""Retrospective observed-data committee and numerical candidate diagnostics.
This is not a new held-out validation, confidence calibration or model release.
"""
import json,sys
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
sys.path.insert(0,'/workspace/active-next-diagnostics')
from engine_retrospective import ROOT,OUT,RESULTS,episode,observations,blocks,summary,scalar_score
from active_study import engine
from l1_study import engine as sparse

def eligible_minimum(member,model):
 for p in member['provenance']:
  pool=model['full_data_candidate_supports'] if p['role']=='full_training' else model['uncertainty']['records'][p['replicate']]['candidate_supports']
  if pool and member['support'] in pool and len(member['support'])==min(map(len,pool)):return True
 return False

def main():
 rows=[]
 for budget in [5,6,9]:
  for path in sorted((RESULTS/'episodes').glob('*structural_disagreement.json')):
   ep=json.loads(path.read_text());model=ep['checkpoints'][budget-5]['model'];train,_=observations(ep,budget);tb=blocks(train);geometry=engine._training_geometry(tb)
   selected=model['selected']
   if selected is None:
    rows.append({'id':ep['task_id'],'budget':budget,'before_supports':model['candidate_supports'], 'minimum_near_supports':[], 'minimum_and_observed_gate_supports':[], 'selected_retained_in_both':False,'status':'original_insufficient','failure_reason':model.get('failure_reason')})
    continue
   score,_=scalar_score(selected['terms'],tb,geometry['target_scale']);limit=score+max(.05*score,.001)
   small=[];both=[]
   for c in model['committee']:
    if eligible_minimum(c,model):
     small.append(c);cscore,_=scalar_score(c['terms'],tb,geometry['target_scale'])
     if cscore<=limit:both.append(c)
   rows.append({'id':ep['task_id'],'budget':budget,'before_supports':model['candidate_supports'],
    'minimum_near_supports':[list(s) for s in sorted({tuple(c['support']) for c in small})],
    'minimum_and_observed_gate_supports':[list(s) for s in sorted({tuple(c['support']) for c in both})],
    'minimum_members':len(small),'minimum_and_gate_members':len(both),'selected_support':selected['support'],
    'selected_retained_in_both':any(c['support']==selected['support'] for c in both)})
 aggregates={}
 for b in [5,6,9]:
  subset=[r for r in rows if r['budget']==b]
  aggregates[b]={k:float(np.mean([len(r[k]) for r in subset])) for k in ['before_supports','minimum_near_supports','minimum_and_observed_gate_supports']}
  aggregates[b]['original_insufficient']=sum(r.get('status')=='original_insufficient' for r in subset)
  aggregates[b]['selected_retained_in_both']=sum(r['selected_retained_in_both'] for r in subset)
 ep=episode('task-07--clean--structural_disagreement');train,dev=observations(ep,9);tb,db=blocks(train),blocks(dev)
 record=next(r for r in ep['checkpoints'][-1]['model']['uncertainty']['records'] if r['status']=='failed')
 sample=[b for b,n in zip(tb,record['multiplicities']) for _ in range(n)];geometry=engine._training_geometry(sample)
 fit=engine._fit_once(sample,[sparse.validate_block(b) for b in db]);candidate=fit['candidates'][0]
 contribution={t['name']:float(t['coefficient']*geometry['scales'][engine.TERM_NAMES.index(t['name'])]/geometry['target_scale']) for t in candidate['terms']}
 reduced=[]
 for dropped in candidate['support']:
  keep=tuple(engine.TERM_NAMES.index(n) for n in candidate['support'] if n!=dropped)
  c=engine._alias_candidate(keep,geometry,{'origin':'retrospective_drop_one_OLS_refit'})
  reduced.append({'dropped':dropped,'candidate':summary(engine._score_candidate(c,[sparse.validate_block(b) for b in db],geometry['target_scale']))})
 omission=[]
 for tid in ['task-00--clean','task-08--clean']:
  ep=episode(tid+'--structural_disagreement');m=ep['checkpoints'][-1]['model'];modes=[r['retained_modes'] for r in m['diagnostics']['preprocessing']['training']]
  alternatives=[]
  for c in m['committee']:
   if c['support']==m['selected']['support']:continue
   for p in c['provenance']:
    if p['role']=='bootstrap':
     rec=m['uncertainty']['records'][p['replicate']]
     alternatives.append({'support':c['support'],'replicate':p['replicate'],'multiplicities':rec['multiplicities'],
      'sampled_mode_union':sorted({k for modes_i,n in zip(modes,rec['multiplicities']) if n for k in modes_i})})
  omission.append({'task_id':tid,'actions':[q['action']['id'] for q in ep['queries']],'all_training_modes':modes,'alternatives':alternatives})
 result={'label':'retrospective, existing observed arrays only, no new final evidence',
  'prior':'compact acquisition ensemble uses minimum-cardinality near supports from each fit, retaining all equal-complexity aliases',
  'catalog':'full broad candidates and uncertainty receipts must remain unchanged and available',
  'diagnostic_gate':'existing whole-training weak residual <= selected score + max(.05*score,.001); not independent validation',
  'singleton_is_not_identifiability':True,'truth_used_in_probe_rule':False,
  'aggregate':aggregates,'records':rows,'task07_drop_one':{'old_candidate':summary(candidate),'normalized_contributions':contribution,'debiased_zero_threshold':sparse.DEBIAS_ZERO_THRESHOLD,'reduced_candidates':reduced},
  'omission_of_informative_designs':omission}
 (OUT/'committee_retrospective.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
 print(json.dumps({'aggregate':aggregates,'task07_drop_one':result['task07_drop_one'],'omissions':omission},indent=2))
if __name__=='__main__':
 with threadpool_limits(limits=1):main()
