"""Read-only retrospective diagnostics of previously viewed active study records.
No provider calls, new experiments, final arrays or true coefficients enter probes.
"""
from pathlib import Path
import sys,json,hashlib,itertools
from collections import Counter
import numpy as np
from threadpoolctl import threadpool_limits
ROOT=Path('/workspace/pde-discovery-active')
OUT=Path('/workspace/active-next-diagnostics')
sys.path.insert(0,str(ROOT))
from active_study import engine
from l1_study import engine as sparse
from weak_study import engine as weak
RESULTS=ROOT/'active_study/results/first_run'

def load(path):
 with np.load(path,allow_pickle=False) as a:
  return {k:a[k].copy() for k in ('x','t','u')}|{'boundary':'periodic'}
def episode(identity):
 return json.loads((RESULTS/'episodes'/f'{identity}.json').read_text())
def observations(ep,budget):
 tid=ep['task_id'];path=ROOT/'active_study/data/initial'/tid
 train=[load(path/f'training-{i}.npz') for i in range(3)]
 dev=[load(path/f'development-{i}.npz') for i in range(2)]
 train += [load(RESULTS/c['file']) for c in ep['provider_calls'][:budget-5]]
 return train,dev
def blocks(obs):
 return [weak.weak_block(engine.preprocess_observation(o)[0],variant=engine.VARIANT) for o in obs]
def summary(c):
 return {k:c.get(k) for k in ('support','terms','fit_success','fit_relative_error','eligible','development','failure_reason','rejection_reason','near_optimal')}
def scalar_score(terms,bs,scale):
 vals={t['name']:t['coefficient'] for t in terms};v=np.array([vals.get(n,0.) for n in engine.TERM_NAMES])
 each=[float(np.sqrt(np.mean((b['y']-b['X']@v)**2))/scale) for b in bs]
 return float(np.sqrt(np.mean(np.square(each)))),each

def main():
 out={'label':'retrospective diagnostic; already viewed old observations and saved results',
 'new_formal_experiments':False,'final_arrays_loaded':False,'worktree_modified':False,
 'candidate_rules_use_true_coefficients':False,'source_commit':'40f0241'}
 rows=json.loads((RESULTS/'comparison.json').read_text())['records']
 out['budget6_failures']=[{k:r.get(k) for k in ('id','family','equation','recovery','mean_clean_relative_l2','offline_success')} for r in rows if r['budget']==6 and r['policy'] in ('structural_disagreement','fixed_rich') and not r['offline_success']]
 out['budget9_policies']={}
 for policy in ('fixed_rich','random','ensemble_variance','structural_disagreement'):
  subset=[r for r in rows if r['budget']==9 and r['policy']==policy]
  out['budget9_policies'][policy]={'mean_set_size':float(np.mean([r['set_size'] for r in subset])), 'full_selected_exact_support':sum(r['recovery']['exact_support'] for r in subset),'count':len(subset)}
 bad=episode('task-04--noise1--structural_disagreement');train,dev=observations(bad,6)
 tb,db=blocks(train),blocks(dev);geo=engine._training_geometry(tb);dba=[sparse.validate_block(b) for b in db]
 cp=bad['checkpoints'][1]['model'];proposal=bad['checkpoints'][0]['next_proposal']
 cand=[]
 for names in [('u','u_xx'),('u','u_xxxx'),('u_xx','u_xxxx')]:
  support=tuple(engine.TERM_NAMES.index(n) for n in names)
  c=engine._alias_candidate(support,geo,{'origin':'retrospective_explicit_fixed_support'})
  cand.append(summary(engine._score_candidate(c,dba,geo['target_scale'])))
 out['budget6_failed_case']={'actions':[q['action']['id'] for q in bad['queries']],
  'full_candidate_supports':cp['full_data_candidate_supports'], 'full_candidates':[summary(c) for c in cp['candidates']],
  'near_threshold':cp['near_threshold'],'fixed_support_ols_probe':cand,
  'observed_retained_modes':[x['retained_modes'] for x in cp['diagnostics']['preprocessing']['training']],
  'development_retained_modes':[x['retained_modes'] for x in cp['diagnostics']['preprocessing']['development']],
  'acquisition_scores':[{k:s.get(k) for k in ('action_id','score','between_structure_variance','within_structure_variance','status')} for s in sorted(proposal['scores'],key=lambda s:-(s['score'] if s['score'] is not None else -1))]}
 out['task04_observed_even_library_geometry']=[]
 for tid in ['task-04--clean','task-04--noise1']:
  for policy in ['structural_disagreement','fixed_rich']:
   e=episode(tid+'--'+policy);tr,_=observations(e,6);g=engine._training_geometry(blocks(tr))
   ids=[engine.TERM_NAMES.index(n) for n in ['u','u_xx','u_xxxx']]
   singular=np.linalg.svd(g['normalized'][:,ids],compute_uv=False)
   out['task04_observed_even_library_geometry'].append({'task_id':tid,'policy':policy,
    'columns':['u','u_xx','u_xxxx'],'singular_values_over_largest':(singular/singular[0]).tolist(),
    'action':e['queries'][0]['action']['id'],'observation_only':True})
 # A single failed bootstrap replay, using exactly the old draw multiplicity.
 ep=episode('task-07--clean--structural_disagreement');train,dev=observations(ep,9);tb,db=blocks(train),blocks(dev)
 rec=next(r for r in ep['checkpoints'][-1]['model']['uncertainty']['records'] if r['status']=='failed')
 sample=[b for b,n in zip(tb,rec['multiplicities']) for _ in range(n)]
 fit=engine._fit_once(sample,[sparse.validate_block(b) for b in db])
 out['task07_failed_bootstrap']={'record':rec,'actions':[q['action']['id'] for q in ep['queries']],
  'training_modes':[engine.preprocess_observation(o)[1]['retained_modes'] for o in train],
  'fit_diagnostics':fit['diagnostics'],'failure_reason':fit['failure_reason'],
  'candidates':[{**summary(c),'diagnostics':c['diagnostics']} for c in fit['candidates']]}
 # Committee audit and fixed, observation-only full-acquired-training gate.
 # This is a diagnostic compatibility filter, not independent validation.
 audits=[]
 for row in [r for r in rows if r['budget']==9 and r['policy']=='structural_disagreement']:
  ep=episode(row['task_id']+'--structural_disagreement');cp=ep['checkpoints'][-1]['model'];train,dev=observations(ep,9)
  tb=blocks(train);geo=engine._training_geometry(tb);scale=geo['target_scale'];selected=cp['selected']
  sel_score,sel_each=scalar_score(selected['terms'],tb,scale)
  threshold=sel_score+max(.05*sel_score,.001)
  full_supports={tuple(s) for s in cp['full_data_candidate_supports']};kept=[];details=[]
  for member in cp['committee']:
   score,each=scalar_score(member['terms'],tb,scale)
   accepted=score <= threshold
   if accepted:kept.append(member['support'])
   provenance=[]
   for p in member['provenance']:
    pr=dict(p)
    if p['role']=='bootstrap':
     r=cp['uncertainty']['records'][p['replicate']]
     pr['multiplicities']=r['multiplicities'];pr['omitted_all_added']=sum(r['multiplicities'][3:])==0
     pr['omitted_added_indices']=[i for i in range(3,len(train)) if not r['multiplicities'][i]]
    provenance.append(pr)
   details.append({'support':member['support'],'full_training_weak_score':score,'per_trajectory_weak_score':each,
    'gate_accepted':accepted,'provenance':provenance})
  kept_supports={tuple(s) for s in kept};supportset={tuple(s) for s in cp['candidate_supports']}
  records=cp['uncertainty']['records']
  audits.append({'task_id':row['task_id'],'saved_selected_exact_support':row['recovery']['exact_support'],
   'selected_support':selected['support'],'selected_training_score':sel_score,'threshold':threshold,
   'before_supports':cp['candidate_supports'],'full_data_supports':cp['full_data_candidate_supports'],
   'bootstrap_only_supports':[list(s) for s in sorted(supportset-full_supports)],
   'after_gate_supports':[list(s) for s in sorted(kept_supports)],
   'bootstrap_omitting_all_added':sum(sum(r['multiplicities'][3:])==0 for r in records),
   'bootstrap_omitting_some_added':sum(any(n==0 for n in r['multiplicities'][3:]) for r in records),
   'failed_bootstrap':cp['uncertainty']['failed_replicates'], 'members':details})
 out['budget9_structural_committee_audit']=audits
 out['diagnostic_gate']={'rule':'training weak score <= full selected training weak score + max(.05*selected score,.001)',
  'training_reuse_not_independent_validation':True,'truth_used_for_rule':False,
  'before_mean_supports':float(np.mean([len(r['before_supports']) for r in audits])),
  'full_data_mean_supports':float(np.mean([len(r['full_data_supports']) for r in audits])),
  'after_mean_supports':float(np.mean([len(r['after_gate_supports']) for r in audits])),
  'selected_retained':sum(r['selected_support'] in r['after_gate_supports'] for r in audits),
  'singleton_count':sum(len(r['after_gate_supports'])==1 for r in audits),
  'bootstrap_draws':sum(len(episode(r['task_id']+'--structural_disagreement')['checkpoints'][-1]['model']['uncertainty']['records']) for r in audits),
  'draws_omitting_all_added':sum(r['bootstrap_omitting_all_added'] for r in audits),
  'draws_omitting_some_added':sum(r['bootstrap_omitting_some_added'] for r in audits)}
 (OUT/'engine_retrospective.json').write_text(json.dumps(out,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
 print(json.dumps({'budget9':out['budget9_policies'],'gate':out['diagnostic_gate'],
  'task07_failure':out['task07_failed_bootstrap']['failure_reason'],
  'task07_candidates':[{k:c.get(k) for k in ('support','fit_success','failure_reason','rejection_reason')} for c in out['task07_failed_bootstrap']['candidates']],
  'task04_fixed_supports':cand},indent=2,ensure_ascii=False))
if __name__=='__main__':
 with threadpool_limits(limits=1):main()
