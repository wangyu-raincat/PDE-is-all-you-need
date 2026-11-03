"""Retrospective information-geometry diagnostic; no counterfactual measurements.

Reads existing observed training arrays and previously fitted candidate equations.
Never imports/calls provider, never generates reference or formal experiment data,
and never changes the existing repository. Results are design diagnostics only.
"""
from __future__ import annotations
import json
from pathlib import Path
import sys
import time
import numpy as np
from threadpoolctl import threadpool_limits
from scipy.integrate import newton_cotes

ROOT = Path('/workspace/pde-discovery-active')
sys.path.insert(0, str(ROOT))
from active_study import acquisition, actions, engine
from weak_study.engine import _spectra

OUT = Path('/workspace/active-next-diagnostics')
EPS = 1e-6


def observed(path):
    with np.load(path, allow_pickle=False) as z:
        return {key: z[key].copy() for key in ('x', 't', 'u')} | {'boundary': 'periodic'}


def short_features(obs):
    filtered, _ = engine.preprocess_observation(obs)
    short = dict(filtered)
    short['t'] = short['t'][:11]-short['t'][0]
    short['u'] = short['u'][:11]
    return feature_only(short)


def feature_only(data):
    """Existing weak Fourier features on an 11-frame .1 interval, no target.

    The production weak transform requires >=16 samples for target fitting;
    this diagnostic computes its same eight-subinterval feature integrals only.
    No data derivatives or observed future values are manufactured.
    """
    x,t,u=data['x'],data['t'],data['u']
    n=len(x); modes=np.arange(n//4+1)
    wave=2*np.pi*modes/(n*(x[1]-x[0]))
    uh,square,cube=_spectra(u)
    uh,square,cube=uh[:,modes],square[:,modes],cube[:,modes]
    constant=np.zeros_like(uh);constant[:,0]=1
    library=np.stack([constant,uh,square,cube,*[(1j*wave)**order*uh for order in range(1,5)],.5j*wave*square],axis=-1)
    features=[]
    for start in range(0,len(t)-8,2):
        local=t[start:start+9];half=(local[-1]-local[0])/2
        z=(local-(local[-1]+local[0])/2)/half
        weights=newton_cotes(8,equal=1)[0]*(local[1]-local[0])
        w=weights*(1-z*z)**4
        X=np.einsum('t,tmj->mj',w,library[start:start+9])/w.sum()
        features.extend([X[:1].real,np.sqrt(2)*X[1:].real,np.sqrt(2)*X[1:].imag])
    return np.vstack(features)


def gram(X):
    return X.T@X/len(X)


def compute(task, budget):
    source = ROOT/'active_study/results/first_run'
    ep = json.loads((source/f'episodes/{task}--structural_disagreement.json').read_text())
    cp = next(cp for cp in ep['checkpoints'] if cp['budget'] == budget)
    model = cp['model']
    train = [observed(ROOT/f'active_study/data/initial/{task}/training-{i}.npz') for i in range(3)]
    train.extend(observed(source/r['file']) for r in ep['provider_calls'][:budget-5] if r['status']=='observed')
    Xs = [short_features(d) for d in train]
    full_G = sum(map(gram, Xs))
    scales = np.sqrt(np.diag(full_G)/len(Xs))
    scale_floor = max(scales.max()*1e-10, 1e-12)
    scales = np.maximum(scales, scale_floor)
    minimum = min(len(c['support']) for c in model['candidates'] if c.get('near_optimal'))
    representatives = [c for c in model['candidates'] if c.get('near_optimal') and len(c['support'])==minimum]
    minimum_support_union = [n for n in engine.TERM_NAMES if any(n in c['support'] for c in representatives)]
    union = list(engine.TERM_NAMES)
    columns = [engine.TERM_NAMES.index(n) for n in union]
    G0 = full_G[np.ix_(columns, columns)]/np.outer(scales[columns], scales[columns])
    ridge = EPS*np.eye(len(columns))
    log0 = np.linalg.slogdet(G0+ridge)[1]
    pool = actions.make_action_pool(856000+ep['spec_index'])
    used = {q['action']['id'] for q in ep['queries'][:budget-5]}
    rows = []
    started = time.perf_counter()
    for a in pool:
        if a['id'] in used:
            continue
        contributions, failures = [], []
        for c in representatives:
            try:
                pred = acquisition.candidate_forecast(c['terms'], train[0]['x'], np.linspace(0,.2,21),
                                                       actions.action_field(a, train[0]['x']))
                # Candidate-only first usable field window, matching observed
                # arrays whose first frame is already after the .1 prefix.
                block = feature_only({'x':train[0]['x'], 't':np.linspace(0,.1,11),
                                      'u':pred[10:], 'boundary':'periodic'})
                X = block[:, columns]/scales[columns]
                contributions.append(gram(X))
            except (ValueError, RuntimeError, FloatingPointError, np.linalg.LinAlgError) as exc:
                failures.append(str(exc))
        if failures:
            rows.append({'action':a['id'], 'failures':failures, 'score':None})
            continue
        gains = [(np.linalg.slogdet(G0+g+ridge)[1]-log0)/len(columns) for g in contributions]
        rows.append({'action':a['id'], 'score':float(np.mean(gains)),
                     'minimum_normalized_eigenvalue':float(min(np.linalg.eigvalsh(G0+g)[0] for g in contributions)),
                     'per_candidate_logdet_gain':list(map(float,gains))})
    rows.sort(key=lambda r:(-(r['score'] if r['score'] is not None else -1e100),r['action']))
    proposal = cp.get('next_proposal') or {}
    return {'task':task,'budget':budget,'scope':'retrospective design diagnostic, not validation',
            'conditioning_window':'observed first .1; candidate .1 to .2 from lawful IC',
            'normalization':'fixed observed training column RMS, shared across all actions',
            'proxy':'mean normalized logdet Gram gain over full-fit minimum-complexity near models',
            'regularization':EPS,'union_terms':union,'minimum_support_union':minimum_support_union,
            'representative_count':len(representatives),
            'full_committee_count':len(model['committee']), 'full_structure_count':len(model['candidate_supports']),
            'training_normalized_eigenvalues':np.linalg.eigvalsh(G0).tolist(),
            'old_next_action':proposal.get('selected_action_id'), 'scores':rows,
            'cpu_seconds':time.perf_counter()-started,'candidate_rollouts_only':True,
            'observed_counterfactual_outcomes':False,'reference_used_for_action_score':False}


if __name__=='__main__':
    with threadpool_limits(limits=1):
        results = [compute('task-04--noise1',5), compute('task-05--noise1',6)]
    (OUT/'acquisition_information_probe.json').write_text(json.dumps(results,indent=2,allow_nan=False)+'\n')
    for r in results:
        print(r['task'],r['budget'],'old',r['old_next_action'],'new information top',[(x['action'],round(x['score'],4)) for x in r['scores'][:5]],'seconds',round(r['cpu_seconds'],3))
