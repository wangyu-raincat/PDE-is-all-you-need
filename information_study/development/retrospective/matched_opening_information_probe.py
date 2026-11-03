"""Observed-checkpoint replay only: no new measurements or final validation."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
from threadpoolctl import threadpool_limits

SOURCE = Path('/workspace/pde-discovery-active')
IMPLEMENTATION = Path('/workspace/pde-discovery-information')
OUTPUT = Path('/workspace/active-next-diagnostics/matched_opening_information_probe.json')
sys.path.insert(0, str(IMPLEMENTATION))
from active_study import actions
from information_study import acquisition


def observed(path):
    with np.load(path, allow_pickle=False) as z:
        return {key:z[key].copy() for key in ('x','t','u')} | {'boundary':'periodic'}


def replay(identity):
    directory = SOURCE/'active_study/results/first_run'
    path = directory/f'episodes/{identity}--fixed_rich.json'
    episode = json.loads(path.read_text())
    checkpoint = next(c for c in episode['checkpoints'] if c['budget']==6)
    training = [observed(SOURCE/f'active_study/data/initial/{identity}/training-{i}.npz') for i in range(3)]
    training.append(observed(directory/episode['provider_calls'][0]['file']))
    start = time.perf_counter()
    receipt = acquisition.select_action('rich_then_information', checkpoint['model']['committee'],
        actions.make_action_pool(856000+episode['spec_index']), training,
        seed=episode['seed'], used_action_ids=[episode['queries'][0]['action']['id']],
        selected=checkpoint['model']['selected'])
    return {'task_id':identity, 'existing_budget':6, 'read_episode':str(path),
            'episode_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
            'observed_first_action':episode['queries'][0]['action']['id'],
            'retrospective':True, 'new_physical_or_formal_data_generated':False,
            'counterfactual_outcome_observed':False, 'quality_improvement_claimed':False,
            'wall_seconds':time.perf_counter()-start, 'receipt':receipt}


if __name__=='__main__':
    if OUTPUT.exists():
        raise RuntimeError('Preserve the existing retrospective receipt.')
    with threadpool_limits(limits=1):
        results = [replay('task-04--noise1'), replay('task-05--noise1')]
    with OUTPUT.open('x') as handle:
        json.dump({'scope':'retrospective candidate-only acquisition diagnostic',
                   'implementation_sha256':hashlib.sha256((IMPLEMENTATION/'information_study/acquisition.py').read_bytes()).hexdigest(),
                   'cases':results},handle,indent=2,allow_nan=False)
    for r in results:
        receipt = r['receipt']
        print(r['task_id'],receipt['status'],receipt['selected_action_id'],
              'failed',receipt['model_forecast_failures'],
              'calls',receipt['forecast_solver_calls_attempted'],
              'seconds',round(r['wall_seconds'],3))
