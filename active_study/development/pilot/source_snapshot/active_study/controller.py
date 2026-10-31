"""Budgeted observation-only loop, with an injected selected-query capability.

This module never imports the reference provider, sees its task specification,
or uses final scores. Model snapshots are not claims of a unique physical law.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import time
import numpy as np

from pde_discovery import generic_engine
from prefix_study.forecast import validate_prefix
from . import actions, acquisition, engine

POLICIES = ('fixed_rich', 'random', 'ensemble_variance', 'structural_disagreement')
INITIAL_TRAINING = 3
INITIAL_DEVELOPMENT = 2
MAX_ADDITIONS = 4
PROTOCOL = {
    'version': 1, 'policies': list(POLICIES),
    'initial_training': INITIAL_TRAINING, 'initial_development': INITIAL_DEVELOPMENT,
    'maximum_additional_experiments': MAX_ADDITIONS,
    'total_experiment_budgets': [5, 6, 7, 8, 9],
    'measurement_cost': 'future field plus history except shared endpoint; all training and development charged',
    'query_failure_cost': 'a legal started query consumes one attempt and its full reserved measurement budget',
    'invalid_action': 'reject before provider invocation; no free retries or replacement queries',
    'candidate_rollouts': 'model computation only, never observed experiments',
    'repeated_actions': False,
    'benchmark_stopping': 'run every budget checkpoint, no truth-based or final-based stopping',
    'one_candidate': 'candidate only, never proof of uniqueness; permit coverage exploration',
    'initial_fit_cache': 'same common initial observations and ensemble seed shared by all four policies',
    'bootstrap_seed': 'same seed+round counter for every policy within the paired task',
    'final_role_access': False, 'private_reference_spec_access': False,
}


def observation_view(value):
    """Only measured fields and their coordinates cross the learner boundary."""
    if not isinstance(value, dict):
        raise ValueError('An observation dictionary is required.')
    if any(np.iscomplexobj(np.asarray(value.get(k))) for k in
           ('x', 't', 'u', 'prefix_t', 'prefix_u')):
        raise ValueError('Observation coordinates, fields, and history must be real-valued.')
    view = {k: value[k] for k in ('x', 't', 'u', 'boundary') if k in value}
    x, t, u = generic_engine._validate(view)
    _, _, pt, pu = validate_prefix(x, t, value.get('prefix_t'), value.get('prefix_u'))
    if not np.array_equal(u[0], pu[-1]):
        raise ValueError('The shared history endpoint must be exactly the same observation.')
    return {'x': x.copy(), 't': t.copy(), 'u': u.copy(), 'boundary': 'periodic',
            'prefix_t': pt.copy(), 'prefix_u': pu.copy()}


def observation_digest(value):
    d = observation_view(value)
    digest = hashlib.sha256()
    for key in ('x', 't', 'u', 'prefix_t', 'prefix_u'):
        a = np.asarray(d[key], dtype='<f8')
        digest.update(key.encode()); digest.update(str(a.shape).encode()); digest.update(a.tobytes())
    return digest.hexdigest()


def measurement_cost(value):
    d = observation_view(value)
    return int(d['u'].size + d['prefix_u'][:-1].size)


def _seed(value):
    if type(value) is not int or not 0 <= value <= 2**31 - 100:
        raise ValueError('Need an integer seed in the bounded declared range.')
    return value


def _model_failure(exc):
    return {'selected': None, 'candidates': [], 'committee': [], 'candidate_supports': [],
            'uncertainty': {'calibrated': False, 'probability_claimed': False},
            'diagnostics': {'status': 'insufficient', 'failure_reason': str(exc),
                            'unique_physical_equation_established': False}}


def run_episode(training, development, pool, query, *, policy, seed=0,
                max_additions=MAX_ADDITIONS, initial_model=None):
    """Run all allowed rounds; callback sees only the chosen action and index.

The callback returns {'observation': ..., 'receipt': ...}. Neither a clean
future field nor provider parameters are retained or forwarded to discovery.
Every acquisition and legal query attempt has an explicit receipt.
"""
    start = time.monotonic(); seed = _seed(seed)
    if policy not in POLICIES or type(max_additions) is not int or not 0 <= max_additions <= MAX_ADDITIONS:
        raise ValueError('Unknown policy or out-of-budget addition count.')
    if not callable(query) or len(training) != INITIAL_TRAINING or len(development) != INITIAL_DEVELOPMENT:
        raise ValueError('Need three training trajectories, two development trajectories and a query capability.')
    train = [observation_view(d) for d in training]
    dev = [observation_view(d) for d in development]
    pool = [actions.validate_action(a) for a in pool]
    if not pool or len({a['id'] for a in pool}) != len(pool):
        raise ValueError('The action pool must contain unique identities.')
    by_id = {a['id']: a for a in pool}
    shapes = {(tuple(d['u'].shape), tuple(d['prefix_u'].shape)) for d in train + dev}
    if len(shapes) != 1 or any(not np.array_equal(d['x'], train[0]['x'])
                              or not np.array_equal(d['t'], train[0]['t'])
                              or not np.array_equal(d['prefix_t'], train[0]['prefix_t']) for d in train + dev):
        raise ValueError('All experiments must use the same declared measured geometry.')
    per_query_cost = measurement_cost(train[0])
    initial_cost = sum(measurement_cost(d) for d in train + dev)
    used = []; ledger = []; checkpoints = []; terminal_reason = None
    for round_index in range(max_additions + 1):
        try:
            model = deepcopy(initial_model) if round_index == 0 and initial_model is not None else engine.discover(
                train, dev, seed=seed + round_index)
        except (ValueError, RuntimeError, FloatingPointError, np.linalg.LinAlgError) as exc:
            model = _model_failure(exc)
        checkpoint = {
            'budget': INITIAL_TRAINING + INITIAL_DEVELOPMENT + len(ledger),
            'training_trajectories': len(train), 'development_trajectories': len(dev),
            'charged_measurement_scalars': initial_cost + len(ledger) * per_query_cost,
            'available_measurement_scalars': initial_cost + sum(q['returned_measurement_scalars'] for q in ledger),
            'training_hashes': [observation_digest(d) for d in train],
            'development_hashes': [observation_digest(d) for d in dev],
            'model': model, 'next_proposal': None,
            'final_used_for_selection': False,
        }
        checkpoints.append(checkpoint)
        if round_index == max_additions:
            break
        try:
            proposal = acquisition.select_action(policy, model.get('committee', []), pool, train,
                seed=seed + round_index, used_action_ids=tuple(used), cost_per_action=per_query_cost)
            checkpoint['next_proposal'] = deepcopy(proposal)
            chosen = actions.validate_action(proposal.get('selected_action'))
            if chosen['id'] not in by_id or chosen != by_id[chosen['id']] or chosen['id'] in used:
                raise ValueError('Selection is not an unused declared action.')
        except (ValueError, RuntimeError, FloatingPointError, np.linalg.LinAlgError, TypeError) as exc:
            terminal_reason = 'acquisition_failed: ' + str(exc)
            checkpoint['acquisition_failure'] = terminal_reason
            break
        # Charge a started legal query before executing it, including failures.
        used.append(chosen['id'])
        entry = {'query_index': round_index, 'action': chosen,
                 'action_sha256': actions.action_digest(chosen), 'status': 'started',
                 'charged_experiments': 1, 'charged_measurement_scalars': per_query_cost,
                 'returned_measurement_scalars': 0, 'observation_sha256': None}
        ledger.append(entry)
        try:
            response = query(deepcopy(chosen), round_index)
            observed = observation_view(response['observation'])
            if (measurement_cost(observed) != per_query_cost or not np.array_equal(observed['x'], train[0]['x'])
                    or not np.array_equal(observed['t'], train[0]['t'])
                    or not np.array_equal(observed['prefix_t'], train[0]['prefix_t'])):
                raise ValueError('A returned experiment changed the paid observation budget or geometry.')
            entry.update(status='observed', observation_sha256=observation_digest(observed),
                         returned_measurement_scalars=per_query_cost,
                         provider_receipt=deepcopy(response.get('receipt', {})))
            train.append(observed)
        except (ValueError, RuntimeError, FloatingPointError, OSError, KeyError, TypeError) as exc:
            entry.update(status='failed', failure_reason=str(exc))
    result = {'policy': policy, 'seed': seed, 'initial_training_trajectories': INITIAL_TRAINING,
              'initial_development_trajectories': INITIAL_DEVELOPMENT,
              'maximum_additions': max_additions, 'initial_measurement_scalars': initial_cost,
              'per_experiment_measurement_scalars': per_query_cost,
              'total_experiment_attempts': INITIAL_TRAINING + INITIAL_DEVELOPMENT + len(ledger),
              'charged_measurement_scalars': initial_cost + len(ledger) * per_query_cost,
              'queries': ledger, 'checkpoints': checkpoints,
              'status': 'completed_budget' if terminal_reason is None else 'stopped_failure',
              'terminal_reason': terminal_reason, 'compute_seconds': time.monotonic() - start,
              'final_used': False, 'reference_coefficients_used': False,
              'benchmark_does_not_stop_using_truth': True, 'unique_physical_equation_established': False}
    # Scientific snapshots must remain serializable and finite, including failure paths.
    json.dumps(result, allow_nan=False)
    return result
