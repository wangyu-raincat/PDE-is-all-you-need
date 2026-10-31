"""Freeze a complete budgeted policy study before generating final references."""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from threadpoolctl import threadpool_limits

from pde_discovery.generic_experiment import reference_metrics, forecast_metrics
from l1_study.engine import TERM_NAMES
from . import actions, provider, engine, acquisition, controller

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT/'active_study'
DATA = STUDY/'data'
OUTPUT = STUDY/'results/first_run'
POLICIES = controller.POLICIES
INDICES = tuple(range(12))
NOISES = (('clean', 0.), ('noise1', .01))
BUDGETS = tuple(range(5, 10))
FINAL_COUNT = 3
PROTOCOL = {
    'version': 1, 'indices': list(INDICES), 'noise_conditions': list(NOISES),
    'policies': list(POLICIES), 'budgets': list(BUDGETS), 'formal_conditions': 24,
    'independent_parameter_tasks': 12, 'final_trajectories_per_parameter_task': FINAL_COUNT,
    'same_clean_final_shared_between_noise_conditions': True,
    'initial_observations': '3 mode-one training plus2 mode-one development, all5 charged',
    'pool_seed': '856000+spec_index', 'initial_action_seed': '857000+spec_index',
    'learner_seed': '858000+spec_index, identical for paired policies and noise conditions',
    'query_noise_seed': 'SHA256(spec_index,role,action_identity) modulo2^31; paired across policies',
    'final_action_seed': '859000+spec_index, predetermined random mixtures16:19',
    'final_noise_seed': '860000+100*spec_index+final_index',
    'global_freeze': 'all96 policy episodes and480 declared budget slots frozen before generating36 clean final references; unavailable checkpoints remain failed slots',
    'final_release_tracking': 'data/final_release.json created exclusively before first final generation; subsequent output directories are reproductions, not unseen validation',
    'candidate_policy': 'structural_disagreement',
    'primary_metrics': 'strict support recovery, extra/missing terms, each true coefficient error versus total experiment budget',
    'offline_success': 'exact support AND worst true coefficient error<=10% AND three-trajectory mean forecast relativeL2<=5%',
    'success_target_fraction_for_cost_comparison': .8,
    'cost_savings': 'only at same predeclared80% offline-success target; unavailable if either policy never reaches it',
    'truth_based_first_success_budget_is_online_stopping': False,
    'uncertainty_metrics': 'support-set coverage and size, singleton-wrong diagnostic, positive/negative term Brier separately; frequencies are not calibrated probabilities',
    'reference_coefficients_used_for_policy_or_fit': False,
    'automatic_merge': False,
    'scope': 'new controlled synthetic initial-condition task, not a fix or replacement for the old KS5.4104% result',
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def canonical(value):
    return json.loads(json.dumps(value, allow_nan=False))


def sources():
    names = [f'active_study/{name}.py' for name in
             ('actions', 'provider', 'engine', 'acquisition', 'controller', 'forecast', 'run')]
    names += ['active_study/protocol.json', 'active_study/references.json',
              'coefficient_study/generators.py', 'coefficient_study/noise.py',
              'weak_study/engine.py', 'l1_study/engine.py', 'pde_discovery/generic_engine.py',
              'pde_discovery/generic_experiment.py', 'prefix_study/forecast.py',
              'resolved_study/forecast.py', 'five_study/initial.py']
    return {n: sha(ROOT/n) for n in sorted(names)}


def task_id(index, noise_label):
    return f'task-{index:02d}--{noise_label}'


def expected_episode_ids():
    return {task_id(i, label)+'--'+policy for i in INDICES for label, _ in NOISES for policy in POLICIES}


def noise_seed(spec, action):
    key = f"{spec['role']}:{spec['index']}:"+actions.action_digest(action)
    return int(hashlib.sha256(key.encode()).hexdigest()[:8], 16) % 2**31


def save_observation(path, observation):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    obs = controller.observation_view(observation)
    np.savez_compressed(path, **{k: obs[k] for k in ('x', 't', 'u', 'prefix_t', 'prefix_u')})


def load_observation(path):
    with np.load(path, allow_pickle=False) as archive:
        d = {k: archive[k].copy() for k in ('x', 't', 'u', 'prefix_t', 'prefix_u')}
    return controller.observation_view(d | {'boundary': 'periodic'})


def _verify_paid_observation(observation, receipt, action):
    """Bind a paid record to its public geometry, selected action and raw bytes."""
    full_t = np.linspace(0., 1.1, provider.FULL_FRAMES)
    if (observation['u'].shape != (provider.FULL_FRAMES-provider.PREFIX_INTERVALS, provider.OBSERVED_NX)
            or not np.allclose(observation['x'], np.arange(provider.OBSERVED_NX)*actions.PERIOD/provider.OBSERVED_NX,
                               rtol=0, atol=1e-12)
            or not np.array_equal(observation['t'], full_t[provider.PREFIX_INTERVALS:])
            or not np.array_equal(observation['prefix_t'], full_t[:provider.PREFIX_INTERVALS+1])):
        raise ValueError('Paid observation geometry differs from the frozen design.')
    full_field = np.concatenate((observation['prefix_u'][:-1], observation['u']))
    if (receipt.get('action_sha256') != actions.action_digest(action)
            or receipt.get('experiments') != 1
            or receipt.get('measurement_values') != provider.FULL_FRAMES*provider.OBSERVED_NX
            or receipt.get('observed_field_sha256') != hashlib.sha256(full_field.tobytes()).hexdigest()):
        raise ValueError('Provider receipt differs from the selected paid observation.')
    return controller.observation_digest(observation)


def _budget_slots(episode, budgets=BUDGETS):
    return [{'budget': budget,
             'checkpoint_index': next((i for i, c in enumerate(episode['checkpoints']) if c['budget'] == budget), None),
             'status': 'frozen_checkpoint' if any(c['budget'] == budget for c in episode['checkpoints'])
                       else 'missing_checkpoint_failure'} for budget in budgets]


def validate_design():
    from . import forecast
    p = read(STUDY/'protocol.json'); cal = read(DATA/'calibration.json')
    modules = {'actions': actions, 'provider': provider, 'engine': engine,
               'acquisition': acquisition, 'controller': controller, 'forecast': forecast}
    if p.get('status') != 'frozen' or p.get('study') != canonical(PROTOCOL):
        raise ValueError('Study rules must be frozen before formal data generation.')
    for key, module in modules.items():
        if p.get(key) != canonical(module.PROTOCOL):
            raise ValueError('Frozen module protocol mismatch: '+key)
    records = cal.get('records', [])
    expected = {(f, a) for f in provider.FAMILIES for a in ('fixed-rich', 'single-6-0')}
    if (cal.get('all_passed') is not True or cal.get('formal_data_generated') is not False
            or len(records) != 12 or {(r.get('family'), r.get('action_id')) for r in records} != expected
            or cal.get('protocol') != canonical(provider.PROTOCOL)
            or cal.get('action_protocol') != canonical(actions.PROTOCOL)):
        raise ValueError('Require all twelve declared reserved source-calibration checks.')
    for r in records:
        if (r.get('status') != 'passed' or not 0 <= r['relative_l2'] <= provider.CALIBRATION_RELATIVE_LIMIT
                or not 0 <= r['max_absolute_error'] <= provider.CALIBRATION_MAX_ABSOLUTE_LIMIT):
            raise ValueError('Calibration record does not satisfy its fixed bounds.')
    for name, h in p['calibration_sources'].items():
        if sha(ROOT/name) != h:
            raise ValueError('Reference source changed after calibration.')
    return p


def prepare():
    if (DATA/'prepared.json').exists() or list((DATA/'initial').rglob('*.npz')):
        raise RuntimeError('Prepared or partial inputs already exist; preserve the first run.')
    protocol = validate_design(); tasks = []
    for index in INDICES:
        spec = provider.get_task_spec('formal', index)
        initial = actions.initial_actions(857000+index)
        pool = actions.make_action_pool(856000+index)
        for label, level in NOISES:
            identity = task_id(index, label)
            record = {'id': identity, 'spec_index': index, 'spec': spec, 'noise_label': label,
                      'noise_level': level, 'pool': pool, 'initial_actions': initial,
                      'observations': {'training': [], 'development': []}, 'failures': []}
            for role in ('training', 'development'):
                for j, action in enumerate(initial[role]):
                    relative = f'active_study/data/initial/{identity}/{role}-{j}.npz'
                    try:
                        response = provider.generate_observation(spec, action, noise_seed(spec, action), level)
                        save_observation(ROOT/relative, response['observation'])
                        record['observations'][role].append({'index': j, 'file': relative, 'sha256': sha(ROOT/relative),
                                                            'action_sha256': actions.action_digest(action),
                                                            'noise_seed': noise_seed(spec, action),
                                                            'receipt': response['receipt']})
                    except (ValueError, RuntimeError, FloatingPointError, OSError) as exc:
                        record['failures'].append({'role': role, 'index': j, 'reason': str(exc),
                                                  'action_sha256': actions.action_digest(action),
                                                  'noise_seed': noise_seed(spec, action),
                                                  'charged_experiments': 1})
            tasks.append(record)
    receipt = {'sources': sources(), 'protocol': protocol, 'tasks': tasks,
               'calibration_sha256': sha(DATA/'calibration.json'), 'final_generated': False,
               'all_initial_conditions_retained': len(tasks) == 24,
               'prepared_at_utc': datetime.now(timezone.utc).isoformat()}
    write(DATA/'prepared.json', receipt)
    print('Prepared24 paired conditions with5 paid initial observations each; no final fields generated.', flush=True)
    return receipt


def verify_preparation():
    prep = read(DATA/'prepared.json')
    if prep['sources'] != sources() or prep['calibration_sha256'] != sha(DATA/'calibration.json'):
        raise ValueError('Code, protocol, or source calibration changed.')
    if prep.get('final_generated') is not False or {t['id'] for t in prep['tasks']} != {
            task_id(i, label) for i in INDICES for label, _ in NOISES} or len(prep['tasks']) != 24:
        raise ValueError('Prepared task allocation or final-role marker differs.')
    if prep.get('protocol') != validate_design():
        raise ValueError('Prepared protocol differs from the frozen design.')
    noise_levels = dict(NOISES)
    for task in prep['tasks']:
        i = task.get('spec_index'); label = task.get('noise_label')
        if (type(i) is not int or i not in INDICES or label not in noise_levels
                or task['id'] != task_id(i, label) or type(task.get('noise_level')) not in (int, float)
                or task['noise_level'] != noise_levels[label]):
            raise ValueError('Prepared noise condition or parameter-task allocation differs.')
        if task['spec'] != provider.get_task_spec('formal', i):
            raise ValueError('Private provider task changed.')
        if (task.get('pool') != actions.make_action_pool(856000+i)
                or task.get('initial_actions') != actions.initial_actions(857000+i)):
            raise ValueError('Prepared action pool or initial-action allocation changed.')
        observations = task.get('observations', {}); failures = task.get('failures')
        if set(observations) != {'training', 'development'} or not isinstance(failures, list):
            raise ValueError('Initial success/failure allocation is malformed.')
        allocation = set()
        for failure in failures:
            role, j = failure.get('role'), failure.get('index')
            limit = 3 if role == 'training' else 2 if role == 'development' else 0
            if (type(j) is not int or not 0 <= j < limit or (role, j) in allocation
                    or failure.get('charged_experiments') != 1 or not isinstance(failure.get('reason'), str)):
                raise ValueError('Initial failed-attempt allocation differs.')
            action = task['initial_actions'][role][j]
            if (failure.get('action_sha256') != actions.action_digest(action)
                    or failure.get('noise_seed') != noise_seed(task['spec'], action)):
                raise ValueError('Initial failed-attempt action or noise binding differs.')
            allocation.add((role, j))
        for role in ('training', 'development'):
            records = observations[role]
            if not isinstance(records, list): raise ValueError('Initial observations must be lists.')
            for r in records:
                j = r.get('index'); limit = 3 if role == 'training' else 2
                if type(j) is not int or not 0 <= j < limit or (role, j) in allocation:
                    raise ValueError('Initial successful-attempt allocation differs.')
                action = task['initial_actions'][role][j]
                expected_path = f"active_study/data/initial/{task['id']}/{role}-{j}.npz"
                if (r.get('file') != expected_path or r.get('action_sha256') != actions.action_digest(action)
                        or r.get('noise_seed') != noise_seed(task['spec'], action)):
                    raise ValueError('Initial observation path, action, or noise binding differs.')
                if sha(ROOT/expected_path) != r['sha256']:
                    raise ValueError('Initial observation bytes changed.')
                observation = load_observation(ROOT/expected_path)
                _verify_paid_observation(observation, r.get('receipt', {}), action)
                allocation.add((role, j))
            if [r['index'] for r in records] != sorted(r['index'] for r in records):
                raise ValueError('Initial observation order changed.')
        expected = {(role, j) for role, count in (('training', 3), ('development', 2)) for j in range(count)}
        if allocation != expected:
            raise ValueError('Each of five initial attempts must have exactly one success or failure record.')
    return prep


def _run_task(task, output, additions=4, development=False):
    output = Path(output).resolve(); records = []
    with threadpool_limits(limits=1):
        initial_failed = bool(task['failures'])
        train = [load_observation(ROOT/r['file']) for r in task['observations']['training']] if not initial_failed else []
        dev = [load_observation(ROOT/r['file']) for r in task['observations']['development']] if not initial_failed else []
        try:
            initial_model = engine.discover(train, dev, seed=858000+task['spec_index']) if not initial_failed else None
        except (ValueError, RuntimeError, FloatingPointError, np.linalg.LinAlgError) as exc:
            initial_model = controller._model_failure(exc)
        for policy in POLICIES:
            identity = task['id']+'--'+policy
            calls = []
            def query(action, query_index):
                relative = f'queries/{identity}/{query_index}.npz'
                call = {'query_index': query_index, 'action': deepcopy(action), 'started': True}
                calls.append(call)
                try:
                    response = provider.generate_observation(task['spec'], action,
                        noise_seed(task['spec'], action), task['noise_level'])
                    save_observation(output/relative, response['observation'])
                    call.update(file=relative, sha256=sha(output/relative), status='observed')
                    return {'observation': response['observation'], 'receipt': response['receipt']}
                except (ValueError, RuntimeError, FloatingPointError, OSError) as exc:
                    call.update(status='failed', reason=str(exc)); raise
            if initial_failed:
                episode = {'policy': policy, 'status': 'initial_source_failure', 'checkpoints': [], 'queries': [],
                           'initial_failures': task['failures'], 'total_experiment_attempts': 5,
                           'charged_measurement_scalars': 5*provider.FULL_FRAMES*provider.OBSERVED_NX,
                           'final_used': False, 'reference_coefficients_used': False}
            else:
                episode = controller.run_episode(train, dev, task['pool'], query, policy=policy,
                    seed=858000+task['spec_index'], max_additions=additions, initial_model=initial_model)
            if len(calls) != len(episode['queries']):
                raise AssertionError('Every started experiment must have one charged query receipt.')
            episode.update(id=identity, task_id=task['id'], spec_index=task['spec_index'],
                           noise_level=task['noise_level'], noise_label=task['noise_label'],
                           provider_calls=calls, retrospective_development=development,
                           source_hashes=sources())
            episode['budget_slots'] = _budget_slots(episode, BUDGETS[:additions+1])
            path = output/'episodes'/f'{identity}.json'; write(path, episode)
            records.append({'id': identity, 'task_id': task['id'], 'file': str(path.relative_to(output)), 'sha256': sha(path)})
            last = episode['checkpoints'][-1]['model'] if episode['checkpoints'] else {}
            print('Frozen '+identity+': '+(last.get('selected') or {}).get('equation', episode['status']), flush=True)
    return records


def freeze(output=OUTPUT, workers=3):
    output = Path(output).resolve()
    if (output/'freeze_index.json').exists() or list((output/'episodes').glob('*.json')):
        raise RuntimeError('Preserve existing/partial episode snapshots; use a new output directory.')
    prep = verify_preparation(); started = sources()
    if workers == 1:
        records = [r for t in prep['tasks'] for r in _run_task(t, output)]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_run_task, t, str(output)) for t in prep['tasks']]
            records = [r for future in futures for r in future.result()]
    if len(records) != 96 or {r['id'] for r in records} != expected_episode_ids():
        raise ValueError('All96 declared policy episodes must be retained before final validation.')
    if verify_preparation() != prep or sources() != started:
        raise ValueError('Scientific inputs or code changed during the closed loop.')
    index = {'records': records, 'sources': started, 'prepared_sha256': sha(DATA/'prepared.json'),
             'all_episodes_frozen_before_final': True, 'final_generated': False,
             'expected_conditions': 24, 'expected_budgets': list(BUDGETS), 'expected_budget_records': 480,
             'missing_budget_checkpoints_are_failed_slots': True,
             'frozen_at_utc': datetime.now(timezone.utc).isoformat()}
    write(output/'freeze_index.json', index)
    return index


def verify_freeze(output):
    output = Path(output).resolve(); index = read(output/'freeze_index.json'); prep = verify_preparation()
    if (index['sources'] != sources() or index['prepared_sha256'] != sha(DATA/'prepared.json')
            or index.get('all_episodes_frozen_before_final') is not True or index.get('final_generated') is not False
            or index.get('expected_conditions') != 24 or index.get('expected_budgets') != list(BUDGETS)
            or index.get('expected_budget_records') != 480
            or index.get('missing_budget_checkpoints_are_failed_slots') is not True
            or len(index['records']) != 96 or {r['id'] for r in index['records']} != expected_episode_ids()):
        raise ValueError('Global frozen study identity differs.')
    prepared_tasks = {task['id']: task for task in prep['tasks']}
    for r in index['records']:
        if r['file'] != f"episodes/{r['id']}.json" or sha(output/r['file']) != r['sha256']:
            raise ValueError('Frozen episode path or bytes changed.')
        ep = read(output/r['file'])
        task = prepared_tasks.get(ep.get('task_id'))
        if (task is None or ep['id'] != r['id'] or ep['source_hashes'] != index['sources']
                or ep['final_used'] is not False or ep.get('reference_coefficients_used') is not False
                or ep.get('policy') not in POLICIES or ep['id'] != task['id']+'--'+ep['policy']
                or ep.get('spec_index') != task['spec_index'] or ep.get('noise_level') != task['noise_level']
                or ep.get('noise_label') != task['noise_label'] or ep.get('retrospective_development') is not False
                or ep.get('budget_slots') != _budget_slots(ep)):
            raise ValueError('Episode science/identity binding differs.')
        _verify_episode_ledger(ep, task, output)
    return index


def _verify_episode_ledger(ep, task, output):
    """Check all policy-visible inputs against the paid provider-call sequence."""
    cost = provider.FULL_FRAMES*provider.OBSERVED_NX
    calls, queries, checkpoints = ep['provider_calls'], ep['queries'], ep['checkpoints']
    if (len(calls) != len(queries) or len(queries) > controller.MAX_ADDITIONS
            or ep.get('total_experiment_attempts') != 5+len(queries)
            or ep.get('charged_measurement_scalars') != cost*(5+len(queries))):
        raise ValueError('Hidden, uncharged or over-budget provider query.')
    if task['failures']:
        if (ep.get('status') != 'initial_source_failure' or queries or checkpoints
                or ep.get('initial_failures') != task['failures']):
            raise ValueError('Initial source failures disappeared or were replaced.')
        return
    if (ep.get('seed') != 858000+task['spec_index'] or ep.get('maximum_additions') != 4
            or [c['budget'] for c in checkpoints] != list(range(5, 6+len(queries)))
            or ep.get('status') not in ('completed_budget', 'stopped_failure')
            or (ep['status'] == 'completed_budget' and len(queries) != 4)):
        raise ValueError('Frozen checkpoint or deterministic seed allocation differs.')
    train_hashes = [controller.observation_digest(load_observation(ROOT/r['file']))
                    for r in task['observations']['training']]
    dev_hashes = [controller.observation_digest(load_observation(ROOT/r['file']))
                  for r in task['observations']['development']]
    pool = {a['id']: a for a in task['pool']}; used = set(); returned = 0
    for round_index, cp in enumerate(checkpoints):
        if (cp.get('training_hashes') != train_hashes or cp.get('development_hashes') != dev_hashes
                or cp.get('training_trajectories') != len(train_hashes) or cp.get('development_trajectories') != 2
                or cp.get('charged_measurement_scalars') != cost*(5+round_index)
                or cp.get('available_measurement_scalars') != cost*5+returned
                or cp.get('final_used_for_selection') is not False):
            raise ValueError('Checkpoint inputs or budget differ from the acquired observations.')
        if round_index == len(queries): continue
        query, call = queries[round_index], calls[round_index]
        action = actions.validate_action(query.get('action'))
        if (action['id'] not in pool or action != pool[action['id']] or action['id'] in used
                or query.get('query_index') != round_index or call.get('query_index') != round_index
                or call.get('action') != action or call.get('started') is not True
                or query.get('action_sha256') != actions.action_digest(action)
                or query.get('charged_experiments') != 1 or query.get('charged_measurement_scalars') != cost
                or (cp.get('next_proposal') or {}).get('selected_action') != action
                or query.get('status') not in ('observed', 'failed') or call.get('status') not in ('observed', 'failed')):
            raise ValueError('Query ledger differs from the chosen unused paid action.')
        used.add(action['id'])
        observed = None
        if call['status'] == 'observed':
            expected_path = f"queries/{ep['id']}/{round_index}.npz"
            if call.get('file') != expected_path or sha(Path(output)/expected_path) != call.get('sha256'):
                raise ValueError('Selected query path or observation bytes changed.')
            observed = load_observation(Path(output)/expected_path)
        if query['status'] == 'observed':
            if observed is None:
                raise ValueError('A successful training query lacks a returned provider observation.')
            h = _verify_paid_observation(observed, query.get('provider_receipt', {}), action)
            if query.get('observation_sha256') != h or query.get('returned_measurement_scalars') != cost:
                raise ValueError('Query training input is not its returned paid observation.')
            train_hashes.append(h); returned += cost
        elif query.get('returned_measurement_scalars') != 0 or query.get('observation_sha256') is not None:
            raise ValueError('Failed query was counted as available training evidence.')


def _score_episode(task):
    from . import forecast
    output, receipt, task_spec, final_refs = task
    output = Path(output).resolve(); ep = read(output/receipt['file']); rows = []
    references = provider.reference_terms(task_spec)
    truth_names = {t['name'] for t in references}
    with threadpool_limits(limits=1):
        for budget in BUDGETS:
            matches = [(i, c) for i, c in enumerate(ep['checkpoints']) if c['budget'] == budget]
            cp_index, cp = matches[0] if len(matches) == 1 else (None, None)
            model = cp['model'] if cp else {}; selected = model.get('selected')
            terms = selected['terms'] if selected else []
            recovery = reference_metrics(terms, references)
            supports = model.get('candidate_supports', [])
            frequencies = model.get('uncertainty', {}).get('term_inclusion_frequencies', {})
            valid_frequency = (set(frequencies) == set(TERM_NAMES) and all(
                type(v) in (int, float) and np.isfinite(v) and 0 <= v <= 1 for v in frequencies.values()))
            positive = [float((frequencies[n]-1)**2) for n in TERM_NAMES if n in truth_names] if valid_frequency else []
            negative = [float(frequencies[n]**2) for n in TERM_NAMES if n not in truth_names] if valid_frequency else []
            predictions = []
            for j, fr in enumerate(final_refs):
                try:
                    if selected is None: raise ValueError('No selected model at this paid budget.')
                    if fr.get('status') != 'generated': raise ValueError('Final source generation failed.')
                    path = output/fr['file']
                    if sha(path) != fr['sha256']: raise ValueError('Final reference changed.')
                    with np.load(path, allow_pickle=False) as z: clean = {k: z[k].copy() for k in ('x', 't', 'u')}
                    rng = np.random.default_rng(np.random.SeedSequence([provider.NOISE_SEED_BASE, 860000+100*ep['spec_index']+j]))
                    observed_field = clean['u'] + actions.INITIAL_RMS*ep['noise_level']*rng.standard_normal(clean['u'].shape)
                    obs = provider.observation_view(clean, observed_field)
                    pred, diag = forecast.forecast_with_diagnostics(terms, obs['x'], obs['t'], obs['prefix_t'], obs['prefix_u'])
                    metric = forecast_metrics(pred, clean['u'][provider.PREFIX_INTERVALS:])
                    predictions.append({'index': j, 'status': 'evaluated', 'clean': metric,
                                        'forecast_duration': float(obs['t'][-1]-obs['t'][0]),
                                        'prefix_frames_used': len(obs['prefix_t']), 'conditioning': 'observed_prefix',
                                        'initial_state_diagnostics': diag})
                except (ValueError, RuntimeError, FloatingPointError, OSError, np.linalg.LinAlgError) as exc:
                    predictions.append({'index': j, 'status': 'failed', 'reason': str(exc)})
            failures = sum(p['status'] != 'evaluated' for p in predictions)
            mean = None if failures else float(np.mean([p['clean']['relative_l2'] for p in predictions]))
            worst = max(c['relative_error'] for c in recovery['coefficient_comparison'] if c['reference'] != 0)
            success = bool(recovery['exact_support'] and worst <= .1 and mean is not None and mean <= .05)
            rows.append({'id': ep['id']+f'--budget-{budget}', 'task_id': ep['task_id'], 'spec_index': ep['spec_index'],
                'family': task_spec['family'], 'noise_level': ep['noise_level'], 'noise_label': ep['noise_label'],
                'policy': ep['policy'], 'budget': budget, 'terms': terms,
                'equation': (selected or {}).get('equation'), 'recovery': recovery,
                'mean_clean_relative_l2': mean, 'forecasts': predictions, 'forecast_failures': failures,
                'candidate_supports': supports, 'set_size': len(supports),
                'set_contains_true_support': any(set(s) == truth_names for s in supports),
                'singleton_wrong': bool(len(supports) == 1 and set(supports[0]) != truth_names),
                'unique_physical_equation_claimed': False,
                'frequencies': frequencies, 'term_brier_positive': float(np.mean(positive)) if positive else None,
                'term_brier_negative': float(np.mean(negative)) if negative else None,
                'frequency_calibrated': False, 'offline_success': success,
                'model_file': receipt['file'], 'checkpoint_index': cp_index,
                'frozen_episode_sha256': receipt['sha256'],
                'charged_measurement_scalars': cp['charged_measurement_scalars'] if cp else None,
                'available_measurement_scalars': cp['available_measurement_scalars'] if cp else None,
                'status': 'evaluated' if cp is not None else 'missing_budget_checkpoint',
                'hypothesis_status': model.get('status', model.get('diagnostics', {}).get('status', 'insufficient')),
                'query_failure_count_so_far': sum(q['status'] == 'failed' for q in ep['queries'][:budget-5])})
    print('Scored '+ep['id'], flush=True)
    return rows


def _register_final_release(output, index):
    """Reserve the first release before any final field can be generated.

An interrupted first release remains consumed. This deliberately labels any
later attempt conservatively as reproduction, even if the first failed before
returning a field. A different output directory cannot reset final blindness.
"""
    output = Path(output).resolve(); path = DATA/'final_release.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    binding = {'prepared_sha256': sha(DATA/'prepared.json'), 'sources': sources(),
               'final_design': {'indices': list(INDICES), 'trajectories_per_task': FINAL_COUNT,
                                'action_seed_base': 859000, 'action_pool_slice': [16, 19]}}
    receipt = dict(binding, first_freeze_index_sha256=sha(output/'freeze_index.json'),
                   first_output=str(output), released_at_utc=datetime.now(timezone.utc).isoformat(),
                   marker_created_before_final_generation=True,
                   interrupted_first_release_remains_consumed=True)
    if index['prepared_sha256'] != binding['prepared_sha256'] or index['sources'] != binding['sources']:
        raise ValueError('Final release and frozen study differ.')
    try:
        with path.open('x') as stream:
            json.dump(receipt, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write('\n')
        first = True
    except FileExistsError:
        receipt = read(path); first = False
        if (any(receipt.get(k) != v for k, v in binding.items())
                or receipt.get('marker_created_before_final_generation') is not True
                or not isinstance(receipt.get('first_output'), str)
                or not isinstance(receipt.get('first_freeze_index_sha256'), str)
                or not isinstance(receipt.get('released_at_utc'), str)):
            raise ValueError('Existing final release is not bound to this frozen study; preserve it.')
    return {'file': str(path.relative_to(ROOT)), 'sha256': sha(path),
            'is_first_final_release': first, 'released_at_utc': receipt['released_at_utc'],
            'first_output': receipt['first_output'],
            'first_freeze_index_sha256': receipt['first_freeze_index_sha256']}


def evaluate(output=OUTPUT, workers=3):
    output = Path(output).resolve()
    if (output/'comparison.json').exists() or list((output/'final_reference').rglob('*.npz')):
        raise RuntimeError('Preserve original final outputs; later runs are reproduction, not unseen validation.')
    index = verify_freeze(output); frozen_sha = sha(output/'freeze_index.json')
    release = _register_final_release(output, index)
    final = {}
    for i in INDICES:
        spec = provider.get_task_spec('formal', i); pool = actions.make_action_pool(859000+i)
        final[i] = []
        for j, action in enumerate(pool[16:19]):
            relative = f'final_reference/task-{i:02d}/{j}.npz'
            try:
                clean = provider.generate_clean(spec, action)
                p = output/relative; p.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(p, **{k: clean[k] for k in ('x', 't', 'u')})
                final[i].append({'status': 'generated', 'file': relative, 'sha256': sha(p), 'action': action})
            except (ValueError, RuntimeError, FloatingPointError, OSError) as exc:
                final[i].append({'status': 'failed', 'reason': str(exc), 'action': action})
    tasks = []
    for receipt in index['records']:
        ep = read(output/receipt['file']); i = ep['spec_index']
        tasks.append((str(output), receipt, provider.get_task_spec('formal', i), final[i]))
    if workers == 1:
        records = [r for task in tasks for r in _score_episode(task)]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            records = [r for rows in pool.map(_score_episode, tasks) for r in rows]
    verify_freeze(output)
    if sha(output/'freeze_index.json') != frozen_sha or len(records) != 480 or len({r['id'] for r in records}) != 480:
        raise ValueError('Frozen models changed or evaluation omitted a declared condition.')
    if sha(ROOT/release['file']) != release['sha256']:
        raise ValueError('Global final-release record changed during evaluation.')
    report = {'records': records, 'expected_conditions': 24, 'expected_budgets': list(BUDGETS),
              'policies': list(POLICIES), 'expected_records': 480, 'independent_parameter_tasks': 12,
              'final_reference_records': {str(k): v for k, v in final.items()},
              'freeze_index_sha256': frozen_sha, 'final_used_for_tuning': False,
              'final_release': release,
              'is_first_final_release': release['is_first_final_release'],
              'all_data_synthetic': True, 'frequency_calibrated': False, 'reference_used_for_online_stopping': False,
              'policy': PROTOCOL, 'all_declared_records_present': True,
              'final_validation_experiments_per_condition_policy': 3,
              'final_validation_measurement_scalars_per_condition_policy': 3*111*64,
              'decision': 'independent_experiment_not_merged',
              'audit': {'status': 'pending_independent_audit'},
              'evaluation_label': ('first frozen same-family synthetic validation'
                  if release['is_first_final_release'] else
                  'reproduction on previously released final tasks; retrospective, not unseen validation')}
    write(output/'comparison.json', report)
    return report


def develop(output=STUDY/'development/pilot', workers=3):
    """Known development-only tasks. Never called a final or unseen benchmark."""
    output = Path(output).resolve()
    if output.exists(): raise RuntimeError('Use a new pilot directory to preserve development history.')
    tasks = []
    for i in range(6):
        spec = provider.get_task_spec('development', i); initial = actions.initial_actions(857000+i)
        task = {'id': f'development-{i:02d}', 'spec_index': i, 'spec': spec,
                'noise_label': 'noise1', 'noise_level': .01, 'pool': actions.make_action_pool(856000+i),
                'initial_actions': initial, 'observations': {'training': [], 'development': []}, 'failures': []}
        for role in ('training', 'development'):
            for j, action in enumerate(initial[role]):
                response = provider.generate_observation(spec, action, noise_seed(spec, action), .01)
                p = output/'inputs'/task['id']/f'{role}-{j}.npz'; save_observation(p, response['observation'])
                task['observations'][role].append({'file': str(p), 'sha256': sha(p)})
        tasks.append(task)
    start_sources = sources()
    if workers == 1: records = [r for t in tasks for r in _run_task(t, output, additions=2, development=True)]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_run_task, t, str(output), 2, True) for t in tasks]
            records = [r for f in futures for r in f.result()]
    if sources() != start_sources: raise ValueError('Science changed during the development probe.')
    summary = []
    for r in records:
        ep = read(output/r['file']); spec = tasks[ep['spec_index']]['spec']
        for cp in ep['checkpoints']:
            terms = (cp['model'].get('selected') or {}).get('terms', [])
            summary.append({'id': ep['id'], 'budget': cp['budget'], 'policy': ep['policy'],
                            'recovery': reference_metrics(terms, provider.reference_terms(spec)),
                            'set_size': len(cp['model'].get('candidate_supports', []))})
    result = {'development_only': True, 'final_generated': False, 'records': records,
              'source_hashes': start_sources, 'diagnostics': summary}
    write(output/'pilot.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('phase', choices=['prepare', 'freeze', 'evaluate', 'develop'])
    parser.add_argument('--output', type=Path)
    parser.add_argument('--workers', type=int, choices=[1, 2, 3], default=3)
    args = parser.parse_args()
    if args.phase == 'prepare': prepare()
    elif args.phase == 'develop': develop(args.output or STUDY/'development/pilot', args.workers)
    elif args.phase == 'freeze': freeze(args.output or OUTPUT, args.workers)
    else: evaluate(args.output or OUTPUT, args.workers)


if __name__ == '__main__':
    main()
