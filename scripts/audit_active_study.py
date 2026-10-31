#!/usr/bin/env python3
"""Read-only independent audit of an already completed active PDE study.

Never fit a model, generate a reference field, or alter comparison.json.  The
only written file is a new audit JSON.  At most twelve predictions are checked
with the original prefix solver, after validating the saved global freeze.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
POLICIES = ('fixed_rich', 'random', 'ensemble_variance', 'structural_disagreement')
BUDGETS = (5, 6, 7, 8, 9)
NOISES = {'clean': 0., 'noise1': .01}
FAMILIES = ('decay', 'diffusion', 'hyperdiffusion', 'advection_diffusion', 'reaction_diffusion', 'burgers')
TERMS = ('1', 'u', 'u^2', 'u^3', 'u_x', 'u_xx', 'u_xxx', 'u_xxxx', 'u*u_x')
UNIT_COST = 111*64
SOURCE_FILES = {f'active_study/{name}.py' for name in
                ('actions', 'provider', 'engine', 'acquisition', 'controller', 'forecast', 'run')} | {
    'active_study/protocol.json', 'active_study/references.json',
    'coefficient_study/generators.py', 'coefficient_study/noise.py',
    'weak_study/engine.py', 'l1_study/engine.py', 'pde_discovery/generic_engine.py',
    'pde_discovery/generic_experiment.py', 'prefix_study/forecast.py',
    'resolved_study/forecast.py', 'five_study/initial.py'}


class AuditError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise AuditError(message)


def finite(value):
    return type(value) in (float, int) and math.isfinite(value)


def close(actual, expected, label='', *, rtol=2e-11, atol=2e-13):
    require(finite(actual) and finite(expected) and math.isclose(actual, expected, rel_tol=rtol, abs_tol=atol),
            f'{label}: {actual!r} != {expected!r}')


def compare(actual, expected, label=''):
    """Compare required recursively derived fields, allowing extra diagnostics."""
    if isinstance(expected, dict):
        require(isinstance(actual, dict), label+': expected dictionary')
        for key, value in expected.items():
            require(key in actual, label+': missing '+key)
            compare(actual[key], value, label+':'+key)
    elif isinstance(expected, list):
        require(isinstance(actual, list) and len(actual) == len(expected), label+': list length')
        for index, (a, b) in enumerate(zip(actual, expected)):
            compare(a, b, f'{label}[{index}]')
    elif type(expected) is float:
        close(actual, expected, label)
    else:
        require(actual == expected and (not isinstance(expected, bool) or actual is expected),
                f'{label}: {actual!r} != {expected!r}')


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            value.update(block)
    return value.hexdigest()


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'),
                      parse_constant=lambda value: (_ for _ in ()).throw(AuditError('Nonfinite JSON: '+value)))


def safe_path(root, relative):
    require(isinstance(relative, str) and not Path(relative).is_absolute(), 'Need a relative artifact path')
    root = Path(root).resolve()
    result = (root/relative).resolve()
    require(result.is_relative_to(root), 'Artifact path escapes its root')
    return result


def task_id(index, noise):
    return f'task-{index:02d}--{noise}'


def expected_conditions():
    return {task_id(i,n) for i in range(12) for n in NOISES}


def expected_episodes():
    return {condition+'--'+policy for condition in expected_conditions() for policy in POLICIES}


def expected_rows():
    return {episode+f'--budget-{budget}' for episode in expected_episodes() for budget in BUDGETS}


def check_unique(records, expected, label):
    ids = [r['id'] for r in records]
    require(len(ids) == len(expected) and len(set(ids)) == len(ids) and set(ids) == expected,
            label+': missing, duplicate, or unexpected declared slots')


def coefficient_map(items):
    result = {}
    for term in items:
        name, value = term['name'], term['coefficient']
        require(name in TERMS and name not in result and finite(value), 'Invalid/duplicate PDE coefficient')
        result[name] = value
    return {name: value for name, value in result.items() if value != 0}


def recovery(terms, references):
    found, truth = coefficient_map(terms), coefficient_map(references)
    require(bool(truth), 'Empty reference equation')
    selected, true = set(found), set(truth)
    common = len(selected & true)
    precision, recall = common/len(selected) if selected else 0., common/len(true)
    rows = []
    for name in TERMS:
        if name in selected | true:
            error = abs(found.get(name,0.)-truth.get(name,0.))
            rows.append({'name': name, 'reference': truth.get(name,0.), 'discovered': found.get(name,0.),
                         'absolute_error': error, 'relative_error': error/abs(truth[name]) if name in truth else None})
    return {'status': 'evaluated', 'exact_support': selected == true,
            'precision': precision, 'recall': recall,
            'f1': 2*precision*recall/(precision+recall) if precision+recall else 0.,
            'true_positive_terms': sorted(selected & true), 'missing_terms': sorted(true-selected),
            'spurious_terms': sorted(selected-true), 'coefficient_comparison': rows,
            'coefficient_relative_l2': math.sqrt(sum(r['absolute_error']**2 for r in rows)/sum(v*v for v in truth.values())),
            'reference_terms': references}


def support_set(model):
    committee = model.get('committee', [])
    supports = sorted({tuple(name for name in TERMS if name in coefficient_map(member['terms']))
                       for member in committee})
    result = [list(s) for s in supports]
    require(model.get('candidate_supports', []) == result, 'Support set differs from frozen committee')
    return result


def inclusion_frequencies(model):
    uncertainty = model.get('uncertainty', {})
    saved = uncertainty.get('term_inclusion_frequencies', {})
    if 'records' not in uncertainty:
        require(not saved and model.get('selected') is None, 'Missing bootstrap evidence for frequency')
        return {}
    records = uncertainty['records']; requested = uncertainty['requested_replicates']
    require(set(saved) == set(TERMS),'Frequency vector must use exactly the common library')
    require(type(requested) is int and requested >= 0 and len(records) == requested, 'Bootstrap denominator')
    require([r['replicate'] for r in records] == list(range(requested)), 'Bootstrap replicate identities')
    counts = Counter(); successes = 0
    for record in records:
        require(record['whole_trajectory_resampling'] is True and record['structure_selection_repeated'] is True,
                'Bootstrap must rerun whole-trajectory structure selection')
        if record['status'] == 'selected':
            names = record['selected_support']
            require(len(names) == len(set(names)) and set(names) <= set(TERMS), 'Invalid bootstrap support')
            counts.update(names); successes += 1
        else:
            require(record['status'] == 'failed' and record['selected_support'] is None,
                    'Failed bootstrap slot omitted or mislabeled')
    failures = requested-successes
    result = {name: counts[name]/requested if requested and not failures else None for name in TERMS}
    compare(saved, result, 'bootstrap frequency')
    compare(uncertainty, {'raw_counts': {name: counts[name] for name in TERMS},
                         'completed_replicates': successes, 'failed_replicates': failures,
                         'calibrated': False, 'probability_claimed': False,
                         'reference_used': False, 'final_used': False}, 'uncertainty')
    return result


def uncertainty_metrics(supports, frequencies, references):
    truth = set(coefficient_map(references))
    valid = set(frequencies) == set(TERMS) and all(finite(v) and 0 <= v <= 1 for v in frequencies.values())
    positive = [(frequencies[n]-1)**2 for n in TERMS if n in truth] if valid else []
    negative = [frequencies[n]**2 for n in TERMS if n not in truth] if valid else []
    return {'candidate_supports': supports, 'set_size': len(supports),
            'set_contains_true_support': any(set(s) == truth for s in supports),
            'singleton_wrong': len(supports) == 1 and set(supports[0]) != truth,
            'frequencies': frequencies,
            'term_brier_positive': statistics.fmean(positive) if positive else None,
            'term_brier_negative': statistics.fmean(negative) if negative else None,
            'frequency_calibrated': False, 'unique_physical_equation_claimed': False}


def forecast_summary(row, final_refs):
    forecasts = row['forecasts']
    require(len(forecasts) == 3 and [f['index'] for f in forecasts] == [0,1,2], 'All three forecast slots required')
    failures = 0; values = []
    for forecast, reference in zip(forecasts, final_refs):
        if forecast['status'] != 'evaluated':
            require(forecast['status'] == 'failed' and isinstance(forecast.get('reason'),str) and forecast['reason'],
                    'Unexplained failed forecast')
            failures += 1
            continue
        require(reference['status'] == 'generated', 'Failed reference scored as successful forecast')
        compare(forecast, {'forecast_duration': 1., 'prefix_frames_used': 11,
                          'conditioning': 'observed_prefix'}, 'forecast geometry')
        metric = forecast['clean']
        compare(metric, {'excluded_initial_frame': True, 'scored_frames': 100, 'space_points': 64}, 'metric geometry')
        for name in ('rmse','mae','max_absolute_error','relative_l2'):
            require(finite(metric[name]) and metric[name] >= 0, 'Nonfinite or negative forecast metric')
        diagnostic = forecast['initial_state_diagnostics']
        compare(diagnostic, {'future_observations_used': False, 'equal_frame_weights': True,
                             'independent_estimates_claimed': False, 'prefix_frames': 11,
                             'prefix_duration': .1, 'forecast_duration': 1.}, 'prefix diagnostic')
        values.append(metric['relative_l2'])
    mean = None if failures else statistics.fmean(values)
    compare(row, {'forecast_failures': failures, 'mean_clean_relative_l2': mean}, 'complete forecast denominator')
    return mean


def array_digest(observation):
    result = hashlib.sha256()
    for name in ('x','t','u','prefix_t','prefix_u'):
        a = np.asarray(observation[name],dtype='<f8')
        result.update(name.encode()); result.update(str(a.shape).encode()); result.update(a.tobytes())
    return result.hexdigest()


def load_observation(path):
    with np.load(path,allow_pickle=False) as archive:
        require(set(archive.files) == {'x','t','u','prefix_t','prefix_u'}, 'Unexpected observed arrays')
        result = {k: archive[k].copy() for k in archive.files}
    shapes = {'x':(64,), 't':(101,), 'u':(101,64), 'prefix_t':(11,), 'prefix_u':(11,64)}
    for key, shape in shapes.items():
        a = result[key]
        require(a.shape == shape and not np.iscomplexobj(a) and np.isfinite(a).all(), 'Invalid observation '+key)
    require(np.array_equal(result['u'][0],result['prefix_u'][-1]), 'Shared history endpoint differs')
    require(np.allclose(result['x'],np.arange(64)*2*np.pi/64,rtol=0,atol=1e-13), 'Spatial geometry differs')
    require(np.allclose(result['t'],np.linspace(.1,1.1,101),rtol=0,atol=1e-13) and
            np.allclose(result['prefix_t'],np.linspace(0,.1,11),rtol=0,atol=1e-13), 'Temporal geometry differs')
    require(result['u'].size+result['prefix_u'][:-1].size == UNIT_COST, 'Measurement cost differs')
    return result


def action_digest(action):
    require(set(action) == {'id','a','b'} and isinstance(action['id'],str), 'Action identity/schema')
    a,b = np.asarray(action['a']),np.asarray(action['b'])
    require(a.shape == b.shape == (6,) and not np.iscomplexobj(a) and not np.iscomplexobj(b)
            and np.isfinite(a).all() and np.isfinite(b).all(), 'Invalid Fourier action')
    close(float(np.sqrt(.5*(a@a+b@b))), .4, 'Action RMS')
    grid = np.arange(4096)*2*np.pi/4096
    modes = np.arange(1,7)
    field = np.sin(grid[:,None]*modes)@a+np.cos(grid[:,None]*modes)@b
    require(float(np.max(np.abs(field))+np.dot(modes,np.hypot(a,b))*np.pi/4096) <= 1.2,
            'Continuous action amplitude exceeds fixed bound')
    canonical = {'id':action['id'],'a':a.astype(float).tolist(),'b':b.astype(float).tolist()}
    return hashlib.sha256(json.dumps(canonical,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def paid_receipt(observation, receipt, action):
    full = np.concatenate((observation['prefix_u'][:-1],observation['u']))
    compare(receipt,{'experiments':1,'measurement_values':UNIT_COST,
                     'history_endpoint_counted_once':True,'action_sha256':action_digest(action),
                     'observed_field_sha256':hashlib.sha256(full.tobytes()).hexdigest()},'Paid provider receipt')


def verify_episode(episode, task, load_checked):
    """Rebuild the paid ledger and observed-input lineage without refitting."""
    queries, calls, checkpoints = episode['queries'],episode['provider_calls'],episode['checkpoints']
    require(len(queries) == len(calls) <= 4, 'Uncharged/hidden or excessive provider call')
    require(episode['policy'] in POLICIES and episode['task_id'] == task['id'] and
            episode['spec_index'] == task['spec_index'], 'Episode identity binding')
    compare(episode, {'noise_label':task['noise_label'], 'noise_level':task['noise_level'],
                      'final_used':False, 'reference_coefficients_used':False,
                      'retrospective_development':False,
                      'total_experiment_attempts':5+len(queries),
                      'charged_measurement_scalars':(5+len(queries))*UNIT_COST}, 'episode budget')
    slots = []
    for budget in BUDGETS:
        matches = [j for j,c in enumerate(checkpoints) if c['budget'] == budget]
        require(len(matches) <= 1,'Duplicate frozen checkpoint')
        slots.append({'budget':budget,'checkpoint_index':matches[0] if matches else None,
                      'status':'frozen_checkpoint' if matches else 'missing_checkpoint_failure'})
    require(episode['budget_slots'] == slots,'Missing frozen budget failure slot')
    if task['failures']:
        require(not checkpoints and not queries and episode['status'] == 'initial_source_failure'
                and episode['initial_failures'] == task['failures'],
                'Initial source failure must remain in denominator')
        return {'queries':0,'failed_queries':0,'missing_checkpoints':5}
    train = [array_digest(load_checked('root',r)) for r in task['observations']['training']]
    dev = [array_digest(load_checked('root',r)) for r in task['observations']['development']]
    require(len(train) == 3 and len(dev) == 2, 'Initial train/development allocation')
    compare(episode,{'seed':858000+task['spec_index'],'maximum_additions':4,
                     'initial_measurement_scalars':5*UNIT_COST,
                     'per_experiment_measurement_scalars':UNIT_COST},'Episode fixed design')
    require([c['budget'] for c in checkpoints] == list(range(5,5+len(checkpoints))) and
            len(checkpoints) == len(queries)+1, 'Checkpoint/charged-query sequence')
    pool = {a['id']:a for a in task['pool']}; used = set(); returned = 0
    for i, checkpoint in enumerate(checkpoints):
        compare(checkpoint, {'budget':5+i,'charged_measurement_scalars':(5+i)*UNIT_COST,
                             'available_measurement_scalars':5*UNIT_COST+returned,
                             'training_hashes':train,'development_hashes':dev,
                             'training_trajectories':len(train),'development_trajectories':2,
                             'final_used_for_selection':False}, 'checkpoint paid inputs')
        if i == len(queries):
            continue
        query, call = queries[i], calls[i]
        chosen = checkpoint['next_proposal']['selected_action']
        require(chosen['id'] not in used and pool.get(chosen['id']) == chosen, 'Illegal or repeated selected action')
        require(query['action'] == call['action'] == chosen, 'Provider query was not the recorded selected action')
        used.add(chosen['id'])
        compare(query, {'query_index':i,'charged_experiments':1,'charged_measurement_scalars':UNIT_COST,
                        'action_sha256':action_digest(chosen)}, 'query charge')
        require(call['query_index'] == i and call['started'] is True and call['status'] in ('observed','failed'),
                'Missing/unfinished provider call')
        if call['status'] == 'observed':
            require(call['file'] == f"queries/{episode['id']}/{i}.npz",'Query artifact path allocation')
            observation = load_checked('output',call)
        else:
            require(isinstance(call.get('reason'),str) and call['reason'], 'Missing provider failure reason')
            require(query['status'] == 'failed', 'Failed provider call treated as observed')
        if query['status'] == 'observed':
            require(call['status'] == 'observed', 'Observed ledger lacks provider artifact')
            require(query['observation_sha256'] == array_digest(observation), 'Query content digest differs')
            require(query['returned_measurement_scalars'] == UNIT_COST, 'Returned observation cost')
            paid_receipt(observation,query['provider_receipt'],chosen)
            train.append(query['observation_sha256']); returned += UNIT_COST
        else:
            require(query['status'] == 'failed' and query['returned_measurement_scalars'] == 0
                    and query['observation_sha256'] is None and query.get('failure_reason'),
                    'Failed query omitted, refunded, or counted as measured evidence')
    if len(queries) < 4:
        require(episode['status'] == 'stopped_failure' and episode.get('terminal_reason')
                and checkpoints[-1].get('acquisition_failure'), 'Missing budget needs explicit acquisition failure')
    else:
        require(episode['status'] == 'completed_budget', 'Complete budget status differs')
    return {'queries':len(queries),'failed_queries':sum(q['status']=='failed' for q in queries),
            'missing_checkpoints':5-len(checkpoints)}


def independent_forecast_metrics(predicted, truth):
    require(predicted.shape == truth.shape == (101,64) and np.isfinite(predicted).all() and np.isfinite(truth).all(),
            'Independent forecast shape/state')
    difference = predicted[1:]-truth[1:]
    norm, denominator = float(np.linalg.norm(difference)),float(np.linalg.norm(truth[1:]))
    return {'rmse':float(np.sqrt(np.mean(difference**2))), 'mae':float(np.mean(np.abs(difference))),
            'max_absolute_error':float(np.max(np.abs(difference))),
            'relative_l2':norm/denominator if denominator > 1e-12 else (0. if norm < 1e-12 else None),
            'excluded_initial_frame':True,'scored_frames':100,'space_points':64}


def worst_final_budget_rows(rows):
    selected = []
    for policy in POLICIES:
        group = [r for r in rows if r['policy'] == policy and r['budget'] == 9]
        require(len(group) == 24, 'Last-budget forecast sample denominator')
        selected.append(max(group,key=lambda r:(float('inf') if r['mean_clean_relative_l2'] is None
                                                else r['mean_clean_relative_l2'],r['id'])))
    return selected


class Checks:
    def __init__(self):
        self.records = []

    def run(self, name, function, *, keep_details=True):
        try:
            detail = function()
            self.records.append({'name':name,'status':'passed','details':detail if keep_details else None})
            return detail
        except (AuditError,ValueError,TypeError,KeyError,IndexError,OSError,OverflowError,RuntimeError) as exc:
            self.records.append({'name':name,'status':'failed','reason':f'{type(exc).__name__}: {exc}'})
            return None


def audit(root, output):
    root,output = Path(root).resolve(),Path(output).resolve()
    checks = Checks(); summary = {}; forecast_rechecks = []
    report = checks.run('completed_comparison_required_before_artifact_access',lambda:load(output/'comparison.json'),keep_details=False)
    if report is None:
        return {'status':'failed','checks':checks.records,'prediction_calls':0,
                'no_data_generation':True,'comparison_modified':False}
    original_comparison_sha = digest(output/'comparison.json')
    prepared = checks.run('read_prepared',lambda:load(root/'active_study/data/prepared.json'),keep_details=False)
    frozen = checks.run('read_global_freeze',lambda:load(output/'freeze_index.json'),keep_details=False)
    if prepared is None or frozen is None:
        return {'status':'failed','checks':checks.records,'prediction_calls':0,
                'comparison_sha256':original_comparison_sha,'no_data_generation':True,'comparison_modified':False}

    def allocation():
        check_unique(prepared['tasks'],expected_conditions(),'24 paired conditions')
        check_unique(frozen['records'],expected_episodes(),'96 globally frozen episodes')
        check_unique(report['records'],expected_rows(),'480 budget slots')
        compare(report,{'expected_conditions':24,'expected_budgets':list(BUDGETS),
                        'expected_records':480,'independent_parameter_tasks':12,
                        'policies':list(POLICIES),'all_declared_records_present':True,
                        'final_used_for_tuning':False,'all_data_synthetic':True,
                        'frequency_calibrated':False,'reference_used_for_online_stopping':False,
                        'final_validation_experiments_per_condition_policy':3,
                        'final_validation_measurement_scalars_per_condition_policy':3*UNIT_COST},'report allocation')
        compare(frozen,{'all_episodes_frozen_before_final':True,'final_generated':False,
                        'expected_conditions':24,'expected_budgets':list(BUDGETS),'expected_budget_records':480,
                        'missing_budget_checkpoints_are_failed_slots':True},'freeze allocation')
        require(prepared['final_generated'] is False, 'Prepared marker claims final access')
        return {'conditions':24,'independent_parameter_tasks':12,'episodes':96,'budget_slots':480,
                'nominal_measurement_costs':{str(b):b*UNIT_COST for b in BUDGETS}}
    summary['allocation'] = checks.run('fixed_complete_denominators',allocation)

    def source_hashes():
        source_map = frozen['sources']
        require(set(source_map) == SOURCE_FILES and prepared['sources'] == source_map, 'Scientific dependency closure differs')
        for relative,value in source_map.items():
            require(digest(safe_path(root,relative)) == value,'Changed scientific source: '+relative)
        require(frozen['prepared_sha256'] == digest(root/'active_study/data/prepared.json'),'Prepared manifest hash')
        require(report['freeze_index_sha256'] == digest(output/'freeze_index.json'),'Global freeze hash')
        require(prepared['calibration_sha256'] == digest(root/'active_study/data/calibration.json'),'Calibration hash')
        protocol = load(root/'active_study/protocol.json')
        require(prepared['protocol'] == protocol and protocol['status'] == 'frozen','Protocol binding')
        require(report['policy'] == protocol['study'],'Reported study rules differ from frozen protocol')
        calibration = load(root/'active_study/data/calibration.json')
        require(calibration['all_passed'] is True and calibration['formal_data_generated'] is False,'Source calibration failed')
        require(calibration['protocol'] == protocol['provider'] and calibration['action_protocol'] == protocol['actions'],
                'Calibration protocol differs from frozen design')
        expected = {(f,a) for f in FAMILIES for a in ('fixed-rich','single-6-0')}
        require(len(calibration['records']) == 12 and
                {(r['family'],r['action_id']) for r in calibration['records']} == expected,'Calibration denominator')
        for record in calibration['records']:
            require(record['status'] == 'passed' and finite(record['relative_l2']) and
                    0 <= record['relative_l2'] <= protocol['provider']['calibration_relative_l2_limit'] and
                    finite(record['max_absolute_error']) and
                    0 <= record['max_absolute_error'] <= protocol['provider']['calibration_max_absolute_limit'],
                    'Source refinement bounds failed')
        for relative,value in protocol['calibration_sources'].items():
            require(digest(safe_path(root,relative)) == value,'Changed calibrated source '+relative)
        return {'scientific_files':len(source_map)}
    sources_ok = checks.run('source_protocol_and_manifest_hashes',source_hashes)

    tasks = {t['id']:t for t in prepared.get('tasks',[])}
    truth_by_index = {}
    def paired_tasks():
        for i in range(12):
            pair = [tasks[task_id(i,n)] for n in NOISES]
            require(pair[0]['spec'] == pair[1]['spec'],'Noise conditions changed private parameters')
            spec = pair[0]['spec']
            require(spec['role'] == 'formal' and spec['index'] == i and spec['family'] == FAMILIES[i%6],
                    'Private parameter task identity')
            references = [{'name':n,'coefficient':v} for n,v in spec['coefficients'].items()]
            coefficient_map(references); require(bool(references),'Empty frozen truth')
            ranges = prepared['protocol']['provider']['coefficient_ranges'][spec['family']]
            require(set(spec['coefficients']) == set(ranges) and
                    all(bounds[0] <= spec['coefficients'][name] <= bounds[1] for name,bounds in ranges.items()),
                    'Truth outside predeclared family coefficient bounds')
            require(spec['parameter_seed'] == [prepared['protocol']['provider']['role_seed_bases']['formal'],i],
                    'Formal parameter seed allocation')
            truth_by_index[i] = references
            for task in pair:
                require(task['id'] == task_id(i,task['noise_label']),'Noise condition identity differs')
                compare(task,{'spec_index':i,'noise_level':NOISES[task['noise_label']]},'Paired condition')
                require(task['pool'] == pair[0]['pool'] and len(task['pool']) == 32 and
                        len({a['id'] for a in task['pool']}) == 32,'Common action pool')
                require(task['initial_actions'] == pair[0]['initial_actions'],'Common initial experiments')
                for action in task['pool']:
                    action_digest(action)
                for role, count in (('training',3),('development',2)):
                    failed = [f for f in task['failures'] if f['role'] == role]
                    require(len(task['initial_actions'][role]) == count and
                            len(task['observations'][role])+len(failed) == count,'Initial success/failure denominator')
                    require(len({f['index'] for f in failed}) == len(failed) and
                            all(0 <= f['index'] < count and f['charged_experiments'] == 1 and f['reason'] for f in failed),
                            'Uncharged or unidentified initial source failure')
                    allocated = []
                    for record in task['observations'][role]+failed:
                        index = record['index']; allocated.append(index)
                        require(type(index) is int and 0 <= index < count,'Initial experiment index')
                        action = task['initial_actions'][role][index]
                        h = action_digest(action)
                        key = f"formal:{i}:"+h
                        seed = int(hashlib.sha256(key.encode()).hexdigest()[:8],16)%2**31
                        require(record['action_sha256'] == h and record['noise_seed'] == seed,'Initial action/noise seed binding')
                        if 'file' in record:
                            require(record['file'] == f"active_study/data/initial/{task['id']}/{role}-{index}.npz",'Initial path identity')
                    require(sorted(allocated) == list(range(count)),'Initial attempts repeated or omitted')
        return {'unique_frozen_parameter_tasks':len(truth_by_index),'noise_pairs':12}
    checks.run('private_truth_and_paired_design',paired_tasks)

    observation_cache = {}; accessed_query_files = set(); initial_files = set()
    def load_checked(kind, record):
        base = root if kind == 'root' else output
        path = safe_path(base,record['file'])
        require(digest(path) == record['sha256'],'Changed observation '+record['file'])
        (initial_files if kind == 'root' else accessed_query_files).add(record['file'])
        if str(path) not in observation_cache:
            observation_cache[str(path)] = load_observation(path)
        return observation_cache[str(path)]
    for identity,task in sorted(tasks.items()):
        def initial_check(task=task):
            for role in ('training','development'):
                for record in task['observations'][role]:
                    load_checked('root',record)
                    if 'receipt' in record:
                        paid_receipt(load_checked('root',record),record['receipt'],task['initial_actions'][role][record['index']])
            return {'successful':sum(len(v) for v in task['observations'].values()),'failed':len(task['failures'])}
        checks.run('initial_observations:'+identity,initial_check)

    episodes = {}; receipt_by_id = {}
    for receipt in frozen.get('records',[]):
        identity = receipt['id']; receipt_by_id[identity] = receipt
        def episode_check(receipt=receipt,identity=identity):
            require(receipt['file'] == f'episodes/{identity}.json','Frozen episode filename')
            path = safe_path(output,receipt['file'])
            require(digest(path) == receipt['sha256'],'Changed frozen episode')
            episode = load(path)
            require(episode['id'] == identity and episode['source_hashes'] == frozen['sources'],'Episode hash/identity binding')
            require(identity == episode['task_id']+'--'+episode['policy'],'Episode policy identity')
            detail = verify_episode(episode,tasks[episode['task_id']],load_checked)
            episodes[identity] = episode
            return detail
        checks.run('frozen_episode_and_paid_lineage:'+identity,episode_check)

    final_refs = report.get('final_reference_records',{}); final_files = set()
    def final_inventory():
        require(set(final_refs) == {str(i) for i in range(12)},'Final parameter-task allocation')
        generated = failed = 0
        for i in range(12):
            entries = final_refs[str(i)]
            require(len(entries) == 3,'Three final source slots required')
            for j,entry in enumerate(entries):
                action_digest(entry['action'])
                if entry['status'] == 'generated':
                    relative = f'final_reference/task-{i:02d}/{j}.npz'
                    require(entry['file'] == relative and digest(safe_path(output,relative)) == entry['sha256'],
                            'Final source identity/hash')
                    final_files.add(relative); generated += 1
                    with np.load(output/relative,allow_pickle=False) as archive:
                        require(set(archive.files) == {'x','t','u'},'Unexpected final arrays')
                        x,t,u = archive['x'],archive['t'],archive['u']
                        require(x.shape == (64,) and t.shape == (111,) and u.shape == (111,64) and
                                np.isfinite(u).all() and not np.iscomplexobj(u),'Invalid final field')
                        require(np.allclose(x,np.arange(64)*2*np.pi/64,rtol=0,atol=1e-13) and
                                np.allclose(t,np.linspace(0,1.1,111),rtol=0,atol=1e-13),'Final geometry changed')
                else:
                    require(entry['status'] == 'failed' and entry.get('reason'),'Missing final source failure evidence')
                    failed += 1
        require(generated+failed == 36,'Final source denominator')
        return {'source_slots':36,'generated':generated,'failed':failed,'shared_across_noise_conditions':True}
    summary['final_sources'] = checks.run('36_final_reference_slots_and_hashes',final_inventory)

    def artifact_inventory():
        for subdirectory,expected in [('queries',accessed_query_files),('final_reference',final_files)]:
            actual = {str(p.relative_to(output)) for p in (output/subdirectory).rglob('*.npz')}
            require(actual == expected,'Unaccounted or missing '+subdirectory+' artifact')
        actual_initial = {str(p.relative_to(root)) for p in (root/'active_study/data/initial').rglob('*.npz')}
        require(actual_initial == initial_files,'Unaccounted or missing initial artifact')
        return {'initial_files':len(initial_files),'query_files':len(accessed_query_files),'final_files':len(final_files)}
    checks.run('no_unaccounted_observation_files',artifact_inventory)

    row_derived = []
    for row in report.get('records',[]):
        def row_check(row=row):
            identity = row['id']; budget = row['budget']
            episode_id = row['task_id']+'--'+row['policy']
            episode = episodes[episode_id]; receipt = receipt_by_id[episode_id]
            require(identity == episode_id+f'--budget-{budget}' and budget in BUDGETS,'Budget row identity')
            matches = [(j,c) for j,c in enumerate(episode['checkpoints']) if c['budget'] == budget]
            require(len(matches) <= 1,'Duplicate checkpoint')
            position,checkpoint = matches[0] if matches else (None,None)
            model = checkpoint['model'] if checkpoint else {}; selected = model.get('selected')
            selected_terms = selected['terms'] if selected else []
            compare(row,{'terms':selected_terms,'equation':(selected or {}).get('equation'),
                         'checkpoint_index':position,'model_file':receipt['file'],
                         'frozen_episode_sha256':receipt['sha256'],'spec_index':episode['spec_index'],
                         'noise_label':episode['noise_label'],'noise_level':episode['noise_level'],
                         'family':tasks[episode['task_id']]['spec']['family'],
                         'status':'evaluated' if checkpoint else 'missing_budget_checkpoint',
                         'charged_measurement_scalars':checkpoint['charged_measurement_scalars'] if checkpoint else None,
                         'available_measurement_scalars':checkpoint['available_measurement_scalars'] if checkpoint else None,
                         'query_failure_count_so_far':sum(q['status']=='failed' for q in episode['queries'][:budget-5])},'Frozen row binding')
            references = truth_by_index[episode['spec_index']]
            recovered = recovery(selected_terms,references)
            compare(row['recovery'],recovered,'Independent equation recovery')
            supports,frequencies = support_set(model),inclusion_frequencies(model)
            compare(row,uncertainty_metrics(supports,frequencies,references),'Independent structure uncertainty')
            mean = forecast_summary(row,final_refs[str(episode['spec_index'])])
            worst = max(c['relative_error'] for c in recovered['coefficient_comparison'] if c['reference'] != 0)
            success = bool(recovered['exact_support'] and worst <= .1 and mean is not None and mean <= .05)
            require(row['offline_success'] is success,'Offline success gate differs')
            if selected is None:
                require(row['forecast_failures'] == 3 and mean is None and not success,'Missing model escaped failure denominator')
            row_derived.append({'id':identity,'policy':row['policy'],'budget':budget,'offline_success':success,
                                'exact_support':recovered['exact_support'],'forecast_failures':row['forecast_failures']})
            return {'exact_support':recovered['exact_support'],'offline_success':success,'forecast_failures':row['forecast_failures']}
        checks.run('independent_metrics:'+str(row.get('id')),row_check)

    # Lifecycle receipt details are checked separately from numerical metrics.
    checks.run('global_final_release_lifecycle',lambda:check_release(root,output,prepared,frozen,report))
    checks.run('selection_precedes_provider_call_in_frozen_source',lambda:check_code_order(root))

    prediction_calls = 0
    integrity_ok = all(c['status'] == 'passed' for c in checks.records)
    if integrity_ok and sources_ok:
        legacy_forecast = None
        sampled = checks.run('four_worst_last_budget_conditions',lambda:worst_final_budget_rows(report['records']),keep_details=False)
        for row in sampled or []:
            episode_id = row['task_id']+'--'+row['policy']
            episode = episodes[episode_id]
            model = next((c['model'] for c in episode['checkpoints'] if c['budget'] == 9),{})
            selected = model.get('selected')
            for j,entry in enumerate(final_refs[str(row['spec_index'])]):
                name = row['id']+f'--final-{j}'
                saved = row['forecasts'][j]
                evidence = {'id':name,'status':'not_started'}
                if selected is None or entry['status'] != 'generated':
                    evidence.update(status='retained_failure',reason='No frozen model or no successful final source')
                    checks.run('forecast_recheck:'+name,lambda saved=saved:require(saved['status']=='failed','Missing model/source was scored'))
                    forecast_rechecks.append(evidence)
                    continue
                prediction_calls += 1
                if legacy_forecast is None:
                    sys.path.insert(0,str(root))
                    from prefix_study.forecast import forecast_with_diagnostics as legacy_forecast
                def recompute(row=row,j=j,entry=entry,selected=selected,saved=saved,evidence=evidence):
                    with np.load(safe_path(output,entry['file']),allow_pickle=False) as archive:
                        x,t,u = (archive[k].copy() for k in ('x','t','u'))
                    rng = np.random.default_rng(np.random.SeedSequence([854000,860000+100*row['spec_index']+j]))
                    noise = .4*row['noise_level']*rng.standard_normal(u.shape)
                    prefix = u[:11]+noise[:11]
                    try:
                        prediction,diagnostic = legacy_forecast(selected['terms'],x,t[10:],t[:11],prefix)
                    except (ValueError,RuntimeError,FloatingPointError,np.linalg.LinAlgError) as exc:
                        require(saved['status']=='failed','Independent solver failed a recorded successful forecast')
                        evidence.update(status='confirmed_solver_failure',reason=str(exc))
                        return evidence.copy()
                    require(saved['status']=='evaluated','Independent solver succeeds but recorded forecast failed')
                    metrics = independent_forecast_metrics(prediction,u[10:])
                    for key,value in metrics.items():
                        if type(value) is float:
                            close(saved['clean'][key],value,key,rtol=2e-8,atol=2e-10)
                        else:
                            require(saved['clean'][key] == value,'Independent metric '+key)
                    compare(diagnostic,{'prefix_frames':11,'forecast_duration':1.,'future_observations_used':False},'Legacy solver conditioning')
                    evidence.update(status='recomputed',independent_metrics=metrics,
                                    original_relative_l2=saved['clean']['relative_l2'])
                    return evidence.copy()
                checks.run('forecast_recheck:'+name,recompute)
                if checks.records[-1]['status'] == 'failed':
                    evidence.update(status='failed',reason=checks.records[-1]['reason'])
                forecast_rechecks.append(evidence)
    else:
        checks.records.append({'name':'bounded_independent_forecast_recheck','status':'failed',
                               'reason':'Skipped numerical replay because saved-artifact integrity checks failed.'})
    checks.run('comparison_bytes_unchanged',lambda:require(digest(output/'comparison.json') == original_comparison_sha,'Comparison was modified'))
    checks.run('forecast_replay_budget',lambda:require(prediction_calls <= 12,'More than twelve forecast calls'))
    summary['independently_rederived_rows'] = len(row_derived)
    summary['all_forecast_slots'] = sum(len(r.get('forecasts',[])) for r in report.get('records',[]))
    summary['arms'] = {policy:{str(budget):{
        'denominator':24,'records_verified':sum(r['policy']==policy and r['budget']==budget for r in row_derived),
        'exact_support':sum(r['policy']==policy and r['budget']==budget and r['exact_support'] for r in row_derived),
        'offline_success':sum(r['policy']==policy and r['budget']==budget and r['offline_success'] for r in row_derived)}
        for budget in BUDGETS} for policy in POLICIES}
    first_target = {policy:next((b for b in BUDGETS if
        summary['arms'][policy][str(b)]['records_verified'] == 24 and
        summary['arms'][policy][str(b)]['offline_success']/24 >= .8),None) for policy in POLICIES}
    candidate_budget = first_target['structural_disagreement']
    summary['cost_at_same_80pct_offline_success_target'] = {
        'denominator_per_policy_budget':24,'first_budget_reaching_target':first_target,
        'online_stopping_rule':False,
        'candidate_fraction_saved_vs':{policy:(None if candidate_budget is None or first_target[policy] is None
                                               else 1-candidate_budget/first_target[policy])
                                       for policy in POLICIES if policy != 'structural_disagreement'},
        'unavailable_if_either_policy_never_reaches_target':True}
    return {'status':'passed' if all(c['status']=='passed' for c in checks.records) else 'failed',
            'checks':checks.records,'summary':summary,'comparison_sha256':original_comparison_sha,
            'audit_script_sha256':digest(__file__),'prediction_calls':prediction_calls,
            'forecast_rechecks':forecast_rechecks,'no_data_generation':True,'comparison_modified':False,
            'runtime_chronology_independently_authenticated':False,
            'limitations':['Hashes, persisted lifecycle receipts and source order establish artifact consistency; they cannot independently authenticate all past process execution.',
                           'Only four worst final-budget condition rows, at most twelve forecasts, are numerically replayed; remaining saved forecast metrics are checked for complete arithmetic and failure accounting.',
                           'The 24 noise conditions share twelve parameter tasks and 36 clean final trajectories; policy/condition forecasts are paired, not independent samples.',
                           'This is controlled synthetic validation with an observed prefix and full 1.0 future horizon, not physical measurements or calibrated posterior probabilities.']}


def check_release(root,output,prepared,frozen,report):
    """Schema supplied by the separate lifecycle implementation."""
    path = root/'active_study/data/final_release.json'
    release = load(path)
    require(release['sources'] == frozen['sources'],'Final release source hashes')
    design = {'indices':list(range(12)),'noise_conditions':[[name,level] for name,level in NOISES.items()],
              'provider_role':'formal','initial_action_seed_base':857000,'pool_seed_base':856000,
              'trajectories_per_task':3,'final_action_seed_base':859000,'final_action_pool_slice':[16,19]}
    design_hash = hashlib.sha256(json.dumps(design,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
    compare(release,{'marker_created_before_final_generation':True,
                     'interrupted_first_release_remains_consumed':True,
                     'task_design':design,'task_design_sha256':design_hash},'Final release design')
    first = report['is_first_final_release']
    require(type(first) is bool,'First final release flag')
    compare(report['final_release'],{'file':'active_study/data/final_release.json','sha256':digest(path),
                                     'prepared_sha256':digest(root/'active_study/data/prepared.json'),
                                     'first_prepared_sha256':release['first_prepared_sha256'],
                                     'task_design_sha256':design_hash,
                                     'is_first_final_release':first,'released_at_utc':release['released_at_utc'],
                                     'first_output':release['first_output'],
                                     'first_freeze_index_sha256':release['first_freeze_index_sha256']},'Final release report binding')
    if first:
        require(release['first_freeze_index_sha256'] == digest(output/'freeze_index.json'),'First release global freeze hash')
        require(release['first_prepared_sha256'] == digest(root/'active_study/data/prepared.json'),'First release prepared hash')
    frozen_time = datetime.fromisoformat(frozen['frozen_at_utc'])
    released_time = datetime.fromisoformat(release['released_at_utc'])
    prepared_time = datetime.fromisoformat(prepared['prepared_at_utc'])
    require(prepared_time <= frozen_time,'Global freeze predates prepared inputs')
    if first:
        require(frozen_time <= released_time,'Final release predates global episode freeze')
    expected_label = ('first frozen same-family synthetic validation' if first else
                      'reproduction on previously released final tasks; retrospective, not unseen validation')
    require(report['evaluation_label'] == expected_label,'Repeated final release mislabeled as unseen validation')
    return {'is_first_final_release':first,'release_sha256':digest(path),
            'chronology_is_persisted_evidence_not_independent_authentication':True}


def check_code_order(root):
    source = ast.parse((root/'active_study/controller.py').read_text())
    function = next(n for n in ast.walk(source) if isinstance(n,ast.FunctionDef) and n.name=='run_episode')
    selection = []; charge = []; query = []
    for node in ast.walk(function):
        if not isinstance(node,ast.Call):
            continue
        if isinstance(node.func,ast.Attribute) and isinstance(node.func.value,ast.Name):
            if (node.func.value.id,node.func.attr)==('acquisition','select_action'):selection.append(node.lineno)
            if (node.func.value.id,node.func.attr)==('ledger','append'):charge.append(node.lineno)
        if isinstance(node.func,ast.Name) and node.func.id=='query':query.append(node.lineno)
    require(len(selection)==len(charge)==len(query)==1 and selection[0] < charge[0] < query[0],
            'Frozen controller no longer selects then charges then invokes provider')
    for name in ('controller','engine','acquisition'):
        tree = ast.parse((root/f'active_study/{name}.py').read_text())
        imports = []
        for node in ast.walk(tree):
            if isinstance(node,ast.Import):imports += [a.name for a in node.names]
            elif isinstance(node,ast.ImportFrom):imports += [node.module or '']+[a.name for a in node.names]
        require(not any(value=='provider' or value.endswith('.provider') for value in imports),
                'Learner imports private provider: '+name)
    return {'selection_line':selection[0],'charge_line':charge[0],'query_line':query[0]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=ROOT)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--write-json',type=Path,help='New audit file; default OUTPUT/audit.json. Never overwrite.')
    args = parser.parse_args()
    output = (args.output or args.root/'active_study/results/first_run').resolve()
    destination = args.write_json or output/'audit.json'
    # Preflight exclusively reserving the output prevents wasted numerical
    # replays and ensures an existing audit or comparison is never overwritten.
    destination.parent.mkdir(parents=True,exist_ok=True)
    with destination.open('x',encoding='utf-8') as stream:
        try:
            result = audit(args.root,output)
        except (AuditError,ValueError,TypeError,KeyError,IndexError,OSError,RuntimeError) as exc:
            result = {'status':'failed','checks':[{'name':'unexpected_audit_failure','status':'failed',
                       'reason':f'{type(exc).__name__}: {exc}'}],'no_data_generation':True,'comparison_modified':False}
        text = json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n'
        stream.write(text)
    print(json.dumps({'status':result['status'],'audit_file':str(destination),
                      'checks':len(result['checks']),'prediction_calls':result.get('prediction_calls',0)},ensure_ascii=False))
    return 0 if result['status']=='passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
