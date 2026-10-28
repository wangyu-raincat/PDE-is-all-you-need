"""Frozen isolated L1 comparison; run prepare, freeze, evaluate in order."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from pde_discovery import generic_engine as baseline
from pde_discovery.generic_experiment import forecast_metrics, reference_metrics
from . import engine, fixtures

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / 'l1_study'
DATA = STUDY / 'data'
OUTPUT = STUDY / 'results' / 'first_run'
METHODS = ('plain', 'adaptive', 'irl1')
MODES = ('integrated', 'pointwise_estimated', 'pointwise_exact')
PUBLIC = ('burgers', 'kdv', 'ks')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def read(path):
    return json.loads(Path(path).read_text())


def sources():
    files = list(STUDY.glob('*.py')) + [STUDY/'protocol.json',
        ROOT/'pde_discovery/generic_engine.py', ROOT/'pde_discovery/generic_experiment.py',
        ROOT/'experiments/generic_reference_equations.json',
        STUDY/'reference_review/source_manifest.json',
        STUDY/'reference_review/redo_two_gaussian_third_derivative.patch']
    return {str(p.relative_to(ROOT)): sha(p) for p in sorted(files)}


def load_array(path, exact=False):
    with np.load(path, allow_pickle=False) as z:
        fields = ('x', 't', 'u', 'exact_features', 'exact_ut') if exact else ('x', 't', 'u')
        result = {k: z[k].copy() for k in fields}
    return {**result, 'boundary': 'periodic'}


def data_hashes():
    files = list(DATA.glob('*/*/*.npz'))
    for case in PUBLIC:
        files += list((ROOT/f'pde_discovery/data/generic_pde/prepared/{case}').glob('*.npz'))
        files.append(ROOT/f'artifacts/generic_pde/{case}/frozen_model.json')
    return {str(p.relative_to(ROOT)): sha(p) for p in sorted(files)}


def prepare():
    if (DATA/'prepared.json').exists():
        raise RuntimeError('Prepared study already exists; preserve it rather than overwrite.')
    for family in fixtures.FAMILIES:
        for role in ('training', 'development'):
            for i in range(fixtures.ROLE_COUNTS[role]):
                data = fixtures.generate(family, role, i)
                path = DATA/family/role/f'{i}.npz'
                path.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(path, **{k:data[k] for k in
                    ('x','t','u','exact_features','exact_ut')})
    record = {'sources': sources(), 'data_sha256': data_hashes(),
        'fixture_protocol': fixtures.FIXTURE_PROTOCOL, 'engine_protocol': engine.PROTOCOL,
        'final_analytic_fields_generated': False, 'public_is_retrospective': True}
    write(DATA/'prepared.json', record)
    print('Prepared training/development only; analytic final fields not generated.', flush=True)
    return record


def verify_preparation():
    record = read(DATA/'prepared.json')
    if record['sources'] != sources() or record['data_sha256'] != data_hashes():
        raise ValueError('Study code/protocol or prepared/public bytes changed after preparation.')
    return record


def budget_blocks(blocks, count):
    """Same total rows for one and three trajectories within a backend."""
    budget = 3*(min(len(b['y']) for b in blocks)//3)
    per = budget//count
    selected = []
    for b in blocks[:count]:
        indices = np.linspace(0, len(b['y'])-1, per, dtype=int)
        if len(np.unique(indices)) != per:
            raise ValueError('Row budget requires unique observations.')
        selected.append({'X':b['X'][indices], 'y':b['y'][indices],
            'metadata':{**b.get('metadata', {}), 'available_rows':len(b['y']),
                'selected_rows':per, 'common_total_budget':budget,
                'row_indices_sha256':hashlib.sha256(indices.tobytes()).hexdigest()}})
    return selected


def model_id(suite, case, mode, count, method):
    return f'{suite}--{case}--{mode}--n{count}--{method}'


def expected_ids():
    return {model_id('public', case, 'integrated', 1, method)
            for case in PUBLIC for method in METHODS} | {
        model_id('analytic', family, mode, count, method)
        for family in fixtures.FAMILIES for mode in MODES
        for count in (1, 3) for method in METHODS}


def complete_records(records):
    identities = [r.get('id') for r in records]
    return len(identities) == len(set(identities)) == 63 and set(identities) == expected_ids()


def fit_record(suite, case, mode, count, method, blocks, dev):
    ident = model_id(suite, case, mode, count, method)
    try:
        result = engine.select_candidates(engine.fit_blocks(blocks, method), dev)
    except (ValueError, RuntimeError, FloatingPointError, np.linalg.LinAlgError) as exc:
        result = {'selected':None, 'candidates':[], 'near_optimal_supports':[],
            'failure_reason':str(exc)}
    return {'id':ident, 'suite':suite, 'case':case, 'derivatives':mode,
        'training_trajectories':count, 'method':method, **result,
        'reference_used_for_selection':False, 'final_loaded_during_selection':False,
        'retrospective':suite=='public'}


def freeze(output=OUTPUT):
    output = Path(output)
    if (output/'freeze_index.json').exists():
        raise RuntimeError('Frozen run exists; use a different output for explicit reproduction.')
    prep = verify_preparation()
    records = []
    def save(record):
        path = output/'models'/f"{record['id']}.json"
        write(path, record)
        records.append({'id':record['id'], 'file':str(path.relative_to(output)), 'sha256':sha(path)})
        selected = record['selected']
        print(f"Frozen {len(records):02d}/63 {record['id']}: " +
            (selected['equation'] if selected else record['failure_reason']), flush=True)
    for case in PUBLIC:
        base = ROOT/f'pde_discovery/data/generic_pde/prepared/{case}'
        training = load_array(base/'training.npz')
        dev = [load_array(base/'development.npz')]
        block = engine.integrated_block(training)
        for method in METHODS:
            record = fit_record('public',case,'integrated',1,method,[block],dev)
            if method == 'plain':
                archived = read(ROOT/f'artifacts/generic_pde/{case}/frozen_model.json')['selected']
                actual = record['selected']
                if actual is None or [t['name'] for t in actual['terms']] != [t['name'] for t in archived['terms']]:
                    raise ValueError('Plain adapter changed archived baseline selection.')
                np.testing.assert_allclose([t['coefficient'] for t in actual['terms']],
                    [t['coefficient'] for t in archived['terms']], rtol=1e-9, atol=1e-12)
                record['archived_plain_equation_parity'] = True
            save(record)
    for family in fixtures.FAMILIES:
        train = [load_array(DATA/family/'training'/f'{i}.npz',exact=True) for i in range(3)]
        dev = [load_array(DATA/family/'development'/f'{i}.npz') for i in range(2)]
        for mode in MODES:
            blocks = [engine.integrated_block(d) if mode=='integrated' else
                engine.pointwise_block(d, exact=mode=='pointwise_exact') for d in train]
            for count in (1,3):
                chosen = budget_blocks(blocks,count)
                for method in METHODS:
                    save(fit_record('analytic',family,mode,count,method,chosen,dev))
    assert complete_records(records)
    index = {'sources':sources(), 'preparation_sha256':sha(DATA/'prepared.json'),
        'data_sha256':prep['data_sha256'], 'protocol':read(STUDY/'protocol.json'),
        'engine_protocol':engine.PROTOCOL, 'fixture_protocol':fixtures.FIXTURE_PROTOCOL,
        'records':records, 'all_models_frozen_before_final':True,
        'analytic_final_fields_generated':False}
    write(output/'freeze_index.json',index)
    return index


def verify_freeze(output):
    output = Path(output)
    index = read(output/'freeze_index.json')
    if not complete_records(index['records']) or index['sources']!=sources():
        raise ValueError('Incomplete freeze or changed code/protocol.')
    if index['preparation_sha256']!=sha(DATA/'prepared.json') or index['data_sha256']!=data_hashes():
        raise ValueError('Prepared data or receipt changed after freeze.')
    for record in index['records']:
        if record['file'] != f"models/{record['id']}.json":
            raise ValueError('Frozen model path must match its declared identity.')
        if record['sha256']!=sha(output/record['file']):
            raise ValueError('Frozen model bytes changed: '+record['id'])
        model = read(output/record['file'])
        if model.get('id') != record['id'] or model_id(model['suite'], model['case'],
                model['derivatives'], model['training_trajectories'], model['method']) != record['id']:
            raise ValueError('Frozen model metadata does not match its identity.')
    return index


def promotion_gate(records):
    """Predeclared engineering gate; evaluates only the fixed IRL1 candidate."""
    if not complete_records(records):
        return {'candidate':'irl1', 'numerical_gate_passed':False, 'merged':False,
            'decision':'keep_baseline_do_not_merge',
            'failure_reason':'Missing, duplicate, or undeclared model identities.',
            'checks':[{'name':'all_63_unique_declared_models','passed':False}],
            'regression_tests_required_separately':True}
    by_id = {r['id']:r for r in records}
    candidate = [r for r in records if r['method']=='irl1' and r['derivatives']!='pointwise_exact']
    checks=[]
    public=[r for r in candidate if r['suite']=='public']
    checks.append(('public_at_least_2_exact',sum(r['recovery']['exact_support'] for r in public)>=2))
    checks.append(('public_extras_at_most_2',sum(len(r['recovery']['spurious_terms']) for r in public)<=2))
    analytic_gain=0
    for r in candidate:
        old=by_id[model_id(r['suite'],r['case'],r['derivatives'],r['training_trajectories'],'plain')]
        a,b=r['recovery'],old['recovery']
        checks.append((r['id']+':no_structure_loss',not b['exact_support'] or a['exact_support']))
        checks.append((r['id']+':no_more_missing',len(a['missing_terms'])<=len(b['missing_terms'])))
        checks.append((r['id']+':no_more_forecast_failures',r['forecast_failures']<=old['forecast_failures']))
        checks.append((r['id']+':coefficient',a['coefficient_relative_l2']<=max(1.10*b['coefficient_relative_l2'],1e-4)))
        va,vb=r.get('mean_final_relative_l2'),old.get('mean_final_relative_l2')
        checks.append((r['id']+':prediction',va is not None and vb is not None and va<=max(1.25*vb,1e-3)))
        if r['suite']=='analytic' and a['exact_support'] and not b['exact_support']:
            analytic_gain+=1
    checks.append(('analytic_non_oracle_strict_structure_gain',analytic_gain>=1))
    return {'candidate':'irl1','numerical_gate_passed':all(v for _,v in checks),
        'strict_non_oracle_analytic_gains':analytic_gain,
        'checks':[{'name':k,'passed':bool(v)} for k,v in checks],
        'regression_tests_required_separately':True,'merged':False,
        'decision':'eligible_for_review_only' if all(v for _,v in checks) else 'keep_baseline_do_not_merge'}


def evaluate(output=OUTPUT):
    output=Path(output)
    if (output/'comparison.json').exists():
        raise RuntimeError('Final scores already exist; do not relabel a rerun as unseen validation.')
    index=verify_freeze(output)
    # First final-reference parse and first generation of held-out analytic fields.
    public_refs=read(ROOT/'experiments/generic_reference_equations.json')['references']
    finals={('public',case):[load_array(ROOT/f'pde_discovery/data/generic_pde/prepared/{case}/final.npz')]
        for case in PUBLIC}
    refs={('public',case):public_refs[case]['terms'] for case in PUBLIC}
    final_hashes={}
    for family in fixtures.FAMILIES:
        refs[('analytic',family)]=fixtures.reference_terms(family)
        finals[('analytic',family)]=[]
        for i in range(2):
            d=fixtures.generate(family,'final',i)
            values={k:d[k] for k in ('x','t','u')}
            path=output/'final_analytic'/family/f'{i}.npz'
            path.parent.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(path,**values)
            final_hashes[str(path.relative_to(output))]=sha(path)
            finals[('analytic',family)].append({**values,'boundary':'periodic'})
    records=[]
    for item in index['records']:
        model=read(output/item['file'])
        selected=model['selected']; terms=selected['terms'] if selected else []
        recovery=reference_metrics(terms,refs[(model['suite'],model['case'])])
        metrics=[]
        for d in finals[(model['suite'],model['case'])]:
            try:
                if selected is None: raise ValueError('No selected model.')
                pred=baseline.forecast(terms,d['x'],d['t'],d['u'][0])
                metrics.append({'status':'evaluated',**forecast_metrics(pred,d['u'])})
            except (ValueError,RuntimeError,FloatingPointError,np.linalg.LinAlgError) as exc:
                metrics.append({'status':'failed','reason':str(exc)})
        failures=sum(m['status']!='evaluated' for m in metrics)
        result={k:model[k] for k in ('id','suite','case','derivatives','training_trajectories','method','retrospective')}
        result.update({'equation':selected['equation'] if selected else None,'terms':terms,
            'recovery':recovery,'forecasts':metrics,'forecast_failures':failures,
            'mean_final_relative_l2':float(np.mean([m['relative_l2'] for m in metrics])) if not failures else None,
            'candidate_count':len(model['candidates']),'failure_reason':model['failure_reason'],
            'frozen_model_sha256':item['sha256']})
        records.append(result)
        print(f"Scored {len(records):02d}/63 {model['id']}: exact={recovery['exact_support']} extras={len(recovery['spurious_terms'])}",flush=True)
    summary={'freeze_index_sha256':sha(output/'freeze_index.json'),'records':records,
        'gate':promotion_gate(records),'analytic_final_sha256':final_hashes,
        'public_retrospective':True,'exact_is_oracle_control':True,
        'final_scores_used_to_tune':False,'all_declared_models_reported':len(records)==63}
    write(output/'comparison.json',summary)
    return summary


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('phase',choices=['prepare','freeze','evaluate'])
    parser.add_argument('--output',type=Path,default=OUTPUT)
    args=parser.parse_args()
    with threadpool_limits(limits=1):
        if args.phase=='prepare': prepare()
        elif args.phase=='freeze': freeze(args.output)
        else: evaluate(args.output)


if __name__=='__main__':
    main()
