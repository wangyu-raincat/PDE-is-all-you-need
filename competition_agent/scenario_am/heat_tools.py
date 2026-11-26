"""Training-only reproduction and read-only historical evaluation access."""
import hashlib
import json
from importlib.metadata import version

import numpy as np

from .catalog import ROOT, evidence, sha
from pde_discovery.real_experiment import load_measurements
from pde_discovery.real_heat import evaluate
from pde_discovery.real_heat_refined import fit_candidates, predict
from pde_discovery.additive_experiment import NEAR_OPTIMAL_FACTOR

ADDITIVE = ('xn', 'xp', 'yn', 'yp')
METALS = tuple(f'{t}{m}' for t in (50, 75, 100) for m in ('ss', 'al', 'cu'))
METHOD = '给定实测边界的四内点条件预测（物理先验 D+h+c）'


def _verify_text_hash(path, expected):
    """Accept only exact bytes or the documented Windows CRLF conversion."""
    actual = sha(path)
    mode = 'exact_bytes'
    if actual != expected:
        normalized = hashlib.sha256(path.read_bytes().replace(b'\r\n', b'\n')).hexdigest()
        if normalized != expected:
            raise ValueError(f'历史文件校验失败：{path.name}')
        mode = 'crlf_to_lf_only'
    return {'actual_sha256': actual, 'historical_sha256': expected, 'match_mode': mode}


def _paths(group):
    if group not in ADDITIVE + METALS:
        raise ValueError('asset_group 必须是 xn/xp/yn/yp 或 50/75/100 + ss/al/cu')
    data = ROOT / 'pde_discovery/data' / ('additive_heat' if group in ADDITIVE else 'mendeley_heat')
    saved = ROOT / 'artifacts' / ('additive_heat' if group in ADDITIVE else 'refined_heat')
    return data, saved


def _frozen(group):
    data, saved = _paths(group)
    path = saved / group / 'frozen_model.json'
    frozen = json.loads(path.read_text(encoding='utf-8'))
    index = json.loads((saved / 'freeze_index.json').read_text(encoding='utf-8'))
    records = index.get('records', [])
    record = next(r for r in records if r.get('group', r.get('case')) == group)
    if record['manifest_sha256'] != frozen['manifest_sha256']:
        raise ValueError('冻结索引与模型的数据划分引用不一致')
    _verify_text_hash(path, record['frozen_model_sha256'])
    _verify_text_hash(data / group / 'manifest.json', frozen['manifest_sha256'])
    return frozen, path


def heat_identify(asset_group: str, split: str = 'train') -> dict:
    """Refit repeat1 and select with repeat2 only; reject final/dev refitting."""
    if split != 'train':
        raise ValueError('仅接受 train：编号1拟合、编号2选模；禁止用开发/留出记录重新拟合')
    data, _ = _paths(asset_group)
    baseline, baseline_path = _frozen(asset_group)
    manifest_path = data / asset_group / 'manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    directory = manifest_path.parent
    training = manifest['splits']['training']
    development = manifest['splits']['development']
    t_train, u_train = load_measurements(directory, training, manifest['csv_format'])
    t_dev, u_dev = load_measurements(directory, development, manifest['csv_format'])
    dx = float(manifest['spacing_m'])
    candidates = fit_candidates(t_train, u_train, dx)
    eligible = []
    for candidate in candidates:
        diagnostics = candidate.get('diagnostics', {})
        rank, count = diagnostics.get('jacobian_rank'), diagnostics.get('parameter_count')
        if not candidate.get('fit_success') or rank is None or count is None or rank < count:
            continue
        try:
            prediction = predict(candidate['terms'], t_dev, u_dev[:, [0, 5]], u_dev[0, 1:5], dx)
            candidate['development'] = evaluate(prediction, u_dev[:, 1:5])
            eligible.append(candidate)
        except (ValueError, RuntimeError, FloatingPointError):
            continue
    selected = None
    if eligible:
        best = min(c['development']['rmse'] for c in eligible)
        near = [c for c in eligible if c['development']['rmse'] <= best * NEAR_OPTIMAL_FACTOR + 1e-12]
        selected = min(near, key=lambda c: (c['support_size'], c['development']['rmse'], c['model_name']))
    previous = baseline['selected']
    same_terms = bool(selected and previous and
                      [t['name'] for t in selected['terms']] == [t['name'] for t in previous['terms']])
    consistent = same_terms and selected['model_name'] == previous['model_name'] and bool(np.allclose(
        [t['coefficient'] for t in selected['terms']], [t['coefficient'] for t in previous['terms']],
        rtol=1e-6, atol=1e-12))
    sources = [evidence(manifest_path), evidence(baseline_path)]
    for role in ('training', 'development'):
        path = (manifest_path.parent / manifest['splits'][role]['file']).resolve()
        sources.append(evidence(path, path.read_text(encoding='utf-8').splitlines()[0]))
    return {'asset_group': asset_group, 'method': METHOD, 'selected': selected,
            'failure_reason': None if selected else '训练/开发阶段没有可用候选', 'terms': selected['terms'] if selected else [],
            'frozen_consistency': {'matches': consistent, 'rtol': 1e-6, 'atol': 1e-12,
                                   'reference_terms': previous['terms'] if previous else [],
                                   'source_path': baseline_path.relative_to(ROOT).as_posix()},
            'engine_versions': {name: version(name) for name in ('numpy', 'scipy')},
            'historical_manifest_integrity': _verify_text_hash(manifest_path, baseline['manifest_sha256']),
            'final_numeric_values_loaded': False, 'new_blind_validation': False,
            'limitations': '六点有效系数；h 为衰减率，非对流换热系数。条件预测不认证真实物性。',
            'evidence': sources}


def heat_holdout_summary(asset_group: str) -> dict:
    """Read saved scores only; 'all' returns exactly twelve additive records."""
    groups = ADDITIVE if asset_group == 'all' else (asset_group,)
    records, sources = [], []
    for group in groups:
        _, saved = _paths(group)
        _, frozen_path = _frozen(group)
        paths = [saved / group / f'repeat{r}/report.json' for r in (3, 4, 5)] if group in ADDITIVE else [saved / group / 'report.json']
        for path in paths:
            report = json.loads(path.read_text(encoding='utf-8'))
            integrity = _verify_text_hash(frozen_path, report['frozen_model_sha256'])
            source = evidence(path)
            records.append({'case': report['case'], 'validation': report['validation'],
                            'source_path': path.relative_to(ROOT).as_posix(),
                            'frozen_integrity': integrity, 'evidence': [source]})
            sources.append(source)
    return {'method': METHOD, 'records': records, 'recomputed': False,
            'new_blind_validation': False, 'evidence': sources}


def heat_compare(group_a: str, group_b: str) -> dict:
    """Compare original selections and, for additive rods, separate descriptive five-repeat D/h fits."""
    a, b = heat_identify(group_a), heat_identify(group_b)
    parameters_a = (a.get('selected') or {}).get('parameters') or {}
    parameters_b = (b.get('selected') or {}).get('parameters') or {}
    differences = [{'parameter': name, 'unit': unit, 'group_a': parameters_a[name],
                    'group_b': parameters_b[name],
                    'difference_b_minus_a': parameters_b[name] - parameters_a[name]}
                   for name, unit in (('D', 'm^2/s'), ('h', 's^-1'), ('c', 'K/s'))
                   if name in parameters_a and name in parameters_b]
    return {'method': METHOD, 'group_a': a, 'group_b': b,
            'parameter_comparison': differences,
            'repeatability': {g: heat_repeatability(g) for g in dict.fromkeys((group_a, group_b)) if g in ADDITIVE},
            'bootstrap_intervals': None,
            'comparison_scope': '训练/开发点估计与增材杆五次独立拟合分别返回。描述性复测离散度不能确立材料物性差异。',
            'evidence': a['evidence'] + b['evidence']}


def heat_repeatability(group: str) -> dict:
    """Descriptive separate fixed D+h+c fits to the five acquisitions of one additive rod."""
    from pde_discovery.real_heat_refined import fit_candidate

    if group not in ADDITIVE:
        raise ValueError('仅支持增材杆 xn/xp/yn/yp')
    data, _ = _paths(group)
    _frozen(group)
    manifest_path = data / group / 'manifest.json'
    manifest = json.loads(manifest_path.read_bytes())
    records = [*manifest['splits'].values(), *manifest['holdouts']]
    if sorted(record['repeat'] for record in records) != list(range(1, 6)):
        raise ValueError('需要五条独立复测记录')
    fits = []
    for record in sorted(records, key=lambda r: r['repeat']):
        path = (manifest_path.parent / record['file']).resolve()
        t, u = load_measurements(manifest_path.parent, record, manifest['csv_format'])
        fit = fit_candidate(t, u, float(manifest['spacing_m']), 'D+h+c')
        ok = fit['fit_success'] and fit['diagnostics']['jacobian_rank'] == 3
        fits.append({'repeat': record['repeat'], 'fit_success': bool(ok),
                     'parameters': fit['parameters'] if ok else None,
                     'reason': None if ok else fit.get('failure_reason') or 'rank or convergence',
                     'evidence': evidence(path, path.read_text(encoding='utf8').splitlines()[0])})
    summary = {}
    if all(row['fit_success'] for row in fits):
        for name, unit in [('D','m^2/s'), ('h','s^-1'), ('c','K/s')]:
            values = [row['parameters'][name] for row in fits]
            summary[name] = {'unit': unit, 'mean': float(np.mean(values)), 'min': min(values),
                             'max': max(values), 'range': max(values)-min(values),
                             'sample_std': float(np.std(values, ddof=1))}
    pairs = [(row['parameters']['D'], row['parameters']['h']) for row in fits if row['fit_success']]
    correlation = None
    if len(pairs) == 5 and summary['D']['range'] > 0 and summary['h']['range'] > 0:
        correlation = float(np.corrcoef(np.array(pairs).T)[0, 1])
    return {'D_h_correlation': {'method': '描述性 Pearson r', 'n': len(pairs), 'r': correlation},
            'group': group, 'scope': '描述性复测离散度', 'model': 'D+h+c', 'fits': fits,
            'summary': summary, 'manifest': evidence(manifest_path, manifest['assignment']),
            'frozen_scores_modified': False}
