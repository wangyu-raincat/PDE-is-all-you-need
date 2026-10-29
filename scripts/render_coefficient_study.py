"""Render immutable coefficient-study JSON receipts; never fit or load field arrays.

The prospective candidate is always noise_stable. Ablations do not replace it
after final scoring. Previously viewed AC probes are reported separately.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / 'coefficient_study/results/first_run'
RETROSPECTIVE = ROOT / 'coefficient_study/development/ac_probe'
METHODS = ('prefix_baseline', 'stable_selection', 'noise_corrected', 'noise_stable')
LABELS = ('前缀基线', '稳定性选择', '噪声矩修正', '矩修正＋稳定性')
PLOT_LABELS = ('Prefix baseline', 'Stability selection', 'Moment correction', 'Correction + stability')
FAMILIES = ('allen_cahn', 'low_viscosity_burgers', 'ks_unstable')
FAMILY_LABELS = ('Allen–Cahn', 'Low-viscosity Burgers', 'KS')
CONDITIONS = ('clean', 'noise0p1_s79', 'noise0p1_s109', 'noise1_s79', 'noise1_s109')
SAMPLINGS = ('full', 'coarse')
PREFIX_FRAMES = {'full': 11, 'coarse': 6}
DURATIONS = {'allen_cahn': .5, 'low_viscosity_burgers': 1.5, 'ks_unstable': 20.}
CANDIDATE = 'noise_stable'
COLORS = ('#777d8b', '#3f78a8', '#ce963b', '#087b65')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    def reject(value):
        raise ValueError('Nonfinite JSON constant: ' + value)
    return json.loads(Path(path).read_text(), parse_constant=reject)


def number(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or value < 0:
        raise ValueError(f'{name} must be finite and nonnegative.')
    return float(value)


def coefficient_rows(record):
    rows = record['recovery']['coefficient_comparison']
    truth = [row for row in rows if row['reference'] != 0]
    if not truth or len({row['name'] for row in truth}) != len(truth):
        raise ValueError('Every true coefficient must have one saved comparison.')
    for row in truth:
        number(row['relative_error'], 'True coefficient error')
    return truth


def noise_receipts(model):
    """One receipt per training trajectory, not one per candidate/weak variant."""
    by_trajectory = {}
    for candidate in model.get('candidates', []):
        for i, block in enumerate(candidate.get('diagnostics', {}).get('block_diagnostics', [])):
            noise = block.get('noise_estimation')
            if noise is None:
                continue
            number(noise['estimated_spatial_noise_sigma'], 'Estimated sigma')
            number(noise['estimated_spatial_noise_variance'], 'Estimated variance')
            if noise.get('true_noise_used') is not False or noise.get('reference_used') is not False:
                raise ValueError('Noise receipt must explicitly exclude true noise and reference inputs.')
            if i in by_trajectory and by_trajectory[i] != noise:
                raise ValueError('Noise estimates disagree across variants of the same training trajectory.')
            by_trajectory[i] = noise
    return [{'training_trajectory_index': i, **value} for i, value in sorted(by_trajectory.items())]


def flat_row(record, model):
    if record['id'] != model['id']:
        raise ValueError('Frozen model and score identities disagree.')
    method = record['method']
    for key in ('conditioning', 'final_predictor', 'initial_filter', 'coefficient_selection', 'coefficient_noise_correction'):
        if record.get(key) != model.get(key):
            raise ValueError('Model/score stage metadata differ: ' + key)
    if record.get('conditioning') != 'observed_prefix':
        raise ValueError('All four arms must receive the same observed prefix.')
    if record.get('coefficient_noise_correction') is not (method in ('noise_corrected', 'noise_stable')):
        raise ValueError('Noise-correction arm metadata disagree.')
    forecasts = record['forecasts']
    if len(forecasts) != 3 or any(f.get('status') not in ('evaluated', 'failed') for f in forecasts):
        raise ValueError('Retain exactly three evaluated/failed final trajectory slots.')
    evaluated = [f for f in forecasts if f['status'] == 'evaluated']
    errors = [number(f['clean']['relative_l2'], 'Trajectory prediction error') for f in evaluated]
    failures = 3 - len(evaluated)
    if type(record['forecast_failures']) is not int or record['forecast_failures'] != failures:
        raise ValueError('Forecast failure counter disagrees with saved trajectory slots.')
    mean = record['mean_clean_relative_l2']
    if failures:
        if mean is not None:
            raise ValueError('An incomplete three-trajectory prediction mean must remain null.')
    elif not np.isclose(number(mean, 'Prediction mean'), np.mean(errors), rtol=1e-12, atol=1e-14):
        raise ValueError('Prediction mean disagrees with the complete three-trajectory mean.')
    for forecast in evaluated:
        if (forecast.get('conditioning') != 'observed_prefix'
                or forecast.get('prefix_frames_used') != PREFIX_FRAMES[record['sampling']]
                or not np.isclose(forecast.get('forecast_duration', np.nan), DURATIONS[record['family']], rtol=0, atol=1e-10)):
            raise ValueError('Prefix budget or complete prediction horizon changed.')
    recovery = record['recovery']
    truth = coefficient_rows(record)
    vector = number(recovery['coefficient_relative_l2'], 'Coefficient-vector error')
    exact = recovery['exact_support']
    if type(exact) is not bool or (exact and (recovery['missing_terms'] or recovery['spurious_terms'])):
        raise ValueError('Exact-support flag contradicts missing/extra terms.')
    selected = model.get('selected')
    selected_id = selected['candidate_id'] if selected else None
    if record.get('terms') != (selected['terms'] if selected else []):
        raise ValueError('Scored equation terms differ from frozen selected coefficients.')
    receipt = model.get('reliability_selection')
    if receipt != record.get('reliability_selection'):
        raise ValueError('Selection diagnostics differ from the frozen model.')
    if method in ('stable_selection', 'noise_stable') and selected is not None and not receipt:
        raise ValueError('Stable selector requires an explicit success/fallback receipt.')
    receipt = receipt or {}
    diagnostics = receipt.get('candidate_diagnostics', {})
    chosen = diagnostics.get(selected_id, {})
    noises = noise_receipts(model)
    sigmas = [n['estimated_spatial_noise_sigma'] for n in noises]
    limit = .25 if record['noise_level'] == .01 else .10
    return {
        'id': record['id'], 'family': record['family'], 'sampling': record['sampling'],
        'condition': record['condition'], 'method': method,
        'noise_level': record['noise_level'], 'noise_seed': record['noise_seed'],
        'equation': record.get('equation'), 'selected_candidate_id': selected_id,
        'selected_support_median': bool(selected_id and selected_id.startswith('support-median::')),
        'exact_support': exact, 'extra_count': len(recovery['spurious_terms']),
        'missing_count': len(recovery['missing_terms']),
        'extra_terms': ';'.join(recovery['spurious_terms']), 'missing_terms': ';'.join(recovery['missing_terms']),
        'true_reference_coefficients': json.dumps({r['name']: r['reference'] for r in truth}, sort_keys=True),
        'true_discovered_coefficients': json.dumps({r['name']: r['discovered'] for r in truth}, sort_keys=True),
        'true_term_relative_error_percent': json.dumps({r['name']: 100 * r['relative_error'] for r in truth}, sort_keys=True),
        'extra_term_absolute_errors': json.dumps({r['name']: r['absolute_error'] for r in recovery['coefficient_comparison'] if r['reference'] == 0}, sort_keys=True),
        'coefficient_vector_error_percent': 100 * vector,
        'worst_true_term_error_percent': 100 * max(r['relative_error'] for r in truth),
        'each_true_term_accuracy_passed': all(r['relative_error'] <= limit for r in truth),
        'mean_final_prediction_l2_percent': None if mean is None else 100 * mean,
        'max_individual_prediction_l2_percent': 100 * max(errors) if errors else None,
        'condition_mean_target_passed': mean is not None and mean <= .05 and not failures,
        'forecast_failures': failures, 'evaluated_trajectories': len(errors),
        'conditioning': 'observed_prefix', 'prefix_frames_used': PREFIX_FRAMES[record['sampling']],
        'historical_prefix_duration': .1 * DURATIONS[record['family']],
        'unchanged_future_duration': DURATIONS[record['family']], 'prefix_in_scored_future': False,
        'selection_status': receipt.get('status', 'not_applicable'),
        'selector_fallback_used': receipt.get('fallback_used', False),
        'selector_fallback_reason': receipt.get('fallback_reason'),
        'selector_changed_candidate': bool(receipt and selected_id != receipt.get('baseline_candidate_id')),
        'selector_failed_diagnostics': sum(d.get('status') == 'failed' for d in diagnostics.values()),
        'prediction_pool_size': len(receipt.get('prediction_pool_ids', [])),
        'weak_pool_size': len(receipt.get('weak_pool_ids', [])),
        'selected_weak_validation_score': chosen.get('weak_score'),
        'selected_worst_relative_omission_rms': chosen.get('worst_relative_omission_rms'),
        'noise_receipts_available': bool(noises), 'noise_training_trajectories_reported': len(noises),
        'estimated_training_sigma_min': min(sigmas) if sigmas else None,
        'estimated_training_sigma_max': max(sigmas) if sigmas else None,
        'failure_reason': record.get('failure_reason'),
    }


def group_summary(rows, records):
    means = [r['mean_final_prediction_l2_percent'] / 100 for r in rows if r['mean_final_prediction_l2_percent'] is not None]
    worst = [r['worst_true_term_error_percent'] / 100 for r in rows]
    vectors = [r['coefficient_vector_error_percent'] / 100 for r in rows]
    terms = {}
    for record in records:
        for term in coefficient_rows(record):
            terms.setdefault(term['name'], []).append(term['relative_error'])
    return {
        'configurations': len(rows), 'exact': sum(r['exact_support'] for r in rows),
        'extra': sum(r['extra_count'] for r in rows), 'missing': sum(r['missing_count'] for r in rows),
        'median_vector_error': float(np.median(vectors)), 'max_vector_error': max(vectors),
        'median_worst_true_term_error': float(np.median(worst)), 'max_worst_true_term_error': max(worst),
        'per_true_term': {term: {'count': len(values), 'median_error': float(np.median(values)), 'max_error': max(values)} for term, values in sorted(terms.items())},
        'max_condition_mean_prediction_l2': max(means) if means else None,
        'median_condition_mean_prediction_l2': float(np.median(means)) if means else None,
        'complete_prediction_conditions': len(means),
        'condition_mean_target_passed_count': sum(r['condition_mean_target_passed'] for r in rows),
        'all_condition_means_at_most5pct': all(r['condition_mean_target_passed'] for r in rows),
        'forecast_failures': sum(r['forecast_failures'] for r in rows),
        'evaluated_trajectories': sum(r['evaluated_trajectories'] for r in rows),
        'selector_fallback_count': sum(r['selector_fallback_used'] for r in rows),
        'selector_changed_candidate_count': sum(r['selector_changed_candidate'] for r in rows),
        'selector_failed_diagnostics': sum(r['selector_failed_diagnostics'] for r in rows),
        'selector_status_counts': dict(Counter(r['selection_status'] for r in rows)),
        'selector_fallback_reasons': dict(Counter(r['selector_fallback_reason'] for r in rows if r['selector_fallback_used'])),
        'noise_receipt_conditions': sum(r['noise_receipts_available'] for r in rows),
        'selected_support_median_count': sum(r['selected_support_median'] for r in rows),
    }


def validate_gate(report, methods, rows):
    gate = report['gate']
    if gate.get('candidate') != CANDIDATE or gate.get('merged') is not False:
        raise ValueError('Report must retain the frozen candidate and must not claim a merge.')
    checks = gate.get('checks', [])
    if not checks:
        if gate.get('passed') is not False or not gate.get('reason'):
            raise ValueError('Missing gate results require an explicit failed-gate reason.')
        return {'expected_checks': 126, 'reported_checks': 0, 'checks_passed': 0,
                'passed': False, 'failed_checks': [], 'unscored_reason': gate['reason']}
    if len(checks) != 126 or len({c['name'] for c in checks}) != 126 or any(type(c['passed']) is not bool for c in checks):
        raise ValueError('Exactly 126 unique boolean gate checks are required.')
    checks_by_name = {c['name']: c['passed'] for c in checks}
    expected = {}
    by_id = {row['id']: row for row in rows}
    for row in rows:
        if row['method'] != CANDIDATE:
            continue
        stem = row['id']
        old = by_id['--'.join((row['family'], row['sampling'], row['condition'], 'prefix_baseline'))]
        expected[stem + ':exact_structure'] = row['exact_support'] and not row['extra_count'] and not row['missing_count']
        expected[stem + ':each_true_term_accuracy'] = row['each_true_term_accuracy_passed']
        expected[stem + ':mean_prediction_at_most_5pct'] = row['condition_mean_target_passed']
        expected[stem + ':vs_prefix_baseline:coefficient'] = row['coefficient_vector_error_percent'] <= max(1.25 * old['coefficient_vector_error_percent'], 1.)
    maxima = {f: {m: methods[m]['by_family'][f]['max_worst_true_term_error'] for m in METHODS} for f in FAMILIES}
    for family in FAMILIES:
        expected[family + ':vs_prefix_baseline:worst_true_term_nonregression'] = maxima[family][CANDIDATE] <= max(1.25 * maxima[family]['prefix_baseline'], .01)
    ac_old, ac_new = maxima['allen_cahn']['prefix_baseline'], maxima['allen_cahn'][CANDIDATE]
    expected['allen_cahn:worst_true_term_relative_reduction_at_least_20pct'] = ac_old > 0 and ac_new <= .8 * ac_old
    expected['all_families:maximum_true_term_error_not_increased'] = max(maxima[f][CANDIDATE] for f in FAMILIES) <= max(maxima[f]['prefix_baseline'] for f in FAMILIES)
    expected['source_quality'] = all(q.get('status') == 'passed' for q in report['quality'].values()) and not any(e is not None for errors in report['final_reference_failures'].values() for e in errors)
    if checks_by_name != expected or gate['passed'] is not all(expected.values()):
        raise ValueError('Saved promotion gate disagrees with independently summarized evidence.')
    if gate['prediction_target_passed'] is not methods[CANDIDATE]['all_condition_means_at_most5pct']:
        raise ValueError('Saved prediction target disagrees with complete condition means.')
    return {'expected_checks': 126, 'reported_checks': 126, 'checks_passed': sum(expected.values()),
            'passed': all(expected.values()), 'failed_checks': [k for k, v in expected.items() if not v]}


def load_final(output):
    path = output / 'comparison.json'
    if not path.is_file():
        raise ValueError(f'Final comparison does not exist; rendering never starts fitting/scoring: {path}')
    report = read_json(path)
    expected = {'--'.join((f, s, n, m)) for f in FAMILIES for s in SAMPLINGS for n in CONDITIONS for m in METHODS}
    records = report['records']
    if len(records) != 120 or {r['id'] for r in records} != expected:
        raise ValueError('Require all 120 unique declared final records, including failures.')
    for key, value in {'all_declared_models_reported': True, 'final_scores_used_for_tuning': False,
                       'all_data_numerical_not_real_measurement': True}.items():
        if report.get(key) is not value:
            raise ValueError('Missing final provenance marker: ' + key)
    frozen_path = output / 'freeze_index.json'
    frozen = read_json(frozen_path)
    if digest(frozen_path) != report['freeze_index_sha256'] or frozen.get('all_120_frozen_before_final') is not True:
        raise ValueError('Global pre-final freeze receipt does not match final scores.')
    if len(frozen['records']) != 120 or {r['id'] for r in frozen['records']} != expected:
        raise ValueError('Require all 120 frozen models.')
    frozen_records = {r['id']: r for r in frozen['records']}
    for relative, checksum in frozen['sources'].items():
        source = (ROOT / relative).resolve()
        if not source.is_relative_to(ROOT) or digest(source) != checksum:
            raise ValueError('Frozen scientific source changed: ' + relative)
    models = {}
    rows = []
    for record in records:
        identity = record['id']
        if identity != '--'.join(record[k] for k in ('family', 'sampling', 'condition', 'method')):
            raise ValueError('Record identity fields contradict its ID.')
        frozen_model = frozen_records[identity]
        if frozen_model['file'] != f'models/{identity}.json':
            raise ValueError('Unexpected frozen model path.')
        model_path = output / frozen_model['file']
        if digest(model_path) != frozen_model['sha256'] or frozen_model['sha256'] != record['frozen_model_sha256']:
            raise ValueError('Frozen model bytes disagree with final scoring receipt.')
        model = read_json(model_path)
        if model.get('sources') != frozen['sources'] or model.get('input_hashes') != frozen['input_hashes']:
            raise ValueError('Model dependencies differ from the global freeze.')
        models[identity] = model
        rows.append(flat_row(record, model))
    methods = {}
    for method in METHODS:
        group = [r for r in records if r['method'] == method]
        selected_rows = [r for r in rows if r['method'] == method]
        methods[method] = group_summary(selected_rows, group)
        methods[method]['by_family'] = {
            family: group_summary([r for r in selected_rows if r['family'] == family], [r for r in group if r['family'] == family])
            for family in FAMILIES}
    gates = validate_gate(report, methods, rows)
    return {'record_count': 120, 'candidate_frozen_before_final': CANDIDATE, 'final_winner_switch_allowed': False,
            'metrics_unit': 'fractions in summary groups; percent in CSV/condition rows',
            'primary': 'exact support, extra/missing terms, per-true-coefficient errors',
            'secondary': 'maximum condition mean of three full-horizon prediction errors',
            'all_data_numerical_not_real_measurement': True, 'all_arms_identical_observed_prefix': True,
            'prefix_frames': PREFIX_FRAMES, 'future_durations': DURATIONS, 'prefix_in_scored_future': False,
            'methods': methods, 'conditions': rows, 'gate': report['gate'], 'gate_summary': gates,
            'noise_diagnostics': {identity: noise_receipts(model) for identity, model in models.items() if model['method'] in ('noise_corrected', 'noise_stable')},
            'source_hashes': {'comparison.json': digest(path), 'freeze_index.json': digest(frozen_path),
                              'renderer': digest(__file__), 'scientific_sources': frozen['sources'],
                              'frozen_models': {r['file']: r['sha256'] for r in frozen['records']},
                              'training_development_input_hashes_from_freeze': frozen['input_hashes'],
                              'final_reference_hashes_from_scoring': report['final_reference_sha256']},
            'array_files_loaded_by_renderer': False}, records


def load_retrospective(directory):
    path = directory / 'comparison.json'
    if not path.is_file():
        return {'available': False, 'reason': 'No saved AC retrospective comparison.', 'pooled_with_final': False}
    report = read_json(path)
    frozen_path = directory / 'freeze.json'
    frozen = read_json(frozen_path)
    if (report.get('retrospective') is not True or report.get('final_fields_loaded') is not False
            or report.get('formal_fields_generated') is not False or digest(frozen_path) != report['freeze_sha256']):
        raise ValueError('Retrospective provenance/freeze receipt is inconsistent.')
    records = report['records']
    expected = {'--'.join(('allen_cahn', 'coarse', n, m)) for n in ('noise1_s71', 'noise1_s103') for m in METHODS}
    if len(records) != 8 or {r['id'] for r in records} != expected:
        raise ValueError('Retrospective report must retain two AC cases for all four arms.')
    receipts = {r['id']: r for r in frozen['records']}
    rows = []
    for record in records:
        receipt = receipts[record['id']]
        model_path = directory / 'models' / (record['id'] + '.json')
        if digest(model_path) != receipt['sha256']:
            raise ValueError('Retrospective model bytes differ from saved freeze.')
        model = read_json(model_path)
        if record['terms'] != (model['selected']['terms'] if model['selected'] else []):
            raise ValueError('Retrospective scored coefficients differ from the frozen model.')
        true_terms = coefficient_rows(record)
        rows.append({'id': record['id'], 'condition': record['condition'], 'method': record['method'],
                     'exact_support': record['recovery']['exact_support'],
                     'extra_count': len(record['recovery']['spurious_terms']), 'missing_count': len(record['recovery']['missing_terms']),
                     'worst_true_term_error_percent': 100 * max(c['relative_error'] for c in true_terms),
                     'development_rmse': record.get('development_rmse'),
                     'selected_candidate_id': record.get('selected_id'),
                     'coefficient_comparison': true_terms})
    return {'available': True, 'retrospective': True, 'pooled_with_final': False, 'case_count': 2,
            'prediction_scope': 'development RMSE only; no final prediction score', 'conditions': rows,
            'source_hashes': {'comparison.json': digest(path), 'freeze.json': digest(frozen_path),
                              'frozen_models': {r['file']: r['sha256'] for r in frozen['records']},
                              'historical_sources_from_freeze': frozen['sources']}}


def write_csv(path, rows):
    if not rows:
        raise ValueError('Do not emit an empty results table.')
    with Path(path).open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def percent(value):
    return '缺失/失败' if value is None else f'{100 * value:.4f}%'


def render_markdown(summary):
    gate = summary['gate_summary']
    conclusion = '通过冻结的效果门槛，仅可进入回归检查与合入评审；本报告未执行合并。' if gate['passed'] else '未通过冻结门槛，保留前缀基线，不合并。'
    lines = ['# PDE 系数可靠性对照实验', '', conclusion, '',
             f"冻结候选为 `noise_stable`，门槛通过 **{gate['checks_passed']}/{gate['expected_checks']}**；最终验证不根据四臂排名更换候选。回归测试须另行通过。", '',
             '## 任务与隔离', '',
             '四臂共享相同训练/开发/最终划分、观测噪声、11 帧（完整网格）或 6 帧（粗网格）历史观测，以及完整预测时长：Allen–Cahn 0.5、Burgers 1.5、KS 20。历史长度为预测时长的 10%，历史不计入预测评分，起始帧不计入误差。每条件有 3 条最终轨迹，三者完整均值才可判定 5% 目标。', '',
             '共 3 类方程 × 2 种采样 × 5 种噪声条件 × 4 臂 = 120 条记录。数据是数值生成场；不是实测或新方程族的泛化证明。训练 3 条、开发 2 条、最终 3 条轨迹；全部 120 个模型冻结后才生成最终场。所有报告数字只读取保存的评分与模型 JSON。', '',
             '## 主要指标：结构与系数', '',
             '| 方法 | 完整恢复 | 多余项 | 缺失项 | 系数向量误差中位数 | 最坏真实项误差 |',
             '|---|---:|---:|---:|---:|---:|']
    for method, label in zip(METHODS, LABELS):
        row = summary['methods'][method]
        lines.append(f"| {label} | {row['exact']}/30 | {row['extra']} | {row['missing']} | {percent(row['median_vector_error'])} | {percent(row['max_worst_true_term_error'])} |")
    lines += ['', '系数向量范数可能掩盖量级很小的扩散项，因此逐真实项误差为主要依据；逐条件、逐项的参考值、识别值、误差见 `coefficients.csv`。以下每格为该方程 10 个条件中该项的最大相对误差。', '',
              '| 方程 / 真实项 | 前缀基线 | 稳定性选择 | 噪声矩修正 | 矩修正＋稳定性 |', '|---|---:|---:|---:|---:|']
    for family, label in zip(FAMILIES, FAMILY_LABELS):
        for term in summary['methods']['prefix_baseline']['by_family'][family]['per_true_term']:
            values = [percent(summary['methods'][m]['by_family'][family]['per_true_term'][term]['max_error']) for m in METHODS]
            lines.append('| ' + label + ' / `' + term + '` | ' + ' | '.join(values) + ' |')
    lines += ['', '![逐方程系数与结构比较](coefficient_comparison.png)', '', '## 辅助指标：完整时长预测', '',
              '| 方法 | 最坏条件平均误差 | ≤5% 条件 | 完整均值条件 | 失败轨迹 |', '|---|---:|---:|---:|---:|']
    for method, label in zip(METHODS, LABELS):
        row = summary['methods'][method]
        lines.append(f"| {label} | {percent(row['max_condition_mean_prediction_l2'])} | {row['condition_mean_target_passed_count']}/30 | {row['complete_prediction_conditions']}/30 | {row['forecast_failures']}/90 |")
    lines += ['', '出现失败时，表内最大值仅覆盖完整均值，不能据此声称全部达标；失败条件仍保留在 30 条分母中。5% 指每条件三条轨迹的平均误差，不要求每条轨迹、每个系数都低于 5%。', '',
              '![完整时长预测比较](prediction_comparison.png)', '', '## 冻结门槛', '',
              '126 项 = 1 项源质量 + 30 ×（结构完整、逐真实项精度、预测均值 ≤5%、系数向量不退化）+ 3 项方程族尾部不退化 + 2 项系数改善。无噪声/0.1% 噪声逐项上限 10%，1% 噪声上限 25%；系数不退化阈值为 max(1.25 × 基线, 1%)。改善要求 AC 最坏真实项相对降低至少 20%，且所有方程中的最坏真实项误差不增加。', '']
    improvement = summary['gate'].get('coefficient_improvement', {})
    if improvement:
        lines.append(f"AC 最坏真实项：{percent(improvement['ac_baseline_worst'])} → {percent(improvement['ac_candidate_worst'])}，相对降低 {percent(improvement['ac_relative_reduction'])}。全局最坏：{percent(improvement['global_baseline_worst'])} → {percent(improvement['global_candidate_worst'])}。")
        lines.append('')
    if gate.get('unscored_reason'):
        lines += ['门槛未能完整评分：' + gate['unscored_reason'], '']
    if gate['failed_checks']:
        lines += ['失败项：', ''] + ['- `' + name + '`' for name in gate['failed_checks']] + ['']
    lines += ['## 选择与噪声诊断', '',
              '稳定性选择只在开发预测 RMSE 最优值的 1.05 倍近优池中比较最小支持，再以共同弱形式验证残差筛选 1.05 倍近优池，最后选择整条训练轨迹留一扰动最小的系数。该扰动是敏感性诊断，不是置信区间；诊断失败显式回退到原预测选择。', '',
              '| 方法 | 改变选择 | 回退 | 失败的候选诊断 | 有训练噪声回执 |', '|---|---:|---:|---:|---:|']
    for method, label in zip(METHODS, LABELS):
        row = summary['methods'][method]
        lines.append(f"| {label} | {row['selector_changed_candidate_count']}/30 | {row['selector_fallback_count']}/30 | {row['selector_failed_diagnostics']} | {row['noise_receipt_conditions']}/30 |")
    for method, label in zip(METHODS, LABELS):
        row = summary['methods'][method]
        for reason, count in row['selector_fallback_reasons'].items():
            lines += ['', f'回退原因（{label}，{count} 次）：{reason}', '']
    lines += ['', '冻结融合候选的观测噪声估计诊断如下，范围来自各训练轨迹，不是独立噪声重复实验。σ 使用各方程场的原始单位，跨方程不直接比较。', '',
              '| 方程 | 无噪声条件估计 σ 范围 | 所有条件估计 σ 范围 |', '|---|---:|---:|']
    for family, label in zip(FAMILIES, FAMILY_LABELS):
        clean, all_sigma = [], []
        for row in summary['conditions']:
            if row['method'] != CANDIDATE or row['family'] != family:
                continue
            estimates = [r['estimated_spatial_noise_sigma'] for r in summary['noise_diagnostics'].get(row['id'], [])]
            all_sigma.extend(estimates)
            if row['condition'] == 'clean':
                clean.extend(estimates)
        ranges = [f'{min(values):.4g} — {max(values):.4g}' if values else '无保存回执' for values in (clean, all_sigma)]
        lines.append('| ' + label + ' | ' + ' | '.join(ranges) + ' |')
    lines += ['', '噪声矩修正只从观测高频谱估计方差，按 2N Fourier 插值后的点方差修正 u²−v 与 u³−3vu；u·uₓ 同样使用修正平方。需要零均值、空间白高斯、轨迹内恒定方差及高频信号较弱的假设；真实高频物理成分会污染估计，干净场也可能得到正噪声估计。它不消除全部变量含噪回归偏差。`summary.json` 保存按训练轨迹去重的回执；不同弱形式、插值点和重叠窗口不作为独立样本。', '',
              '## 已查看 AC 数据的回顾性诊断', '']
    retro = summary['retrospective_diagnostic']
    if retro['available']:
        lines += ['以下为此前已查看的两种 AC 粗网格 1% 噪声条件，仅用于问题诊断，不计入上方最终验证、126 项门槛或 5% 达标结果；开发 RMSE 不是最终预测误差。', '',
                  '| 条件 | 方法 | 完整恢复 | 最坏真实项误差 | 开发 RMSE |', '|---|---|---:|---:|---:|']
        for row in retro['conditions']:
            rmse = row['development_rmse']
            lines.append(f"| {row['condition']} | {LABELS[METHODS.index(row['method'])]} | {'是' if row['exact_support'] else '否'} | {row['worst_true_term_error_percent']:.4f}% | {rmse:.8g} |" if rmse is not None else f"| {row['condition']} | {row['method']} | 否 | {row['worst_true_term_error_percent']:.4f}% | 缺失/失败 |")
    else:
        lines.append('没有可验证的已保存回顾性记录；未填造结果。')
    lines += ['', '## 可追溯性与重现', '',
              '渲染不读取场数组、不重新训练、重新选模或评分。`summary.json` 记录本报告输入、全局冻结、全部模型、科学代码、原始输入与最终参考场的 SHA-256；后两类数组哈希来自冻结/评分回执，渲染阶段不重新读取数组。CSV 使用 LF 换行。', '',
              '```bash', 'python scripts/render_coefficient_study.py --output coefficient_study/results/first_run', '```', '',
              '该命令需要已有 `comparison.json` 与匹配的冻结模型。缺文件、哈希不匹配、改变历史预算、部分轨迹伪装完整均值或门槛矛盾会明确失败。新实验结果不可反过来调整本次协议后继续称为未见验证。', '']
    return '\n'.join(lines)


def render_figures(summary, output):
    plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False})
    x = np.arange(3)
    width = .19
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.65))
    for i, (method, label, color) in enumerate(zip(METHODS, PLOT_LABELS, COLORS)):
        values = [100 * summary['methods'][method]['by_family'][f]['max_worst_true_term_error'] for f in FAMILIES]
        supports = [summary['methods'][method]['by_family'][f]['exact'] for f in FAMILIES]
        bars = axes[0].bar(x + (i - 1.5) * width, values, width, label=label, color=color)
        axes[0].bar_label(bars, fmt='%.2f', fontsize=7, padding=2, rotation=90)
        axes[1].bar(x + (i - 1.5) * width, supports, width, color=color)
    axes[0].set(ylabel='Worst true-term coefficient error (%)', ylim=(0, max(1., axes[0].get_ylim()[1]) * 1.28), title='Primary: each true coefficient')
    axes[1].set(ylabel='Exact equation recovery (of 10)', ylim=(0, 11.2), title='Primary: complete support')
    for ax in axes:
        ax.set_xticks(x, FAMILY_LABELS)
        ax.grid(axis='y', alpha=.2)
        ax.set_axisbelow(True)
    fig.legend(*axes[0].get_legend_handles_labels(), loc='lower center', ncol=4, frameon=False, fontsize=8)
    fig.suptitle('Frozen four-arm comparison; identical observed-prefix budget', fontsize=11)
    fig.tight_layout(rect=(0, .08, 1, .95))
    fig.savefig(output / 'coefficient_comparison.png', dpi=180)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8.6, 3.8))
    for i, (method, label, color) in enumerate(zip(METHODS, PLOT_LABELS, COLORS)):
        group = [summary['methods'][method]['by_family'][f] for f in FAMILIES]
        values = [100 * r['max_condition_mean_prediction_l2'] if r['max_condition_mean_prediction_l2'] is not None else np.nan for r in group]
        bars = ax.bar(x + (i - 1.5) * width, values, width, label=label, color=color)
        for bar, value, row in zip(bars, values, group):
            annotation = (f'{value:.2f}' if np.isfinite(value) else 'N/A') + (f"\n{row['complete_prediction_conditions']}/10" if row['complete_prediction_conditions'] < 10 else '')
            ax.text(bar.get_x() + bar.get_width() / 2, value + .08 if np.isfinite(value) else .1, annotation, ha='center', va='bottom', fontsize=8)
    ax.axhline(5, color='#a12838', linestyle='--', linewidth=1, label='5% condition-mean target')
    ax.set(ylabel='Maximum complete condition mean relative L2 (%)', title='Secondary: full future AC 0.5 / Burgers 1.5 / KS 20', ylim=(0, max(5., ax.get_ylim()[1]) * 1.2))
    ax.set_xticks(x, FAMILY_LABELS)
    ax.grid(axis='y', alpha=.2)
    ax.set_axisbelow(True)
    fig.legend(*ax.get_legend_handles_labels(), loc='lower center', ncol=3, frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, .13, 1, 1))
    fig.savefig(output / 'prediction_comparison.png', dpi=180)
    plt.close(fig)


def main(output=DEFAULT_OUTPUT, retrospective=RETROSPECTIVE):
    output = Path(output)
    summary, records = load_final(output)
    summary['retrospective_diagnostic'] = load_retrospective(Path(retrospective))
    term_rows = []
    for record in records:
        for term in record['recovery']['coefficient_comparison']:
            term_rows.append({k: record[k] for k in ('id', 'family', 'sampling', 'condition', 'method')} |
                             {'term': term['name'], 'scope': 'true_term' if term['reference'] != 0 else 'extra_term',
                              'reference': term['reference'], 'discovered': term['discovered'],
                              'absolute_error': term['absolute_error'],
                              'relative_error_percent': None if term['relative_error'] is None else 100 * term['relative_error']})
    write_csv(output / 'summary.csv', summary['conditions'])
    write_csv(output / 'coefficients.csv', term_rows)
    render_figures(summary, output)
    (output / 'REPORT.md').write_text(render_markdown(summary), encoding='utf-8')
    summary['artifact_hashes'] = {name: digest(output / name) for name in ('summary.csv', 'coefficients.csv', 'coefficient_comparison.png', 'prediction_comparison.png', 'REPORT.md')}
    (output / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps({'output': str(output), 'candidate': CANDIDATE, 'gate': summary['gate_summary'], 'records': len(records)}, ensure_ascii=False))
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--retrospective', type=Path, default=RETROSPECTIVE)
    args = parser.parse_args()
    main(args.output, args.retrospective)
