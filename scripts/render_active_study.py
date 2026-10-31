"""Summarize saved active-study receipts, without importing discovery or field data.

Failures and absent conditions retain their place in each planned denominator.
Bootstrap frequencies are descriptive stability measurements, not calibrated
probabilities. Rendering never selects another equation or changes a checkpoint.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / 'active_study/results/first_run'
POLICIES = ('fixed_rich', 'random', 'ensemble_variance', 'structural_disagreement')
LABELS = {'fixed_rich': '固定丰富激励', 'random': '随机实验',
          'ensemble_variance': '集成预测方差', 'structural_disagreement': '结构分歧'}
PLOT_LABELS = ('Fixed rich', 'Random', 'Ensemble variance', 'Structural disagreement')
BUDGETS = (5, 6, 7, 8, 9)
COLORS = ('#687181', '#c58132', '#477aaa', '#007f6a')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    def reject(value):
        raise ValueError('Nonfinite JSON constant: ' + value)
    return json.loads(Path(path).read_text(encoding='utf-8'), parse_constant=reject)


def number(value, name, nullable=False):
    if value is None and nullable:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError(f'{name} must be finite and nonnegative.')
    return float(value)


def safe_child(directory, name):
    root = Path(directory).resolve()
    path = (root / name).resolve()
    if path == root or root not in path.parents:
        raise ValueError('Receipt path must stay inside the selected result directory.')
    return path


def flat_row(record):
    recovery = record.get('recovery') or {}
    exact = recovery.get('exact_support', False)
    if type(exact) is not bool:
        raise ValueError('exact_support must be a boolean.')
    missing, extra = recovery.get('missing_terms', []), recovery.get('spurious_terms', [])
    if exact and (missing or extra):
        raise ValueError('Exact-support flag contradicts missing or spurious terms.')
    truth = [r for r in recovery.get('coefficient_comparison', []) if r['reference'] != 0]
    if len({r['name'] for r in truth}) != len(truth):
        raise ValueError('Duplicate reference coefficient.')
    worst = max((number(r['relative_error'], 'True coefficient error') for r in truth), default=None)
    forecasts = record.get('forecasts', [])
    if len(forecasts) != 3 or any(f.get('status') not in ('evaluated', 'failed') for f in forecasts):
        raise ValueError('Every record must retain three evaluated/failed final trajectory slots.')
    errors = [number(f['clean']['relative_l2'], 'Forecast error') for f in forecasts if f['status'] == 'evaluated']
    for forecast in forecasts:
        if forecast['status'] == 'evaluated' and (forecast.get('conditioning') != 'observed_prefix'
                or forecast.get('prefix_frames_used') != 11
                or not math.isclose(forecast.get('forecast_duration', -1), 1., abs_tol=1e-12)):
            raise ValueError('Final forecast must retain the declared 11-frame prefix and full horizon.')
    mean = number(record.get('mean_clean_relative_l2'), 'Mean forecast error', nullable=True)
    if len(errors) < 3:
        if mean is not None:
            raise ValueError('Incomplete final trajectory mean must be null.')
    elif mean is None or not math.isclose(mean, sum(errors)/3, rel_tol=1e-10, abs_tol=1e-12):
        raise ValueError('Saved mean differs from its complete three-trajectory mean.')
    supports = record.get('candidate_supports', [])
    set_size = record.get('set_size', len(supports))
    if type(set_size) is not int or set_size < 0 or set_size != len({tuple(sorted(s)) for s in supports}):
        raise ValueError('Candidate set size must count distinct saved structures.')
    coverage = record.get('set_contains_true_support', False)
    if type(coverage) is not bool or (coverage and set_size == 0):
        raise ValueError('Invalid candidate set coverage.')
    frequencies = record.get('frequencies', {})
    if not isinstance(frequencies, dict):
        raise ValueError('Term frequencies must be a dictionary.')
    for value in frequencies.values():
        if value is not None and number(value, 'Term frequency') > 1:
            raise ValueError('Term frequency must lie in [0,1].')
    positive = number(record.get('term_brier_positive'), 'Positive-term Brier', nullable=True)
    negative = number(record.get('term_brier_negative'), 'Negative-term Brier', nullable=True)
    if any(value is not None and value > 1 for value in (positive, negative)):
        raise ValueError('Brier score must lie in [0,1].')
    budget = record['budget']
    if type(budget) is not int or budget not in BUDGETS:
        raise ValueError('Unknown experiment budget.')
    if record['policy'] not in POLICIES:
        raise ValueError('Unknown acquisition policy.')
    cost = record.get('charged_measurement_scalars')
    absent = record.get('status') == 'missing_budget_checkpoint'
    if not (cost is None and absent) and (type(cost) is not int or cost < 0):
        raise ValueError('Measurement cost must be a nonnegative integer.')
    if cost is not None and cost != budget*7104:
        raise ValueError('Charged measurements differ from the declared equal-cost experiment budget.')
    return {
        'task_id': record['task_id'], 'spec_index': record['spec_index'], 'family': record['family'],
        'noise_level': record['noise_level'], 'policy': record['policy'], 'budget': budget,
        'status': record.get('status', 'unknown'), 'exact_support': exact,
        'missing_count': len(missing), 'extra_count': len(extra),
        'missing_terms': ';'.join(missing), 'extra_terms': ';'.join(extra),
        'worst_true_coefficient_relative_error': worst,
        'coefficient_relative_l2': number(recovery.get('coefficient_relative_l2'), 'Coefficient vector error', nullable=True),
        'mean_clean_relative_l2': mean, 'forecast_failures': 3-len(errors),
        'success': bool(exact and worst is not None and worst <= .10 and mean is not None and mean <= .05),
        'set_contains_true_support': coverage, 'set_size': set_size,
        'singleton_set': set_size == 1, 'wrong_singleton_set': set_size == 1 and not coverage,
        'term_brier_positive': positive, 'term_brier_negative': negative,
        'balanced_term_brier': (positive+negative)/2 if positive is not None and negative is not None else None,
        'charged_measurement_scalars': cost,
        'frequencies': json.dumps(frequencies, ensure_ascii=False, sort_keys=True),
        'model_file': record.get('model_file'), 'checkpoint_index': record.get('checkpoint_index'),
    }


def summarize(comparison, *, expected_conditions=None):
    records = comparison.get('records')
    if not isinstance(records, list) or not records:
        raise ValueError('No saved final-evaluation records are available.')
    expected = expected_conditions if expected_conditions is not None else comparison.get('expected_conditions', 24)
    if type(expected) is not int or expected < 1:
        raise ValueError('Expected condition count must be positive.')
    rows = [flat_row(record) for record in records]
    identities = [(r['task_id'], r['policy'], r['budget']) for r in rows]
    if len(identities) != len(set(identities)):
        raise ValueError('Duplicate task/policy/budget result; refusing double counting.')
    tasks = {r['task_id'] for r in rows}
    if len(tasks) > expected:
        raise ValueError('More condition identities than the planned denominator.')
    metadata = {}
    for row in rows:
        value = row['spec_index'], row['family'], row['noise_level']
        if row['task_id'] in metadata and metadata[row['task_id']] != value:
            raise ValueError('Condition metadata differ between paired policy/budget records.')
        metadata[row['task_id']] = value
    groups = []
    for policy in POLICIES:
        for budget in BUDGETS:
            group = [r for r in rows if r['policy'] == policy and r['budget'] == budget]
            available = lambda key: [r[key] for r in group if r[key] is not None]
            brier, forecasts = available('balanced_term_brier'), available('mean_clean_relative_l2')
            coefficients = available('worst_true_coefficient_relative_error')
            positive, negative = available('term_brier_positive'), available('term_brier_negative')
            groups.append({
                'policy': policy, 'budget': budget, 'planned_conditions': expected,
                'present_conditions': len(group), 'missing_conditions': expected-len(group),
                'exact_count': sum(r['exact_support'] for r in group),
                'exact_rate': sum(r['exact_support'] for r in group)/expected,
                'success_count': sum(r['success'] for r in group),
                'success_rate': sum(r['success'] for r in group)/expected,
                'coverage_count': sum(r['set_contains_true_support'] for r in group),
                'coverage_rate': sum(r['set_contains_true_support'] for r in group)/expected,
                'mean_set_size': sum(r['set_size'] for r in group)/len(group) if group else None,
                'singleton_count': sum(r['singleton_set'] for r in group),
                'wrong_singleton_count': sum(r['wrong_singleton_set'] for r in group),
                'wrong_singleton_rate_all_conditions': sum(r['wrong_singleton_set'] for r in group)/expected,
                'extra_terms_total': sum(r['extra_count'] for r in group),
                'missing_terms_total': sum(r['missing_count'] for r in group),
                'forecast_failure_slots': sum(r['forecast_failures'] for r in group),
                'complete_forecast_conditions': len(forecasts),
                'worst_mean_clean_relative_l2': max(forecasts) if forecasts else None,
                'coefficient_available_conditions': len(coefficients),
                'mean_worst_true_coefficient_relative_error': sum(coefficients)/len(coefficients) if coefficients else None,
                'worst_true_coefficient_relative_error': max(coefficients) if coefficients else None,
                'balanced_term_brier_mean': sum(brier)/len(brier) if brier else None,
                'balanced_term_brier_available_conditions': len(brier),
                'positive_term_brier_mean': sum(positive)/len(positive) if positive else None,
                'negative_term_brier_mean': sum(negative)/len(negative) if negative else None,
            })
    complete = len(rows) == expected * len(POLICIES) * len(BUDGETS)
    threshold = .8
    attainment = {}
    for policy in POLICIES:
        passed = [g['budget'] for g in groups if g['policy'] == policy and g['present_conditions'] == expected
                  and g['success_rate'] >= threshold]
        attainment[policy] = min(passed) if passed and complete else None
    active = attainment['structural_disagreement']
    savings = {policy: (1-active/attainment[policy]) if active is not None and attainment[policy] is not None else None
               for policy in POLICIES if policy != 'structural_disagreement'}
    return {
        'schema_version': 1, 'result_state': 'complete' if complete else 'partial',
        'expected_conditions': expected, 'present_conditions': len(tasks),
        'independent_parameter_tasks_present': len({r['spec_index'] for r in rows}),
        'expected_records': expected*len(POLICIES)*len(BUDGETS), 'present_records': len(rows),
        'success_definition': 'Exact support AND worst true coefficient relative error <= 10% AND mean of all three final forecast relative L2 errors <= 5%.',
        'evaluation_scope': comparison.get('evaluation_scope', comparison.get('evaluation_label', 'Frozen synthetic active-design benchmark; see run audit for prospective status.')),
        'budget_definition': '3 initial training + 2 development + acquired experiments; failed acquisitions are charged. Final test measurements are separate.',
        'noise_definition': '1% = sigma 0.004 relative to fixed initial-field RMS 0.4, not each trajectory RMS.',
        'uncertainty_note': 'Bootstrap inclusion frequencies are uncalibrated descriptive stability, not probabilities of true terms. A singleton candidate set does not prove identification.',
        'groups': groups, 'target_success_rate': threshold, 'first_observed_budget_at_target': attainment,
        'total_budget_savings_vs_comparator': savings,
        'cost_comparison_note': 'Post-hoc readout at saved budget checkpoints, not truth-based online stopping; null means no complete same-quality comparison. Negative savings mean higher cost.',
        'audit': comparison.get('audit', {'status': 'not_reported', 'passed': None}),
        'limitations': ['Synthetic controlled periodic scalar PDEs; no new real-world validation.',
            '24 paired noise conditions represent 12 parameter tasks, not 24 independent physical systems.',
            'Only initial-condition choice is optimized; costs are equal and sensors/boundaries are fixed.',
            'Balanced Brier scores are descriptive and do not establish posterior calibration.',
            'This experiment does not resolve the previous KS benchmark 5% failure.'],
    }, rows


def load_results(directory):
    directory = Path(directory)
    comparison = read_json(directory/'comparison.json')
    summary, rows = summarize(comparison)
    summary['comparison_sha256'] = digest(directory/'comparison.json')
    audit_file = directory/'audit.json'
    if audit_file.is_file():
        summary['audit'] = read_json(audit_file)
        summary['audit_sha256'] = digest(audit_file)
    return comparison, summary, rows


def load_episode(directory, task_id, policy, comparison=None):
    if policy not in POLICIES:
        raise ValueError('Unknown policy.')
    comparison = comparison if comparison is not None else read_json(Path(directory)/'comparison.json')
    matching = [r for r in comparison['records'] if r['task_id'] == task_id and r['policy'] == policy]
    if not matching:
        raise KeyError('No saved episode for this task and policy.')
    path = safe_child(directory, f'episodes/{task_id}--{policy}.json')
    episode = read_json(path)
    episode_hash = digest(path)
    if episode.get('policy') != policy:
        raise ValueError('Episode policy differs from comparison identity.')
    for record in matching:
        if record.get('model_file') != f'episodes/{task_id}--{policy}.json':
            raise ValueError('Score receipt points to a different episode identity.')
        if record.get('frozen_episode_sha256') is not None and record['frozen_episode_sha256'] != episode_hash:
            raise ValueError('Saved episode differs from its frozen digest.')
        index = record.get('checkpoint_index')
        if type(index) is not int or index < 0 or index >= len(episode['checkpoints']):
            # Failed acquisitions may leave later scored slots without a checkpoint.
            if index is None and record.get('status') in ('failed', 'unavailable', 'missing_budget_checkpoint'):
                continue
            raise ValueError('Invalid saved checkpoint index.')
        checkpoint = episode['checkpoints'][index]
        if checkpoint['budget'] != record['budget']:
            raise ValueError('Checkpoint budget differs from score receipt.')
        selected = checkpoint['model'].get('selected')
        if record.get('terms', []) != (selected['terms'] if selected else []):
            raise ValueError('Scored coefficients differ from the saved checkpoint.')
    return episode


def render(directory=DEFAULT_OUTPUT):
    directory = Path(directory)
    comparison, summary, rows = load_results(directory)
    sources = {}
    for task_id, policy in sorted({(r['task_id'], r['policy']) for r in rows}):
        load_episode(directory, task_id, policy, comparison)
        name = f'episodes/{task_id}--{policy}.json'
        sources[name] = digest(safe_child(directory, name))
    summary['verified_saved_episode_sha256'] = sources
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    paths = {name: directory/name for name in ('summary.json', 'comparison.csv', 'budget_curves.png', 'REPORT.md')}
    paths['summary.json'].write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    with paths['comparison.csv'].open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader(); writer.writerows(rows)
    with plt.rc_context({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False}):
        fig, axes = plt.subplots(2, 2, figsize=(10.5, 7), layout='constrained')
        for policy, label, color in zip(POLICIES, PLOT_LABELS, COLORS):
            group = [g for g in summary['groups'] if g['policy'] == policy]
            for ax, metric in zip(axes.flat, ('exact_rate', 'success_rate', 'coverage_rate', 'mean_set_size')):
                ax.plot(BUDGETS, [g[metric] for g in group], marker='o', color=color, label=label)
        for ax, title in zip(axes.flat, ('Exact support / planned conditions', 'Joint success / planned conditions',
                                          'True support in candidate set', 'Mean candidate set size (available records)')):
            ax.set_title(title); ax.set_xlabel('Charged experiments (3 train + 2 dev + additions)')
            ax.set_xticks(BUDGETS); ax.grid(alpha=.18)
        for ax in list(axes.flat)[:3]:
            ax.set_ylim(-.02, 1.04)
        axes[0, 1].axhline(.8, ls=':', color='#444', lw=1)
        axes[0, 0].legend(fontsize=8, loc='best')
        fig.suptitle(f"Frozen active-design evaluation — {summary['result_state']} ({summary['present_records']}/{summary['expected_records']} records)")
        fig.savefig(paths['budget_curves.png'], dpi=160)
        plt.close(fig)
    lines = ['# 主动 PDE 识别：已完成实验的只读报告', '',
        f"结果状态：**{summary['result_state']}**；保存 {summary['present_records']}/{summary['expected_records']} 条预算记录。",
        f"条件 {summary['present_conditions']}/{summary['expected_conditions']}；参数任务 {summary['independent_parameter_tasks_present']}。四策略共享识别器、初始训练/开发数据、预算和候选初值池。", '',
        '主成功指标：结构完全恢复、每个真项系数相对误差 ≤10%、3 条最终轨迹平均相对 L2 ≤5%，三者同时满足。预测使用 11 帧历史前缀，保留完整 1.0 时长预测。',
        '预算包括 3 次初始训练＋2 次开发＋追加实验；追加失败仍收费。每次实验 111×64＝7104 个标量，最终验证成本单列。',
        summary['noise_definition'], '',
        '|策略|预算|完全恢复|联合成功|候选集覆盖|平均集合大小|最差真项误差（条件均值 / 最坏）|遗漏 / 多余项总数|完整预测条件|',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for group in summary['groups']:
        size = '—' if group['mean_set_size'] is None else f"{group['mean_set_size']:.2f}"
        n = group['planned_conditions']
        coefficient = '—' if group['mean_worst_true_coefficient_relative_error'] is None else f"{group['mean_worst_true_coefficient_relative_error']:.2%} / {group['worst_true_coefficient_relative_error']:.2%}"
        lines.append(f"|{LABELS[group['policy']]}|{group['budget']}|{group['exact_count']}/{n}|{group['success_count']}/{n}|{group['coverage_count']}/{n}|{size}|{coefficient}|{group['missing_terms_total']} / {group['extra_terms_total']}|{group['complete_forecast_conditions']}/{n}|")
    lines += ['', '未完成、失败和缺失结果保留在计划条件分母中；平均集合大小仅针对已有记录。错误单集合指候选集中仅有一个结构且该结构错误，不表示系统声称物理唯一性。', '',
              '同质量成本：以 ≥80% 条件达到联合成功为目标，在完整结果中读取首次达到目标的预算；这不是使用真值的在线停止规则。']
    for policy in POLICIES:
        target = summary['first_observed_budget_at_target'][policy]
        lines.append(f"- {LABELS[policy]}：" + (str(target)+' 次实验。' if target is not None else '未获得可比较的达标预算（null）。'))
    for policy, saving in summary['total_budget_savings_vs_comparator'].items():
        lines.append(f"- 结构分歧相对{LABELS[policy]}的总预算节省：" + (f'{saving:.1%}。' if saving is not None else '不可估计（null）。'))
    lines += ['', '最大预算下的不确定性描述：', '',
              '|策略|错误单集合 / 全部条件|单集合总数|正项 Brier|负项 Brier|平衡 Brier|可计算条件|',
              '|---|---:|---:|---:|---:|---:|---:|']
    for group in summary['groups']:
        if group['budget'] != max(BUDGETS):
            continue
        values = [('—' if group[key] is None else f"{group[key]:.4f}") for key in ('positive_term_brier_mean', 'negative_term_brier_mean', 'balanced_term_brier_mean')]
        lines.append(f"|{LABELS[group['policy']]}|{group['wrong_singleton_count']}/{group['planned_conditions']}|{group['singleton_count']}|"+'|'.join(values)+f"|{group['balanced_term_brier_available_conditions']}/{group['planned_conditions']}|")
    lines += ['', '自助法项包含频率仅表示稳定性，不是真项概率。正/负项平衡 Brier 为描述性误差；不宣称概率已校准。结构集覆盖必须同时结合集合大小阅读。',
              '同一参数任务的两档噪声是配对条件，不能视为独立的 24 个物理系统；这里没有给出独立同分布意义的置信区间。', '',
              '审计状态（原始回执）：', '```json', json.dumps(summary['audit'], ensure_ascii=False, indent=2), '```', '',
              '范围：受控合成周期 PDE、固定传感器和边界、等成本初值选择；没有新增真实测量验证，也没有解决旧 KS 实验的 5% 失败。已有数据再分析必须标为回顾性。',
              '本页不运行识别、选模或预测，只汇总 comparison.json。闭环动作的逐步回放见 `python -m active_study.app --port 8767`。', '',
              f"comparison.json SHA-256：`{summary['comparison_sha256']}`", '',
              '产物：`comparison.csv`（逐条件原始指标）、`summary.json`（所有分母与缺失计数）、`budget_curves.png`（预算曲线）。', '']
    paths['REPORT.md'].write_text('\n'.join(lines), encoding='utf-8')
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT, help='Existing scored run directory.')
    args = parser.parse_args()
    result = render(args.output)
    print(json.dumps({'result_state': result['result_state'], 'records': result['present_records'],
                      'output': str(args.output)}, ensure_ascii=False))
