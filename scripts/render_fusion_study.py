"""Render saved fusion scores only; never load fields, fit, or score forecasts.

The final JSON must already exist. Model JSON is read only for fusion screening
receipts absent from the score rows. Viewed retrospective results, if present,
are summarized separately and never pooled with fresh validation.
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
DEFAULT_OUTPUT = ROOT/'fusion_study/results/first_run'
RETROSPECTIVE = ROOT/'fusion_study/development/retrospective/comparison.json'
METHODS = ('weak_original', 'weak_resolved', 'anchored_best', 'fused_median')
LABELS = ('Weak / original', 'Weak / refined', 'Anchored best', 'Guarded median')
SHORT_LABELS = ('Weak /\noriginal', 'Weak /\nrefined', 'Anchored\nbest', 'Guarded\nmedian')
COLORS = ('#959da5', '#4796b3', '#d9982e', '#00796e')
FAMILIES = ('allen_cahn', 'low_viscosity_burgers', 'ks_unstable')
FAMILY_LABELS = ('Allen–Cahn', 'Low-viscosity Burgers', 'KS')


def read_json(path):
    path = Path(path)
    if path.suffix != '.json':
        raise ValueError('This renderer reads saved JSON evidence only.')
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def worst_true_error(record):
    values = [c['relative_error'] for c in record['recovery']['coefficient_comparison']
              if c['reference'] != 0]
    if not values or not all(np.isfinite(v) and v >= 0 for v in values):
        raise ValueError('Saved reference comparison needs finite errors for every true term.')
    return float(max(values))


def fusion_evidence(record, output):
    if isinstance(record.get('fusion'), dict):
        return record['fusion']
    if record['method'] not in ('anchored_best', 'fused_median'):
        return {}
    path = output/'models'/f"{record['id']}.json"
    if not path.is_file():
        raise ValueError(f'Missing frozen fusion receipt: {path}')
    if record.get('frozen_model_sha256') and digest(path) != record['frozen_model_sha256']:
        raise ValueError('Frozen model receipt does not match scored model digest.')
    model = read_json(path)
    if model.get('id') != record['id']:
        raise ValueError('Frozen model receipt identity differs from scored record.')
    return model.get('fusion', {})


def flat_row(record, output):
    fusion = fusion_evidence(record, output)
    prediction = record['mean_clean_relative_l2']
    return {
        'id': record['id'], 'family': record['family'], 'sampling': record['sampling'],
        'condition': record['condition'], 'method': record['method'],
        'noise_level': record.get('noise_level'), 'noise_seed': record.get('noise_seed'),
        'structural_selection_predictor': record.get('structural_selection_predictor'),
        'coefficient_selection_predictor': record.get('coefficient_selection_predictor'),
        'final_predictor': record.get('final_predictor'), 'equation': record.get('equation'),
        'exact_support': bool(record['recovery']['exact_support']),
        'extra_terms': ';'.join(record['recovery']['spurious_terms']),
        'extra_count': len(record['recovery']['spurious_terms']),
        'missing_terms': ';'.join(record['recovery']['missing_terms']),
        'missing_count': len(record['recovery']['missing_terms']),
        'coefficient_vector_error_percent': 100*record['recovery']['coefficient_relative_l2'],
        'worst_true_term_error_percent': 100*worst_true_error(record),
        'mean_final_prediction_l2_percent': None if prediction is None else 100*prediction,
        'prediction_above10pct': prediction is not None and prediction > .1,
        'forecast_failures': record['forecast_failures'],
        'median_status': fusion.get('median_status'),
        'median_reason': fusion.get('median_reason'),
        'median_member_variants': ';'.join(fusion.get('member_variants', [])),
        'median_member_candidate_ids': ';'.join(fusion.get('member_candidate_ids', [])),
        'anchor_candidate_id': fusion.get('anchor_candidate_id'),
    }


def group_summary(records, rows):
    predictions = [r['mean_clean_relative_l2'] for r in records if r['mean_clean_relative_l2'] is not None]
    individual = [f['clean']['relative_l2'] for r in records for f in r.get('forecasts', [])
                  if f.get('status') == 'evaluated']
    worst = [worst_true_error(r) for r in records]
    vector = [r['recovery']['coefficient_relative_l2'] for r in records]
    statuses = Counter(row['median_status'] for row in rows if row['median_status'] is not None)
    return {
        'configurations': len(records),
        'exact': sum(r['recovery']['exact_support'] for r in records),
        'extra': sum(len(r['recovery']['spurious_terms']) for r in records),
        'missing': sum(len(r['recovery']['missing_terms']) for r in records),
        'median_vector_error': float(np.median(vector)) if vector else None,
        'max_vector_error': max(vector) if vector else None,
        'median_worst_true_term_error': float(np.median(worst)) if worst else None,
        'max_worst_true_term_error': max(worst) if worst else None,
        'median_prediction_l2': float(np.median(predictions)) if predictions else None,
        'max_condition_mean_prediction_l2': max(predictions) if predictions else None,
        'max_individual_prediction_l2': max(individual) if individual else None,
        'configs_above10pct': sum(value > .1 for value in predictions),
        'finite_prediction_configs': len(predictions),
        'configs_with_forecast_failure': sum(r['forecast_failures'] > 0 for r in records),
        'forecast_failures': sum(r['forecast_failures'] for r in records),
        'median_status_counts': dict(sorted(statuses.items())),
        'median_accept_count': statuses.get('accepted', 0),
        'median_fallback_count': sum(count for status, count in statuses.items()
                                     if status not in ('accepted', 'not_requested')),
    }


def summarize(report, output, expected_total=120):
    records = report['records']
    if len(records) != expected_total or len({r['id'] for r in records}) != expected_total:
        raise ValueError('Expected all 120 unique declared configurations; do not silently omit failures.')
    if set(r['method'] for r in records) != set(METHODS):
        raise ValueError('Expected exactly the four declared fusion comparison arms.')
    rows = [flat_row(r, output) for r in records]
    methods = {}
    for method in METHODS:
        group = [r for r in records if r['method'] == method]
        selected_rows = [r for r in rows if r['method'] == method]
        if len(group) != 30:
            raise ValueError('Each method must retain all 30 configurations.')
        methods[method] = group_summary(group, selected_rows)
        methods[method]['by_family'] = {}
        for family in FAMILIES:
            family_records = [r for r in group if r['family'] == family]
            if len(family_records) != 10:
                raise ValueError('Each family/method must retain ten conditions.')
            methods[method]['by_family'][family] = group_summary(
                family_records, [r for r in selected_rows if r['family'] == family])
    return {'record_count': len(records), 'methods': methods, 'conditions': rows,
            'gate': report.get('gate'), 'retrospective': bool(report.get('retrospective', False))}


def save_figure(fig, output, filename):
    fig.savefig(output/filename, dpi=170, bbox_inches='tight', pad_inches=.15)
    plt.close(fig)


def render_figures(summary, output):
    methods = summary['methods']
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), layout='constrained')
    exact = [methods[m]['exact'] for m in METHODS]
    bars = axes[0].bar(range(4), exact, color=COLORS)
    axes[0].bar_label(bars, labels=[f'{v}/30' for v in exact], padding=3, fontsize=11)
    axes[0].set(ylim=(0, 34), ylabel='Exact term sets — higher is better')
    extra, missing = [methods[m]['extra'] for m in METHODS], [methods[m]['missing'] for m in METHODS]
    a = axes[1].bar(np.arange(4)-.18, extra, .36, color=COLORS)
    b = axes[1].bar(np.arange(4)+.18, missing, .36, color='white', edgecolor=COLORS, hatch='//')
    axes[1].bar_label(a, labels=[f'{v} extra' for v in extra], padding=3, fontsize=9)
    axes[1].bar_label(b, labels=[f'{v} missing' for v in missing], padding=3, fontsize=9, rotation=90)
    axes[1].set(ylim=(0, max(3, max(extra+missing)*1.35+1)), ylabel='Total extra / missing terms — lower is better')
    for ax in axes:
        ax.set_xticks(range(4), SHORT_LABELS)
        ax.grid(axis='y', alpha=.2)
        ax.set_axisbelow(True)
    fig.suptitle('Fresh fusion validation: equation structure\n30 paired conditions per method; predictions cannot replace term recovery', fontsize=13)
    save_figure(fig, output, 'support_recovery.png')

    fig, axes = plt.subplots(2, 3, figsize=(14.5, 8), layout='constrained')
    for col, (family, name) in enumerate(zip(FAMILIES, FAMILY_LABELS)):
        for row, (key, label) in enumerate([
                ('max_worst_true_term_error', 'Maximum true-coefficient error (%)'),
                ('median_worst_true_term_error', 'Median of condition-wise worst errors (%)')]):
            values = [100*methods[m]['by_family'][family][key] for m in METHODS]
            ax = axes[row, col]
            bars = ax.bar(range(4), values, color=COLORS)
            ax.bar_label(bars, labels=[f'{v:.3g}' for v in values], padding=3, fontsize=10)
            ax.set_xticks(range(4), SHORT_LABELS)
            ax.set_ylim(0, max(1e-9, max(values)*1.25))
            ax.grid(axis='y', alpha=.2); ax.set_axisbelow(True)
            if col == 0:
                ax.set_ylabel(label)
            if row == 0:
                ax.set_title(name)
    fig.suptitle('Coefficient accuracy — lower is better\nEach condition reports its worst relative error over true terms; missing terms count as 100%', fontsize=13)
    save_figure(fig, output, 'coefficient_comparison.png')

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.4), layout='constrained')
    for ax, key, title in zip(axes, ['median_prediction_l2', 'max_condition_mean_prediction_l2'],
                              ['Median condition error', 'Worst condition error']):
        largest = 0.
        for j, method in enumerate(METHODS):
            raw = [methods[method]['by_family'][f][key] for f in FAMILIES]
            values = [np.nan if v is None else 100*v for v in raw]
            finite = [v for v in values if np.isfinite(v)]
            largest = max([largest, *finite])
            bars = ax.bar(np.arange(3)+(j-1.5)*.2, values, .2, color=COLORS[j], label=LABELS[j])
            ax.bar_label(bars, labels=['failed' if not np.isfinite(v) else f'{v:.2g}' for v in values],
                         padding=3, fontsize=9)
        ax.axhline(10, linestyle='--', linewidth=1, color='#777777', label='10% diagnostic line')
        ax.set_xticks(range(3), FAMILY_LABELS)
        ax.set(title=title, ylabel='Final forecast relative L2 (%)', ylim=(0, max(12., largest*1.25)))
        ax.grid(axis='y', alpha=.2); ax.set_axisbelow(True)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='outside lower center', ncols=3, frameon=False)
    fig.suptitle('Fresh fusion validation: forecast comparison\nEach condition averages 3 final trajectories; 9 underlying initial conditions shared across all arms', fontsize=13)
    save_figure(fig, output, 'prediction_comparison.png')


def main(output=DEFAULT_OUTPUT):
    output = Path(output)
    source = output/'comparison.json'
    if not source.is_file():
        raise FileNotFoundError('Final comparison.json must exist before rendering; this script never starts experiments.')
    report = read_json(source)
    summary = summarize(report, output)
    summary.update(source_sha256=digest(source),
                   error_units='fraction in JSON method summaries; percent in CSV and condition rows',
                   renderer_reads='saved comparison/model JSON only; no raw fields, fitting or forecast scoring',
                   independence_note='120 configurations / 360 planned forecasts share 9 underlying final initial conditions')
    if RETROSPECTIVE.is_file():
        retrospective_report = read_json(RETROSPECTIVE)
        if retrospective_report.get('retrospective') is not True:
            raise ValueError('Previously viewed data must be labeled retrospective.')
        summary['retrospective_comparison'] = summarize(retrospective_report, RETROSPECTIVE.parent)
        summary['retrospective_comparison']['source_sha256'] = digest(RETROSPECTIVE)
        summary['retrospective_comparison']['not_pooled_with_fresh'] = True
    with (output/'summary.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary['conditions'][0]))
        writer.writeheader(); writer.writerows(summary['conditions'])
    (output/'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
    render_figures(summary, output)
    print(json.dumps({'output': str(output), 'rows': summary['record_count'],
                      'figures': ['support_recovery.png', 'coefficient_comparison.png', 'prediction_comparison.png']}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    main(parser.parse_args().output)
