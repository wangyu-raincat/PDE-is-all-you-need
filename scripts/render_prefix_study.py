"""Render saved prefix-conditioned study scores only; never load fields, fit, or score forecasts.

The final JSON must already exist. Model JSON is read only for selection and
median-generation receipts absent from score rows. Previously viewed records
are kept separate from fresh validation. The 5% target concerns every condition
mean over its three trajectories, not each individual trajectory or coefficient.
Prefix methods receive 11 full-grid or 6 coarse-grid historical frames ending
at the prediction cutoff. Single-frame controls receive only that shared last
frame. All arms retain future durations AC=.5, Burgers=1.5 and KS=20; the
historical prefix is not included in the scored future or counted as prediction.
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
DEFAULT_OUTPUT = ROOT/'prefix_study/results/first_run'
RETROSPECTIVE = ROOT/'prefix_study/development/retrospective/comparison.json'
METHODS = ('weak_resolved', 'denoised_median', 'prefix_weak', 'prefix_median')
LABELS = ('Single frame: raw weak', 'Single frame: filtered + median', 'Observed prefix: weak', 'Observed prefix: + median')
SHORT_LABELS = ('Single frame\nraw weak', 'Single frame\nfiltered + median', 'Observed prefix\nweak', 'Observed prefix\n+ median')
COLORS = ('#4796b3', '#d9982e', '#496fc6', '#00796e')
FAMILIES = ('allen_cahn', 'low_viscosity_burgers', 'ks_unstable')
FAMILY_LABELS = ('Allen–Cahn', 'Low-viscosity Burgers', 'KS')
LIMIT = .05
FUTURE_DURATION = {'allen_cahn': .5, 'low_viscosity_burgers': 1.5, 'ks_unstable': 20.}
PREFIX_FRAMES = {'full': 11, 'coarse': 6}
CONDITIONING_NOTE = ('Prefix arms use 11 full / 6 coarse historical frames ending at the forecast cutoff; '
                     'single-frame controls use only the shared endpoint; extra prefix observations are explicit.')
HORIZON_NOTE = 'Unchanged future duration: AC 0.5, Burgers 1.5, KS 20 (benchmark time units).'


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


def model_evidence(record, output, retrospective=False):
    path = output/'models'/f"{record['id']}.json"
    # The shared cutoff and full future differ from earlier studies. Do not
    # substitute an older single-frame model or its scores on a different task.
    if not path.is_file():
        raise ValueError(f'Missing frozen model receipt: {path}')
    if not record.get('frozen_model_sha256') or digest(path) != record['frozen_model_sha256']:
        raise ValueError('Frozen model receipt does not match the scored model digest.')
    model = read_json(path)
    if model.get('id') != record['id']:
        raise ValueError('Frozen model receipt identity differs from scored record.')
    return model


def flat_row(record, output, retrospective=False):
    model = model_evidence(record, output, retrospective)
    selected = model.get('selected')
    selected_id = selected['candidate_id'] if selected else None
    generation = model.get('median_augmentation') or model.get('median_generation') or {}
    groups = generation.get('groups', [])
    added = sum(g['status'] == 'added' for g in groups)
    if added != generation.get('added_candidate_count', 0):
        raise ValueError('Median-generation group counts contradict the saved aggregate.')
    aliases = [g for support in groups for g in support.get('distinct_vector_groups', [])]
    prediction = record['mean_clean_relative_l2']
    if prediction is not None and (not np.isfinite(prediction) or prediction < 0):
        raise ValueError('Prediction means must be finite nonnegative values or explicit failure nulls.')
    target_passed = prediction is not None and prediction <= LIMIT and record['forecast_failures'] == 0
    true_rows = [r for r in record['recovery']['coefficient_comparison'] if r['reference'] != 0]
    coefficient_limit = .25 if record.get('noise_level') == .01 else .10
    evaluated = [r for r in record.get('forecasts', []) if r.get('status') == 'evaluated']
    individual = [r['clean']['relative_l2'] for r in evaluated]
    prefix_method = record['method'].startswith('prefix_')
    conditioning = 'observed_prefix' if prefix_method else 'last_frame_only'
    frames = PREFIX_FRAMES[record['sampling']] if prefix_method else 1
    duration = FUTURE_DURATION[record['family']]
    if record.get('conditioning') != conditioning or model.get('conditioning') != conditioning:
        raise ValueError('Model/score conditioning must explicitly distinguish prefix and single-frame methods.')
    for forecast in evaluated:
        if (forecast.get('conditioning') != conditioning
                or forecast.get('prefix_frames_used') != frames
                or not np.isclose(forecast.get('forecast_duration', np.nan), duration, rtol=0, atol=1e-10)):
            raise ValueError('Saved forecast conditioning or complete future horizon differs from the declared task.')
    return {
        'id': record['id'], 'family': record['family'], 'sampling': record['sampling'],
        'condition': record['condition'], 'method': record['method'],
        'noise_level': record.get('noise_level'), 'noise_seed': record.get('noise_seed'),
        'structural_selection_predictor': record.get('structural_selection_predictor'),
        'coefficient_selection_predictor': record.get('coefficient_selection_predictor'),
        'final_predictor': record.get('final_predictor'), 'equation': record.get('equation'),
        'initial_filter': record.get('initial_filter', model.get('initial_filter', 'none')),
        'conditioning': conditioning,
        'observed_conditioning_frames': frames,
        'available_prefix_frames': PREFIX_FRAMES[record['sampling']],
        'available_historical_prefix_duration': .1*duration,
        'observed_conditioning_duration': .1*duration if prefix_method else 0.,
        'unchanged_future_duration': duration,
        'prefix_in_scored_future': False,
        'exact_support': bool(record['recovery']['exact_support']),
        'extra_terms': ';'.join(record['recovery']['spurious_terms']),
        'extra_count': len(record['recovery']['spurious_terms']),
        'missing_terms': ';'.join(record['recovery']['missing_terms']),
        'missing_count': len(record['recovery']['missing_terms']),
        'coefficient_vector_error_percent': 100*record['recovery']['coefficient_relative_l2'],
        'worst_true_term_error_percent': 100*worst_true_error(record),
        'each_true_term_accuracy_passed': bool(true_rows) and all(c['relative_error'] <= coefficient_limit for c in true_rows),
        'mean_final_prediction_l2_percent': None if prediction is None else 100*prediction,
        'max_individual_prediction_l2_percent': 100*max(individual) if individual else None,
        'individual_above5pct_diagnostic_only': sum(value > LIMIT for value in individual),
        'evaluated_trajectories': len(individual),
        'condition_mean_above5pct': prediction is not None and prediction > LIMIT,
        'condition_mean_target_passed': target_passed,
        'forecast_failures': record['forecast_failures'],
        'selected_candidate_id': selected_id,
        'selected_support_median': bool(selected_id and selected_id.startswith('support-median::')),
        'median_support_groups': len(groups),
        'generated_median_candidates': added,
        'invalid_median_candidates': sum(g['status'] == 'invalid_median' for g in groups),
        'too_few_distinct_vector_supports': sum(g['status'] == 'too_few_distinct_vectors' for g in groups),
        'variant_representatives': sum(len(g.get('variant_representative_ids', [])) for g in groups),
        'distinct_coefficient_vectors': sum(g.get('distinct_vector_count', 0) for g in groups),
        'exact_alias_groups': sum(len(g['variants']) > 1 for g in aliases),
        'duplicate_vector_aliases_removed': sum(max(0, len(g['variants'])-1) for g in aliases),
    }


def group_summary(records, rows):
    predictions = [r['mean_clean_relative_l2'] for r in records if r['mean_clean_relative_l2'] is not None]
    individual = [f['clean']['relative_l2'] for r in records for f in r.get('forecasts', [])
                  if f.get('status') == 'evaluated']
    worst = [worst_true_error(r) for r in records]
    vector = [r['recovery']['coefficient_relative_l2'] for r in records]
    return {
        'configurations': len(records),
        'conditioning_modes': dict(sorted(Counter(row['conditioning'] for row in rows).items())),
        'conditioning_frames_by_condition': dict(sorted(Counter(str(row['observed_conditioning_frames']) for row in rows).items())),
        'complete_future_durations': sorted({row['unchanged_future_duration'] for row in rows}),
        'prefix_in_scored_future': False,
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
        'configs_above5pct': sum(value > LIMIT for value in predictions),
        'individual_trajectories_above5pct_diagnostic_only': sum(value > LIMIT for value in individual),
        'evaluated_individual_trajectories': len(individual),
        'individual_trajectory_limit': None,
        'condition_mean_target_passed_count': sum(row['condition_mean_target_passed'] for row in rows),
        'all_condition_means_at_most5pct': bool(records) and all(row['condition_mean_target_passed'] for row in rows),
        'true_term_accuracy_failed_conditions': sum(not row['each_true_term_accuracy_passed'] for row in rows),
        'finite_prediction_configs': len(predictions),
        'configs_with_forecast_failure': sum(r['forecast_failures'] > 0 for r in records),
        'forecast_failures': sum(r['forecast_failures'] for r in records),
        'selected_support_median_count': sum(row['selected_support_median'] for row in rows),
        'median_support_groups': sum(row['median_support_groups'] for row in rows),
        'generated_median_candidates': sum(row['generated_median_candidates'] for row in rows),
        'invalid_median_candidates': sum(row['invalid_median_candidates'] for row in rows),
        'too_few_distinct_vector_supports': sum(row['too_few_distinct_vector_supports'] for row in rows),
        'variant_representatives': sum(row['variant_representatives'] for row in rows),
        'distinct_coefficient_vectors': sum(row['distinct_coefficient_vectors'] for row in rows),
        'exact_alias_groups': sum(row['exact_alias_groups'] for row in rows),
        'duplicate_vector_aliases_removed': sum(row['duplicate_vector_aliases_removed'] for row in rows),
    }


def summarize(report, output, expected_total=120):
    records = report['records']
    if len(records) != expected_total or len({r['id'] for r in records}) != expected_total:
        raise ValueError('Expected all 120 unique declared configurations; do not silently omit failures.')
    if set(r['method'] for r in records) != set(METHODS):
        raise ValueError('Expected exactly the four declared prefix-conditioned comparison arms.')
    retrospective = bool(report.get('retrospective', False))
    rows = [flat_row(r, output, retrospective) for r in records]
    methods = {}
    condition_sets = []
    for method in METHODS:
        group = [r for r in records if r['method'] == method]
        selected_rows = [r for r in rows if r['method'] == method]
        if len(group) != 30:
            raise ValueError('Each method must retain all 30 configurations.')
        condition_sets.append({(r['family'], r['sampling'], r['condition']) for r in group})
        if len(condition_sets[-1]) != 30 or any(r['id'] != '--'.join((r['family'], r['sampling'], r['condition'], method)) for r in group):
            raise ValueError('Saved condition identity is duplicated or inconsistent.')
        methods[method] = group_summary(group, selected_rows)
        methods[method]['by_family'] = {}
        for family in FAMILIES:
            family_records = [r for r in group if r['family'] == family]
            if len(family_records) != 10:
                raise ValueError('Each family/method must retain ten conditions.')
            methods[method]['by_family'][family] = group_summary(
                family_records, [r for r in selected_rows if r['family'] == family])
    if any(s != condition_sets[0] for s in condition_sets[1:]):
        raise ValueError('The methods must use identical paired conditions.')
    candidate = methods['prefix_median']
    target = {'candidate': 'prefix_median', 'relative_l2_limit': LIMIT,
              'metric': 'maximum over all 30 condition means; each mean contains three final-trajectory relative L2 errors',
              'conditioning': CONDITIONING_NOTE,
              'prefix_frames': PREFIX_FRAMES,
              'future_duration': FUTURE_DURATION,
              'extra_observations_vs_single_frame_are_explicit': True,
              'conditions_passed': candidate['condition_mean_target_passed_count'], 'conditions_total': 30,
              'passed': candidate['all_condition_means_at_most5pct'],
              'individual_trajectory_limit': None,
              'failed_conditions': [r['id'] for r in rows if r['method'] == 'prefix_median' and not r['condition_mean_target_passed']],
              'retrospective_diagnostic_only': retrospective}
    gate = report.get('gate')
    if gate and gate.get('prediction_target_passed') != target['passed']:
        raise ValueError('Saved gate and independently summarized 5% target disagree.')
    non_prediction = [c for c in gate.get('checks', []) if not c['name'].endswith(':mean_prediction_at_most_5pct')] if gate else []
    guards = {'available': bool(gate), 'checks_passed': sum(c['passed'] for c in non_prediction),
              'checks_total': len(non_prediction),
              'passed': all(c['passed'] for c in non_prediction) if non_prediction else None,
              'failed_checks': [c['name'] for c in non_prediction if not c['passed']],
              'scope': 'identification, coefficient nonregression and other non-prediction guards; distinct from 5% target'}
    return {'record_count': len(records), 'methods': methods, 'conditions': rows,
            'prediction_target': target, 'identification_and_other_guards': guards,
            'gate': gate, 'retrospective': retrospective}


def save_figure(fig, output, filename):
    fig.savefig(output/filename, dpi=170, bbox_inches='tight', pad_inches=.15)
    plt.close(fig)


def render_figures(summary, output):
    methods = summary['methods']
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.0), layout='constrained')
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
    fig.suptitle('Prefix-conditioned validation: equation structure\n30 paired conditions per method; prefix methods receive 11 full / 6 coarse frames', fontsize=13)
    save_figure(fig, output, 'support_recovery.png')

    fig, axes = plt.subplots(2, 3, figsize=(16.5, 8.7), layout='constrained')
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
    fig.suptitle('Coefficient accuracy — lower is better\nHistorical prefix is observed input, not predicted evidence; missing true terms count as 100%', fontsize=13)
    save_figure(fig, output, 'coefficient_comparison.png')

    fig, axes = plt.subplots(1, 2, figsize=(15.5, 6.6), layout='constrained')
    for ax, key, title in zip(axes, ['median_prediction_l2', 'max_condition_mean_prediction_l2'],
                              ['Median of condition means', 'Worst condition mean']):
        largest = 0.
        for j, method in enumerate(METHODS):
            raw = [methods[method]['by_family'][f][key]
                   if methods[method]['by_family'][f]['finite_prediction_configs'] == 10
                   and methods[method]['by_family'][f]['forecast_failures'] == 0 else None for f in FAMILIES]
            values = [np.nan if v is None else 100*v for v in raw]
            finite = [v for v in values if np.isfinite(v)]
            largest = max([largest, *finite])
            bars = ax.bar(np.arange(3)+(j-1.5)*.2, values, .2, color=COLORS[j], label=LABELS[j])
            ax.bar_label(bars, labels=['incomplete' if not np.isfinite(v) else f'{v:.3g}' for v in values],
                         padding=3, fontsize=9)
        ax.axhline(5, linestyle='--', linewidth=1, color='#777777', label='5% target for every condition mean')
        ax.set_xticks(range(3), FAMILY_LABELS)
        ax.set(title=title, ylabel='Relative L2 vs clean numerical reference (%)', ylim=(0, max(6., largest*1.25)))
        ax.grid(axis='y', alpha=.2); ax.set_axisbelow(True)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='outside lower center', ncols=2, frameon=False)
    fig.suptitle('Prefix-conditioned validation: full-horizon future prediction\n'
                 'Prefix: 11 full / 6 coarse frames; single-frame controls use the same cutoff\n'
                 'Future: AC 0.5, Burgers 1.5, KS 20; 5% applies to condition means, not individual trajectories', fontsize=12)
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
                   independence_note='120 configurations / 360 planned forecasts share 9 underlying final initial conditions',
                   prediction_target_note='all30 condition means <=5%; individual maxima and counts above5% are diagnostic only',
                   exact_alias_note='generation counters count distinct training coefficient vectors; correlated variants are not independent experiments',
                   conditioning_note=CONDITIONING_NOTE, full_future_note=HORIZON_NOTE,
                   future_duration=FUTURE_DURATION, observed_prefix_frames=PREFIX_FRAMES,
                   prefix_is_observed_not_predicted=True,
                   earlier_study_scores_not_substituted=True)
    if RETROSPECTIVE.is_file():
        retrospective_report = read_json(RETROSPECTIVE)
        if retrospective_report.get('retrospective') is not True:
            raise ValueError('Previously viewed data must be labeled retrospective.')
        summary['retrospective_comparison'] = summarize(retrospective_report, RETROSPECTIVE.parent)
        summary['retrospective_comparison']['source_sha256'] = digest(RETROSPECTIVE)
        summary['retrospective_comparison']['not_pooled_with_fresh'] = True
    with (output/'summary.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary['conditions'][0]), lineterminator='\n')
        writer.writeheader(); writer.writerows(summary['conditions'])
    (output/'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
    render_figures(summary, output)
    print(json.dumps({'output': str(output), 'rows': summary['record_count'],
                      'figures': ['support_recovery.png', 'coefficient_comparison.png', 'prediction_comparison.png']}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    main(parser.parse_args().output)
