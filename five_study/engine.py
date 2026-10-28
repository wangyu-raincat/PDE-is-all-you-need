"""Observation-only initial filtering and support-wise coefficient candidates.

Unlike the previous fusion, no old predictor's selected support is mandatory.
Every median is formed before development scoring from training-fit candidates
with exactly the same support. Distinct weak variants with identical physical
coefficient vectors count once; lambda-path multiplicity never supplies votes.
"""
from __future__ import annotations

from copy import deepcopy
import numpy as np

from fusion_study import engine as previous
from l1_study import engine as sparse
from weak_study import engine as weak
from resolved_study import selection as refined
from pde_discovery import generic_engine as baseline
from pde_discovery.generic_experiment import forecast_metrics
from .forecast import forecast_with_diagnostics

METHODS = ('weak_resolved', 'fused_median', 'denoised_weak', 'denoised_median')
NEAR_FACTOR = 1.05
ABSOLUTE_TOLERANCE = 1e-12
PROTOCOL = {
    'version': 1,
    'methods': list(METHODS),
    'candidate_generation': 'unchanged three fixed weak variants, adaptive L1, normalized OLS debias, fit once',
    'old_controls': 'unchanged weak_resolved and original-support-anchored fused_median rules',
    'median_generation': 'training-only; group successful candidate fits by exact term-name set',
    'variant_deduplication': 'one candidate per fixed variant/support by minimum stable training candidate ID',
    'vector_deduplication': 'exact equal physical coefficient vectors within support contribute once, retaining all aliases',
    'minimum_median_contributors': 2,
    'median': 'coordinatewise median of distinct physical coefficient vectors, no development filter',
    'median_failure': 'zero or nonfinite median rejects only the new candidate; no silent term deletion',
    'original_candidates_preserved': True,
    'old_support_anchor_for_new_arms': False,
    'cross_support_averaging': False,
    'path_frequency_weighting': False,
    'development_predictor': 'fixed spectral-hard-filter first frame then resolved factor4',
    'development_score': 'sqrt(mean(trajectory_MSE)); first frame excluded',
    'development_selection': 'within near_factor*best+absolute_tolerance choose fewest terms, error, stable ID',
    'shared_scoring': 'score augmented denoised pool once, then independently select its original-candidate subset for denoised_weak',
    'near_factor': NEAR_FACTOR,
    'absolute_tolerance': ABSOLUTE_TOLERANCE,
    'true_noise_or_family_or_reference_input': False,
    'median_members_are_independent_estimates': False,
}


def _support(candidate):
    support = previous._support(candidate)
    if type(candidate.get('support_size')) is not int or candidate['support_size'] != len(support):
        raise ValueError('Recorded support size must equal the nonzero term set.')
    return support


def generate_support_medians(candidates):
    """Return all original candidates plus training-only support-wise medians.

    Development fields, when present in a historical record, are never read.
    Member identities and exact-vector aliases remain visible in the evidence.
    """
    if not isinstance(candidates, (list, tuple)) or not all(isinstance(c, dict) for c in candidates):
        raise ValueError('Supply a list of training candidate dictionaries.')
    augmented = deepcopy(list(candidates))
    evidence = {'training_only': True, 'reference_used': False, 'development_used': False,
                'path_frequency_weighting': False, 'groups': [], 'excluded_candidates': [],
                'original_candidate_count': len(candidates), 'added_candidate_count': 0}
    groups = {}
    for raw in candidates:
        try:
            if not raw.get('fit_success'):
                raise ValueError('Training fit was not successful.')
            support = _support(raw)
            variant = raw.get('weak_variant')
            identity = raw.get('candidate_id')
            if variant not in weak.VARIANTS:
                raise ValueError('Only declared training weak variants can contribute.')
            if not isinstance(identity, str) or not identity:
                raise ValueError('A stable training candidate ID is required.')
        except (ValueError, TypeError, AttributeError) as exc:
            evidence['excluded_candidates'].append({'candidate_id': raw.get('candidate_id'), 'reason': str(exc)})
            continue
        groups.setdefault(support, []).append(raw)

    for support in sorted(groups, key=lambda s: tuple(sparse.TERM_NAMES.index(n) for n in s)):
        representatives = {}
        for candidate in sorted(groups[support], key=lambda c: c['candidate_id']):
            representatives.setdefault(candidate['weak_variant'], candidate)
        distinct = {}
        for variant in weak.VARIANTS:
            if variant not in representatives:
                continue
            candidate = representatives[variant]
            vector = tuple(float(next(t['coefficient'] for t in candidate['terms'] if t['name'] == name))
                           for name in support)
            group = distinct.setdefault(vector, {'coefficients': list(vector), 'candidate_ids': [], 'variants': []})
            group['candidate_ids'].append(candidate['candidate_id'])
            group['variants'].append(variant)
        record = {
            'support': list(support),
            'variant_representative_ids': [representatives[v]['candidate_id'] for v in weak.VARIANTS if v in representatives],
            'distinct_vector_groups': deepcopy(list(distinct.values())),
            'distinct_vector_count': len(distinct),
            'status': 'too_few_distinct_vectors', 'candidate_id': None,
            'proposed_terms': None, 'reason': None,
        }
        evidence['groups'].append(record)
        if len(distinct) < 2:
            record['reason'] = 'Fewer than two distinct coefficient vectors; original candidates retained.'
            continue
        with np.errstate(over='ignore', invalid='ignore'):
            coefficients = np.median(np.array(list(distinct), float), axis=0)
        record['proposed_terms'] = [{'name': name, 'coefficient': float(value) if np.isfinite(value) else None}
                                    for name, value in zip(support, coefficients)]
        if not np.isfinite(coefficients).all() or np.any(coefficients == 0):
            record.update(status='invalid_median', reason='Median is nonfinite or would delete an anchored term.')
            continue
        identity = 'support-median::' + '-'.join(str(sparse.TERM_NAMES.index(name)) for name in support)
        if any(c.get('candidate_id') == identity for c in augmented):
            raise ValueError('Generated median candidate ID collides with input candidate ID.')
        terms = deepcopy(record['proposed_terms'])
        median = {'candidate_id': identity, 'method': 'denoised_median',
                  'terms': terms, 'equation': baseline._equation(terms), 'support_size': len(support),
                  'fit_success': True, 'coefficient_source': 'coordinatewise median of distinct training-fit vectors',
                  'fit_objective_not_recomputed': True,
                  'diagnostics': {'support': list(support), 'reference_equation_used': False,
                                  'development_used_for_coefficients': False,
                                  'distinct_vector_groups': deepcopy(record['distinct_vector_groups']),
                                  'distinct_vector_count': len(distinct)}}
        augmented.append(median)
        record.update(status='added', candidate_id=identity)
        evidence['added_candidate_count'] += 1
    return augmented, evidence


def select_candidates(candidates, developments):
    """Select globally across supports using only denoised development forecasts."""
    if not isinstance(candidates, (list, tuple)) or not all(isinstance(c, dict) for c in candidates):
        raise ValueError('Supply candidate dictionaries.')
    if not isinstance(developments, (list, tuple)) or not developments:
        raise ValueError('At least one development trajectory is required.')
    arrays = [baseline._validate(d) for d in developments]
    scored = deepcopy(list(candidates))
    for candidate in scored:
        candidate['eligible'] = False
        candidate.pop('development', None)
        candidate.pop('rejection_reason', None)
        try:
            if not candidate.get('fit_success'):
                raise ValueError('Training convergence, finite coefficients or rank check failed.')
            _support(candidate)
            if not isinstance(candidate.get('candidate_id'), str) or not candidate['candidate_id']:
                raise ValueError('A stable candidate ID is required.')
            individual, initial_diagnostics = [], []
            for x, t, u in arrays:
                prediction, initial = forecast_with_diagnostics(candidate['terms'], x, t, u[0])
                individual.append(forecast_metrics(prediction, u))
                initial_diagnostics.append(initial)
            error = float(np.sqrt(np.mean([m['rmse']**2 for m in individual])))
            if not np.isfinite(error):
                raise ValueError('Development aggregate is not finite.')
            candidate['development'] = {'rmse': error, 'per_trajectory': individual,
                'aggregation': 'sqrt(mean(trajectory_MSE))', 'trajectory_count': len(individual),
                'initial_filter_diagnostics': initial_diagnostics}
            candidate['eligible'] = True
        except (ValueError, RuntimeError, FloatingPointError, np.linalg.LinAlgError) as exc:
            candidate['rejection_reason'] = str(exc)
    return choose_scored(scored)


def choose_scored(candidates):
    """Apply the fixed near-optimal rule to already scored candidate records.

    Used only to reuse identical development forecasts across the two new arms.
    It performs no new fitting, denoising or forecasting.
    """
    if not isinstance(candidates, (list, tuple)) or not all(isinstance(c, dict) for c in candidates):
        raise ValueError('Supply scored candidate dictionaries.')
    scored = deepcopy(list(candidates))
    for c in scored:
        if not c.get('eligible'):
            continue
        try:
            if not c.get('fit_success'):
                raise ValueError('Training fit was not successful.')
            _support(c)
            previous._error(c)
            if not isinstance(c.get('candidate_id'), str) or not c['candidate_id']:
                raise ValueError('A stable candidate ID is required.')
        except (ValueError, TypeError, AttributeError) as exc:
            c['eligible'] = False
            c['rejection_reason'] = str(exc)
    eligible = [c for c in scored if c.get('eligible')]
    if not eligible:
        return {'selected': None, 'candidates': scored, 'near_optimal_supports': [],
                'failure_reason': 'No candidate passed training and denoised development checks.'}
    best = min(c['development']['rmse'] for c in eligible)
    pool = [c for c in eligible if c['development']['rmse'] <= NEAR_FACTOR*best+ABSOLUTE_TOLERANCE]
    selected = min(pool, key=lambda c: (c['support_size'], c['development']['rmse'], c['candidate_id']))
    near = sorted({tuple(sorted(t['name'] for t in c['terms'])) for c in pool})
    return {'selected': selected, 'candidates': scored, 'near_optimal_supports': [list(s) for s in near],
            'failure_reason': None}


def stage_metadata(method):
    if method not in METHODS:
        raise ValueError('Unknown method.')
    if method in ('denoised_weak', 'denoised_median'):
        return {'structural_selection_predictor': 'denoised_resolved_factor4',
                'coefficient_selection_predictor': 'denoised_resolved_factor4',
                'final_predictor': 'denoised_resolved_factor4', 'initial_filter': 'spectral_hard',
                'selection_predictor': 'denoised_resolved_factor4'}
    return {**previous._stage_metadata(method), 'initial_filter': 'none'}


def fit_observations(training, developments):
    """Return four methods from shared training candidates, without oracle inputs."""
    if not isinstance(training, (list, tuple)) or not training:
        raise ValueError('At least one training trajectory is required.')
    candidates, trials = previous.generate_candidates(training)
    augmented, evidence = generate_support_medians(candidates)
    original = sparse.select_candidates(candidates, developments)
    resolved = refined.select_candidates(candidates, developments)
    augmented_result = select_candidates(augmented, developments)
    original_ids = {c['candidate_id'] for c in candidates}
    denoised_original = choose_scored([c for c in augmented_result['candidates'] if c['candidate_id'] in original_ids])
    results = {
        'weak_resolved': resolved,
        'fused_median': previous.select_from_scored(original, resolved, 'fused_median', developments),
        'denoised_weak': denoised_original,
        'denoised_median': augmented_result,
    }
    for method, result in results.items():
        result['variant_trials'] = deepcopy(trials)
        result['median_augmentation'] = deepcopy(evidence) if method == 'denoised_median' else None
        result.update(stage_metadata(method))
    return results
