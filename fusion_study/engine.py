"""Support-anchored fusion of existing weak fits, using development data only.

The original predictor anchors a support. The refined predictor compares
coefficients only within that support, and screens an equal-variant median.
These are different declared stages, not a claim that the two predictors agree.
Neither references nor family/noise labels are accepted by these interfaces.
"""
from __future__ import annotations

from copy import deepcopy
import numpy as np

from l1_study import engine as sparse
from weak_study import engine as weak
from resolved_study import selection as refined
from pde_discovery import generic_engine as baseline

METHODS = ('weak_original', 'weak_resolved', 'anchored_best', 'fused_median')
FUSION_MODES = ('anchored_best', 'fused_median')
NEAR_FACTOR = 1.05
ABSOLUTE_TOLERANCE = 1e-12
PROTOCOL = {
    'version': 1,
    'methods': list(METHODS),
    'candidate_generation': 'unchanged three fixed weak variants, adaptive L1, normalized OLS debias',
    'structural_anchor': 'original observed-grid development selector; exact selected support',
    'coefficient_comparison': 'refined factor4 development RMSE, restricted to anchor support',
    'anchored_best': 'minimum refined RMSE, stable candidate ID tie break',
    'median_pool': 'same-support eligible candidates within near_factor*best+absolute_tolerance',
    'variant_deduplication': 'one eligible same-support candidate per declared weak variant, minimum stable training candidate ID before near-optimal filtering',
    'combination': 'coordinatewise median of physical coefficients, equal weight per distinct weak variant',
    'median_near_factor': NEAR_FACTOR,
    'absolute_tolerance': ABSOLUTE_TOLERANCE,
    'single_variant': 'retain anchored best, no fusion',
    'median_acceptance': 'same refined development RMSE <= same threshold; otherwise anchored best',
    'support_preservation': 'reject nonfinite or exactly zero median coefficients, never delete terms',
    'final_predictor': 'original grid for weak_original; factor4 for other three arms',
    'cross_support_averaging': False,
    'path_frequency_weighting': False,
    'reference_used_for_selection': False,
    'noise_scale_used_for_selection': False,
    'variants_are_independent_estimates': False,
    'development_reuse': 'same development observations reused for support, coefficient and median screening; not independent validation',
}


def generate_candidates(training):
    """Generate each existing weak-variant adaptive path exactly once."""
    candidates, trials = [], []
    for variant in weak.VARIANTS:
        try:
            blocks = [weak.weak_block(d, variant) for d in training]
            fitted = sparse.fit_blocks(blocks, 'adaptive')
            for candidate in fitted:
                candidate['candidate_id'] = variant + '::' + candidate['candidate_id']
                candidate['weak_variant'] = variant
                candidate['method'] = 'weak_adaptive'
            candidates.extend(fitted)
            trials.append({'variant': variant, 'candidate_count': len(fitted), 'failure_reason': None})
        except (ValueError, RuntimeError, FloatingPointError, np.linalg.LinAlgError) as exc:
            trials.append({'variant': variant, 'candidate_count': 0, 'failure_reason': str(exc)})
    return candidates, trials


def _support(candidate):
    terms = candidate.get('terms', [])
    if not isinstance(terms, list) or not terms:
        raise ValueError('Candidate needs a nonempty term list.')
    names = []
    for term in terms:
        name, value = term.get('name'), term.get('coefficient')
        if (name not in sparse.TERM_NAMES or name in names
                or isinstance(value, (bool, complex))
                or not isinstance(value, (int, float, np.integer, np.floating))
                or not np.isfinite(value) or value == 0):
            raise ValueError('Candidate terms must be unique, known, finite and nonzero.')
        names.append(name)
    return tuple(name for name in sparse.TERM_NAMES if name in names)


def _error(candidate):
    value = candidate.get('development', {}).get('rmse')
    if (isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating))
            or not np.isfinite(value) or value < 0):
        raise ValueError('A finite nonnegative development RMSE is required.')
    return float(value)


def _stage_metadata(method):
    return {
        'structural_selection_predictor': 'original_observed_grid' if method != 'weak_resolved' else 'resolved_factor4',
        'coefficient_selection_predictor': 'original_observed_grid' if method == 'weak_original' else 'resolved_factor4',
        'final_predictor': 'original_observed_grid' if method == 'weak_original' else 'resolved_factor4',
        'selection_predictor': ('original_support_then_resolved_coefficients' if method in FUSION_MODES
                                else ('original_observed_grid' if method == 'weak_original' else 'resolved_factor4')),
    }


def select_from_scored(original_result, resolved_result, mode, developments=None):
    """Fuse saved scored candidates without fitting or using reference labels.

    ``anchored_best`` can use saved development scores alone. ``fused_median``
    must receive the development observations to score its new coefficient
    vector. Inputs are copied, so previous frozen evidence remains unchanged.
    """
    if mode not in FUSION_MODES:
        raise ValueError('Unknown fusion mode.')
    if mode == 'fused_median' and (not isinstance(developments, (list, tuple)) or not developments):
        raise ValueError('Median fusion requires development observations for its new vector.')
    if not isinstance(original_result, dict) or not isinstance(resolved_result, dict):
        raise ValueError('Both scored selection results are required.')
    evidence = {
        'mode': mode, 'anchor_candidate_id': None, 'anchor_support': [],
        'near_factor': NEAR_FACTOR, 'absolute_tolerance': ABSOLUTE_TOLERANCE,
        'best_same_support_candidate_id': None, 'best_same_support_rmse': None,
        'acceptance_threshold': None, 'same_support_candidate_ids': [],
        'excluded_candidates': [], 'variant_representative_ids': [],
        'member_candidate_ids': [], 'member_variants': [],
        'median_status': 'not_requested' if mode == 'anchored_best' else 'not_attempted',
        'median_reason': None, 'median_candidate': None, 'median_proposed_terms': None,
        'reference_used': False, 'path_frequency_weighting': False,
    }
    result = {'selected': None, 'candidates': [], 'near_optimal_supports': [],
              'failure_reason': None, 'fusion': evidence, **_stage_metadata(mode)}

    anchor = original_result.get('selected')
    if not anchor or not anchor.get('fit_success') or not anchor.get('eligible'):
        result['failure_reason'] = 'No eligible original weak structural anchor.'
        return result
    try:
        support = _support(anchor)
        _error(anchor)
    except (ValueError, TypeError, AttributeError) as exc:
        result['failure_reason'] = 'Invalid original structural anchor: ' + str(exc)
        return result
    evidence.update(anchor_candidate_id=anchor['candidate_id'], anchor_support=list(support))
    if not resolved_result.get('selected'):
        result['failure_reason'] = 'No eligible refined weak selection.'
        return result

    eligible = []
    for raw in resolved_result.get('candidates', []):
        candidate = deepcopy(raw)
        try:
            if not candidate.get('fit_success') or not candidate.get('eligible'):
                raise ValueError('Training or refined development eligibility failed.')
            if _support(candidate) != support:
                raise ValueError('Support differs from the original structural anchor.')
            if candidate.get('weak_variant') not in weak.VARIANTS:
                raise ValueError('Unknown weak variant.')
            if not isinstance(candidate.get('candidate_id'), str) or not candidate['candidate_id']:
                raise ValueError('A stable candidate ID is required.')
            _error(candidate)
        except (ValueError, TypeError, AttributeError) as exc:
            evidence['excluded_candidates'].append({'candidate_id': candidate.get('candidate_id'), 'reason': str(exc)})
            continue
        eligible.append(candidate)
    if not eligible:
        result['failure_reason'] = 'No eligible refined candidate has the anchored support.'
        return result

    eligible.sort(key=lambda c: (_error(c), c['candidate_id']))
    best = eligible[0]
    threshold = NEAR_FACTOR * _error(best) + ABSOLUTE_TOLERANCE
    evidence.update(best_same_support_candidate_id=best['candidate_id'],
                    best_same_support_rmse=_error(best), acceptance_threshold=threshold,
                    same_support_candidate_ids=[c['candidate_id'] for c in eligible])
    result.update(selected=deepcopy(best), candidates=deepcopy(eligible), near_optimal_supports=[list(support)])
    if mode == 'anchored_best':
        return result

    # Candidate multiplicity / lambda-path occurrences never determine weights.
    representatives = {}
    for candidate in sorted(eligible, key=lambda c: c['candidate_id']):
        representatives.setdefault(candidate['weak_variant'], candidate)
    evidence['variant_representative_ids'] = [representatives[v]['candidate_id']
                                               for v in weak.VARIANTS if v in representatives]
    ordered = [representatives[v] for v in weak.VARIANTS
               if v in representatives and _error(representatives[v]) <= threshold]
    evidence['member_candidate_ids'] = [c['candidate_id'] for c in ordered]
    evidence['member_variants'] = [c['weak_variant'] for c in ordered]
    if len(ordered) < 2:
        evidence.update(median_status='single_variant_no_fusion' if ordered else 'no_near_variants_no_fusion',
                        median_reason='Fewer than two distinct near-optimal weak variants; retain anchored best.')
        return result
    values = np.array([[next(t['coefficient'] for t in c['terms'] if t['name'] == name)
                        for name in support] for c in ordered], float)
    with np.errstate(over='ignore', invalid='ignore'):
        coefficients = np.median(values, axis=0)
    evidence['median_proposed_terms'] = [
        {'name': name, 'coefficient': float(value) if np.isfinite(value) else None}
        for name, value in zip(support, coefficients)]
    if not np.isfinite(coefficients).all() or np.any(coefficients == 0):
        evidence.update(median_status='invalid_coefficients',
                        median_reason='Median would remove an anchored term or create nonfinite coefficients; retain anchored best.')
        return result
    terms = [{'name': name, 'coefficient': float(value)} for name, value in zip(support, coefficients)]
    median = {
        'candidate_id': 'fusion-median::' + '-'.join(str(sparse.TERM_NAMES.index(name)) for name in support),
        'method': 'fused_median', 'terms': terms, 'equation': baseline._equation(terms),
        'support_size': len(support), 'fit_success': True,
        'coefficient_source': 'coordinatewise median of distinct training-fit variants',
        'fit_objective_not_recomputed': True,
        'diagnostics': {'member_candidate_ids': evidence['member_candidate_ids'],
                        'member_variants': evidence['member_variants'],
                        'equal_variant_weights': True, 'reference_equation_used': False},
    }
    # Use exactly the same refined forecast and equal-trajectory scoring rule.
    scored = refined.select_candidates([median], developments)
    median = scored['candidates'][0]
    evidence['median_candidate'] = deepcopy(median)
    result['candidates'].append(deepcopy(median))
    if not median.get('eligible'):
        evidence.update(median_status='forecast_rejected',
                        median_reason=median.get('rejection_reason', scored.get('failure_reason')))
    elif _error(median) > threshold:
        evidence.update(median_status='development_rejected',
                        median_reason='Median development RMSE exceeds fixed same-support near-optimal threshold.')
    else:
        result['selected'] = deepcopy(median)
        evidence.update(median_status='accepted', median_reason=None)
    return result


def fit_observations(training, developments):
    """Return all four arms from shared fits and shared development scoring."""
    if not isinstance(training, (list, tuple)) or not training:
        raise ValueError('At least one training trajectory is required.')
    candidates, trials = generate_candidates(training)
    original = sparse.select_candidates(candidates, developments)
    resolved = refined.select_candidates(candidates, developments)
    results = {
        'weak_original': original,
        'weak_resolved': resolved,
        'anchored_best': select_from_scored(original, resolved, 'anchored_best'),
        'fused_median': select_from_scored(original, resolved, 'fused_median', developments),
    }
    for method, result in results.items():
        result['variant_trials'] = deepcopy(trials)
        result.update(_stage_metadata(method))
    return results
