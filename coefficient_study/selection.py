"""Observation-only coefficient reliability selection within a prediction pool.

The caller supplies already scored prefix candidates. This module never
forecasts, changes coefficients, or reads reference coefficients/final fields.
Its leave-one-trajectory-out score is a sensitivity diagnostic, not a confidence
interval; correlated weak windows are never resampled as independent rows.
"""
from __future__ import annotations

from copy import deepcopy
import numpy as np

from five_study import engine as prediction_selection
from l1_study import engine as sparse
from pde_discovery import generic_engine as baseline
from weak_study import engine as weak

PREDICTION_NEAR_FACTOR = 1.05
WEAK_NEAR_FACTOR = 1.05
ABSOLUTE_TOLERANCE = 1e-12
COMMON_VARIANT = 'resolved'
NORMALIZED_COEFFICIENT_FLOOR = sparse.SUPPORT_THRESHOLD
PROTOCOL = {
    'version': 1,
    'prediction_pool': 'existing eligible prefix scores <=1.05*best+1e-12',
    'support_priority': 'minimum nonzero term count within prediction pool',
    'common_validation_variant': COMMON_VARIANT,
    'common_validation_score': 'sqrt(mean(development trajectory weak residual MSE)) / common training target RMS',
    'common_validation_fields': 'development observations only, same resolved block builder for every candidate',
    'weak_pool': 'within minimum support-count pool, weak score <=1.05*best+1e-12',
    'final_order': 'minimum worst-term leave-one-trajectory-out relative RMS displacement, weak score, prediction RMSE, stable ID',
    'prediction_near_factor': PREDICTION_NEAR_FACTOR,
    'weak_near_factor': WEAK_NEAR_FACTOR,
    'absolute_tolerance': ABSOLUTE_TOLERANCE,
    'coefficient_refit': 'fixed support normalized OLS, equal trajectory weighting, rcond1e-10; never refit selected coefficients',
    'stability_resampling': 'omit one complete training trajectory, never independent weak rows',
    'minimum_training_trajectories': 3,
    'stability_denominator': 'max(abs(full coefficient),1e-6*common training target RMS/common training feature RMS)',
    'normalized_coefficient_floor': NORMALIZED_COEFFICIENT_FLOOR,
    'median_refit': 'refit participating variants on each fold, exact equal-vector aliases counted once, then coordinatewise median',
    'median_variants_are_independent': False,
    'stability_is_confidence_interval': False,
    'failure_policy': 'any required diagnostic failure retains original prediction-selected baseline with explicit fallback record',
    'reference_or_final_or_true_noise_input': False,
    'prediction_recomputed': False,
    'limits': 'stable coefficients can remain biased; common weak residuals have measurement error and correlated windows',
}


def _observations(records, minimum, role):
    if not isinstance(records, (list, tuple)) or not minimum <= len(records) <= 32:
        raise ValueError(f'{role} requires {minimum}..32 observation trajectories.')
    result = []
    for record in records:
        # Neither prefix state estimates nor reference/noise metadata reach a
        # caller-provided weak builder. Development u is used for validation.
        view = {key: record[key] for key in ('x', 't', 'u', 'boundary') if key in record}
        x, t, u = baseline._validate(view)
        result.append({'x': x, 't': t, 'u': u, 'boundary': 'periodic'})
    return result


def _coefficient_vector(candidate, support):
    values = {term['name']: float(term['coefficient']) for term in candidate['terms']}
    return np.array([values[name] for name in support], float)


def _participating_variants(candidate):
    variant = candidate.get('weak_variant')
    if variant in weak.VARIANTS:
        return [variant], False
    if not candidate['candidate_id'].startswith('support-median::'):
        raise ValueError('Candidate has no declared weak variant or median provenance.')
    groups = candidate.get('diagnostics', {}).get('distinct_vector_groups')
    if not isinstance(groups, list) or len(groups) < 2:
        raise ValueError('Median needs at least two recorded distinct training vectors.')
    variants = []
    for group in groups:
        current = group.get('variants')
        if not isinstance(current, list) or not current:
            raise ValueError('Median alias group must retain contributing variants.')
        for name in current:
            if name not in weak.VARIANTS or name in variants:
                raise ValueError('Median contains an unknown or repeated variant.')
            variants.append(name)
    return [name for name in weak.VARIANTS if name in variants], True


def _fixed_support_fit(blocks, indices):
    X, y, _ = sparse.pool_blocks(blocks)
    scales = np.sqrt(np.mean(X * X, axis=0))
    target_scale = float(np.sqrt(np.mean(y * y)))
    selected_scales = scales[indices]
    if (not np.isfinite(scales).all() or not np.isfinite(target_scale) or target_scale <= 1e-12
            or np.any(selected_scales <= max(1e-15, float(scales.max()) * 1e-12))):
        raise ValueError('Fixed support has no finite measurable training excitation.')
    normalized, _, rank, _ = np.linalg.lstsq(X[:, indices] / selected_scales, y / target_scale, rcond=1e-10)
    coefficients = normalized * target_scale / selected_scales
    if rank != len(indices) or not np.isfinite(coefficients).all():
        raise ValueError('Leave-one-trajectory-out support is rank deficient or nonfinite.')
    return coefficients


def _combine_variant_vectors(vectors, median):
    # Group exact physical coefficient vectors before every median. Alias
    # multiplicity never supplies votes, including on leave-out folds.
    groups = {}
    for variant, vector in vectors.items():
        groups.setdefault(tuple(float(v) for v in vector), []).append(variant)
    if median:
        coefficient = np.median(np.array(list(groups), float), axis=0)
    else:
        if len(vectors) != 1:
            raise ValueError('An original candidate must have exactly one weak variant.')
        coefficient = next(iter(vectors.values())).copy()
    if not np.isfinite(coefficient).all():
        raise ValueError('Nonfinite fixed-support coefficient combination.')
    evidence = [{'coefficients': list(vector), 'variants': names} for vector, names in groups.items()]
    return coefficient, evidence


def select_candidates(scored_candidates, training, developments, block_builder=weak.weak_block):
    """Select without repeating prediction or modifying fitted coefficients.

    Diagnostic failures explicitly fall back to the existing prediction result.
    The baseline candidate list and all failed diagnostics are preserved.
    """
    original = prediction_selection.choose_scored(scored_candidates)
    candidates = original['candidates']
    ids = [candidate.get('candidate_id') for candidate in candidates]
    if len(ids) != len(set(ids)):
        raise ValueError('Candidate identities must be unique.')
    receipt = {'status': 'pending', 'fallback_used': False, 'fallback_reason': None,
               'baseline_candidate_id': original['selected']['candidate_id'] if original['selected'] else None,
               'prediction_pool_ids': [], 'minimum_support_pool_ids': [], 'weak_pool_ids': [],
               'candidate_diagnostics': {}, 'prediction_recomputed': False,
               'reference_used': False, 'final_used': False,
               'whole_trajectory_omissions': True, 'confidence_interval_claimed': False}

    def result(selected, status, reason=None):
        receipt.update(status=status, fallback_used=status == 'fallback_to_prediction', fallback_reason=reason)
        if selected is not None:
            selected = next(c for c in candidates if c['candidate_id'] == selected['candidate_id'])
        return {'selected': deepcopy(selected), 'candidates': candidates,
                'near_optimal_supports': original['near_optimal_supports'],
                'failure_reason': original['failure_reason'], 'reliability_selection': receipt}

    if original['selected'] is None:
        return result(None, 'no_baseline_candidate')
    try:
        train = _observations(training, 3, 'Training reliability')
        dev = _observations(developments, 1, 'Development reliability')
        eligible = [c for c in candidates if c.get('eligible')]
        prediction_threshold = PREDICTION_NEAR_FACTOR * min(c['development']['rmse'] for c in eligible) + ABSOLUTE_TOLERANCE
        near = [c for c in eligible if c['development']['rmse'] <= prediction_threshold]
        minimum = min(c['support_size'] for c in near)
        pool = [c for c in near if c['support_size'] == minimum]
        receipt.update(prediction_threshold=prediction_threshold,
                       prediction_pool_ids=[c['candidate_id'] for c in near],
                       minimum_support_size=minimum, minimum_support_pool_ids=[c['candidate_id'] for c in pool],
                       training_trajectories=len(train), development_trajectories=len(dev))
        cache = {}

        def training_blocks(variant):
            if variant not in cache:
                cache[variant] = [block_builder(d, variant=variant) for d in train]
                for block in cache[variant]:
                    sparse.validate_block(block)
            return cache[variant]

        common = training_blocks(COMMON_VARIANT)
        common_X, common_y, _ = sparse.pool_blocks(common)
        common_scale = float(np.sqrt(np.mean(common_y ** 2)))
        common_feature_scale = np.sqrt(np.mean(common_X ** 2, axis=0))
        if not np.isfinite(common_scale) or common_scale <= 1e-12 or not np.isfinite(common_feature_scale).all():
            raise ValueError('Common training target scale is not measurable.')
        dev_blocks = [sparse.validate_block(block_builder(d, variant=COMMON_VARIANT)) for d in dev]
        receipt.update(common_validation_variant=COMMON_VARIANT, common_training_target_rms=common_scale,
                       common_training_feature_rms=common_feature_scale.tolist())
        # A reconstruction cache avoids repeated fixed-support OLS for aliases.
        fit_cache = {}

        def fit(variant, support, omit):
            key = (variant, support, omit)
            if key not in fit_cache:
                blocks = training_blocks(variant)
                chosen = [b for i, b in enumerate(blocks) if i != omit]
                indices = np.array([sparse.TERM_NAMES.index(name) for name in support])
                fit_cache[key] = _fixed_support_fit(chosen, indices)
            return fit_cache[key]

        for candidate in pool:
            identity = candidate['candidate_id']
            diagnostic = {'status': 'pending'}
            candidate['coefficient_reliability'] = diagnostic
            receipt['candidate_diagnostics'][identity] = diagnostic
            try:
                support = prediction_selection._support(candidate)
                indices = np.array([sparse.TERM_NAMES.index(name) for name in support])
                coefficients = _coefficient_vector(candidate, support)
                if np.any(common_feature_scale[indices] <= 1e-15):
                    raise ValueError('Selected term has no measurable common training feature.')
                floor = NORMALIZED_COEFFICIENT_FLOOR * common_scale / common_feature_scale[indices]
                denominator = np.maximum(np.abs(coefficients), floor)
                variants, median = _participating_variants(candidate)
                rebuilt, full_aliases = _combine_variant_vectors({v: fit(v, support, None) for v in variants}, median)
                # This guards supplying candidates from a different field,
                # normalization, support fit or noise-correction configuration.
                if np.any(np.abs(rebuilt - coefficients) > 1e-8 * np.maximum(np.abs(coefficients), floor)):
                    raise ValueError('Candidate coefficients do not match the declared training estimator.')
                folds = []
                for omitted in range(len(train)):
                    vector, aliases = _combine_variant_vectors({v: fit(v, support, omitted) for v in variants}, median)
                    folds.append({'omitted_training_trajectory': omitted, 'coefficients': vector.tolist(),
                                  'distinct_vector_groups': aliases})
                fold_coefficients = np.array([fold['coefficients'] for fold in folds])
                per_term = np.sqrt(np.mean(((fold_coefficients - coefficients) / denominator) ** 2, axis=0))
                per_dev = [float(np.sqrt(np.mean((y - X[:, indices] @ coefficients) ** 2)) / common_scale)
                           for X, y in dev_blocks]
                weak_score = float(np.sqrt(np.mean(np.square(per_dev))))
                stability = float(np.max(per_term))
                if not np.isfinite(weak_score) or not np.isfinite(stability):
                    raise ValueError('Nonfinite weak validation or coefficient stability diagnostic.')
                diagnostic.update(status='evaluated', support=list(support), weak_score=weak_score,
                    weak_per_development_trajectory=per_dev, worst_relative_omission_rms=stability,
                    per_term_relative_omission_rms=per_term.tolist(), coefficient_denominators=denominator.tolist(),
                    coefficient_floors=floor.tolist(), participating_variants=variants,
                    median_reconstructed=median, full_distinct_vector_groups=full_aliases, leave_out_folds=folds)
            except (ValueError, RuntimeError, FloatingPointError, np.linalg.LinAlgError, KeyError, TypeError) as exc:
                diagnostic.update(status='failed', reason=str(exc))
                raise ValueError(f'{identity}: {exc}') from exc

        weak_threshold = WEAK_NEAR_FACTOR * min(c['coefficient_reliability']['weak_score'] for c in pool) + ABSOLUTE_TOLERANCE
        weak_pool = [c for c in pool if c['coefficient_reliability']['weak_score'] <= weak_threshold]
        selected = min(weak_pool, key=lambda c: (c['coefficient_reliability']['worst_relative_omission_rms'],
            c['coefficient_reliability']['weak_score'], c['development']['rmse'], c['candidate_id']))
        receipt.update(weak_threshold=weak_threshold, weak_pool_ids=[c['candidate_id'] for c in weak_pool],
                       selected_candidate_id=selected['candidate_id'])
        return result(selected, 'selected')
    except (ValueError, RuntimeError, FloatingPointError, np.linalg.LinAlgError, KeyError, TypeError) as exc:
        return result(original['selected'], 'fallback_to_prediction', str(exc))
