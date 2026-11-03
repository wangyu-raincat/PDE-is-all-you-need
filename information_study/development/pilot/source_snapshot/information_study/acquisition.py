"""Matched rich opening, then observed-anchor relative-coefficient design.

The inverse-Gram calculation is a local, support-conditional design proxy.
Overlapping weak windows are not independent experiments, the feature matrix is
noisy, and the selected support can be wrong. No calibrated covariance, Fisher
information, posterior, or information-gain interpretation is claimed.
"""
from __future__ import annotations

from copy import deepcopy
import time

import numpy as np

from active_study import acquisition as original
from active_study import actions as action_space
from active_study import engine as observed_engine
from l1_study import engine as sparse
from weak_study import engine as weak


LEGACY_POLICIES = original.POLICIES
POLICIES = (*LEGACY_POLICIES, 'rich_then_variance', 'rich_then_information')
PREDICTION_FRAMES = 33
PREDICTION_START = .1
PREDICTION_STOP = 1.1
RIDGE_RELATIVE = 1e-8
PSD_RELATIVE_TOLERANCE = 1e-12
SCORE_TIE_RTOL = 1e-10
SCORE_TIE_ATOL = 1e-12
PROTOCOL = {
    'version': 1,
    'policies': list(POLICIES),
    'legacy_policies': 'delegate original select_action unchanged, including all diagnostics and failure handling',
    'rich_then_variance': 'empty used_action_ids: original fixed_rich; otherwise original ensemble_variance',
    'rich_then_information': 'empty used_action_ids: original fixed_rich; otherwise conditional relative-coefficient inverse-Gram trace reduction',
    'first_action': 'both new policies use precisely the original fixed_rich result on the same action pool',
    'information_anchor': 'one full-data selected candidate supplied by the caller; never a bootstrap mean or reference equation',
    'support_interpretation': 'conditional on the selected support; neither singleton nor well-conditioned Gram proves the support correct',
    'observed_transform': 'unchanged active_study.engine.preprocess_observation then weak_study.engine.weak_block(resolved), all supplied training frames',
    'trajectory_weight': 'each acquired independent trajectory contributes X.T@X divided by its own weak row count; contributions accumulate',
    'column_scale': 's_j = sqrt(mean_trajectory mean_rows X_ij^2), computed once from currently acquired observations',
    'target_scale': 'q = sqrt(mean_trajectory mean_rows y_i^2), computed once from currently acquired observations',
    'relative_coefficient_scale': 'theta_j = abs(selected_coefficient_j) * s_j / q; no coefficient clipping or term deletion',
    'relative_features': 'Z_i = (X_i[:,S]/s_S)*theta_S = X_i[:,S]*abs(selected_coefficients_S)/q',
    'observed_gram': 'G = sum_trajectories Z_i.T@Z_i / row_count_i',
    'prospective_gram': 'G_a = Z_a.T@Z_a / row_count_a; same observed s, q and anchor coefficients for every action',
    'ridge': 'lambda = ridge_relative * max(trace(G)/support_size, 1), shared unchanged by all actions in the round',
    'ridge_relative': RIDGE_RELATIVE,
    'score': '(trace(inv(G+lambda*I))-trace(inv(G+G_a+lambda*I))) / declared_experiment_cost',
    'trace_reduction_computation': 'PSD square-root identity using C=A^-1/2 G_a A^-1/2; sum(mu/(1+mu)*column_norm(A^-1/2 eigenvectors(C))^2)',
    'psd_relative_tolerance': PSD_RELATIVE_TOLERANCE,
    'predicted_postprefix_frames': PREDICTION_FRAMES,
    'predicted_postprefix_start': PREDICTION_START,
    'predicted_postprefix_stop': PREDICTION_STOP,
    'predicted_postprefix_duration': PREDICTION_STOP-PREDICTION_START,
    'predictor': 'unchanged original.candidate_forecast; [0,.1] burn-in then uniform linspace(.1,1.1,33), because the original solver requires uniform time grids',
    'weak_prediction_input': 'only the 33 post-prefix candidate-predicted frames; no prospective measured field',
    'predicted_preprocessing': 'same unchanged observed spectral filtering on the candidate-predicted field; an approximation, no artificial measurement noise is added',
    'predicted_temporal_resolution': '33 frames are a declared design approximation; actual discovery and final validation retain their original 101 frames and full duration',
    'per_action_model_count': 1,
    'per_action_forecast_calls': 2,
    'per_action_forecast_intervals': 33,
    'measurement_cost_is_compute_cost': False,
    'score_tie': 'within score_tie_atol + score_tie_rtol*abs(max_score), original fixed_rich richness/entropy/id ordering',
    'score_tie_rtol': SCORE_TIE_RTOL,
    'score_tie_atol': SCORE_TIE_ATOL,
    'observed_support_rank': 'same 1e-10 singular-value rank rule as the existing normalized support solver; deficient support falls back explicitly',
    'failure': 'retain invalid anchor, observed geometry and per-action prediction failures; choose original fixed_rich if geometry fails or all actions are invalid/uninformative',
    'action_repeats': False,
    'committee_used_for_information_score': False,
    'reference_or_provider_or_future_measurement_access': False,
    'probability_or_calibrated_covariance_or_information_gain_claimed': False,
}


def _eigh_psd(matrix, label):
    matrix = np.asarray(matrix, float)
    if (matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or not len(matrix)
            or not np.isfinite(matrix).all()):
        raise ValueError(label+' must be a finite nonempty square matrix.')
    matrix = (matrix+matrix.T)/2
    values, vectors = np.linalg.eigh(matrix)
    tolerance = PSD_RELATIVE_TOLERANCE*max(float(np.max(np.abs(values))), 1.)
    if float(values.min()) < -tolerance:
        raise ValueError(label+' is not positive semidefinite within the declared numerical tolerance.')
    return np.maximum(values, 0.), vectors


def trace_reduction(gram, additional, ridge):
    """Stable nonnegative inverse-trace decrease, with explicit PSD checks."""
    if not np.isscalar(ridge) or not np.isfinite(ridge) or ridge <= 0:
        raise ValueError('Ridge must be finite and strictly positive.')
    values, vectors = _eigh_psd(gram, 'Observed Gram')
    add_values, add_vectors = _eigh_psd(additional, 'Prospective Gram')
    if np.asarray(gram).shape != np.asarray(additional).shape:
        raise ValueError('Observed and prospective Gram shapes must match.')
    additional_psd = (add_vectors*add_values)@add_vectors.T
    with np.errstate(over='raise', invalid='raise', divide='raise'):
        inverse_root = (vectors/(np.sqrt(values+ridge)))@vectors.T
        transformed = inverse_root@additional_psd@inverse_root
        mu, directions = _eigh_psd(transformed, 'Whitened prospective Gram')
        columns = inverse_root@directions
        reduction = float(np.sum((mu/(1.+mu))*np.sum(columns*columns, axis=0)))
        before = float(np.sum(1./(values+ridge)))
        after_eigenvalues, _ = _eigh_psd((vectors*values)@vectors.T+additional_psd,
                                        'Updated Gram')
        after = float(np.sum(1./(after_eigenvalues+ridge)))
    if not all(np.isfinite(v) and v >= 0 for v in (reduction, before, after)):
        raise ValueError('Inverse-trace proxy is nonfinite or negative.')
    tolerance = PSD_RELATIVE_TOLERANCE*max(before, 1.)
    if reduction > before+tolerance or after > before+tolerance:
        raise ValueError('Inverse-trace proxy violates the PSD monotonicity bound.')
    return {'trace_reduction': min(reduction, before), 'inverse_trace_before': before,
            'inverse_trace_after': after, 'numerical_method': 'PSD_square_root_identity'}


def _anchor(selected):
    if not isinstance(selected, dict):
        raise ValueError('No full-data selected candidate is available.')
    if selected.get('fit_success') is False or selected.get('eligible') is False:
        raise ValueError('The supplied full-data selected candidate is explicitly ineligible.')
    if (not isinstance(selected.get('terms'), (list, tuple))
            or any(not isinstance(term, dict) for term in selected['terms'])):
        raise ValueError('Selected candidate terms must be a list of coefficient dictionaries.')
    support = observed_engine.canonical_support(selected.get('terms'))
    if 'support' in selected and tuple(selected['support']) != support:
        raise ValueError('Selected candidate support disagrees with its coefficient terms.')
    invalid = observed_engine._forward_guard(selected['terms'])
    if invalid:
        raise ValueError(invalid)
    values = {term['name']: float(term['coefficient']) for term in selected['terms']}
    return [{'name': name, 'coefficient': values[name]} for name in support], support


def observed_geometry(observations, selected):
    """All scales and accumulated design evidence come from current training."""
    terms, support = _anchor(selected)
    parsed = observed_engine._observations(observations, 1, 'Information acquisition training')
    arrays, transforms, preprocessing = [], [], []
    for observation in parsed:
        filtered, evidence = observed_engine.preprocess_observation(observation)
        block = weak.weak_block(filtered, variant=observed_engine.VARIANT)
        arrays.append(sparse.validate_block(block))
        transforms.append(block['metadata'])
        preprocessing.append(evidence)
    feature_scale = np.sqrt(np.mean([np.mean(X*X, axis=0) for X, _ in arrays], axis=0))
    target_scale = float(np.sqrt(np.mean([np.mean(y*y) for _, y in arrays])))
    if not np.isfinite(target_scale) or target_scale <= 1e-12:
        raise ValueError('No measurable current-training temporal excitation.')
    indices = np.asarray([sparse.TERM_NAMES.index(name) for name in support], int)
    scales = feature_scale[indices]
    if (not np.isfinite(feature_scale).all()
            or np.any(scales <= max(1e-15, float(feature_scale.max())*1e-12))):
        raise ValueError('A selected support column has no numerically measurable training excitation.')
    coefficients = np.abs(np.asarray([term['coefficient'] for term in terms]))
    relative_scale = coefficients*scales/target_scale
    if not np.isfinite(relative_scale).all() or np.any(relative_scale <= 0):
        raise ValueError('Selected relative-coefficient scaling is nonfinite or underflows; no terms were dropped.')
    normalized_rows = np.vstack([X[:, indices]/scales/np.sqrt(len(X)) for X, _ in arrays])
    condition, rank = original.solver._condition(normalized_rows)
    if rank != len(indices):
        raise ValueError(f'Selected normalized feature support is rank deficient ({rank}/{len(indices)}).')
    with np.errstate(over='raise', invalid='raise', divide='raise'):
        grams = []
        for X, _ in arrays:
            Z = (X[:, indices]/scales)*relative_scale
            grams.append(Z.T@Z/len(Z))
        gram = np.sum(grams, axis=0)
    _eigh_psd(gram, 'Observed Gram')
    ridge = RIDGE_RELATIVE*max(float(np.trace(gram)/len(indices)), 1.)
    if not np.isfinite(ridge):
        raise ValueError('Observed shared ridge is not finite.')
    diagnostics = {'support': list(support), 'support_size': len(support),
        'selected_terms': deepcopy(terms), 'training_trajectories': len(parsed),
        'training_frames_per_trajectory': [len(d['t']) for d in parsed],
        'training_spatial_points': len(parsed[0]['x']),
        'weak_rows_per_trajectory': [len(y) for _, y in arrays],
        'observed_support_column_scales': scales.tolist(), 'observed_target_scale': target_scale,
        'selected_relative_coefficient_scales': relative_scale.tolist(),
        'normalized_support_rank': rank, 'normalized_support_condition_number': condition,
        'observed_gram': gram.tolist(), 'ridge': ridge,
        'observed_gram_eigenvalues': np.linalg.eigvalsh((gram+gram.T)/2).tolist(),
        'preprocessing': preprocessing, 'weak_transform_metadata': transforms,
        'support_correctness_established': False, 'overlapping_rows_are_independent_evidence': False,
        'reference_used': False, 'noncoefficient_selected_metadata_used_for_scaling': False}
    return {'terms': terms, 'support': support, 'indices': indices, 'grid': parsed[0]['x'],
            'scales': scales, 'relative_scale': relative_scale, 'gram': gram,
            'ridge': ridge, 'diagnostics': diagnostics}


def predict_anchor(terms, grid, action):
    """Two uniform-grid calls preserve the existing candidate forward operator."""
    return _predict_with_counter(terms, grid, action, original.candidate_forecast)


def select_action(policy, committee, actions, observations, *, seed=0,
                  used_action_ids=(), cost_per_action=None, selected=None):
    """Original acquisition API plus one optional full-data selected anchor."""
    if policy in LEGACY_POLICIES:
        return original.select_action(policy, committee, actions, observations, seed=seed,
                                      used_action_ids=used_action_ids, cost_per_action=cost_per_action)
    if policy not in POLICIES:
        raise ValueError('Unknown acquisition policy.')
    started = time.perf_counter()
    # This call validates the unchanged shared input contract and supplies the
    # exact original fixed-rich ordering for opening, ties and explicit fallback.
    fixed = original.select_action('fixed_rich', committee, actions, observations, seed=seed,
                                   used_action_ids=used_action_ids, cost_per_action=cost_per_action)
    if not used_action_ids or fixed['status'] == 'exhausted':
        fixed.update(policy=policy, delegated_policy='fixed_rich',
                     information_phase='matched_rich_opening' if not used_action_ids else 'exhausted',
                     first_action_matched_fixed_rich=not bool(used_action_ids),
                     compute_seconds=float(time.perf_counter()-started))
        return fixed
    if policy == 'rich_then_variance':
        result = original.select_action('ensemble_variance', committee, actions, observations,
                                        seed=seed, used_action_ids=used_action_ids,
                                        cost_per_action=cost_per_action)
        result.update(policy=policy, delegated_policy='ensemble_variance',
                      information_phase='post_rich_variance', compute_seconds=float(time.perf_counter()-started))
        return result
    result = deepcopy(fixed)
    result.update(policy=policy, delegated_policy=None, information_phase='post_rich_information',
        status='pending', reason=None, fallback_used=False, scores=[],
        selected_action=None, selected_action_id=None, model_count_used_for_information=0,
        information_anchor_terms=None, anchor_geometry=None, anchor_failure=None,
        model_forecast_attempts=0, model_forecast_failures=0,
        forecast_solver_calls_attempted=0, predicted_postprefix_frames=PREDICTION_FRAMES,
        acquisition_times=[0.]+np.linspace(PREDICTION_START, PREDICTION_STOP, PREDICTION_FRAMES).tolist(),
        acquisition_forecast_duration=PREDICTION_STOP,
        postprefix_design_duration=PREDICTION_STOP-PREDICTION_START,
        acquisition_horizon_is_scored_final_reference=False,
        statistical_covariance_or_information_gain_claimed=False,
        support_correctness_established=False, extra_measured_values_used=0,
        measurement_cost_is_compute_cost=False,
        compute_budget={'forecast_models_per_action': 1, 'forecast_calls_per_action': 2,
                        'forecast_intervals_per_action': PREDICTION_FRAMES,
                        'forecast_substeps_per_interval': original.solver.SUBSTEPS,
                        'weak_design_frames_per_action': PREDICTION_FRAMES})
    pool = sorted((action_space.validate_action(a) for a in actions), key=lambda a: a['id'])
    available = [a for a in pool if a['id'] not in set(used_action_ids)]
    costs = {row['action_id']: row['cost'] for row in fixed['scores']}

    def finish(action, status, reason, fallback=False):
        result.update(selected_action=deepcopy(action), selected_action_id=action['id'] if action else None,
                      status=status, reason=reason, fallback_used=fallback,
                      compute_seconds=float(time.perf_counter()-started))
        return result

    def fallback(reason):
        result['fallback_fixed_rich_action_id'] = fixed['selected_action_id']
        return finish(fixed['selected_action'], 'fallback_fixed_rich', reason, True)

    try:
        geometry = observed_geometry(observations, selected)
        result['model_count_used_for_information'] = 1
        result['information_anchor_terms'] = deepcopy(geometry['terms'])
        result['anchor_geometry'] = geometry['diagnostics']
    except (ValueError, RuntimeError, TypeError, KeyError, FloatingPointError, OverflowError,
            np.linalg.LinAlgError) as exc:
        result['anchor_failure'] = str(exc)
        result['scores'] = [{'action_id': a['id'], 'cost': costs[a['id']], 'score': None,
                             'status': 'not_scored_anchor_failure', 'failure_reason': str(exc)} for a in available]
        return fallback('Information anchor or observed geometry unavailable: '+str(exc))
    for action in available:
        row = {'action_id': action['id'], 'cost': costs[action['id']], 'score': None,
               'status': 'pending', 'model_count_required': 1, 'valid_model_count': 0,
               'failed_models': [], 'ridge': geometry['ridge']}
        result['model_forecast_attempts'] += 1
        calls = 0
        try:
            # Instrument calls individually, including a failed burn-in. This
            # count represents simulation effort, never extra observations.
            def counted_forecast(*args):
                nonlocal calls
                calls += 1
                return predictor(*args)
            predictor = original.candidate_forecast
            prediction = _predict_with_counter(geometry['terms'], geometry['grid'], action,
                                               counted_forecast)
            filtered, preprocessing = observed_engine.preprocess_observation(prediction)
            block = weak.weak_block(filtered, variant=observed_engine.VARIANT)
            X, _ = sparse.validate_block(block)
            with np.errstate(over='raise', invalid='raise', divide='raise'):
                Z = (X[:, geometry['indices']]/geometry['scales'])*geometry['relative_scale']
                additional = Z.T@Z/len(Z)
            metrics = trace_reduction(geometry['gram'], additional, geometry['ridge'])
            score = metrics['trace_reduction']/costs[action['id']]
            if not np.isfinite(score) or score < 0:
                raise ValueError('Cost-normalized information proxy is nonfinite or negative.')
            row.update(metrics, score=float(score), status='scored', valid_model_count=1,
                       prospective_gram=additional.tolist(), predicted_weak_rows=len(Z),
                       predicted_weak_metadata=block['metadata'], predicted_preprocessing=preprocessing)
        except (ValueError, RuntimeError, TypeError, KeyError, FloatingPointError, OverflowError,
                np.linalg.LinAlgError) as exc:
            result['model_forecast_failures'] += 1
            row.update(status='ineligible_prediction_or_design_failure', failure_reason=str(exc),
                       failed_models=[{'model': 'full_data_selected_anchor', 'reason': str(exc)}])
        row['forecast_solver_calls_attempted'] = calls
        result['forecast_solver_calls_attempted'] += calls
        result['scores'].append(row)
    valid = [row for row in result['scores'] if row['score'] is not None]
    if not valid:
        return fallback('Every anchor prediction or prospective design failed; no action was silently rescored with another model.')
    maximum = max(row['score'] for row in valid)
    if maximum <= 0:
        return fallback('All valid prospective inverse-trace reductions vanish; this does not establish identification.')
    tolerance = SCORE_TIE_ATOL+SCORE_TIE_RTOL*abs(maximum)
    tied = {row['action_id'] for row in valid if maximum-row['score'] <= tolerance}
    chosen = min((a for a in available if a['id'] in tied), key=original._richness_key)
    result['numerical_tie_action_ids'] = sorted(tied)
    return finish(chosen, 'selected', 'Maximum local conditional relative-coefficient design proxy; fixed-rich ordering resolves numerical ties.')


def _predict_with_counter(terms, grid, action, predictor):
    """Same prediction as predict_anchor, with an injectable call counter."""
    initial = action_space.action_field(action, grid)
    burn = np.asarray(predictor(terms, grid, np.array([0., PREDICTION_START]), initial))
    if burn.shape != (2, len(grid)) or np.iscomplexobj(burn) or not np.isfinite(burn).all():
        raise ValueError('Selected candidate returned invalid burn-in predictions.')
    times = np.linspace(PREDICTION_START, PREDICTION_STOP, PREDICTION_FRAMES)
    field = np.asarray(predictor(terms, grid, times, burn[-1]))
    if field.shape != (PREDICTION_FRAMES, len(grid)) or np.iscomplexobj(field) or not np.isfinite(field).all():
        raise ValueError('Selected candidate returned invalid post-prefix predictions.')
    return {'x': grid, 't': times, 'u': field, 'boundary': 'periodic'}
