"""Budget-aware initial-condition selection using candidate predictions only.

The structural score is a discrimination proxy, not Bayesian information gain.
No reference-system module is imported here. Observations are used only to
establish their public spatial grid and estimate a measurement-noise floor.
Every candidate action is simulated by the discovered equations, never by the
hidden experimental provider. Invalid candidate forecasts remain in receipts.
"""
from __future__ import annotations

from copy import deepcopy
from numbers import Integral
import time

import numpy as np

from pde_discovery import generic_engine as solver
from . import actions as action_space

POLICIES = ('fixed_rich', 'random', 'ensemble_variance', 'structural_disagreement')
ACQUISITION_TIMES = (0., .1, .2)
MAX_COMMITTEE = 256
MAX_ACTIONS = 128
NUMERIC_VARIANCE_FLOOR = 1e-12
PROTOCOL = {
    'version': 1,
    'policies': list(POLICIES),
    'actions': 'same lawful pool for every policy; already selected ids excluded',
    'fixed_rich': 'descending number of excited modes, descending spectral entropy, stable action id',
    'random': 'uniform over remaining action ids after canonical sorting, supplied deterministic seed',
    'ensemble_variance': 'mean pointwise forecast variance of exact-distinct coefficient vectors, divided by experiment cost',
    'ensemble_variance_reference': 'E-SINDy-inspired variance baseline, not an exact reproduction of published E-SINDy',
    'structural_disagreement': 'between-structure forecast variance / (mean within-structure coefficient variance + observed noise variance + numerical variance floor) / experiment cost',
    'structure_weights': 'equal total weight for each canonical nonzero support; equal distinct-vector weight inside each support',
    'alias_weights': 'exact equal physical coefficient vectors counted once, regardless of path or bootstrap multiplicity',
    'acquisition_times': list(ACQUISITION_TIMES),
    'initial_frame_scored': False,
    'acquisition_horizon_is_final_forecast_horizon': False,
    'predictor': 'same observed periodic grid; exact diagonal Fourier propagation for affine linear candidates, existing ETDRK4 otherwise',
    'forward_guards': 'same dissipative highest even derivative and finite-state/growth budgets as generic_engine.forecast',
    'noise_floor': 'equal-trajectory mean variance estimated from pooled observed upper-band rFFT magnitudes; no true noise parameter',
    'noise_assumption': 'spatially white noise and weak physical upper-band energy; the estimate is not guaranteed unbiased',
    'numeric_variance_floor': NUMERIC_VARIANCE_FLOOR,
    'numerically_zero_prediction_variance': 'numerator variance <= numeric_variance_floor * fixed initial RMS squared is scored as zero',
    'default_cost': 'one experiment, with identical measurement geometry and duration across actions',
    'cost_override': 'positive scalar or complete action-id to positive cost mapping supplied before selection',
    'candidate_failure': 'an action is score-ineligible if any committee member fails; never score a selectively reduced committee',
    'no_structural_disagreement': 'explicit spectral-coverage exploration, never a claim of certain identification',
    'all_scores_invalid': 'explicit spectral-coverage exploration, retaining failures',
    'coverage_fallback': 'maximize minimum spectral-energy-profile distance from previously selected pool actions; richness and id break ties',
    'maximum_exact_distinct_models': MAX_COMMITTEE,
    'maximum_actions': MAX_ACTIONS,
    'reference_or_family_or_future_field_input': False,
    'calibrated_probability_or_information_gain_claimed': False,
}


def _canonical_terms(terms):
    if not isinstance(terms, (list, tuple)) or not 1 <= len(terms) <= len(solver.TERM_NAMES):
        raise ValueError('Each candidate requires one to nine named finite coefficients.')
    coefficients = {name: 0. for name in solver.TERM_NAMES}
    for term in terms:
        if not isinstance(term, dict) or term.get('name') not in coefficients:
            raise ValueError('Candidate contains an unsupported PDE term.')
        value = term.get('coefficient')
        if isinstance(value, (bool, np.bool_, complex)) or not np.isscalar(value):
            raise ValueError('Coefficients must be finite real numbers.')
        try:
            value = float(value)
        except (ValueError, TypeError, OverflowError) as exc:
            raise ValueError('Coefficients must be finite real numbers.') from exc
        if not np.isfinite(value):
            raise ValueError('Coefficients must be finite real numbers.')
        coefficients[term['name']] += value
    if not all(np.isfinite(v) for v in coefficients.values()):
        raise ValueError('Summed candidate coefficients overflowed.')
    result = [{'name': name, 'coefficient': value} for name, value in coefficients.items() if value != 0.]
    if not result:
        raise ValueError('An all-zero candidate does not define a discovery committee member.')
    return result


def canonical_committee(committee):
    """Deduplicate exact coefficient vectors, preserving all alias identities."""
    if not isinstance(committee, (list, tuple)) or len(committee) > 4096:
        raise ValueError('Committee must contain at most 4096 recorded members.')
    distinct = {}
    for index, candidate in enumerate(committee):
        if not isinstance(candidate, dict):
            raise ValueError('Committee entries must be candidate dictionaries.')
        terms = _canonical_terms(candidate.get('terms'))
        values = {term['name']: term['coefficient'] for term in terms}
        key = tuple(values.get(name, 0.) for name in solver.TERM_NAMES)
        identity = candidate.get('candidate_id', f'member-{index}')
        if not isinstance(identity, str):
            raise ValueError('Candidate identifiers must be strings.')
        if key not in distinct:
            distinct[key] = {'terms': terms, 'support': [term['name'] for term in terms],
                             'candidate_ids': [], 'coefficient_vector': list(key)}
        distinct[key]['candidate_ids'].append(identity)
    return [distinct[key] for key in sorted(distinct)]


def candidate_forecast(terms, x, t, initial):
    """Short-horizon forecast, with an exact affine-linear acceleration.

The acceleration does not use a reference operator: its Fourier symbol is
constructed entirely from the discovered coefficients supplied by the caller.
The constant feature is a DC forcing. Nonlinear candidates use the established
ETDRK4 implementation without changing its substep or forward-problem guards.
"""
    terms = _canonical_terms(terms)
    if any(np.iscomplexobj(np.asarray(v)) for v in (x, t, initial)):
        raise ValueError('Candidate prediction requires real coordinates and initial data.')
    x, t = solver._coordinates(x, t)
    initial = np.asarray(initial, float)
    if initial.shape != x.shape or not np.isfinite(initial).all():
        raise ValueError('Finite candidate initial data must match the spatial grid.')
    if (len(t)-1)*solver.SUBSTEPS > solver.MAX_STEPS:
        raise ValueError('Candidate forecast exceeds the established integration budget.')
    values = {name: 0. for name in solver.TERM_NAMES}
    values.update({term['name']: term['coefficient'] for term in terms})
    if values['u_xxxx'] > 0 or (values['u_xxxx'] == 0 and values['u_xx'] < 0):
        raise ValueError('Highest even derivative is antidissipative.')
    if any(values[name] != 0 for name in ('u^2', 'u^3', 'u*u_x')):
        return solver.forecast(terms, x, t, initial)
    wave, retained, _ = solver._geometry(x)
    with np.errstate(over='raise', invalid='raise', divide='raise'):
        linear = np.full(len(x), values['u'], complex)
        for order, name in enumerate(('u_x', 'u_xx', 'u_xxx', 'u_xxxx'), 1):
            linear += values[name]*(1j*wave)**order
        if not np.isfinite(linear).all():
            raise ValueError('Candidate Fourier operator is not finite.')
        for interval in np.unique(np.diff(t)):
            solver._etd_coefficients(linear, float(interval/solver.SUBSTEPS))
        initial_spectrum = np.fft.fft(initial)*retained
        forcing = np.zeros(len(x), complex)
        forcing[0] = len(x)*values['1']
        result = []
        limit = max(100., 1000*float(np.max(np.abs(initial))))
        for elapsed in t-t[0]:
            z = linear*elapsed
            state = (np.exp(z)*initial_spectrum + elapsed*solver._phi(z, 1)*forcing)*retained
            field = np.fft.ifft(state).real
            if (not np.isfinite(state).all() or np.max(np.abs(state)) > len(x)*limit
                    or not np.isfinite(field).all() or np.max(np.abs(field)) > limit):
                raise ValueError('Candidate forecast exceeded the finite-state growth budget.')
            result.append(field)
    return np.asarray(result)


def estimate_observation_noise(observations):
    """Use only already acquired training x,t,u, never noise metadata."""
    if not isinstance(observations, (list, tuple)) or not 1 <= len(observations) <= 32:
        raise ValueError('Acquisition requires one to 32 acquired training observations.')
    grid = None
    receipts = []
    for index, observation in enumerate(observations):
        if not isinstance(observation, dict):
            raise ValueError('Training observations must contain x,t,u.')
        view = {key: observation[key] for key in ('x', 't', 'u', 'boundary') if key in observation}
        if any(np.iscomplexobj(np.asarray(view.get(key))) for key in ('x', 't', 'u')):
            raise ValueError('Acquisition noise observations must be real-valued.')
        x, _, u = solver._validate(view)
        if len(x) > 512:
            raise ValueError('Observed acquisition grid exceeds 512 points.')
        if grid is None:
            grid = x.copy()
        elif not np.array_equal(grid, x):
            raise ValueError('All acquired observations must use the same spatial grid.')
        period = len(x)*(x[1]-x[0])
        if not np.isclose(period, action_space.PERIOD, rtol=0, atol=1e-10):
            raise ValueError('Action space requires a 2π-periodic observed spatial grid.')
        band = np.arange(len(x)//4, len(x)//2)
        magnitude = np.abs((np.fft.rfft(u, axis=1)/len(x))[:, band])
        radial = float(np.median(magnitude)/np.sqrt(np.log(2.)))
        variance = float(len(x)*radial*radial)
        if not np.isfinite(variance):
            raise ValueError('Estimated observation noise is not finite.')
        receipts.append({'training_trajectory': index, 'estimated_spatial_variance': variance,
                         'observed_frames': len(u), 'spatial_points': len(x)})
    return grid, {'estimated_spatial_variance': float(np.mean([r['estimated_spatial_variance'] for r in receipts])),
                  'per_trajectory': receipts, 'aggregation': 'equal trajectory mean variance',
                  'true_noise_used': False, 'physical_upper_band_signal_can_bias_estimate': True}


def _energy_profile(action):
    value = np.asarray(action['a'])**2 + np.asarray(action['b'])**2
    return value/float(np.sum(value))


def _richness_key(action):
    energy = _energy_profile(action)
    positive = energy[energy > 1e-14]
    entropy = float(-np.sum(positive*np.log(positive)))
    # Rounding only makes mathematically equal prescribed spectra tie by id;
    # it never changes predicted scores or coefficients.
    return (-len(positive), -round(entropy, 12), action['id'])


def _coverage_choice(available, pool, used):
    previous = [_energy_profile(action) for action in pool if action['id'] in used]
    distances = {action['id']: min((float(np.linalg.norm(_energy_profile(action)-p)) for p in previous),
                                  default=0.) for action in available}
    selected = min(available, key=lambda action: (-round(distances[action['id']], 12), *_richness_key(action)))
    return selected, distances


def _costs(pool, supplied):
    if supplied is None:
        return {action['id']: 1. for action in pool}, 'experiment'
    values = supplied if isinstance(supplied, dict) else {action['id']: supplied for action in pool}
    if set(values) != {action['id'] for action in pool}:
        raise ValueError('Cost mapping must contain every declared action id exactly once.')
    result = {}
    for identity, value in values.items():
        if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating))
                or not np.isfinite(value) or value <= 0):
            raise ValueError('Each experiment cost must be a positive finite number.')
        result[identity] = float(value)
    return result, 'caller_declared_cost'


def variance_components(predictions, supports):
    """Population variances with equal total weight for each structure."""
    values = np.asarray(predictions, float)
    if (values.ndim != 3 or len(values) != len(supports) or not len(values)
            or not np.isfinite(values).all()):
        raise ValueError('Need finite model-by-time-by-space predictions and matching supports.')
    groups = {}
    for index, support in enumerate(supports):
        groups.setdefault(tuple(support), []).append(index)
    means = np.array([np.mean(values[indices], axis=0) for indices in groups.values()])
    between = float(np.mean((means-np.mean(means, axis=0))**2))
    within = float(np.mean([np.mean((values[indices]-np.mean(values[indices], axis=0))**2)
                            for indices in groups.values()]))
    ensemble = float(np.mean((values-np.mean(values, axis=0))**2))
    return {'between_structure_variance': between, 'within_structure_variance': within,
            'ensemble_variance': ensemble, 'structure_count': len(groups),
            'models_per_structure': [{'support': list(support), 'models': len(indices)}
                                     for support, indices in sorted(groups.items())]}


def select_action(policy, committee, actions, observations, *, seed=0,
                  used_action_ids=(), cost_per_action=None):
    """Select one lawful action without observing any prospective outcome.

``observations`` must contain acquired *training* trajectories only. Root-level
budget accounting is separate; this function never queries an experiment and
does not accept a reference model or future observations. With equal geometry,
default unit costs are one experiment each. Exhaustion returns no action.
"""
    started = time.perf_counter()
    if policy not in POLICIES:
        raise ValueError('Unknown acquisition policy.')
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, Integral) or not 0 <= seed < 2**31:
        raise ValueError('Selection seed must be an integer in [0,2**31).')
    if not isinstance(actions, (list, tuple)) or not 1 <= len(actions) <= MAX_ACTIONS:
        raise ValueError('Provide one to 128 lawful action proposals.')
    pool = sorted((action_space.validate_action(action) for action in actions), key=lambda a: a['id'])
    if len({action['id'] for action in pool}) != len(pool):
        raise ValueError('Action identifiers must be unique.')
    if (not isinstance(used_action_ids, (list, tuple, set, frozenset))
            or any(not isinstance(identity, str) for identity in used_action_ids)):
        raise ValueError('Used action identifiers must be a collection of strings.')
    used = set(used_action_ids)
    available = [action for action in pool if action['id'] not in used]
    costs, units = _costs(pool, cost_per_action)
    members = canonical_committee(committee)
    grid, noise = estimate_observation_noise(observations)
    supports = [tuple(member['support']) for member in members]
    structure_count = len(set(supports))
    receipt = {'policy': policy, 'selected_action': None, 'selected_action_id': None,
        'status': 'pending', 'reason': None, 'fallback_used': False, 'scores': [],
        'raw_committee_members': len(committee), 'distinct_coefficient_vectors': len(members),
        'structure_count': structure_count, 'committee': deepcopy(members),
        'available_actions': len(available), 'excluded_action_ids': sorted(used),
        'cost_units': units, 'noise_estimate': noise,
        'acquisition_times': list(ACQUISITION_TIMES), 'acquisition_forecast_duration': .2,
        'initial_frame_scored': False, 'model_forecast_attempts': 0, 'model_forecast_failures': 0,
        'actual_experiment_queries': 0, 'reference_used': False, 'prospective_observations_used': False,
        'calibrated_probability_or_information_gain_claimed': False,
        'identification_certainty_claimed': False}

    def finish(selected, status, reason, fallback=False):
        receipt.update(status=status, reason=reason, fallback_used=fallback,
                       selected_action=deepcopy(selected),
                       selected_action_id=selected['id'] if selected is not None else None,
                       compute_seconds=float(time.perf_counter()-started))
        return receipt

    if not available:
        return finish(None, 'exhausted', 'Every declared action has already been selected.')
    if policy in ('fixed_rich', 'random'):
        ordered = sorted(available, key=_richness_key) if policy == 'fixed_rich' else available
        selected = ordered[0] if policy == 'fixed_rich' else ordered[int(np.random.default_rng(int(seed)).integers(len(ordered)))]
        receipt['scores'] = [{'action_id': a['id'], 'status': 'not_scored_by_policy', 'score': None,
                              'cost': costs[a['id']]} for a in available]
        if policy == 'fixed_rich':
            receipt['fixed_action_order'] = [a['id'] for a in ordered]
        return finish(selected, 'selected', 'Predetermined rich excitation.' if policy == 'fixed_rich'
                      else 'Uniform random remaining lawful action.')
    fallback_reason = None
    if not members:
        fallback_reason = 'No usable model committee; uncertainty is unresolved.'
    elif len(members) > MAX_COMMITTEE:
        fallback_reason = 'Committee exceeds the declared forecast budget; no member was silently truncated.'
    elif policy == 'structural_disagreement' and structure_count < 2:
        fallback_reason = 'Only one observed structure; no between-structure comparison establishes certainty.'
    if fallback_reason is not None:
        selected, distances = _coverage_choice(available, pool, used)
        receipt['scores'] = [{'action_id': a['id'], 'status': 'coverage_only', 'score': None,
            'cost': costs[a['id']], 'coverage_distance': distances[a['id']]} for a in available]
        return finish(selected, 'fallback_coverage', fallback_reason, True)
    for action in available:
        initial = action_space.action_field(action, grid)
        predictions, failures = [], []
        for index, member in enumerate(members):
            receipt['model_forecast_attempts'] += 1
            try:
                prediction = np.asarray(candidate_forecast(member['terms'], grid, np.asarray(ACQUISITION_TIMES), initial), float)
                if prediction.shape != (len(ACQUISITION_TIMES), len(grid)) or not np.isfinite(prediction).all():
                    raise ValueError('Candidate returned a nonfinite or wrong-shaped forecast.')
                predictions.append(prediction[1:])
            except (ValueError, RuntimeError, FloatingPointError, OverflowError, np.linalg.LinAlgError) as exc:
                failures.append({'member_index': index, 'candidate_ids': member['candidate_ids'], 'reason': str(exc)})
        row = {'action_id': action['id'], 'cost': costs[action['id']], 'score': None,
               'model_count_required': len(members), 'valid_model_count': len(predictions),
               'failed_models': failures, 'status': 'ineligible_candidate_failure' if failures else 'scored'}
        receipt['model_forecast_failures'] += len(failures)
        if not failures:
            component = variance_components(predictions, supports)
            denominator = (component['within_structure_variance'] + noise['estimated_spatial_variance']
                           + NUMERIC_VARIANCE_FLOOR*action_space.INITIAL_RMS**2)
            numerator = (component['between_structure_variance'] if policy == 'structural_disagreement'
                         else component['ensemble_variance'])
            numerically_zero = numerator <= NUMERIC_VARIANCE_FLOOR*action_space.INITIAL_RMS**2
            raw = (0. if numerically_zero else numerator/denominator
                   if policy == 'structural_disagreement' else numerator)
            score = raw/costs[action['id']]
            row.update(component, structural_denominator=denominator, variance_numerically_zero=numerically_zero)
            if not np.isfinite(score) or score < 0:
                row.update(status='ineligible_nonfinite_score', score=None)
            else:
                row['score'] = float(score)
        receipt['scores'].append(row)
    valid = [row for row in receipt['scores'] if row['score'] is not None]
    if not valid:
        selected, distances = _coverage_choice(available, pool, used)
        receipt['coverage_distances'] = distances
        return finish(selected, 'fallback_coverage', 'No action has a valid comparison of the complete committee.', True)
    if max(row['score'] for row in valid) <= 0:
        selected, distances = _coverage_choice(available, pool, used)
        receipt['coverage_distances'] = distances
        return finish(selected, 'fallback_coverage', 'All valid predicted differences vanish; this is not proof of identification.', True)
    best = min(valid, key=lambda row: (-row['score'], row['action_id']))
    selected = next(action for action in available if action['id'] == best['action_id'])
    return finish(selected, 'selected', 'Maximum declared candidate-prediction acquisition score.')
