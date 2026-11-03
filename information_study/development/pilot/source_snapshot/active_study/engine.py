"""Observation-only weak PDE discovery for budgeted active experiments.

The current trajectory set supplies every fitted quantity.  Complete-trajectory
bootstrap draws rerun the same adaptive L1 path, alias expansion and development
selection; the resulting frequencies are not calibrated existence probabilities.
Distinct structures and coefficient vectors are retained for experiment design.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import hashlib
import math

import numpy as np

from coefficient_study.noise import estimate_noise_variance
from l1_study import engine as sparse
from pde_discovery import generic_engine as baseline
from weak_study import engine as weak


TERM_NAMES = tuple(sparse.TERM_NAMES)
VARIANT = 'resolved'
ALIAS_CORRELATION = .995
NEAR_RELATIVE_ALLOWANCE = .05
NEAR_ABSOLUTE_ALLOWANCE = .001
MAX_ACCEPTED_WEAK_ERROR = .15
DEFAULT_BOOTSTRAP_REPLICATES = 8
MAX_BOOTSTRAP_REPLICATES = 64
MIN_TRAINING_TRAJECTORIES = 3
MAX_TRAJECTORIES = 16
PROTOCOL = {
    'version': 1,
    'candidate_library': list(TERM_NAMES),
    'mandatory_terms': [],
    'weak_variant': VARIANT,
    'weak_protocol': deepcopy(weak.PROTOCOL),
    'sparse_protocol': deepcopy(sparse.PROTOCOL),
    'sparse_method': 'adaptive',
    'observed_preprocessing': {
        'noise_estimator': 'pooled observed upper-band Rayleigh scale, per trajectory',
        'spectrum': 'rFFT(u)/Nx',
        'mode_amplitude': 'sqrt(mean over observed frames of abs(spectrum)**2)',
        'threshold': 'max(estimated coefficient-noise RMS, numerical floor) * sqrt(2*log(Nx))',
        'numerical_floor': '32*machine_epsilon*max(observed field RMS,1)',
        'mask': 'one time-shared mask per provided trajectory, DC always retained',
        'noise_model': 'constant-variance white Gaussian measurement noise with mostly noise in upper spatial band',
        'limitation': 'may remove weak physical modes or mistake broadband signal for noise',
        'true_noise_or_family_used': False,
        'prefix_or_final_fields_used': False,
    },
    'alias_expansion': 'closure of one-for-one correlated-column replacements from successful L1 supports',
    'alias_absolute_correlation': ALIAS_CORRELATION,
    'alias_support_limit': 'all nonempty subsets of the fixed nine-column library; no arbitrary first-k truncation',
    'alias_refit': 'equal-trajectory normalized OLS with the same numerical zero and rank rules as the L1 engine',
    'candidate_selection_score': 'sqrt(mean development-trajectory weak residual MSE) / own pooled training target RMS',
    'near_pool': 'score <= best + max(.05*best,.001)',
    'near_relative_allowance': NEAR_RELATIVE_ALLOWANCE,
    'near_absolute_allowance': NEAR_ABSOLUTE_ALLOWANCE,
    'maximum_accepted_weak_error': MAX_ACCEPTED_WEAK_ERROR,
    'selected_order': 'minimum support size, development weak score, canonical candidate ID',
    'committee': 'all full-data near-pool vectors plus bootstrap near-pool vectors passing the full-data development threshold',
    'committee_alias_weighting': 'exact coefficient-vector duplicates merged; no lambda-path count is a vote',
    'bootstrap': 'sample entire training trajectories with replacement; keep development observations fixed',
    'development_reuse': 'development observations are reused for selection across rounds; not final validation',
    'bootstrap_default_replicates': DEFAULT_BOOTSTRAP_REPLICATES,
    'bootstrap_maximum_replicates': MAX_BOOTSTRAP_REPLICATES,
    'bootstrap_caching': 'exact multiplicity vectors share one complete refit; deterministic per-trajectory weak transforms may be reused',
    'inclusion_frequency': 'fraction of requested bootstrap draws selecting the term; null if any draw failed',
    'frequency_interpretation': 'algorithmic resampling frequency, not posterior probability or calibrated confidence',
    'minimum_training_trajectories_with_bootstrap': MIN_TRAINING_TRAJECTORIES,
    'minimum_training_trajectories_without_bootstrap': 1,
    'maximum_training_or_development_trajectories': MAX_TRAJECTORIES,
    'forward_guard': 'highest nonzero even derivative must be dissipative; no truth-specific coefficient bounds',
    'final_or_reference_access': False,
    'unique_physical_equation_claimed': False,
    'cross_lambda_objective_comparison': False,
    'path_frequency_weighting': False,
}


def _observations(records, minimum, role):
    if not isinstance(records, (list, tuple)) or not minimum <= len(records) <= MAX_TRAJECTORIES:
        raise ValueError(f'{role} requires {minimum}..{MAX_TRAJECTORIES} complete observation trajectories.')
    parsed = []
    for record in records:
        if not isinstance(record, dict):
            raise ValueError(f'{role} trajectories must be dictionaries.')
        view = {k: record[k] for k in ('x', 't', 'u', 'boundary') if k in record}
        if any(np.iscomplexobj(np.asarray(view.get(k))) for k in ('x', 't', 'u')):
            raise ValueError('Observation arrays must be real.')
        x, t, u = baseline._validate(view)
        parsed.append({'x': x.copy(), 't': t.copy(), 'u': u.copy(), 'boundary': 'periodic'})
    first_x = parsed[0]['x']
    if any(not np.array_equal(item['x'], first_x) for item in parsed[1:]):
        raise ValueError('Observation trajectories must use the same spatial grid.')
    return parsed


def preprocess_observation(observation):
    """Filter only the provided observation frames, with one shared mode mask."""
    if any(np.iscomplexobj(np.asarray(observation.get(k))) for k in ('x', 't', 'u')):
        raise ValueError('Observation arrays must be real.')
    x, t, u = baseline._validate(observation)
    variance, estimated = estimate_noise_variance(x, u)
    n = len(x)
    spectrum = np.fft.rfft(u, axis=1)/n
    amplitudes = np.sqrt(np.mean(np.abs(spectrum)**2, axis=0))
    coefficient_noise = float(np.sqrt(variance/n))
    numerical_floor = float(32*np.finfo(float).eps*max(float(np.sqrt(np.mean(u*u))), 1.))
    threshold = float(max(coefficient_noise, numerical_floor)*np.sqrt(2*np.log(n)))
    keep = amplitudes >= threshold
    keep[0] = True
    filtered = np.fft.irfft(spectrum*keep[None, :], n=n, axis=1)*n
    if not np.isfinite(filtered).all():
        raise ValueError('Observation preprocessing produced nonfinite values.')
    difference = float(np.linalg.norm(filtered-u)/max(float(np.linalg.norm(u)), 1e-30))
    result = {'x': x.copy(), 't': t.copy(), 'u': filtered, 'boundary': 'periodic'}
    evidence = {
        'method': 'time_shared_observed_spectral_threshold',
        'observed_frames': len(t), 'spatial_points': n,
        'estimated_coefficient_noise_rms': coefficient_noise,
        'numerical_floor': numerical_floor, 'threshold': threshold,
        'retained_modes': np.flatnonzero(keep).tolist(),
        'removed_modes': np.flatnonzero(~keep).tolist(),
        'relative_field_change': difference,
        'removed_observed_energy_fraction': difference**2,
        'noise_estimation': estimated,
        'time_shared_mask': True, 'reference_used': False, 'true_noise_used': False,
        'weak_physical_modes_may_be_removed': True,
    }
    return result, evidence


def canonical_support(terms):
    if not isinstance(terms, (list, tuple)) or not terms:
        raise ValueError('A candidate needs a nonempty term list.')
    values = {}
    for term in terms:
        name, coefficient = term.get('name'), term.get('coefficient')
        if (name not in TERM_NAMES or name in values or isinstance(coefficient, (bool, complex))
                or not isinstance(coefficient, (int, float, np.floating, np.integer))
                or not np.isfinite(coefficient) or coefficient == 0):
            raise ValueError('Candidate terms must be unique, known and finite nonzero coefficients.')
        values[name] = float(coefficient)
    return tuple(name for name in TERM_NAMES if name in values)


def _forward_guard(terms):
    values = {term['name']: term['coefficient'] for term in terms}
    if values.get('u_xxxx', 0.) > 0 or (values.get('u_xxxx', 0.) == 0 and values.get('u_xx', 0.) < 0):
        return 'Highest nonzero even derivative is antidissipative.'
    return None


def _training_geometry(blocks):
    features, target, pooled = sparse.pool_blocks(blocks)
    scales = np.sqrt(np.mean(features*features, axis=0))
    target_scale = float(np.sqrt(np.mean(target*target)))
    if not np.isfinite(target_scale) or target_scale <= 1e-12:
        raise ValueError('No measurable temporal excitation.')
    active = scales > max(1e-15, float(scales.max())*1e-12)
    safe = np.where(active, scales, 1.)
    normalized = features/safe
    norms = np.linalg.norm(normalized, axis=0)
    correlation = normalized.T@normalized/np.maximum(np.outer(norms, norms), 1e-30)
    aliases = [(i, j) for i in range(len(TERM_NAMES)) for j in range(i+1, len(TERM_NAMES))
               if active[i] and active[j] and abs(correlation[i, j]) > ALIAS_CORRELATION]
    condition, rank = baseline._condition(normalized[:, active])
    return {'features': features, 'target': target, 'scales': safe, 'raw_scales': scales,
            'target_scale': target_scale, 'active': active, 'normalized': normalized,
            'correlation': correlation, 'aliases': aliases,
            'condition_number': condition, 'rank': rank, 'pooled': pooled}


def _expand_supports(seeds, alias_pairs):
    """Finite closure; each support is visited once and no path votes are used."""
    evidence = {tuple(support): {'origin': 'adaptive_l1', 'parent': None, 'replacement': None}
                for support in sorted(set(seeds))}
    pending = list(evidence)
    for support in pending:
        current = set(support)
        for i, j in alias_pairs:
            for old, new in ((i, j), (j, i)):
                if old not in current or new in current:
                    continue
                candidate = tuple(sorted((current-{old}) | {new}))
                if candidate not in evidence:
                    evidence[candidate] = {'origin': 'observed_column_alias', 'parent': list(support),
                                           'replacement': [TERM_NAMES[old], TERM_NAMES[new]]}
                    pending.append(candidate)
    if len(evidence) > 2**len(TERM_NAMES)-1:
        raise AssertionError('Alias closure exceeded the complete finite support space.')
    return evidence


def _alias_candidate(support, geometry, provenance):
    indices = np.asarray(support, int)
    normalized = np.linalg.lstsq(geometry['normalized'][:, indices],
                               geometry['target']/geometry['target_scale'], rcond=1e-10)[0]
    keep = np.abs(normalized) > sparse.DEBIAS_ZERO_THRESHOLD
    indices, normalized = indices[keep], normalized[keep]
    if not len(indices):
        return None
    condition, rank = baseline._condition(geometry['normalized'][:, indices])
    coefficients = normalized*geometry['target_scale']/geometry['scales'][indices]
    terms = [{'name': TERM_NAMES[i], 'coefficient': float(value)} for i, value in zip(indices, coefficients)]
    residual = geometry['target']-geometry['features'][:, indices]@coefficients
    success = rank == len(indices) and bool(np.isfinite(coefficients).all())
    return {
        'candidate_id': 'alias-'+'-'.join(str(i) for i in indices),
        'method': 'observed_column_alias_normalized_ols',
        'equation': baseline._equation(terms), 'terms': terms, 'support_size': len(indices),
        'support': [TERM_NAMES[i] for i in indices], 'fit_success': success,
        'fit_relative_error': float(np.linalg.norm(residual)/np.linalg.norm(geometry['target'])),
        'fit_rmse': float(np.sqrt(np.mean(residual*residual))),
        'failure_reason': None if success else 'Rank-deficient or nonfinite alias support.',
        'diagnostics': {'support_rank': rank, 'parameter_count': len(indices),
                        'condition_number': condition, 'reference_equation_used': False},
        'alias_provenance': deepcopy(provenance),
    }


def _score_candidate(candidate, development_blocks, target_scale):
    candidate = deepcopy(candidate)
    candidate['eligible'] = False
    candidate['development'] = None
    candidate['near_optimal'] = False
    try:
        support = canonical_support(candidate['terms'])
        candidate['support'] = list(support)
        if not candidate.get('fit_success'):
            raise ValueError(candidate.get('failure_reason') or 'Failed training candidate.')
        invalid = _forward_guard(candidate['terms'])
        if invalid:
            raise ValueError(invalid)
        values = {term['name']: term['coefficient'] for term in candidate['terms']}
        indices = np.array([TERM_NAMES.index(name) for name in support], int)
        coefficients = np.array([values[name] for name in support], float)
        residual_scores = [float(np.sqrt(np.mean((y-X[:, indices]@coefficients)**2))/target_scale)
                           for X, y in development_blocks]
        score = float(np.sqrt(np.mean(np.square(residual_scores))))
        if not np.isfinite(score):
            raise ValueError('Nonfinite development weak residual.')
        candidate['development'] = {'weak_score': score, 'per_trajectory_weak_score': residual_scores,
                                    'normalizer_training_target_rms': target_scale,
                                    'aggregation': 'sqrt(mean trajectory weak residual MSE)/training target RMS',
                                    'trajectory_count': len(development_blocks), 'variant': VARIANT}
        candidate['eligible'] = True
        candidate['rejection_reason'] = None
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError, np.linalg.LinAlgError) as exc:
        candidate['rejection_reason'] = str(exc)
    return candidate


def _fit_once(training_blocks, development_blocks):
    geometry = _training_geometry(training_blocks)
    originals = sparse.fit_blocks(training_blocks, 'adaptive')
    by_support = {}
    for candidate in originals:
        support = canonical_support(candidate['terms'])
        indices = tuple(TERM_NAMES.index(name) for name in support)
        candidate = deepcopy(candidate)
        candidate['support'] = list(support)
        candidate['alias_provenance'] = {'origin': 'adaptive_l1', 'parent': None, 'replacement': None}
        # Existing sparse code already deduplicates OLS-collapsed supports.  No
        # lambda-path multiplicity is used to weight or select a candidate.
        by_support[indices] = candidate
    seeds = [indices for indices, candidate in by_support.items() if candidate.get('fit_success')]
    closure = _expand_supports(seeds, geometry['aliases'])
    failed_aliases = []
    for support, provenance in sorted(closure.items()):
        if support in by_support:
            continue
        try:
            candidate = _alias_candidate(support, geometry, provenance)
            if candidate is None:
                failed_aliases.append({'support': [TERM_NAMES[i] for i in support],
                                       'reason': 'All normalized alias coefficients are numerical zero.'})
                continue
            key = tuple(TERM_NAMES.index(name) for name in canonical_support(candidate['terms']))
            if key not in by_support:
                by_support[key] = candidate
        except (ValueError, RuntimeError, FloatingPointError, np.linalg.LinAlgError) as exc:
            failed_aliases.append({'support': [TERM_NAMES[i] for i in support], 'reason': str(exc)})
    candidates = [_score_candidate(candidate, development_blocks, geometry['target_scale'])
                  for _, candidate in sorted(by_support.items())]
    eligible = [candidate for candidate in candidates if candidate['eligible']]
    selected, threshold, near = None, None, []
    failure = None
    if not eligible:
        failure = 'No candidate passed training, forward-operator and development checks.'
    else:
        best = min(candidate['development']['weak_score'] for candidate in eligible)
        threshold = best+max(NEAR_RELATIVE_ALLOWANCE*best, NEAR_ABSOLUTE_ALLOWANCE)
        if best > MAX_ACCEPTED_WEAK_ERROR:
            failure = 'Best development weak residual exceeds the fixed adequacy limit.'
        else:
            near = [candidate for candidate in eligible if candidate['development']['weak_score'] <= threshold]
            for candidate in near:
                candidate['near_optimal'] = True
            selected = min(near, key=lambda candidate: (candidate['support_size'],
                candidate['development']['weak_score'], candidate['candidate_id']))
    minimum = min((candidate['support_size'] for candidate in near), default=None)
    minimal_supports = sorted({tuple(candidate['support']) for candidate in near
                              if candidate['support_size'] == minimum})
    return {
        'selected': deepcopy(selected), 'candidates': candidates,
        'candidate_supports': [list(value) for value in sorted({tuple(c['support']) for c in near})],
        'minimal_near_supports': [list(value) for value in minimal_supports],
        'near_threshold': threshold, 'failure_reason': failure,
        'target_scale': geometry['target_scale'],
        'diagnostics': {
            'training_trajectories': len(training_blocks),
            'full_library_rank': geometry['rank'],
            'active_library_columns': int(geometry['active'].sum()),
            'full_library_condition_number': geometry['condition_number'],
            'correlated_columns': [{'terms': [TERM_NAMES[i], TERM_NAMES[j]],
                                    'correlation': float(geometry['correlation'][i, j])}
                                   for i, j in geometry['aliases']],
            'successful_l1_seed_supports': len(seeds), 'alias_closure_supports': len(closure),
            'candidate_count': len(candidates),
            'failed_training_candidates': sum(not c['fit_success'] for c in candidates),
            'rejected_candidates': sum(not c['eligible'] for c in candidates),
            'failed_alias_refits': failed_aliases,
            'cross_lambda_objective_comparison': False, 'path_frequency_weighting': False,
        },
    }


def _vector_key(candidate):
    values = {term['name']: float(term['coefficient']) for term in candidate['terms']}
    return tuple(values.get(name, 0.) for name in TERM_NAMES)


def discover(training, development, *, seed=0, bootstrap_replicates=DEFAULT_BOOTSTRAP_REPLICATES):
    """Fit observed trajectories and retain auditable structural uncertainty.

    No reference, role label, noise parameter, final field or prefix enters the
    estimator. Prefixes in caller dictionaries are ignored. All input arrays
    remain unchanged. An insufficient result is explicit, never a fabricated
    zero-error model. Validation mistakes raise before fitting.
    """
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError('seed must be an integer in [0,2**32).')
    if type(bootstrap_replicates) is not int or not 0 <= bootstrap_replicates <= MAX_BOOTSTRAP_REPLICATES:
        raise ValueError(f'bootstrap_replicates must be an integer in 0..{MAX_BOOTSTRAP_REPLICATES}.')
    train = _observations(training, MIN_TRAINING_TRAJECTORIES if bootstrap_replicates else 1, 'Training')
    dev = _observations(development, 1, 'Development')
    if any(not np.array_equal(item['x'], train[0]['x']) for item in dev):
        raise ValueError('Training and development must use the same spatial grid.')
    preprocessing = {'training': [], 'development': []}
    blocks = {}
    for role, records in (('training', train), ('development', dev)):
        blocks[role] = []
        for record in records:
            filtered, evidence = preprocess_observation(record)
            block = weak.weak_block(filtered, variant=VARIANT)
            blocks[role].append(block)
            preprocessing[role].append(evidence)
    dev_arrays = [sparse.validate_block(block) for block in blocks['development']]
    n = len(train)
    failure_records = []
    try:
        full = _fit_once(blocks['training'], dev_arrays)
    except (ValueError, RuntimeError, FloatingPointError, np.linalg.LinAlgError) as exc:
        full = {'selected': None, 'candidates': [], 'candidate_supports': [],
                'minimal_near_supports': [], 'near_threshold': None, 'target_scale': None,
                'failure_reason': str(exc), 'diagnostics': {'candidate_generation_failure': str(exc)}}
        failure_records.append({'role': 'full_training', 'reason': str(exc)})
    cache = {(1,)*n: full}
    records, bootstrap_results, counts = [], [], Counter()
    rng = np.random.default_rng(seed)
    refits = 1
    for index in range(bootstrap_replicates):
        draw = rng.integers(0, n, size=n)
        multiplicity = tuple(int(np.sum(draw == value)) for value in range(n))
        cached = multiplicity in cache
        if not cached:
            sampled_blocks = [block for block, times in zip(blocks['training'], multiplicity) for _ in range(times)]
            try:
                cache[multiplicity] = _fit_once(sampled_blocks, dev_arrays)
            except (ValueError, RuntimeError, FloatingPointError, np.linalg.LinAlgError) as exc:
                cache[multiplicity] = {'selected': None, 'candidates': [], 'candidate_supports': [],
                    'minimal_near_supports': [], 'near_threshold': None, 'failure_reason': str(exc), 'diagnostics': {}}
            refits += 1
        result = cache[multiplicity]
        selected = result['selected']
        record = {'replicate': index, 'draw': draw.tolist(), 'multiplicities': list(multiplicity),
                  'cached_exact_multiplicity': cached, 'status': 'selected' if selected is not None else 'failed',
                  'selected_support': selected['support'] if selected is not None else None,
                  'candidate_supports': result['candidate_supports'],
                  'failure_reason': result['failure_reason'],
                  'whole_trajectory_resampling': True, 'structure_selection_repeated': True}
        if selected is not None:
            counts.update(selected['support'])
        else:
            failure_records.append({'role': 'bootstrap', 'replicate': index, 'reason': result['failure_reason']})
        records.append(record)
        bootstrap_results.append(result)
    successful = sum(record['status'] == 'selected' for record in records)
    failures = bootstrap_replicates-successful
    frequencies = {name: float(counts[name]/bootstrap_replicates) if bootstrap_replicates and not failures else None
                   for name in TERM_NAMES}
    conditional = {name: float(counts[name]/successful) if successful else None for name in TERM_NAMES}
    bounds = {name: [counts[name]/bootstrap_replicates, (counts[name]+failures)/bootstrap_replicates]
              if bootstrap_replicates else None for name in TERM_NAMES}
    committee, rejected = {}, []
    if full['selected'] is not None:
        sources = [('full_training', None, full)]+[('bootstrap', i, result) for i, result in enumerate(bootstrap_results)]
        for role, index, result in sources:
            for original in result['candidates']:
                if not original.get('near_optimal'):
                    continue
                candidate = _score_candidate(original, dev_arrays, full['target_scale'])
                provenance = {'role': role, 'replicate': index, 'source_candidate_id': original['candidate_id']}
                if not candidate['eligible'] or candidate['development']['weak_score'] > full['near_threshold']:
                    rejected.append({**provenance, 'support': candidate.get('support'),
                        'reason': candidate.get('rejection_reason') or 'Outside full-data development threshold.'})
                    continue
                key = _vector_key(candidate)
                if key not in committee:
                    identity = hashlib.sha256(np.asarray(key, dtype='<f8').tobytes()).hexdigest()[:16]
                    committee[key] = {'candidate_id': 'committee-'+identity,
                        'terms': deepcopy(candidate['terms']), 'support': candidate['support'],
                        'equation': candidate['equation'], 'development': candidate['development'],
                        'provenance': []}
                committee[key]['provenance'].append(provenance)
    members = sorted(committee.values(), key=lambda c: (len(c['support']), c['support'], c['candidate_id']))
    supports = sorted({tuple(member['support']) for member in members})
    if full['selected'] is None:
        status = 'insufficient'
    elif len(supports) > 1:
        status = 'observational_ambiguity'
    else:
        status = 'candidate'
    diagnostics = {
        **full['diagnostics'], 'status': status,
        'training_trajectories': n, 'development_trajectories': len(dev),
        'committee_members': len(members), 'committee_structures': len(supports),
        'minimal_near_supports': full['minimal_near_supports'],
        'candidate_generation_failures': failure_records,
        'unique_physical_equation_established': False,
        'library_incompleteness_cannot_be_excluded': True,
        'singleton_committee_proves_identifiability': False,
        'data_roles_used': ['training', 'development'], 'final_data_used': False,
        'preprocessing': preprocessing,
        'computation': {'adaptive_l1_complete_refits': refits,
                        'bootstrap_requested_replicates': bootstrap_replicates,
                        'unique_bootstrap_multiplicity_vectors': len({tuple(r['multiplicities']) for r in records}),
                        'per_trajectory_weak_transform_reused': True},
    }
    uncertainty = {
        'method': 'complete_trajectory_bootstrap_full_structure_refit',
        'term_inclusion_frequencies': frequencies,
        'conditional_on_success_frequency': conditional,
        'frequency_bounds_including_failures': bounds,
        'raw_counts': {name: counts[name] for name in TERM_NAMES},
        'requested_replicates': bootstrap_replicates, 'completed_replicates': successful,
        'failed_replicates': failures, 'records': records,
        'number_of_available_multiplicity_vectors': math.comb(2*n-1, n-1),
        'calibrated': False, 'probability_claimed': False, 'confidence_interval_claimed': False,
        'independent_physical_experiments': n,
        'resampling_adds_physical_evidence': False,
        'limited_data_warning': 'Few complete trajectories; repeated bootstrap draws do not create independent evidence.' if n < 5 else None,
        'reference_used': False, 'final_used': False,
    }
    return {'selected': full['selected'], 'candidates': full['candidates'], 'committee': members,
        'candidate_supports': [list(support) for support in supports],
        'full_data_candidate_supports': full['candidate_supports'],
        'committee_rejections': rejected, 'uncertainty': uncertainty, 'diagnostics': diagnostics,
        'near_threshold': full['near_threshold'], 'failure_reason': full['failure_reason'],
        'status': status, 'protocol': deepcopy(PROTOCOL)}
