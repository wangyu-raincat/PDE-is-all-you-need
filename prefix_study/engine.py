"""Compare candidate PDEs on future development observations after a prefix."""
from copy import deepcopy
import numpy as np
from five_study import engine as single
from fusion_study.engine import generate_candidates
from resolved_study import selection as raw
from pde_discovery import generic_engine as baseline
from pde_discovery.generic_experiment import forecast_metrics
from .forecast import forecast_with_diagnostics, validate_prefix

METHODS=('weak_resolved','denoised_median','prefix_weak','prefix_median')
PROTOCOL={'version':1,'methods':list(METHODS),'candidate_training':single.PROTOCOL,
    'training_segment':'forecast segment only; prefix observations excluded from weak fitting',
    'median_generation':'same training-only exact-support distinct-vector rule as five_study',
    'selection':'same trajectory-equal RMSE; within1.05best+1e-12 fewestterms,error,stableID',
    'prefix_prediction':'fixed filtered propagation-to-cutoff average followed by full-horizon forecast',
    'single_frame_controls':'same forecast-start timestamp and complete future horizon; only last prefix observation used',
    'reference_or_noise_or_family_input':False,'future_holdout_used_for_selection':False}


def select_candidates(candidates,developments):
    if not isinstance(developments,(list,tuple)) or not developments:
        raise ValueError('Development trajectories with observed prefixes are required.')
    arrays=[]
    for d in developments:
        x,t,u=baseline._validate(d)
        _,_,pt,pu=validate_prefix(x,t,d.get('prefix_t'),d.get('prefix_u'))
        if not np.array_equal(u[0],pu[-1]):raise ValueError('Forecast-start observation must be the shared prefix endpoint.')
        arrays.append((x,t,u,pt,pu))
    scored=deepcopy(candidates)
    for candidate in scored:
        candidate['eligible']=False
        if not candidate.get('fit_success'):
            candidate['rejection_reason']='Training convergence, rank or finite-coefficient check failed.'
            continue
        try:
            individual=[];initial=[]
            for x,t,u,pt,pu in arrays:
                prediction,diagnostic=forecast_with_diagnostics(candidate['terms'],x,t,pt,pu)
                individual.append(forecast_metrics(prediction,u));initial.append(diagnostic)
            error=float(np.sqrt(np.mean([m['rmse']**2 for m in individual])))
            if not np.isfinite(error):raise ValueError('Nonfinite development score.')
            candidate['development']={'rmse':error,'per_trajectory':individual,
                'aggregation':'sqrt(mean(trajectory_MSE))','trajectory_count':len(individual),
                'prefix_state_diagnostics':initial}
            candidate['eligible']=True
        except (ValueError,RuntimeError,FloatingPointError,np.linalg.LinAlgError) as exc:
            candidate['rejection_reason']=str(exc)
    return single.choose_scored(scored)


def stage_metadata(method):
    if method not in METHODS:raise ValueError('Undeclared method.')
    predictor=('prefix_aligned_resolved_factor4' if method.startswith('prefix_') else
               'denoised_resolved_factor4' if method=='denoised_median' else 'resolved_factor4')
    return {'structural_selection_predictor':predictor,'coefficient_selection_predictor':predictor,
        'final_predictor':predictor,'initial_filter':('prefix_propagate_average' if method.startswith('prefix_') else
             'spectral_hard' if method=='denoised_median' else 'none'),
        'conditioning':'observed_prefix' if method.startswith('prefix_') else 'last_frame_only'}


def fit_observations(training,developments):
    if not isinstance(training,(list,tuple)) or not training:raise ValueError('Training trajectories required.')
    # Fitting consumes only post-cutoff x,t,u, even if history is also supplied.
    fit_inputs=[{k:d[k] for k in ('x','t','u','boundary') if k in d} for d in training]
    candidates,trials=generate_candidates(fit_inputs)
    augmented,evidence=single.generate_support_medians(candidates)
    original_ids={c['candidate_id'] for c in candidates}
    prefixed=select_candidates(augmented,developments)
    results={'weak_resolved':raw.select_candidates(candidates,developments),
        'denoised_median':single.select_candidates(augmented,developments),
        'prefix_weak':single.choose_scored([c for c in prefixed['candidates'] if c['candidate_id'] in original_ids]),
        'prefix_median':prefixed}
    for method,result in results.items():
        result.update(stage_metadata(method));result['variant_trials']=deepcopy(trials)
        result['median_augmentation']=deepcopy(evidence) if method in ('prefix_median','denoised_median') else None
    return results
