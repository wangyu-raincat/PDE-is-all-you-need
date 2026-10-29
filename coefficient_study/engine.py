"""Four predeclared coefficient-estimation/selection arms on identical prefixes."""
from copy import deepcopy
import numpy as np
from weak_study import engine as weak
from l1_study import engine as sparse
from fusion_study.engine import generate_candidates
from five_study.engine import generate_support_medians
from prefix_study import engine as prefix
from . import noise,selection

METHODS=('prefix_baseline','stable_selection','noise_corrected','noise_stable')
PROTOCOL={'version':1,'methods':list(METHODS),
    'baseline':'unchanged prefix_median candidate generation and prefix prediction selection',
    'candidate_training':'three fixed weak variants, adaptive L1, normalized OLS debias, training-only support medians',
    'training_segment':'full forecast segment; historical prefix excluded from coefficient fitting',
    'corrected_training':'same weak blocks with observation-estimated white-Gaussian nonlinear moment correction',
    'noise_correction':noise.PROTOCOL,'reliability_selection':selection.PROTOCOL,
    'shared_scoring':'score each raw/corrected augmented pool once using the same fixed prefix predictor',
    'forecast_budget':'unchanged prefix11/6 frames and full future horizon for all arms',
    'reference_or_true_noise_or_family_input':False,'no_final_role_access':True}


def stage_metadata(method):
    if method not in METHODS:raise ValueError('Undeclared method.')
    return {'structural_selection_predictor':'prefix_aligned_resolved_factor4',
        'coefficient_selection_predictor':'prefix_aligned_resolved_factor4',
        'final_predictor':'prefix_aligned_resolved_factor4','initial_filter':'prefix_propagate_average',
        'conditioning':'observed_prefix','coefficient_noise_correction':method in ('noise_corrected','noise_stable'),
        'coefficient_selection':'weak_residual_then_trajectory_stability' if method in ('stable_selection','noise_stable') else 'forecast_only'}


def corrected_candidates(training):
    candidates=[];trials=[]
    for variant in weak.VARIANTS:
        try:
            blocks=[noise.weak_block(d,variant) for d in training]
            fitted=sparse.fit_blocks(blocks,'adaptive')
            for c in fitted:
                c['candidate_id']=variant+'::'+c['candidate_id'];c['weak_variant']=variant
                c['method']='noise_corrected_weak_adaptive'
            candidates.extend(fitted)
            trials.append({'variant':variant,'candidate_count':len(fitted),'failure_reason':None})
        except (ValueError,RuntimeError,FloatingPointError,np.linalg.LinAlgError) as exc:
            trials.append({'variant':variant,'candidate_count':0,'failure_reason':str(exc)})
    return candidates,trials


def fit_observations(training,developments):
    if not isinstance(training,(list,tuple)) or not training:raise ValueError('Training trajectories required.')
    fit_inputs=[{k:d[k] for k in ('x','t','u','boundary') if k in d} for d in training]
    raw,raw_trials=generate_candidates(fit_inputs)
    corrected,corrected_trials=corrected_candidates(fit_inputs)
    raw_pool,raw_medians=generate_support_medians(raw)
    corrected_pool,corrected_medians=generate_support_medians(corrected)
    raw_score=prefix.select_candidates(raw_pool,developments)
    corrected_score=prefix.select_candidates(corrected_pool,developments)
    results={'prefix_baseline':raw_score,
        'stable_selection':selection.select_candidates(raw_score['candidates'],fit_inputs,developments,block_builder=weak.weak_block),
        'noise_corrected':corrected_score,
        'noise_stable':selection.select_candidates(corrected_score['candidates'],fit_inputs,developments,block_builder=noise.weak_block)}
    for method,result in results.items():
        corrected_arm=method in ('noise_corrected','noise_stable')
        result.update(stage_metadata(method))
        result['variant_trials']=deepcopy(corrected_trials if corrected_arm else raw_trials)
        result['median_augmentation']=deepcopy(corrected_medians if corrected_arm else raw_medians)
    return results
