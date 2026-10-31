"""Estimate the forecast-start state from observed history, then forecast.

Every historical frame is independently filtered and propagated to the same
cutoff using the candidate PDE. Their equally weighted states are averaged.
Only timestamps, not future observations, enter the prediction interface.
"""
import numpy as np
from five_study.initial import denoise_initial, PROTOCOL as INITIAL_PROTOCOL
from resolved_study import forecast as resolved
from pde_discovery import generic_engine as baseline

PROTOCOL={'version':1,'name':'prefix_aligned_resolved_factor4',
    'history':'observed frames at or before forecast start only; endpoint shared once',
    'estimator':'independently filter each prefix frame, propagate each to cutoff with candidate PDE, equal mean of terminal states',
    'weights':'fixed equal per observed prefix frame, no development or final fitting',
    'filter':INITIAL_PROTOCOL,'solver':resolved.PROTOCOL,
    'max_prefix_frames':32,'minimum_prefix_frames':2,
    'forecast_horizon':'unchanged after cutoff; prefix not counted as predicted future',
    'conditional_forecast':True,'future_observations_input':False,
    'limitations':'candidate-model bias and sparse-spectrum initial-filter prior; not independent propagated estimates'}


def validate_prefix(x,t,prefix_t,prefix_u):
    if any(np.iscomplexobj(np.asarray(v)) for v in (x,t,prefix_t,prefix_u)):
        raise ValueError('History and coordinates must be real.')
    x,t=baseline._coordinates(x,t)
    pt=np.asarray(prefix_t,float);pu=np.asarray(prefix_u,float)
    if (pt.ndim!=1 or not 2<=len(pt)<=32 or pu.shape!=(len(pt),len(x))
            or not np.isfinite(pt).all() or not np.isfinite(pu).all()
            or np.any(np.diff(pt)<=0) or not np.isclose(pt[-1],t[0],rtol=0,atol=1e-10)
            or np.any(pt>t[0]+1e-10)):
        raise ValueError('Need 2..32 finite historical frames ending exactly at forecast start.')
    return x,t,pt,pu


def estimate_state(terms,x,t,prefix_t,prefix_u):
    x,t,pt,pu=validate_prefix(x,t,prefix_t,prefix_u)
    states=[];filter_evidence=[]
    for i in range(len(pt)):
        filtered,diagnostic=denoise_initial(x,pu[i])
        if i<len(pt)-1:
            state=resolved.forecast(terms,x,pt[i:],filtered)[-1]
        else:state=filtered
        states.append(state);filter_evidence.append(diagnostic)
    state=np.mean(np.asarray(states),axis=0)
    if not np.isfinite(state).all():raise ValueError('Prefix estimate is not finite.')
    diagnostic={'conditioning':'observed_prefix','prefix_frames':len(pt),
        'prefix_start':float(pt[0]),'forecast_start':float(t[0]),
        'prefix_duration':float(pt[-1]-pt[0]),'forecast_duration':float(t[-1]-t[0]),
        'future_observations_used':False,'equal_frame_weights':True,
        'candidate_PDE_used_for_state_estimation':True,
        'last_observation_relative_change':float(np.linalg.norm(state-pu[-1])/max(np.linalg.norm(pu[-1]),1e-30)),
        'per_frame_filter':filter_evidence,'independent_estimates_claimed':False}
    return state,diagnostic


def forecast_with_diagnostics(terms,x,t,prefix_t,prefix_u):
    state,diagnostic=estimate_state(terms,x,t,prefix_t,prefix_u)
    return resolved.forecast(terms,x,t,state),diagnostic


def forecast(terms,x,t,prefix_t,prefix_u):
    return forecast_with_diagnostics(terms,x,t,prefix_t,prefix_u)[0]
