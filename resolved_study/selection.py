"""Same development selection rule, with an explicit consistent predictor."""
from copy import deepcopy
import numpy as np
from l1_study.engine import NEAR_FACTOR
from pde_discovery import generic_engine as baseline
from pde_discovery.generic_experiment import forecast_metrics
from .forecast import forecast


def select_candidates(candidates, developments):
    if not isinstance(developments,(list,tuple)) or not developments:
        raise ValueError('At least one development trajectory is required.')
    arrays=[baseline._validate(d) for d in developments]
    scored=deepcopy(candidates)
    for candidate in scored:
        candidate['eligible']=False
        if not candidate.get('fit_success'):
            candidate['rejection_reason']='Training convergence, finite coefficients or rank check failed.'
            continue
        try:
            individual=[]
            for x,t,u in arrays:
                prediction=forecast(candidate['terms'],x,t,u[0])
                individual.append(forecast_metrics(prediction,u))
            error=float(np.sqrt(np.mean([m['rmse']**2 for m in individual])))
            if not np.isfinite(error):raise ValueError('Development aggregate is not finite.')
            candidate['development']={'rmse':error,'per_trajectory':individual,
                'aggregation':'sqrt(mean(trajectory_MSE))','trajectory_count':len(individual)}
            candidate['eligible']=True
        except (ValueError,RuntimeError,FloatingPointError,np.linalg.LinAlgError) as exc:
            candidate['rejection_reason']=str(exc)
    eligible=[c for c in scored if c['eligible']]
    if not eligible:return {'selected':None,'candidates':scored,'near_optimal_supports':[],
                            'failure_reason':'No candidate passed training and development checks.'}
    best=min(c['development']['rmse'] for c in eligible)
    pool=[c for c in eligible if c['development']['rmse']<=NEAR_FACTOR*best+1e-12]
    selected=min(pool,key=lambda c:(c['support_size'],c['development']['rmse'],c['candidate_id']))
    near=sorted({tuple(sorted(t['name'] for t in c['terms'])) for c in pool})
    return {'selected':selected,'candidates':scored,'near_optimal_supports':[list(s) for s in near],
            'failure_reason':None}
