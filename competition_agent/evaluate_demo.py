"""Declared local integration matrix; descriptive synthetic checks, not promotion."""
import asyncio,json
from pathlib import Path
from competition_agent.demo import run
from competition_agent.service import source_hashes,digest

CASES=[{'name':f'{f}-noise1','fixture':f,'noise':.01} for f in ('decay','diffusion','hyperdiffusion','advection_diffusion','reaction_diffusion','burgers')]+[{'name':'burgers-noise3','fixture':'burgers','noise':.03}]
async def evaluate(output):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    plan={'type':'retrospective engineering integration checks','cases':CASES,'seed':240,'budget':7,'policy':'fixed_rich','tuning_or_promotion':False,'source_hashes':source_hashes()}
    (output/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    rows=[]
    for case in CASES:
        result=await run(output/case['name'],case['fixture'],case['noise'],7,240,'fixed_rich')
        r=result['report'];rows.append({'case':case['name'],'exact_support':r['structure']['exact_support'],'missing_terms':r['structure']['missing_terms'],'spurious_terms':r['structure']['spurious_terms'],'max_coefficient_error':r['max_true_coefficient_relative_error'],'mean_prediction_error':r['mean_prediction_error'],'prediction_failures':r['prediction_failure_slots'],'numerical_joint_success':r['numerical_joint_success'],'charged_discovery_experiments':r['charged_experiments'],'evaluation_reference_trajectories':r['evaluation_reference_trajectories_generated'],'mcp_calls':result['tool_calls']})
    assert plan['source_hashes']==source_hashes(),'Source changed during integration matrix'
    summary={'plan_sha256':digest(plan),'scope':plan['type'],'live_nexent_verified':False,'llm_used':False,'new_algorithm_benefit_tested':False,'rows':rows}
    (output/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(summary,ensure_ascii=False,indent=2))
if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',default='competition_agent/build/verification');a=p.parse_args();asyncio.run(evaluate(a.output))
