"""Actual MCP client exercise with deterministic role routing, WITHOUT an LLM."""
import argparse,asyncio,json,sys,time,uuid
from pathlib import Path
from datetime import timedelta,datetime,timezone
from mcp import ClientSession,StdioServerParameters
from mcp.client.stdio import stdio_client
from competition_agent.bundle import ROLES
from competition_agent.service import Lab
from competition_agent.reporting import save as save_html, render_inputs
from competition_agent.preprocess import prepare

HERE=Path(__file__).resolve().parent
ROLE_TOOLS={name:set(tools) for _,name,_,_,tools,_ in ROLES}
async def run(output,fixture='diffusion',noise=.01,budget=7,seed=240,policy='fixed_rich'):
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=True)
    if any(output.iterdir()):raise FileExistsError('Output directory is not empty; choose a new directory to preserve earlier runs')
    request_id=uuid.uuid4().hex
    with (output/'run-request.json').open('x',encoding='utf-8') as f:json.dump({'request_id':request_id,'fixture':fixture,'noise':noise,'budget':budget,'seed':seed,'policy':policy},f)
    documents=[{'title':p.name,'kind':'csv' if p.suffix=='.csv' else 'text','text':p.read_text()} for p in sorted((HERE/'knowledge').glob('*')) if p.suffix in ('.csv','.txt')]
    duplicate=next((d for d in documents if d['title']=='field-card.csv'),None)
    if duplicate:documents.append(dict(duplicate,title='field-card-copy.csv'))
    prepared=prepare(documents)
    (output/'input-original.json').write_text(json.dumps(documents,ensure_ascii=False,indent=2)+'\n')
    (output/'input-prepared.json').write_text(json.dumps(prepared,ensure_ascii=False,indent=2)+'\n')
    (output/'input-report.html').write_text(render_inputs(prepared),encoding='utf-8')
    if not prepared['summary']['ready_for_demo']:
        raise ValueError('Input preparation incomplete; inspect input-report.html. No MCP case or experiment was started.')
    trace=[]
    params=StdioServerParameters(command=sys.executable,args=['-m','competition_agent.mcp_server','--directory',str(output/'runtime')],cwd=str(HERE.parent))
    async with stdio_client(params) as (read,write):
        async with ClientSession(read,write,read_timeout_seconds=timedelta(seconds=180)) as session:
            init=await session.initialize();catalog=await session.list_tools()
            async def call(planned_role,name,**arguments):
                if name not in ROLE_TOOLS[planned_role]:raise ValueError(f"Tool {name} is outside planned role {planned_role}")
                started=time.monotonic();r=await session.call_tool(name,arguments)
                data=r.structuredContent
                if data is None:data=json.loads(r.content[0].text) if not r.isError else {'error':r.content[0].text}
                trace.append({'step':len(trace)+1,'planned_role':planned_role,'tool':name,'arguments':arguments,'result':data,'is_error':bool(r.isError),'duration_seconds':time.monotonic()-started})
                (output/'tool-trace.json').write_text(json.dumps(trace,ensure_ascii=False,indent=2)+'\n')
                print(f'{name}: '+('ERROR' if r.isError else 'ok'),flush=True)
                if r.isError:raise RuntimeError(data)
                return data
            status=await call('asset_agent','pde_case_create',fixture=fixture,noise=noise,budget=budget,seed=seed,policy=policy,request_id=request_id);case=status['case_id']
            if status['phase']!='ready':raise RuntimeError(status)
            await call('pde_coordinator','case_handoff',case_id=case)
            registered=await call('asset_agent','asset_ingest',case_id=case,documents=documents)
            packet=await call('graph_agent','asset_context',case_id=case,role='graph_agent')
            if packet['input_preparation']['blocked']:raise RuntimeError(packet['input_preparation'])
            hits=await call('graph_agent','asset_search',case_id=case,query='网格 字段 单位 边界',limit=10)
            if packet['next_offset'] is not None:raise RuntimeError('Demo context exceeds one declared page; do not truncate silently')
            facts=[c['fact'] for c in packet['fact_candidates']]
            proposal=await call('graph_agent','graph_propose',case_id=case,facts=facts)
            await call('graph_agent','graph_apply',case_id=case,proposal_id=proposal['proposal_id'])
            evidence_paths=await call('graph_agent','evidence_path',case_id=case,source='demo_grid',target='dimensionless')
            handoff=await call('pde_coordinator','case_handoff',case_id=case)
            if handoff['stop'] or handoff['next_tool']!='pde_identify':raise RuntimeError(handoff)
            status=await call('pde_agent','pde_identify',case_id=case)
            while status['charged_experiments']<budget:
                handoff=await call('pde_coordinator','case_handoff',case_id=case)
                if handoff['stop'] or handoff['next_tool']!='experiment_propose':raise RuntimeError(handoff)
                query=await call('experiment_agent','experiment_propose',case_id=case)
                handoff=await call('pde_coordinator','case_handoff',case_id=case)
                if handoff['next_arguments'].get('proposal_id')!=query['id']:raise RuntimeError(handoff)
                acquired=await call('experiment_agent','experiment_acquire',case_id=case,proposal_id=query['id'])
                if acquired['query']['status']=='started':raise RuntimeError('Uncertain acquisition; do not retry')
                status=await call('pde_agent','pde_identify',case_id=case)
            handoff=await call('pde_coordinator','case_handoff',case_id=case)
            if handoff['next_tool']!='validation_freeze':raise RuntimeError(handoff)
            await call('validation_agent','validation_freeze',case_id=case)
            handoff=await call('pde_coordinator','case_handoff',case_id=case)
            if handoff['next_tool']!='validation_evaluate':raise RuntimeError(handoff)
            report=await call('validation_agent','validation_evaluate',case_id=case)
            handoff=await call('pde_coordinator','case_handoff',case_id=case)
            if not handoff['stop'] or handoff['next_tool'] is not None:raise RuntimeError(handoff)
            evidence=await call('validation_agent','evidence_get',case_id=case)
            result={'created_at_utc':datetime.now(timezone.utc).isoformat(),'execution':'actual MCP stdio with deterministic routing; no LLM','inputs':{'fixture':fixture,'noise':noise,'budget':budget,'seed':seed,'policy':policy},'protocol_version':init.protocolVersion,'tools':sorted(t.name for t in catalog.tools),'tool_calls':len(trace),'report':report,'chain_consistent':evidence['chain_consistent'],'live_nexent_verified':False,'evidence_paths':evidence_paths,'final_handoff':handoff,'pre_agent_input_preparation':prepared['summary']}
            (output/'report.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
            (output/'evidence.json').write_text(json.dumps(evidence,ensure_ascii=False,indent=2)+'\n')
            save_html(output/'report.html',result,evidence,trace)
            return result


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    d=sub.add_parser('run');d.add_argument('--output',default='competition_agent/build/demo');d.add_argument('--fixture',default='diffusion');d.add_argument('--noise',type=float,default=.01);d.add_argument('--budget',type=int,default=7);d.add_argument('--seed',type=int,default=240);d.add_argument('--policy',default='fixed_rich')
    o=sub.add_parser('review-ontology');o.add_argument('--directory',required=True);o.add_argument('--case-id',required=True);o.add_argument('--proposal-id',required=True);o.add_argument('--decision',required=True,choices=['accept','reject'])
    a=p.parse_args()
    if a.command=='run':r=asyncio.run(run(a.output,a.fixture,a.noise,a.budget,a.seed,a.policy));print(json.dumps(r['report'],ensure_ascii=False,indent=2))
    else:print(json.dumps(Lab(a.directory).review_ontology(a.case_id,a.proposal_id,a.decision=='accept'),ensure_ascii=False,indent=2))
if __name__=='__main__':main()
