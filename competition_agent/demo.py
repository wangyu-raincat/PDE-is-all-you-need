"""Actual MCP client exercise with deterministic role routing, WITHOUT an LLM."""
import argparse,asyncio,json,os,sys,time
from pathlib import Path
from datetime import timedelta,datetime,timezone
from mcp import ClientSession,StdioServerParameters
from mcp.client.stdio import stdio_client
from competition_agent.bundle import ROLES
from competition_agent.service import digest,Lab

HERE=Path(__file__).resolve().parent
ROLE={tool:name for _,name,_,_,tools,_ in ROLES for tool in tools}
async def run(output,fixture='diffusion',noise=.01,budget=7,seed=240,policy='fixed_rich'):
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=True)
    trace=[]
    params=StdioServerParameters(command=sys.executable,args=['-m','competition_agent.mcp_server','--directory',str(output/'runtime')],cwd=str(HERE.parent))
    async with stdio_client(params) as (read,write):
        async with ClientSession(read,write,read_timeout_seconds=timedelta(seconds=180)) as session:
            init=await session.initialize();catalog=await session.list_tools()
            async def call(name,**arguments):
                started=time.monotonic();r=await session.call_tool(name,arguments)
                data=r.structuredContent
                if data is None:data=json.loads(r.content[0].text) if not r.isError else {'error':r.content[0].text}
                trace.append({'step':len(trace)+1,'planned_role':ROLE[name],'tool':name,'arguments':arguments,'result':data,'is_error':bool(r.isError),'duration_seconds':time.monotonic()-started})
                (output/'tool-trace.json').write_text(json.dumps(trace,ensure_ascii=False,indent=2)+'\n')
                print(f'{name}: '+('ERROR' if r.isError else 'ok'),flush=True)
                if r.isError:raise RuntimeError(data)
                return data
            status=await call('pde_case_create',fixture=fixture,noise=noise,budget=budget,seed=seed,policy=policy);case=status['case_id']
            if status['phase']!='ready':raise RuntimeError(status)
            documents=[{'title':p.name,'kind':'csv' if p.suffix=='.csv' else 'text','text':p.read_text()} for p in sorted((HERE/'knowledge').glob('*')) if p.suffix in ('.csv','.txt')]
            registered=await call('asset_ingest',case_id=case,documents=documents)
            hits=await call('asset_search',case_id=case,query='demo_field demo_grid',limit=6)
            facts=[]
            for subject,predicate,obj in [('demo_field','uses_unit','dimensionless'),('demo_grid','has_boundary','periodic')]:
                hit=next(h for h in hits['hits'] if f'{subject} {predicate} {obj}.'==h['quote'])
                facts.append({'subject':subject,'predicate':predicate,'object':obj,'source_id':hit['source_id'],'quote':hit['quote']})
            proposal=await call('graph_propose',case_id=case,facts=facts)
            await call('graph_apply',case_id=case,proposal_id=proposal['proposal_id'])
            status=await call('pde_identify',case_id=case)
            while status['charged_experiments']<budget:
                query=await call('experiment_propose',case_id=case)
                acquired=await call('experiment_acquire',case_id=case,proposal_id=query['id'])
                if acquired['query']['status']=='started':raise RuntimeError('Uncertain acquisition; do not retry')
                status=await call('pde_identify',case_id=case)
            await call('validation_freeze',case_id=case)
            report=await call('validation_evaluate',case_id=case)
            evidence=await call('evidence_get',case_id=case)
            result={'created_at_utc':datetime.now(timezone.utc).isoformat(),'execution':'actual MCP stdio with deterministic routing; no LLM','inputs':{'fixture':fixture,'noise':noise,'budget':budget,'seed':seed,'policy':policy},'protocol_version':init.protocolVersion,'tools':sorted(t.name for t in catalog.tools),'tool_calls':len(trace),'report':report,'chain_consistent':evidence['chain_consistent'],'live_nexent_verified':False}
            (output/'report.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
            (output/'evidence.json').write_text(json.dumps(evidence,ensure_ascii=False,indent=2)+'\n')
            return result


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    d=sub.add_parser('run');d.add_argument('--output',default='competition_agent/build/demo');d.add_argument('--fixture',default='diffusion');d.add_argument('--noise',type=float,default=.01);d.add_argument('--budget',type=int,default=7);d.add_argument('--seed',type=int,default=240);d.add_argument('--policy',default='fixed_rich')
    o=sub.add_parser('review-ontology');o.add_argument('--directory',required=True);o.add_argument('--case-id',required=True);o.add_argument('--proposal-id',required=True);o.add_argument('--decision',required=True,choices=['accept','reject'])
    a=p.parse_args()
    if a.command=='run':r=asyncio.run(run(a.output,a.fixture,a.noise,a.budget,a.seed,a.policy));print(json.dumps(r['report'],ensure_ascii=False,indent=2))
    else:print(json.dumps(Lab(a.directory).review_ontology(a.case_id,a.proposal_id,a.decision=='accept'),ensure_ascii=False,indent=2))
if __name__=='__main__':main()
