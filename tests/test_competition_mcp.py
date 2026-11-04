"""Real stdio and Streamable HTTP MCP negotiation; no mock transport or LLM."""
import asyncio,json,socket,subprocess,sys,tempfile,unittest
from datetime import timedelta
from pathlib import Path
import httpx
from mcp import ClientSession,StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamable_http_client
from competition_agent.bundle import ROLES

ROOT=Path(__file__).resolve().parents[1]
EXPECTED={t for row in ROLES for t in row[4]}

def value(result):return result.structuredContent or json.loads(result.content[0].text)

class CompetitionMCPTests(unittest.IsolatedAsyncioTestCase):
    async def check_catalog_and_errors(self,session):
        init=await session.initialize();self.assertTrue(init.protocolVersion)
        listing=await session.list_tools();self.assertEqual({t.name for t in listing.tools},EXPECTED)
        self.assertNotIn('review_ontology',EXPECTED)
        for t in listing.tools:self.assertEqual(t.inputSchema['type'],'object')
        bad=await session.call_tool('case_status',{'case_id':'../escape'});self.assertTrue(bad.isError)
        bad=await session.call_tool('pde_case_create',{'fixture':'not_supported'});self.assertTrue(bad.isError)
        return listing
    async def test_stdio_end_to_end_lifecycle_and_frozen_error(self):
        with tempfile.TemporaryDirectory() as d:
            params=StdioServerParameters(command=sys.executable,args=['-m','competition_agent.mcp_server','--directory',d],cwd=str(ROOT))
            async with stdio_client(params) as (r,w):
                async with ClientSession(r,w,read_timeout_seconds=timedelta(seconds=120)) as s:
                    await self.check_catalog_and_errors(s)
                    create=await s.call_tool('pde_case_create',{'budget':5,'seed':241});self.assertFalse(create.isError)
                    case=value(create)['case_id']
                    bad=await s.call_tool('asset_ingest',{'case_id':case,'documents':[{'title':'x','kind':'binary','text':'no'}]});self.assertTrue(bad.isError)
                    fit=await s.call_tool('pde_identify',{'case_id':case});self.assertFalse(fit.isError)
                    self.assertFalse(value(fit)['final_opened'])
                    frozen=await s.call_tool('validation_freeze',{'case_id':case});self.assertFalse(frozen.isError)
                    failed=await s.call_tool('pde_identify',{'case_id':case});self.assertTrue(failed.isError)
                    report=await s.call_tool('validation_evaluate',{'case_id':case});self.assertFalse(report.isError)
                    repeat=await s.call_tool('validation_evaluate',{'case_id':case});self.assertEqual(value(report),value(repeat))
                    self.assertFalse(value(report)['nexent_live_llm_verified'])
    async def test_streamable_http_actual_server(self):
        with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
        with tempfile.TemporaryDirectory() as d:
            with open(Path(d)/'server.log','wb') as log:
                proc=subprocess.Popen([sys.executable,'-m','competition_agent.mcp_server','--directory',d,'--transport','streamable-http','--port',str(port)],cwd=ROOT,stdout=log,stderr=log)
                try:
                    ready=False
                    for _ in range(100):
                        if proc.poll() is not None:raise RuntimeError((Path(d)/'server.log').read_text())
                        try:
                            r,w=await asyncio.open_connection('127.0.0.1',port);w.close();await w.wait_closed();ready=True;break
                        except OSError:await asyncio.sleep(.1)
                    self.assertTrue(ready,'MCP HTTP server did not start in 10 seconds')
                    async with httpx.AsyncClient(trust_env=False,timeout=30) as client:
                        async with streamable_http_client(f'http://127.0.0.1:{port}/mcp',http_client=client) as (r,w,_):
                            async with ClientSession(r,w,read_timeout_seconds=timedelta(seconds=30)) as s:await self.check_catalog_and_errors(s)
                finally:
                    proc.terminate()
                    try:proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:proc.kill();proc.wait(timeout=5)

if __name__=='__main__':unittest.main()
