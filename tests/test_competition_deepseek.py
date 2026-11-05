import asyncio,json,tempfile,unittest
from pathlib import Path
from unittest.mock import AsyncMock,patch
import httpx
from competition_agent.deepseek_acceptance import (AcceptanceError,Journal,DeepSeek,Gateway,credential,redact,compare_final,run,example_documents)
from competition_agent.preprocess import prepare

class DeepSeekContracts(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.key='sk-testcredential000000000000'
        self.log=Journal(self.root/'trace.jsonl',self.key)
    async def test_live_client_preserves_tool_calls_and_records_usage_without_key(self):
        def response(req):
            self.assertEqual(req.url.host,'api.deepseek.com');self.assertEqual(req.headers['Authorization'],'Bearer '+self.key)
            payload=json.loads(req.content);self.assertEqual(payload['thinking'],{'type':'disabled'})
            return httpx.Response(200,json={'model':'deepseek-flash','choices':[{'finish_reason':'tool_calls','message':{'role':'assistant','content':None,'tool_calls':[{'id':'call-1','type':'function','function':{'name':'case_status','arguments':'{}'}}],'reasoning_content':'not retained'}}],'usage':{'prompt_tokens':4,'completion_tokens':3,'total_tokens':7}})
        c=DeepSeek(self.key,self.log,transport=httpx.MockTransport(response))
        try:r=await c.complete('asset_agent',[{'role':'user','content':'test'}],[])
        finally:await c.close()
        self.assertEqual(r['tool_calls'][0]['id'],'call-1');self.assertEqual(c.usage['total_tokens'],7)
        self.assertNotIn(self.key,self.log.path.read_text());self.assertNotIn('reasoning_content',self.log.path.read_text())
    async def test_http_error_does_not_log_body_or_retry(self):
        c=DeepSeek(self.key,self.log,transport=httpx.MockTransport(lambda req:httpx.Response(401,text=self.key)))
        try:
            with self.assertRaisesRegex(AcceptanceError,'HTTP 401'):await c.complete('asset_agent',[],[])
        finally:await c.close()
        self.assertEqual(c.calls,1);self.assertNotIn(self.key,self.log.path.read_text())
    async def test_global_request_budget_is_enforced_before_network(self):
        transport=httpx.MockTransport(lambda req:self.fail('No request allowed'))
        c=DeepSeek(self.key,self.log,max_calls=0,transport=transport)
        try:
            with self.assertRaisesRegex(AcceptanceError,'request limit'):await c.complete('asset_agent',[],[])
        finally:await c.close()
    async def test_timeout_marks_uncertain_without_retry(self):
        def response(req):raise httpx.ReadTimeout('private token '+self.key,request=req)
        c=DeepSeek(self.key,self.log,transport=httpx.MockTransport(response))
        try:
            with self.assertRaisesRegex(AcceptanceError,'uncertain'):await c.complete('asset_agent',[],[])
        finally:await c.close()
        self.assertEqual(c.calls,1);self.assertNotIn(self.key,self.log.path.read_text())
    async def test_truncated_completion_is_not_executed(self):
        c=DeepSeek(self.key,self.log,transport=httpx.MockTransport(lambda req:httpx.Response(200,json={'choices':[{'finish_reason':'length','message':{'role':'assistant','content':'partial'}}]})))
        try:
            with self.assertRaisesRegex(AcceptanceError,'truncated'):await c.complete('asset_agent',[],[])
        finally:await c.close()
    def gateway(self):
        docs=example_documents();session=AsyncMock();g=Gateway(session,self.log,docs,prepare(docs),{'budget':6});return g,session
    async def test_role_scope_rejects_before_mcp(self):
        g,s=self.gateway();r=await g.call('graph_agent','pde_identify',{'case_id':'x'})
        self.assertEqual(r['error'],'gateway_denied');s.call_tool.assert_not_called()
    async def test_case_binding_and_config_are_not_model_editable(self):
        g,s=self.gateway()
        r=await g.call('asset_agent','pde_case_create',{'budget':9});self.assertEqual(r['error'],'gateway_denied')
        g.case_id='current';r=await g.call('asset_agent','case_status',{'case_id':'other'});self.assertEqual(r['error'],'gateway_denied')
        r=await g.call('asset_agent','pde_case_create',{'budget':6});self.assertEqual(r['error'],'gateway_denied');s.call_tool.assert_not_called()
    async def test_unprepared_document_cannot_replace_original(self):
        g,s=self.gateway();g.case_id='current'
        r=await g.call('asset_agent','asset_ingest',{'case_id':'current','documents':[{'title':'invented','kind':'text','text':'field uses_unit m.'}]})
        self.assertEqual(r['error'],'gateway_denied');s.call_tool.assert_not_called()
    async def test_readonly_denial_is_a_model_attempt_not_a_success(self):
        g,s=self.gateway();g.case_id='current';g.read_only=True
        await g.call('pde_agent','pde_identify',{'case_id':'current'})
        self.assertEqual(g.denials,1);self.assertEqual(g.calls,0);s.call_tool.assert_not_called()
    async def test_invalid_input_stops_before_api_and_mcp(self):
        with patch('competition_agent.deepseek_acceptance.DeepSeek') as client,patch('competition_agent.deepseek_acceptance.stdio_client') as mcp:
            with self.assertRaisesRegex(AcceptanceError,'Input preparation'):
                await run(self.root/'bad',self.key,documents=[{'title':'bad','kind':'csv','text':'subject,predicate,object\na,b\n'}])
            client.assert_not_called();mcp.assert_not_called()
            self.assertTrue((self.root/'bad/input-report.html').exists());self.assertFalse((self.root/'bad/runtime').exists())
    async def test_existing_output_is_not_overwritten(self):
        path=self.root/'keep';path.mkdir();(path/'original').write_text('preserve')
        with self.assertRaisesRegex(AcceptanceError,'not empty'):await run(path,self.key)
        self.assertEqual((path/'original').read_text(),'preserve')
    def test_private_credential_file_and_redaction(self):
        p=self.root/'key';p.write_text(self.key);p.chmod(0o644)
        with self.assertRaises(AcceptanceError):credential(p)
        p.chmod(0o600);self.assertEqual(credential(p),self.key)
        self.assertNotIn(self.key,str(redact({'authorization':self.key,'nested':['Bearer '+self.key]},self.key)))
    def test_final_report_requires_exact_sources_and_metrics(self):
        report={'case_id':'a','structure':{'exact_support':True},'evidence_ready':True,'charged_experiments':6,'max_true_coefficient_relative_error':.001,'mean_prediction_error':.002,'evidence_sources':['source']}
        reply={'decision':'completed','case_id':'a','exact_support':True,'evidence_ready':True,'charged_experiments':6,'max_true_coefficient_relative_error':.001,'mean_prediction_error':.002,'source_ids':['source'],'limitations':['合成数据，本地运行，非 Nexent 平台验收']}
        self.assertTrue(compare_final(json.dumps(reply),report)['passed'])
        reply['mean_prediction_error']=0;self.assertFalse(compare_final(json.dumps(reply),report)['passed'])
        reply['mean_prediction_error']=.002;reply['source_ids']=[];self.assertFalse(compare_final(json.dumps(reply),report)['passed'])

if __name__=='__main__':unittest.main()
