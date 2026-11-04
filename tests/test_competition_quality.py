"""Behavior-focused fixtures for retrieval, evidence, handoff and export."""
import asyncio,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from competition_agent.service import Lab
from competition_agent import evidence_quality as quality
from competition_agent.demo import run as demo
from competition_agent.quality_probe import probe
from tests.test_active_controller import fabricated_observation

def response(*a,**kw):
    return {'observation':fabricated_observation(),'receipt':{'manufactured':True}}

class CompetitionQualityTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.lab=Lab(self.temp.name)
        self.p=patch('competition_agent.service.provider.generate_observation',side_effect=response)
        self.p.start();self.addCleanup(self.p.stop)
        self.case=self.lab.create_demo(budget=5)['case_id']
    def metadata(self):
        for title,triples in [('grid',[quality.REQUIREMENTS[0],quality.REQUIREMENTS[2]]),('field',[quality.REQUIREMENTS[1]])]:
            text='\n'.join(f'{s} {p} {o}.' for s,p,o in triples)
            source=self.lab.ingest(self.case,[{'title':title,'kind':'text','text':text}])['sources'][0]['source_id']
            facts=[dict(subject=s,predicate=p,object=o,source_id=source,quote=f'{s} {p} {o}.') for s,p,o in triples]
            p=self.lab.graph_propose(self.case,facts);self.lab.graph_apply(self.case,p['proposal_id'])
    def fit(self):
        with patch('competition_agent.service.engine.discover',return_value={'selected':None,'committee':[]}):self.lab.identify(self.case)
    def test_all_curated_contract_regressions(self):
        result=probe(Lab)
        self.assertEqual(result['total'],14);self.assertEqual(result['passed'],14,result)
    def test_creation_key_cannot_change_config(self):
        self.lab.create_demo(request_id='operation')
        with self.assertRaisesRegex(ValueError,'different configuration'):self.lab.create_demo(request_id='operation',budget=8)
    def test_failed_initial_creation_replay_does_not_resubmit(self):
        with patch('competition_agent.service.provider.generate_observation',side_effect=OSError('outage')) as p:
            a=self.lab.create_demo(request_id='failed');b=self.lab.create_demo(request_id='failed')
        self.assertEqual(p.call_count,1);self.assertEqual(a['case_id'],b['case_id']);self.assertEqual(b['phase'],'initial_failed')
    def test_blank_search_and_unicode_offsets(self):
        docs={'id':{'title':'fixture','text':'prefix\nＦＩＥＬＤ uses_unit dimensionless.'}}
        h=quality.retrieve(docs,'field',5)['hits'][0]
        self.assertEqual(docs['id']['text'][h['start_char']:h['end_char']],h['quote'])
        self.assertEqual(h['line'],2);self.assertEqual(quality.retrieve(docs,'？？',5)['hits'],[])
    def test_negated_container_cannot_support_extracted_affirmation(self):
        f={'subject':'field','predicate':'uses_unit','object':'kelvin','source_id':'id','quote':'field uses_unit kelvin.'}
        self.assertFalse(quality.assertion(f,{'id':{'text':'It is false that field uses_unit kelvin.'}}))
    def test_citation_prefers_whole_line_assertion_over_negated_substring(self):
        doc={'text':'It is false that field uses_unit kelvin.\nfield uses_unit kelvin.'}
        quote='field uses_unit kelvin.'
        citation=quality.span(doc,quote)
        self.assertEqual(citation['line'],2)
        self.assertEqual(doc['text'][citation['start_char']:citation['end_char']],quote)
    def test_paths_are_bounded_and_do_not_invent_missing_edges(self):
        self.metadata();p=self.lab.evidence_path(self.case,'demo_grid','dimensionless')
        self.assertEqual(len(p['paths'][0]['source_ids']),2)
        self.assertFalse(self.lab.evidence_path(self.case,'demo_grid','dimensionless',max_hops=1)['found'])
        self.assertFalse(self.lab.evidence_path(self.case,'dimensionless','demo_grid')['found'])
        with self.assertRaises(ValueError):self.lab.evidence_path(self.case,'demo_grid','dimensionless',max_hops=20)
    def test_handoff_tracks_required_inputs_and_budget(self):
        h=self.lab.handoff(self.case);self.assertEqual(h['next_role'],'asset_agent');self.assertFalse(h['arguments_complete'])
        self.metadata();h=self.lab.handoff(self.case);self.assertEqual(h['next_tool'],'pde_identify')
        self.fit();h=self.lab.handoff(self.case);self.assertEqual(h['next_tool'],'validation_freeze')
        self.lab.freeze(self.case);self.assertEqual(self.lab.handoff(self.case)['next_tool'],'validation_evaluate')
        self.lab.validate(self.case);h=self.lab.handoff(self.case)
        self.assertTrue(h['stop']);self.assertTrue(h['terminal']);self.assertIsNone(h['next_tool'])
    def test_conflict_handoff_stops_automatic_progress(self):
        self.metadata()
        source=self.lab.ingest(self.case,[{'title':'conflict','kind':'text','text':'demo_field uses_unit kelvin.'}])['sources'][0]['source_id']
        f=dict(subject='demo_field',predicate='uses_unit',object='kelvin',source_id=source,quote='demo_field uses_unit kelvin.')
        p=self.lab.graph_propose(self.case,[f]);self.lab.graph_apply(self.case,p['proposal_id'])
        h=self.lab.handoff(self.case);self.assertTrue(h['stop']);self.assertEqual(h['next_role'],'local_operator');self.assertFalse(h['terminal'])
    def test_frozen_model_change_rejected_by_evidence_read(self):
        self.fit();self.lab.freeze(self.case)
        with self.lab.case(self.case) as s:s['model']['changed']=True
        with self.assertRaisesRegex(ValueError,'Frozen model'):self.lab.evidence(self.case)
    def test_cached_final_report_tampering_is_not_returned(self):
        self.fit();self.lab.freeze(self.case);self.lab.validate(self.case)
        with self.lab.case(self.case) as s:s['report']['numerical_joint_success']=True
        with self.assertRaisesRegex(ValueError,'Final report'):self.lab.validate(self.case)
    def test_demo_preserves_existing_output_before_starting_server(self):
        path=Path(self.temp.name)/'existing';path.mkdir();(path/'report.json').write_text('preserve')
        with self.assertRaises(FileExistsError):asyncio.run(demo(path))
        self.assertEqual((path/'report.json').read_text(),'preserve')
    def test_html_exports_escape_untrusted_evidence(self):
        from competition_agent.reporting import render
        self.fit();self.lab.freeze(self.case);r=self.lab.validate(self.case);e=self.lab.evidence(self.case)
        e['facts']=[dict(id='id',subject='<script>alert(1)</script>',predicate='measures',object='x',source_id='source',quote='<img src=x onerror=alert(1)>')]
        output=render({'report':r},e,[])
        self.assertNotIn('<script>',output);self.assertNotIn('<img src=x',output)
        self.assertIn('&lt;script&gt;',output);self.assertIn('未选出可接受的方程',output)

if __name__=='__main__':unittest.main()
