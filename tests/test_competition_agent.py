"""Integration invariants with manufactured fixtures, not scientific outcomes."""
import json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from copy import deepcopy
import numpy as np
from competition_agent.service import Lab,source_hashes
from competition_agent.bundle import build,ROLES
from competition_agent.verification.nexent_schema_reference import ExportAndImportDataFormat
from tests.test_active_controller import fabricated_observation,fabricated_model


def response(*args,**kwargs):
    obs=fabricated_observation()
    return {'observation':obs,'clean':{'u':np.concatenate((obs['prefix_u'][:-1],obs['u']))},'receipt':{'manufactured':True}}
def model():
    m=fabricated_model();m['selected']['equation']='u_t = -u';return m

class CompetitionAgentTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.lab=Lab(self.temp.name)
        self.generator=patch('competition_agent.service.provider.generate_observation',side_effect=response).start();self.addCleanup(patch.stopall)
        self.learner=patch('competition_agent.service.engine.discover',side_effect=lambda *a,**k:model()).start()
        self.case=self.lab.create_demo(budget=6)['case_id']
    def identified(self):return self.lab.identify(self.case)
    def ingest(self,text='field uses_unit dimensionless.'):
        return self.lab.ingest(self.case,[{'title':'fixture','kind':'text','text':text}])['sources'][0]['source_id']
    def fact(self,source,predicate='uses_unit',obj='dimensionless',quote=None):
        return {'subject':'field','predicate':predicate,'object':obj,'source_id':source,'quote':quote or f'field {predicate} {obj}.'}
    def proposal(self):
        self.identified();return self.lab.propose(self.case)
    def test_observations_only_enter_discovery(self):
        self.identified();args=self.learner.call_args.args
        for obs in args[0]+args[1]:self.assertEqual(set(obs),{'x','t','u','prefix_t','prefix_u','boundary'})
        public=json.dumps(self.lab.status(self.case));self.assertNotIn('_spec',public);self.assertNotIn('reference_terms',public)
        self.assertFalse(self.lab.status(self.case)['final_opened'])
    def test_source_closure_covers_integration_and_numerical_engine(self):
        hashes=source_hashes()
        for f in ['competition_agent/service.py','competition_agent/mcp_server.py','active_study/engine.py','pde_discovery/generic_experiment.py','coefficient_study/generators.py']:self.assertIn(f,hashes)
    def test_failed_acquisition_charged_once_and_retry_idempotent(self):
        p=self.proposal();self.generator.side_effect=OSError('manufactured outage')
        a=self.lab.acquire(self.case,p['id']);b=self.lab.acquire(self.case,p['id'])
        self.assertEqual(a['query']['status'],'failed');self.assertTrue(b['idempotent_replay'])
        s=self.lab.status(self.case);self.assertEqual(s['charged_experiments'],6);self.assertEqual(s['training_trajectories'],3)
        self.identified()
        with self.assertRaisesRegex(ValueError,'exhausted'):self.lab.propose(self.case)
    def test_success_acquisition_replay_does_not_generate_or_charge(self):
        p=self.proposal();self.lab.acquire(self.case,p['id']);calls=self.generator.call_count
        self.lab.acquire(self.case,p['id']);self.assertEqual(self.generator.call_count,calls)
        self.assertEqual(self.lab.status(self.case)['training_trajectories'],4)
    def test_interruption_keeps_charge_without_resubmission(self):
        p=self.proposal();self.generator.side_effect=KeyboardInterrupt()
        with self.assertRaises(KeyboardInterrupt):self.lab.acquire(self.case,p['id'])
        s=self.lab.status(self.case);self.assertTrue(s['recovery_required']);self.assertEqual(s['charged_experiments'],6)
        calls=self.generator.call_count;self.assertEqual(self.lab.acquire(self.case,p['id'])['query']['status'],'started')
        self.assertEqual(self.generator.call_count,calls)
        with self.assertRaises(ValueError):self.lab.identify(self.case)
    def test_stale_proposal_does_not_charge(self):
        self.proposal()
        with self.assertRaises(ValueError):self.lab.acquire(self.case,'stale')
        self.assertEqual(self.lab.status(self.case)['charged_experiments'],5)
    def test_identification_idempotent_until_new_observation(self):
        self.identified();self.identified();self.assertEqual(self.learner.call_count,1)
    def test_validation_requires_freeze_and_no_future_opening(self):
        calls=self.generator.call_count
        with self.assertRaises(ValueError):self.lab.validate(self.case)
        self.assertEqual(calls,self.generator.call_count);self.assertFalse(self.lab.status(self.case)['final_opened'])
    def test_frozen_model_prevents_learning_and_document_mutation(self):
        self.identified();self.lab.freeze(self.case)
        with self.assertRaises(ValueError):self.lab.identify(self.case)
        with self.assertRaises(ValueError):self.ingest()
        with self.assertRaises(ValueError):self.lab.graph_propose(self.case,[])
        self.assertFalse(self.lab.status(self.case)['final_opened'])
    def test_empty_model_retains_all_failure_slots_and_caches_evaluation(self):
        self.learner.side_effect=None;self.learner.return_value={'selected':None,'committee':[],'candidate_supports':[],'failure_reason':'manufactured rejection'}
        self.identified();self.lab.freeze(self.case);calls=self.generator.call_count
        r=self.lab.validate(self.case)
        self.assertEqual(r['prediction_failure_slots'],3);self.assertIsNone(r['mean_prediction_error']);self.assertFalse(r['numerical_joint_success'])
        self.assertEqual(calls,self.generator.call_count);self.assertEqual(r,self.lab.validate(self.case))
        with self.assertRaises(ValueError):self.lab.identify(self.case)
    def test_final_forecast_failure_retained_with_separate_eval_cost(self):
        self.identified();self.lab.freeze(self.case)
        with patch('competition_agent.service.forecast.forecast_with_diagnostics',side_effect=ValueError('manufactured numerical failure')):r=self.lab.validate(self.case)
        self.assertEqual(r['evaluation_reference_trajectories_generated'],3);self.assertEqual(r['prediction_failure_slots'],3)
        self.assertEqual(r['charged_experiments'],5);self.assertEqual(r['evaluation_measurement_scalars'],21312)
    def test_freeze_tampering_rejected_before_final(self):
        self.identified();self.lab.freeze(self.case)
        with self.lab.case(self.case) as s:s['model']['selected']['terms'][0]['coefficient']=2
        with self.assertRaisesRegex(ValueError,'model changed'):self.lab.validate(self.case)
        self.assertFalse(self.lab.status(self.case)['final_opened'])
    def test_document_tampering_rejected_before_final(self):
        self.ingest();self.identified();self.lab.freeze(self.case)
        with self.lab.case(self.case) as s:next(iter(s['documents'].values()))['text']='tampered'
        with self.assertRaisesRegex(ValueError,'setup or documents'):self.lab.validate(self.case)
    def test_observation_tampering_rejected_before_discovery(self):
        with self.lab.case(self.case) as s:p=Path(self.temp.name)/s['train'][0]['file']
        p.write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError,'Observation changed'):self.identified()
    def test_source_change_rejected_before_final(self):
        self.identified();self.lab.freeze(self.case)
        with patch('competition_agent.service.source_hashes',return_value={'changed':'code'}):
            with self.assertRaisesRegex(ValueError,'source changed'):self.lab.validate(self.case)
    def test_source_grounding_rejects_fabricated_quote_and_unknown_id(self):
        source=self.ingest()
        for f in [self.fact(source,quote='field uses_unit kelvin.'),self.fact('unknown')]:
            with self.assertRaises(ValueError):self.lab.graph_propose(self.case,[f])
        self.assertEqual(self.lab.status(self.case)['graph_facts'],0)
    def test_unknown_relation_requires_operator_review(self):
        src=self.ingest('field is_measured_by sensor.')
        p=self.lab.graph_propose(self.case,[self.fact(src,'is_measured_by','sensor')])
        with self.assertRaisesRegex(ValueError,'operator review'):self.lab.graph_apply(self.case,p['proposal_id'])
        self.lab.review_ontology(self.case,p['pending_ontology_changes'][0]['id'],True)
        r=self.lab.graph_apply(self.case,p['proposal_id']);self.assertEqual(r['ontology_version'],2)
    def test_rejected_ontology_cannot_be_applied(self):
        src=self.ingest('field is_measured_by sensor.');p=self.lab.graph_propose(self.case,[self.fact(src,'is_measured_by','sensor')])
        self.lab.review_ontology(self.case,p['pending_ontology_changes'][0]['id'],False)
        with self.assertRaises(ValueError):self.lab.graph_apply(self.case,p['proposal_id'])
    def test_rejected_identical_proposal_is_not_reopened_by_replay(self):
        src=self.ingest('field is_measured_by sensor.');facts=[self.fact(src,'is_measured_by','sensor')]
        p=self.lab.graph_propose(self.case,facts);self.lab.review_ontology(self.case,p['pending_ontology_changes'][0]['id'],False)
        self.lab.graph_propose(self.case,facts)
        self.assertEqual(self.lab.status(self.case)['pending_ontology_changes'],0)
        with self.assertRaises(ValueError):self.lab.graph_apply(self.case,p['proposal_id'])
    def test_duplicate_facts_within_proposal_are_deduplicated(self):
        src=self.ingest();fact=self.fact(src);p=self.lab.graph_propose(self.case,[fact,deepcopy(fact)])
        self.lab.graph_apply(self.case,p['proposal_id']);self.assertEqual(self.lab.status(self.case)['graph_facts'],1)
    def test_conflicting_facts_retained_and_readiness_false(self):
        src=self.ingest('field uses_unit dimensionless.\nfield uses_unit kelvin.')
        p=self.lab.graph_propose(self.case,[self.fact(src),self.fact(src,obj='kelvin')]);r=self.lab.graph_apply(self.case,p['proposal_id'])
        self.assertEqual(len(r['facts']),2);self.assertEqual(len(r['conflicts']),1)
        self.learner.side_effect=None;self.learner.return_value={'selected':None};self.identified();self.lab.freeze(self.case)
        self.assertFalse(self.lab.validate(self.case)['evidence_ready'])
    def test_fact_application_is_idempotent(self):
        src=self.ingest();p=self.lab.graph_propose(self.case,[self.fact(src)])
        self.lab.graph_apply(self.case,p['proposal_id']);self.lab.graph_apply(self.case,p['proposal_id'])
        self.assertEqual(self.lab.status(self.case)['graph_facts'],1)
    def test_lexical_retrieval_contains_exact_source_and_line(self):
        src=self.ingest();r=self.lab.search(self.case,'dimensionless')['hits'][0]
        self.assertEqual(r['source_id'],src);self.assertEqual(r['line'],1);self.assertEqual(r['quote'],'field uses_unit dimensionless.')
    def test_event_tampering_detected(self):
        with self.lab.case(self.case) as s:s['events'][0]['details']['experiments']=0
        with self.assertRaisesRegex(ValueError,'chain changed'):self.lab.evidence(self.case)
    def test_invalid_inputs_rejected(self):
        for kw in [{'budget':4},{'budget':True},{'seed':801},{'noise':float('nan')},{'fixture':'ks'},{'policy':'rich_information'}]:
            with self.subTest(kw=kw),self.assertRaises(ValueError):self.lab.create_demo(**kw)
        for c in ['../cases.sqlite','', 'x'*32]:
            with self.assertRaises(ValueError):self.lab.status(c)
        with self.assertRaises(ValueError):self.lab.ingest(self.case,[{'title':'x','kind':'binary','text':'x'}])
    def test_initial_failure_is_recorded_without_success_claim(self):
        self.generator.side_effect=OSError('initial outage');s=self.lab.create_demo()
        self.assertEqual(s['phase'],'initial_failed');self.assertEqual(s['charged_experiments'],5);self.assertEqual(s['failure'],'initial outage')
    def test_nexent_schema_skills_roles_and_model_pending(self):
        with tempfile.TemporaryDirectory() as d:
            payload,meta=build(d);ExportAndImportDataFormat.model_validate(payload)
            self.assertEqual(len(payload['agent_info']),6);self.assertFalse(meta['nexent_live_verified'])
            self.assertEqual(payload['agent_info']['1']['managed_agents'],[11,12,13,14,15])
            for a in payload['agent_info'].values():self.assertEqual(a['model_names'],[])
            self.assertEqual(len(list(Path(d).glob('*.zip'))),6)
            tools=set(t['class_name'] for a in payload['agent_info'].values() for t in a['tools'])
            self.assertEqual(len(tools),12);self.assertNotIn('review_ontology',tools)
            self.assertEqual({t['class_name'] for t in payload['agent_info']['15']['tools']},{'case_status','validation_freeze','validation_evaluate','evidence_get'})
    def test_nexent_schema_rejects_missing_actual_required_field(self):
        with tempfile.TemporaryDirectory() as d:
            payload,_=build(d);del payload['agent_info']['11']['provide_run_summary']
            with self.assertRaises(ValueError):ExportAndImportDataFormat.model_validate(payload)

if __name__=='__main__':unittest.main()
