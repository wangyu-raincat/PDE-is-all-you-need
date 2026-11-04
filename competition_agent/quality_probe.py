"""Curated contract regression fixtures; NOT numerical/LLM efficacy evaluation.

Run this exact file against a selected checkout so both versions see the same
fixtures. A manufactured empty learner isolates evidence and workflow checks.
"""
import argparse
import inspect
import json
from pathlib import Path
import sqlite3
import sys
import tempfile
from unittest.mock import patch


def probe(Lab):
    results=[]
    def run(name, check):
        with tempfile.TemporaryDirectory() as d:
            lab=Lab(d);case=lab.create_demo(budget=5)['case_id']
            try:
                passed=bool(check(lab,case))
                results.append({'check':name,'passed':passed,'error':None})
            except Exception as exc:
                results.append({'check':name,'passed':False,'error':f'{type(exc).__name__}: {exc}'})
    def ingest(lab,case,text,title='fixture'):
        return lab.ingest(case,[{'title':title,'kind':'text','text':text}])['sources'][0]['source_id']
    def apply(lab,case,source,triples,quotes=None):
        facts=[dict(subject=s,predicate=p,object=o,source_id=source,
                    quote=quotes[i] if quotes else f'{s} {p} {o}.') for i,(s,p,o) in enumerate(triples)]
        return lab.graph_apply(case,lab.graph_propose(case,facts)['proposal_id'])
    def complete(lab,case,boundary='periodic'):
        a=[('demo_grid','records_field','demo_field'),('demo_grid','has_boundary',boundary)]
        b=[('demo_field','uses_unit','dimensionless')]
        for name,items in [('grid',a),('field',b)]:
            source=ingest(lab,case,'\n'.join(f'{s} {p} {o}.' for s,p,o in items),name)
            apply(lab,case,source,items)
    def finish(lab,case):
        # Deliberately manufactured no-model output: only evidence/phase gates
        # are being tested, no coefficient or prediction results are asserted.
        with patch('competition_agent.service.engine.discover',return_value={'selected':None,'committee':[]}):
            lab.identify(case)
        lab.freeze(case)
        return lab.validate(case)
    def search(lab,case,text,query,target):
        source=ingest(lab,case,text)
        hits=lab.search(case,query)['hits']
        if not hits or hits[0]['source_id']!=source or target not in hits[0]['quote']:
            return False
        h=hits[0]
        return 'start_char' not in h or text[h['start_char']:h['end_char']]==h['quote']
    run('english_retrieval_control',lambda l,c:search(l,c,'field uses_unit dimensionless.','dimensionless','dimensionless'))
    run('chinese_question_retrieval',lambda l,c:search(l,c,'观测单位为无量纲。','这组观测的单位是什么？','单位'))
    run('late_english_match_is_in_quote',lambda l,c:search(l,c,'x'*4000+' field uses_unit dimensionless.','dimensionless','dimensionless'))
    run('late_chinese_match_is_in_quote',lambda l,c:search(l,c,'甲'*2400+'温度字段单位为无量纲。','单位','单位'))
    def replay(l,c):
        docs=[{'title':f'doc{i}','kind':'text','text':f'unique {i}'} for i in range(50)]
        for i in range(0,50,10):l.ingest(c,docs[i:i+10])
        l.ingest(c,[docs[0]])
        with l.case(c) as s:return len(s['documents'])==50
    run('ingest_replay_at_document_limit',replay)
    def unrelated(l,c):
        src=ingest(l,c,'sensor measures field.');apply(l,c,src,[('sensor','measures','field')])
        return not finish(l,c)['evidence_ready']
    run('unrelated_fact_does_not_establish_readiness',unrelated)
    def valid(l,c):
        complete(l,c);return finish(l,c)['evidence_ready']
    run('complete_metadata_control',valid)
    def negation(l,c):
        complete(l,c);src=ingest(l,c,'sensor does not measure field.','negation')
        apply(l,c,src,[('sensor','measures','field')],['sensor does not measure field.'])
        return not finish(l,c)['evidence_ready']
    run('negated_relation_is_not_verified',negation)
    def wrong(l,c):
        complete(l,c,'nonperiodic');return not finish(l,c)['evidence_ready']
    run('metadata_must_match_simulator',wrong)
    def cross(l,c):
        complete(l,c)
        if not hasattr(l,'evidence_path'):return False
        p=l.evidence_path(c,'demo_grid','dimensionless')['paths'][0]
        return len(p['edges'])==2 and len(p['source_ids'])==2 and p['all_relations_asserted']
    run('cross_document_path_has_both_sources',cross)
    def report(l,c):
        finish(l,c)
        with l.case(c) as s:s['report']['equation']='tampered'
        try:l.evidence(c)
        except ValueError:return True
        return False
    run('changed_final_report_is_rejected',report)
    def seal(l,c):
        with patch('competition_agent.service.engine.discover',return_value={'selected':None,'committee':[]}):l.identify(c)
        l.freeze(c)
        with l.case(c) as s:s['freeze']['sha256']='0'*64
        try:l.validate(c)
        except ValueError:return not l.status(c)['final_opened']
        return False
    run('changed_freeze_seal_rejected_before_final',seal)
    def request(l,c):
        if 'request_id' not in inspect.signature(l.create_demo).parameters:return False
        a=l.create_demo(request_id='retry-test');b=l.create_demo(request_id='retry-test')
        return a['case_id']==b['case_id'] and b['charged_experiments']==5
    run('repeated_create_does_not_charge_twice',request)
    def read(l,c):
        root=l.root
        l.connect=lambda:sqlite3.connect(root/'cases.sqlite',timeout=.05)
        blocker=l.connect();blocker.execute('BEGIN IMMEDIATE')
        try:return l.status(c)['case_id']==c
        finally:blocker.rollback();blocker.close()
    run('status_read_during_reserved_write',read)
    return {'scope':'14 curated regression fixtures, includes 2 positive controls; manufactured empty learner where scoring is needed',
            'is_generalization_benchmark':False,'checks':results,'passed':sum(x['passed'] for x in results),'total':len(results)}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--repository',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();sys.path.insert(0,str(Path(a.repository).resolve()))
    from competition_agent.service import Lab
    result=probe(Lab)
    Path(a.output).parent.mkdir(parents=True,exist_ok=True)
    Path(a.output).write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
