"""Transactional local tools; no LLM, hardware or hidden future in discovery outputs."""
from __future__ import annotations
import hashlib,json,re,sqlite3,uuid
from pathlib import Path
from contextlib import contextmanager
from datetime import datetime,timezone
import numpy as np
from active_study import actions,provider,engine,acquisition,forecast,controller
from competition_agent import evidence_quality as quality
from pde_discovery.generic_experiment import reference_metrics,forecast_metrics

ROOT=Path(__file__).resolve().parents[1]
PREDICATES=('uses_unit','has_boundary','records_field','measures','derived_from','supported_by')
POLICIES=('fixed_rich','random','ensemble_variance','structural_disagreement')
def canonical(v):return json.dumps(v,sort_keys=True,ensure_ascii=False,allow_nan=False,separators=(',',':')).encode()
def digest(v):return hashlib.sha256(canonical(v)).hexdigest()
def file_sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def require(ok,message):
    if not ok:raise ValueError(message)
def integer(v,lo,hi):return type(v) is int and lo<=v<=hi

def source_hashes():
    packages=('active_study','pde_discovery','coefficient_study','prefix_study','five_study',
              'resolved_study','weak_study','l1_study','fusion_study','challenge_study','competition_agent')
    return {str(p.relative_to(ROOT)):file_sha(p) for name in packages for p in sorted((ROOT/name).glob('*.py'))}

def setup_hash(s):
    return digest({k:s[k] for k in ('seed','policy','budget','charged','_spec','_noise','train','dev','documents')})

class Lab:
    def __init__(self,directory=None):
        self.root=Path(directory or ROOT/'competition_agent/runtime').resolve()
        self.root.mkdir(parents=True,exist_ok=True)
        with self.connect() as db:
            db.execute('CREATE TABLE IF NOT EXISTS cases (id TEXT PRIMARY KEY,payload TEXT NOT NULL)')
            db.execute('CREATE TABLE IF NOT EXISTS requests (request_id TEXT PRIMARY KEY, config_sha256 TEXT NOT NULL, case_id TEXT NOT NULL)')
    def connect(self):return sqlite3.connect(self.root/'cases.sqlite',timeout=120)
    @contextmanager
    def case(self,case_id,readonly=False):
        require(isinstance(case_id,str) and re.fullmatch(r'[0-9a-f]{32}',case_id),'Invalid case id')
        db=self.connect()
        try:
            if not readonly:db.execute('BEGIN IMMEDIATE')
            row=db.execute('SELECT payload FROM cases WHERE id=?',(case_id,)).fetchone()
            require(row is not None,'Unknown case id'); state=json.loads(row[0]);yield state
            if not readonly:
                db.execute('UPDATE cases SET payload=? WHERE id=?',(canonical(state).decode(),case_id));db.commit()
        except BaseException:db.rollback();raise
        finally:db.close()
    def event(self,s,actor,operation,details):
        events=s['events']; item={'sequence':len(events)+1,'actor':actor,'operation':operation,
          'details':details,'previous':events[-1]['sha256'] if events else None,
          'at_utc':datetime.now(timezone.utc).isoformat()}
        item['sha256']=digest(item);events.append(item)
    def save_observation(self,s,obs,name):
        p=self.root/s['id']/f'{name}.npz';p.parent.mkdir(exist_ok=True)
        np.savez_compressed(p,**{k:obs[k] for k in ('x','t','u','prefix_t','prefix_u')})
        return {'file':str(p.relative_to(self.root)),'sha256':file_sha(p)}
    def load_observation(self,item):
        p=self.root/item['file'];require(file_sha(p)==item['sha256'],'Observation changed after acquisition')
        with np.load(p,allow_pickle=False) as z:obs={k:z[k].copy() for k in z.files}
        return controller.observation_view(obs|{'boundary':'periodic'})
    def view(self,s):
        selected=(s.get('model') or {}).get('selected')
        return {'case_id':s['id'],'phase':s['phase'],'origin':'controlled_synthetic_demo',
          'benchmark_claim':'engineering demonstration, not a new blind scientific evaluation',
          'budget_limit':s['budget'],'charged_experiments':s['charged'],
          'measurement_scalars_charged':s['charged']*7104,'training_trajectories':len(s['train']),
          'development_trajectories':len(s['dev']),'policy':s['policy'],
          'selected_equation':None if selected is None else selected['equation'],
          'selected_terms':[] if selected is None else selected['terms'],
          'candidate_supports':(s.get('model') or {}).get('candidate_supports',[]),
          'identification_status':(s.get('model') or {}).get('status'),'identification_failure':(s.get('model') or {}).get('failure_reason'),
          'uncertainty':{k:v for k,v in (s.get('model') or {}).get('uncertainty',{}).items() if k!='records'},
          'minimal_near_supports':(s.get('model') or {}).get('diagnostics',{}).get('minimal_near_supports',[]),
          'recovery_required':s['phase'] in ('creating','acquiring','validating'),'failure':s.get('failure'),'frequency_calibrated':False,'final_opened':s.get('final_opened',False),
          'ontology_version':s['ontology']['version'],'graph_facts':len(s['facts']),
          'evidence_quality':quality.assess(s,self.conflicts(s)),'graph_conflicts':self.conflicts(s),'pending_ontology_changes':sum(p['status']=='pending' for p in s['ontology_proposals']),
          'event_count':len(s['events']),'event_head':s['events'][-1]['sha256'] if s['events'] else None}
    def status(self,case_id):
        with self.case(case_id,readonly=True) as s:return self.view(s)
    def create_demo(self,fixture='diffusion',noise=.01,budget=7,seed=240,policy='fixed_rich',request_id=None):
        require(fixture in provider.FAMILIES,'Unsupported fixture')
        require(type(noise) in (int,float) and np.isfinite(noise) and 0<=noise<=.03,'Noise must be in [0,.03]')
        require(integer(budget,5,9) and integer(seed,0,800),'Budget must be5..9; seed0..800')
        require(policy in POLICIES,'Unsupported policy; rejected information candidate is not a default')
        require(request_id is None or isinstance(request_id,str) and re.fullmatch(r'[A-Za-z0-9_-]{1,80}',request_id),'Invalid request_id')
        config_sha=digest({'fixture':fixture,'noise':float(noise),'budget':budget,'seed':seed,'policy':policy})
        identity=uuid.uuid4().hex;index=seed*6+provider.FAMILIES.index(fixture)
        s={'id':identity,'phase':'creating','budget':budget,'charged':5,'policy':policy,'seed':seed,
          '_spec':provider.get_task_spec('development',index),'_noise':noise,'train':[],'dev':[],
          'queries':[],'model':None,'documents':{},'facts':[],'graph_proposals':[],
          'ontology':{'version':1,'predicates':list(PREDICATES)},'ontology_proposals':[],'events':[]}
        self.event(s,'asset_agent','initial_budget_charged',{'experiments':5,'is_physical_data':False})
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            previous=db.execute('SELECT config_sha256,case_id FROM requests WHERE request_id=?',(request_id,)).fetchone() if request_id else None
            if previous:
                require(previous[0]==config_sha,'request_id already belongs to a different configuration')
                old=json.loads(db.execute('SELECT payload FROM cases WHERE id=?',(previous[1],)).fetchone()[0])
                return dict(self.view(old),idempotent_replay=True)
            db.execute('INSERT INTO cases VALUES (?,?)',(identity,canonical(s).decode()))
            if request_id:db.execute('INSERT INTO requests VALUES (?,?,?)',(request_id,config_sha,identity))
        with self.case(identity) as s:
            try:
                initial=actions.initial_actions(2100000+seed)
                for role,dest in [('training','train'),('development','dev')]:
                    for j,action in enumerate(initial[role]):
                        response=provider.generate_observation(s['_spec'],action,2200000+seed*10+j+(5 if dest=='dev' else 0),noise)
                        s[dest].append(self.save_observation(s,response['observation'],f'{dest}-{j}'))
                s['phase']='ready';self.event(s,'asset_agent','initial_observations_registered',{'observations':5})
            except Exception as exc:
                s['phase']='initial_failed';s['failure']=str(exc);self.event(s,'asset_agent','initial_failed',{'reason':str(exc)})
            return self.view(s)
    def ingest(self,case_id,documents):
        require(isinstance(documents,list) and 1<=len(documents)<=10,'Need1..10 documents')
        prepared=[]
        for d in documents:
            require(isinstance(d,dict) and set(d)=={'title','kind','text'},'Document requires title,kind,text')
            require(d['kind'] in ('text','csv') and isinstance(d['text'],str) and 1<=len(d['text'].encode())<=131072,'Only bounded text/CSV input is supported')
            require(isinstance(d['title'],str) and 1<=len(d['title'])<=160,'Invalid title')
            prepared.append((digest(d),d))
        with self.case(case_id) as s:
            require(s['phase'] not in ('frozen','validating','validated'),'Evidence is frozen')
            require(len(set(s['documents'])|{key for key,d in prepared})<=50,'Document count limit reached')
            for key,d in prepared:s['documents'][key]=d
            self.event(s,'asset_agent','documents_ingested',{'source_ids':[k for k,d in prepared]})
            return {'sources':[{'source_id':k,'title':d['title'],'kind':d['kind']} for k,d in prepared]}
    def search(self,case_id,query,limit=5):
        require(isinstance(query,str) and 1<=len(query)<=300 and integer(limit,1,10),'Invalid search')
        with self.case(case_id,readonly=True) as s:return quality.retrieve(s['documents'],query,limit)
    def graph_propose(self,case_id,facts):
        require(isinstance(facts,list) and 1<=len(facts)<=40,'Need1..40 source-backed facts')
        with self.case(case_id) as s:
            require(s['phase'] not in ('frozen','validating','validated'),'Evidence is frozen')
            checked=[]
            for f in facts:
                require(isinstance(f,dict) and set(f)=={'subject','predicate','object','source_id','quote'},'Invalid fact fields')
                require(all(isinstance(v,str) and 1<=len(v)<=2000 for v in f.values()),'Invalid fact value')
                require(f['source_id'] in s['documents'] and f['quote'] in s['documents'][f['source_id']]['text'],'Quote must match a registered source exactly')
                require(f['subject'] in f['quote'] and f['object'] in f['quote'],'Subject and object must occur in cited evidence')
                require(re.fullmatch(r'[a-z][a-z0-9_]{0,63}',f['predicate']),'Invalid relation name')
                checked.append(dict(f,id=digest(f),status='source_span_verified',citation=quality.span(s['documents'][f['source_id']],f['quote']),relation_assertion_verified=quality.assertion(f,s['documents'])))
            checked=list({f['id']:f for f in checked}.values())
            proposal={'id':digest(checked),'facts':checked,'status':'proposed','ontology_version':s['ontology']['version']}
            if not any(p['id']==proposal['id'] for p in s['graph_proposals']):s['graph_proposals'].append(proposal)
            unknown=sorted({f['predicate'] for f in checked}-set(s['ontology']['predicates']))
            for name in unknown:
                change_id=digest({'predicate':name,'proposal':proposal['id']})
                if not any(p['id']==change_id or (p['predicate']==name and p['status']=='pending') for p in s['ontology_proposals']):
                    s['ontology_proposals'].append({'id':change_id,'predicate':name,'source_proposal':proposal['id'],'status':'pending'})
            self.event(s,'graph_agent','graph_proposed',{'proposal_id':proposal['id'],'unknown_predicates':unknown})
            return {'proposal_id':proposal['id'],'unknown_predicates':unknown,'pending_ontology_changes':s['ontology_proposals'],'semantic_truth_verified':False}
    def graph_apply(self,case_id,proposal_id):
        with self.case(case_id) as s:
            require(s['phase'] not in ('frozen','validating','validated'),'Evidence is frozen')
            p=next((p for p in s['graph_proposals'] if p['id']==proposal_id),None);require(p is not None,'Unknown graph proposal')
            require(all(f['predicate'] in s['ontology']['predicates'] for f in p['facts']),'New ontology relations require local operator review')
            existing={f['id'] for f in s['facts']}
            require(len(existing|{f['id'] for f in p['facts']})<=500,'Graph fact limit reached')
            s['facts'].extend(f for f in p['facts'] if f['id'] not in existing);p['status']='applied'
            self.event(s,'graph_agent','source_grounded_graph_applied',{'proposal_id':proposal_id,'human_semantic_review':False})
            return {'facts':s['facts'],'conflicts':self.conflicts(s),'ontology_version':s['ontology']['version']}
    def review_ontology(self,case_id,proposal_id,accept):
        # Operator CLI only. Deliberately absent from the MCP tool list.
        require(type(accept) is bool,'Review must explicitly accept or reject')
        with self.case(case_id) as s:
            require(s['phase'] not in ('frozen','validating','validated'),'Evidence is frozen')
            p=next((p for p in s['ontology_proposals'] if p['id']==proposal_id),None)
            require(p is not None and p['status']=='pending','No pending proposal')
            p['status']='accepted' if accept else 'rejected'
            if accept:s['ontology']['predicates'].append(p['predicate']);s['ontology']['version']+=1
            self.event(s,'local_operator','ontology_review',{'proposal_id':proposal_id,'accepted':accept})
            return s['ontology']
    @staticmethod
    def conflicts(s):
        groups={}
        for f in s['facts']:
            if f['predicate'] in ('uses_unit','has_boundary'):groups.setdefault((f['subject'],f['predicate']),[]).append(f)
        return [{'subject':k[0],'predicate':k[1],'facts':v,'status':'requires_review_not_automatically_resolved'} for k,v in groups.items() if len({f['object'] for f in v})>1]
    def identify(self,case_id):
        with self.case(case_id) as s:
            require(s['phase'] in ('ready','observed','identified'),'Discovery is unavailable after freeze or while an operation is pending')
            if s['phase']=='identified':return self.view(s)
            train=[self.load_observation(r) for r in s['train']];dev=[self.load_observation(r) for r in s['dev']]
            s['model']=engine.discover(train,dev,seed=2300000+s['seed']+len(s['queries']))
            s['phase']='identified';self.event(s,'pde_agent','identify',{'model_sha256':digest(s['model']),'training_hashes':[r['sha256'] for r in s['train']],'final_used':False})
            return self.view(s)
    def propose(self,case_id):
        with self.case(case_id) as s:
            require(s['phase'] in ('identified','proposed'),'Identify before proposing an experiment')
            require(s['charged']<s['budget'],'Experiment budget exhausted')
            if s['phase']=='proposed':return s['proposal']
            train=[self.load_observation(r) for r in s['train']]
            proposal=acquisition.select_action(s['policy'],s['model'].get('committee',[]),actions.make_action_pool(2400000+s['seed']),train,seed=2500000+s['seed']+len(s['queries']),used_action_ids=tuple(q['action']['id'] for q in s['queries']),cost_per_action=7104)
            s['proposal']={'id':digest({'case':s['id'],'round':len(s['queries']),'action':proposal['selected_action']}),'action':proposal['selected_action'],'reason':proposal.get('reason'),'policy':s['policy'],'diagnostics':proposal,'measurement_cost':7104,'is_physical_instruction':False}
            s['phase']='proposed';self.event(s,'experiment_agent','propose',{'proposal_id':s['proposal']['id']})
            return s['proposal']
    def acquire(self,case_id,proposal_id):
        with self.case(case_id) as s:
            previous=next((q for q in s['queries'] if q['proposal_id']==proposal_id),None)
            if previous is not None:return {'query':previous,'idempotent_replay':True}
            require(s['phase']=='proposed' and s['proposal']['id']==proposal_id,'Missing or stale proposal')
            require(s['charged']<s['budget'],'Experiment budget exhausted')
            q={'proposal_id':proposal_id,'action':s['proposal']['action'],'status':'started','charged_experiments':1};s['queries'].append(q)
            s['charged']+=1;s['phase']='acquiring';s['model']=None
            self.event(s,'experiment_agent','query_charged_before_execution',{'proposal_id':proposal_id,'charged_experiments':s['charged']})
        # Charge is durable before simulator execution; failures stay charged.
        with self.case(case_id) as s:
            q=s['queries'][-1]
            try:
                response=provider.generate_observation(s['_spec'],q['action'],2600000+s['seed']*10+len(s['queries']),s['_noise'])
                record=self.save_observation(s,response['observation'],f"query-{len(s['queries'])}")
                s['train'].append(record);q.update(status='observed',observation=record,receipt=response['receipt'])
            except Exception as exc:q.update(status='failed',reason=str(exc))
            s['phase']='observed';self.event(s,'experiment_agent','query_completed',{'proposal_id':proposal_id,'status':q['status']})
            return {'query':q,'idempotent_replay':False}
    def freeze(self,case_id):
        with self.case(case_id) as s:
            require(s['phase'] in ('identified','frozen'),'Identify current observations before freezing')
            if s['phase']=='frozen':return s['freeze']
            for r in s['train']+s['dev']:self.load_observation(r)
            frozen={'case_id':s['id'],'model_sha256':digest(s['model']),'training':s['train'],'development':s['dev'],'charged':s['charged'],'facts_sha256':digest(s['facts']),'ontology_sha256':digest(s['ontology']),'setup_sha256':setup_hash(s),'sources':source_hashes(),'final_opened':False}
            s['freeze']=dict(frozen,sha256=digest(frozen));s['phase']='frozen'
            self.event(s,'validation_agent','freeze',{'sha256':s['freeze']['sha256']});return s['freeze']
    def validate(self,case_id):
        with self.case(case_id) as s:
            if s['phase']=='validated':
                self.verify_seals(s);return s['report']
            require(s['phase']=='frozen','Final evaluation requires a frozen model; no training after final opening')
            self.verify_seals(s)
            frozen=s['freeze'];require(setup_hash(s)==frozen['setup_sha256'],'Frozen setup or documents changed');require(digest(s['model'])==frozen['model_sha256'],'Frozen model changed')
            require(digest(s['facts'])==frozen['facts_sha256'] and digest(s['ontology'])==frozen['ontology_sha256'],'Frozen evidence changed')
            require(source_hashes()==frozen['sources'],'Numerical source changed since freeze')
            for r in s['train']+s['dev']:self.load_observation(r)
            s['phase']='validating';s['final_opened']=True
            self.event(s,'validation_agent','final_opened',{'freeze_sha256':frozen['sha256']})
        with self.case(case_id) as s:
            selected=s['model'].get('selected');terms=(selected or {}).get('terms',[]);rows=[];generated=0
            pool=actions.make_action_pool(2700000+s['seed'])
            for j,action in enumerate(pool[16:19]):
                try:
                    require(selected is not None,'No selected model; failure is retained')
                    response=provider.generate_observation(s['_spec'],action,2800000+s['seed']*10+j,s['_noise'])
                    generated+=1;obs=response['observation'];pred,diag=forecast.forecast_with_diagnostics(terms,obs['x'],obs['t'],obs['prefix_t'],obs['prefix_u'])
                    metrics=forecast_metrics(pred,response['clean']['u'][10:]);require(metrics['relative_l2'] is not None and np.isfinite(metrics['relative_l2']),'Undefined prediction metric')
                    rows.append({'index':j,'status':'evaluated','metrics':metrics,'prefix_frames':11,'duration':1.})
                except Exception as exc:rows.append({'index':j,'status':'failed','reason':str(exc)})
            recovery=reference_metrics(terms,provider.reference_terms(s['_spec']))
            failures=sum(r['status']=='failed' for r in rows);mean=None if failures else float(np.mean([r['metrics']['relative_l2'] for r in rows]))
            max_error=max(r['relative_error'] for r in recovery['coefficient_comparison'] if r['relative_error'] is not None)
            joint_success=recovery['exact_support'] and max_error<=.10 and mean is not None and mean<=.05
            conflicts=self.conflicts(s);pending=any(p['status']=='pending' for p in s['ontology_proposals'])
            evidence_quality=quality.assess(s,conflicts);evidence_ready=evidence_quality['metadata_ready']
            s['report']={'case_id':s['id'],'origin':'controlled_synthetic_demo','nexent_live_llm_verified':False,'equation':(selected or {}).get('equation'),'terms':terms,'structure':recovery,'max_true_coefficient_relative_error':max_error,'numerical_joint_success':joint_success,'joint_success_rule':'exact support AND max true coefficient error <=10% AND all 3 predictions valid AND mean prediction error <=5%','predictions':rows,'evaluation_slots':3,'evaluation_reference_trajectories_generated':generated,'evaluation_measurement_scalars':generated*7104,'discovery_budget_excludes_final_evaluation':True,'prediction_failure_slots':failures,'mean_prediction_error':mean,'prediction_5pct_passed':mean is not None and mean<=.05,'evidence_ready':evidence_ready,'evidence_quality':evidence_quality,'evidence_conflicts':conflicts,'pending_ontology_review':pending,'evidence_sources':sorted(s['documents']),'graph':s['facts'],'freeze_sha256':s['freeze']['sha256'],'measurement_scalars_charged':s['charged']*7104,'charged_experiments':s['charged'],'frequency_is_probability':False,'real_industry_validation':False,'final_used_for_discovery':False,'note':'This local run verifies tool contracts, not multi-agent LLM reasoning quality or industrial value.'}
            s['phase']='validated';self.event(s,'validation_agent','validation_completed',{'report_sha256':digest(s['report'])})
            return s['report']
    @staticmethod
    def verify_seals(s):
        previous=None
        for sequence,e in enumerate(s['events'],1):
            require(e['sequence']==sequence and e['previous']==previous and digest({k:v for k,v in e.items() if k!='sha256'})==e['sha256'],'Event chain changed')
            previous=e['sha256']
        frozen=s.get('freeze')
        if frozen:
            require(digest({k:v for k,v in frozen.items() if k!='sha256'})==frozen['sha256'],'Freeze seal changed')
            seals=[e['details'].get('sha256') for e in s['events'] if e['operation']=='freeze']
            require(seals and seals[-1]==frozen['sha256'],'Freeze event does not bind this seal')
            require(digest(s['model'])==frozen['model_sha256'],'Frozen model changed')
            require(setup_hash(s)==frozen['setup_sha256'],'Frozen setup or documents changed')
            require(digest(s['facts'])==frozen['facts_sha256'] and digest(s['ontology'])==frozen['ontology_sha256'],'Frozen evidence changed')
        if s['phase']=='validated':
            seals=[e['details'].get('report_sha256') for e in s['events'] if e['operation']=='validation_completed']
            require(seals and seals[-1]==digest(s['report']),'Final report changed after validation')
    def evidence_path(self,case_id,source,target,max_hops=4,limit=3):
        require(all(isinstance(x,str) and 1<=len(x)<=2000 for x in (source,target)) and source!=target,'Distinct bounded source and target required')
        require(integer(max_hops,1,6) and integer(limit,1,5),'Invalid path bounds')
        with self.case(case_id,readonly=True) as s:
            result=quality.paths(s,source,target,max_hops,limit)
            return dict(result,case_id=case_id,conflicts=self.conflicts(s),evidence_quality=quality.assess(s,self.conflicts(s)))
    def handoff(self,case_id):
        with self.case(case_id,readonly=True) as s:
            self.verify_seals(s)
            phase=s['phase'];evidence=quality.assess(s,self.conflicts(s));budget=s['budget']-s['charged']
            mapping={'ready':('pde_agent','pde_identify'),'observed':('pde_agent','pde_identify'),
                     'identified':('experiment_agent','experiment_propose') if budget else ('validation_agent','validation_freeze'),
                     'proposed':('experiment_agent','experiment_acquire'),'frozen':('validation_agent','validation_evaluate')}
            role,tool=mapping.get(phase,(None,None))
            stop=phase in ('validated','initial_failed','creating','acquiring','validating')
            issue=None
            if stop:issue='terminal' if phase=='validated' else 'failed_or_pending_operation_do_not_resubmit'
            elif phase in ('ready','observed','identified') and not evidence['metadata_ready']:
                role,tool=('asset_agent','asset_ingest') if not s['documents'] else ('graph_agent','graph_propose')
                issue='evidence_review_required; numerical diagnostics remain possible but cannot imply metadata readiness'
                if any(b['code']!='missing_required_metadata' for b in evidence['blockers']):
                    role,tool,stop='local_operator',None,True
                    issue='manual_evidence_review_required; no automatic conflict resolution'
            args={'case_id':case_id}
            if tool=='experiment_acquire':args['proposal_id']=s['proposal']['id']
            return {'contract_version':1,'case_id':case_id,'phase':phase,'state_head':s['events'][-1]['sha256'],
                    'remaining_experiments':budget,'next_role':role,'next_tool':tool,
                    'next_arguments':args if tool else {},'required_inputs':{'asset_ingest':['documents'],'graph_propose':['facts']}.get(tool,[]),
                    'arguments_complete':tool is not None and tool not in ('asset_ingest','graph_propose'),'stop':stop,'terminal':phase in ('validated','initial_failed'),'reason':issue,
                    'evidence_quality':evidence,'final_opened':s.get('final_opened',False),
                    'source_ids':sorted(s['documents']),'freeze_sha256':s.get('freeze',{}).get('sha256'),
                    'authentication_claimed':False,'final_metrics_must_not_guide_new_selection':s.get('final_opened',False)}
    def evidence(self,case_id):
        with self.case(case_id,readonly=True) as s:
            self.verify_seals(s)
            return {'status':self.view(s),'events':s['events'],'facts':s['facts'],'ontology':s['ontology'],'report':s.get('report'),'chain_consistent':True,'historical_runtime_independently_authenticated':False}
