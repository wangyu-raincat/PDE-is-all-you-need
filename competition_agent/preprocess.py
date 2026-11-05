"""P0 deterministic input preparation, before agent/MCP orchestration.

Only text assertions, metadata triples CSV and experiment-register CSV have
declared schemas. Numerical CSV is quarantined, never silently used for training.
"""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import re
import unicodedata

VERSION = 'input-preparation-v1'
REQUIREMENTS = (
    ('demo_grid', 'records_field', 'demo_field'),
    ('demo_field', 'uses_unit', 'dimensionless'),
    ('demo_grid', 'has_boundary', 'periodic'),
)
HEADER_ALIASES = {
    'subject':'subject','entity':'subject','实体':'subject','主体':'subject',
    'predicate':'predicate','relation':'predicate','关系':'predicate',
    'object':'object','value':'object','值':'object','客体':'object',
}
RELATIONS = {'单位':'uses_unit','边界':'has_boundary','记录字段':'records_field',
             '测量':'measures','派生自':'derived_from','证据来源':'supported_by'}
UNITS = {'无量纲':'dimensionless','dimensionless':'dimensionless','1':'dimensionless',
         '开尔文':'K','kelvin':'K','K':'K','摄氏度':'degC','℃':'degC','°C':'degC',
         'degC':'degC','米':'m','m':'m','秒':'s','s':'s'}
BOUNDARIES = {'周期':'periodic','周期边界':'periodic','periodic':'periodic',
              '非周期':'nonperiodic','nonperiodic':'nonperiodic'}
TEXT_TRIPLE = re.compile(r'([^\W\d][\w:-]*)\s+([a-z][a-z0-9_]{0,63})\s+(.+?)\.?$', re.UNICODE)
ROLES = ('pde_coordinator','asset_agent','graph_agent','pde_agent','experiment_agent','validation_agent')


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,allow_nan=False,separators=(',',':')).encode()).hexdigest()


def normalized_content(text):
    return unicodedata.normalize('NFC',text.removeprefix('\ufeff').replace('\r\n','\n').replace('\r','\n')).rstrip('\n')


def validate_document(doc):
    if not isinstance(doc,dict) or set(doc)!={'title','kind','text'}:
        raise ValueError('Document requires title,kind,text')
    if not isinstance(doc['title'],str) or not 1<=len(doc['title'])<=160:
        raise ValueError('Invalid title')
    if doc['kind'] not in ('text','csv') or not isinstance(doc['text'],str) or not 1<=len(doc['text'].encode())<=131072:
        raise ValueError('Only bounded text/CSV input is supported')


def normalize_triple(subject,predicate,obj):
    original=(subject,predicate,obj)
    subject,predicate,obj=subject.strip(),predicate.strip(),obj.strip()
    transformed=[{'field':name,'from':old,'to':new,'rule':'trim_outer_whitespace'} for name,old,new in zip(('subject','predicate','object'),original,(subject,predicate,obj)) if old!=new]
    relation=RELATIONS.get(predicate,predicate)
    if relation!=predicate:transformed.append({'field':'predicate','from':predicate,'to':relation,'rule':'explicit_relation_alias'})
    result=(UNITS if relation=='uses_unit' else BOUNDARIES if relation=='has_boundary' else {}).get(obj,obj)
    if result!=obj:transformed.append({'field':'object','from':obj,'to':result,'rule':'explicit_label_alias_no_numeric_conversion'})
    return subject,relation,result,transformed


def parse_document(doc,source=None):
    validate_document(doc);source=source or digest(doc);text=doc['text']
    result={'source_id':source,'title':doc['title'],'kind':doc['kind'],
            'content_id':digest({'kind':doc['kind'],'text':normalized_content(text)}),
            'status':'ready','classification':'text','issues':[],'issue_count':0,'candidates':[],
            'parser_version':VERSION,'raw_text_changed':False,'numerical_training_input':False}
    def issue(code,priority='P0',**detail):
        result['issue_count']+=1
        if len(result['issues'])<20:result['issues'].append({'priority':priority,'code':code,'source_id':source,**detail})
        if priority=='P0':result['status']='blocked'
    def candidate(subject,predicate,obj,quote,start,end,line,**extra):
        subject,predicate,obj,changes=normalize_triple(subject,predicate,obj)
        if not subject or not obj or not re.fullmatch(r'[a-z][a-z0-9_]{0,63}',predicate):
            issue('invalid_metadata_triple',line=line);return
        if max(len(subject),len(obj),len(quote))>2000:
            issue('metadata_fact_exceeds_limit',line=line);return
        if len(result['candidates'])>=400:
            issue('metadata_fact_count_exceeds_limit',line=line);return
        fact={'subject':subject,'predicate':predicate,'object':obj,'source_id':source,'quote':quote}
        result['candidates'].append({'fact':fact,'citation':{'start_char':start,'end_char':end,
            'line':line,'offset_unit':'Unicode code points, half-open',**extra},
            'transformations':changes,'parser_version':VERSION,
            'verification':'declared format assertion only, not external truth'})
    if not normalized_content(text).strip():
        issue('empty_document')
    if any(ord(c)<32 and c not in '\r\n\t' for c in text):
        issue('unsupported_control_character')
    if doc['kind']=='text':
        offset=0
        for line_number,raw in enumerate(text.splitlines(keepends=True),1):
            line=raw.rstrip('\r\n');stripped=line.strip().removeprefix('\ufeff')
            match=TEXT_TRIPLE.fullmatch(stripped)
            if match and (match.group(2) in ('uses_unit','has_boundary','records_field','measures','derived_from','supported_by') or '_' in match.group(2)):
                a,b,c=match.groups()
                start=offset+line.index(stripped)
                candidate(a,b,c,stripped,start,start+len(stripped),line_number,format='text_triple')
            offset+=len(raw)
        if not result['candidates'] and result['status']!='blocked':
            issue('unstructured_text_for_retrieval_only','P1')
    else:
        result['classification']='table'
        # csv.reader counts CR/LF physical lines; Unicode line separators are data.
        lines=re.split(r'(?<=\n)|(?<=\r)(?!\n)',text)
        starts=[0]
        for line in lines:starts.append(starts[-1]+len(line))
        reader=csv.reader(io.StringIO(text.removeprefix('\ufeff'),newline=''),strict=True)
        rows=[];previous=0
        try:
            for values in reader:
                current=reader.line_num
                start=starts[previous]+(1 if previous==0 and text.startswith('\ufeff') else 0)
                end=starts[current]
                raw=text[start:end].rstrip('\r\n')
                if values:rows.append((values,raw,start,start+len(raw),previous+1,current))
                previous=current
                if len(rows)>1001:
                    issue('table_row_limit_exceeded');break
        except csv.Error as exc:issue('invalid_csv',detail=str(exc))
        if not rows:
            issue('empty_table')
        else:
            headers=[unicodedata.normalize('NFKC',x).strip().casefold() for x in rows[0][0]]
            headers=[HEADER_ALIASES.get(x,x) for x in headers]
            result['columns']=headers;result['raw_columns']=rows[0][0];result['row_count']=len(rows)-1
            result['register_records']=[]
            if not all(headers) or len(headers)!=len(set(headers)) or len(headers)>64:
                issue('invalid_or_duplicate_headers')
            if len(rows)==1:issue('table_has_no_records')
            if set(headers)=={'subject','predicate','object'}:
                result['classification']='metadata_triples'
            elif set(headers)=={'asset_id','group','trajectories','source_kind'}:
                result['classification']='experiment_register'
            elif {'t','x','u'}<=set(headers):
                result['classification']='numerical_data'
                issue('numerical_adapter_and_split_assignment_required')
            else:issue('unregistered_table_schema','P1')
            preview=[]
            for values,raw,start,end,line,last_line in rows[1:]:
                if len(values)!=len(headers):
                    issue('ragged_csv_record',line=line,expected_columns=len(headers),actual_columns=len(values));continue
                record=dict(zip(headers,values))
                if len(preview)<5:preview.append({'values':record,'line':line,'end_line':last_line})
                if result['classification']=='metadata_triples':
                    candidate(record['subject'],record['predicate'],record['object'],raw,start,end,line,
                              end_line=last_line,format='metadata_triples_csv',columns=headers,raw_columns=rows[0][0],header_quote=rows[0][1])
                elif result['classification']=='experiment_register':
                    if (not record['asset_id'].strip() or record['group'] not in ('training','development','final')
                        or not record['trajectories'].isdigit() or int(record['trajectories'])<1
                        or not record['source_kind'].strip()):
                        issue('invalid_experiment_register_record',line=line)
                    else:result['register_records'].append({'asset_id':record['asset_id'],'group':record['group'],'trajectories':int(record['trajectories']),'source_kind':record['source_kind'],'line':line})
            result['preview_rows']=preview
    if result['status']=='blocked':result['candidates']=[]
    return result


def prepare(documents):
    if not isinstance(documents,list) or len(documents)>50:
        raise ValueError('Need a list of at most 50 documents')
    # Original title/kind/text identity retained; same payload replay is one record.
    originals={}
    for doc in documents:validate_document(doc);originals[digest(doc)]=doc
    records={key:parse_document(doc,key) for key,doc in sorted(originals.items())}
    canonical={}
    for key,r in records.items():
        r['duplicate_of']=canonical.setdefault(r['content_id'],key)
        if r['duplicate_of']==key:r['duplicate_of']=None
    candidates=[c for r in records.values() if r['status']=='ready' and not r['duplicate_of'] for c in r['candidates']]
    issues=[i for r in records.values() for i in r['issues']]
    groups={}
    for c in candidates:
        f=c['fact']
        if f['predicate'] in ('uses_unit','has_boundary'):
            groups.setdefault((f['subject'],f['predicate']),[]).append(c)
    for key,values in groups.items():
        if len({v['fact']['object'] for v in values})>1:
            issues.append({'priority':'P0','code':'conflicting_input_metadata','subject':key[0],'predicate':key[1],
                           'evidence':[v['fact'] for v in values[:5]],'evidence_count':len(values),'evidence_truncated':len(values)>5})
    known={(c['fact']['subject'],c['fact']['predicate'],c['fact']['object']) for c in candidates}
    missing=[dict(zip(('subject','predicate','object'),t)) for t in REQUIREMENTS if t not in known]
    if missing:issues.append({'priority':'P1','code':'missing_demo_metadata','requirements':missing})
    issues.sort(key=lambda i:(i['priority'],i['code'],i.get('source_id','')))
    summary={'parser_version':VERSION,'priority_order':['P0: stop and resolve format/conflicts','P1: fill schema/metadata gaps','P2: optional enrichment'],
             'input_documents':len(documents),'stored_sources':len(records),
             'unique_contents':len(canonical),'duplicate_sources':sum(r['duplicate_of'] is not None for r in records.values()),
             'quarantined_sources':sum(r['status']=='blocked' for r in records.values()),
             'candidate_facts':len(candidates),'blocked':any(i['priority']=='P0' for i in issues),
             'issues':issues[:40],'issues_truncated':len(issues)>40 or any(r['issue_count']>len(r['issues']) for r in records.values()),
             'ready_for_demo':not missing and not any(i['priority']=='P0' for i in issues),'numerical_training_input':False,'llm_used':False}
    return {'summary':summary,'records':records,'candidates':candidates,
            'usable_source_ids':[k for k,r in records.items() if r['status']=='ready' and not r['duplicate_of']]}


def matching_fact(fact,documents,record=None):
    doc=documents.get(fact.get('source_id'))
    if doc is None:return None
    parsed=record if record is not None else parse_document(doc,fact['source_id'])
    for candidate in parsed['candidates']:
        if candidate['fact']=={k:fact[k] for k in ('subject','predicate','object','source_id','quote')}:
            return candidate
    return None


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();path=Path(a.input)
    if path.stat().st_size>7*1024*1024:raise ValueError('Input manifest exceeds 7 MiB')
    payload=json.loads(path.read_text(encoding='utf-8-sig'))
    prepared=prepare(payload['documents'] if isinstance(payload,dict) else payload)
    with Path(a.output).open('x',encoding='utf-8') as f:json.dump(prepared,f,ensure_ascii=False,indent=2)
    print(json.dumps(prepared['summary'],ensure_ascii=False,indent=2))
    if prepared['summary']['blocked']:raise SystemExit(2)

if __name__=='__main__':main()
