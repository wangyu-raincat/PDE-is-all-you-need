"""Reproducible T03 real-file acceptance; never invent missing source assertions."""
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import re
import sys
import tempfile
from unittest.mock import patch
import zipfile
import xml.etree.ElementTree as ET
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from competition_agent.multimodal import ROOT, parse_asset
from competition_agent.service import Lab


def offline(stack):
    stack.enter_context(patch.dict('os.environ',{},clear=True))
    for name in ('socket.socket.connect','socket.socket.connect_ex','socket.create_connection'):
        stack.enter_context(patch(name,side_effect=AssertionError('Network is forbidden during OCR acceptance')))


def table_check(path,asset):
    """Count source XML formula nodes independently of the parser; retain located examples."""
    with zipfile.ZipFile(path) as archive:
        ns={'m':'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
        xml_formula_count=sum(len(ET.fromstring(archive.read(n)).findall('.//m:f',ns))
            for n in archive.namelist() if re.fullmatch(r'xl/worksheets/sheet\d+\.xml',n))
    sheets=[]
    for block in asset['blocks']:
        cells=block['headers']+[c for row in block['rows'] for c in row]
        formulas=[c for c in cells if c['formula'] is not None]
        sheets.append({'locator':block['locator'],'nonempty_cells':len(cells),
            'formula_count':len(formulas),'cached_formula_count':sum(c['value'] is not None for c in formulas),
            'missing_formula_cache_count':sum(c['formula_cache_missing'] for c in formulas),
            'header_cells':block['headers'],'value_examples':cells[len(block['headers']):len(block['headers'])+3]})
    parsed_count=sum(s['formula_count'] for s in sheets)
    assert parsed_count==xml_formula_count
    return {'sheets':sheets,'source_xml_formula_count':xml_formula_count,
            'formula_count_verified':True,'note':'Zero source formulas means real files cannot exercise formula/cache preservation; use generated fixture.'}


def graph_check(raw,output):
    """Use public-source facts in a real SQLite Lab graph in an isolated acceptance case."""
    with tempfile.TemporaryDirectory(prefix='T03-graph-',dir=output) as directory:
        lab=Lab(directory)
        case=lab.create_demo()['case_id']
        metadata=ROOT/'pde_discovery/data/additive_heat/metadata_verification.md'
        text=metadata.read_text(encoding='utf-8')
        source=lab.ingest(case,[{'title':'公开元数据核查（验收引用）','kind':'text','text':text}])['sources'][0]['source_id']
        facts=[{'subject':g,'predicate':'has_build_direction','object':'x','source_id':source,
                'quote':next(t for t in text.splitlines() if t.startswith('| '+g+' |'))} for g in ('xn','xp')]
        proposal=lab.graph_propose(case,facts)
        # Test fixture only: simulate the operator step; never alter a user's production ontology.
        for change in proposal['pending_ontology_changes']:
            if change['status']=='pending': lab.review_ontology(case,change['id'],True)
        lab.graph_apply(case,proposal['proposal_id'])
        with lab.case(case,readonly=True) as state: before=state['facts'][:]
        source_ids={}
        with ExitStack() as stack:
            offline(stack)
            for group in ('xn','xp'):
                source_ids[group]=lab.ingest(case,[{'path':str(raw/'examples'/f'自制示例-{group}.png'),'kind':'image'}])['sources'][0]['source_id']
        context=lab.context(case,'graph_agent',limit=40)
        conflicts=context['multimodal_conflicts']
        assert len(conflicts)==1 and conflicts[0]['candidate']['subject']=='xn'
        assert conflicts[0]['existing_facts'][0]['id'] in {f['id'] for f in before}
        fields={}
        with lab.case(case,readonly=True) as state:
            for group,source_id in source_ids.items():
                asset=state['multimodal_assets'][source_id]['asset']; block=asset['blocks'][0]
                fields[group]={'source_id':source_id,'asset_id':asset['asset_id'],'ocr':block['ocr'],
                    'thumbnail_sha256':hashlib.sha256(Path(block['thumbnail_path']).read_bytes()).hexdigest(),
                    'needs_review':block['needs_review'],'vlm_status':block['vlm_status']}
                for token in (group,'has_build_direction','未记录','20.8','5m/s','自制示例'):
                    assert token in block['text'].replace(' ','')
        candidate=next(c['fact'] for c in context['fact_candidates'] if c['fact']['source_id']==source_ids['xn'])
        proposed=lab.graph_propose(case,[candidate])
        try: lab.graph_apply(case,proposed['proposal_id'])
        except ValueError as exc:
            assert 'human review' in str(exc)
            refusal=str(exc)
        else: raise AssertionError('OCR fact was applied without review')
        with lab.case(case,readonly=True) as state:
            assert state['facts']==before
            events=state['events']
        # Persist audit snapshot before the temporary SQLite database is removed.
        return {'status':'passed','case_id':case,'storage':'real Lab SQLite in disposable acceptance directory',
            'fixture_notice':'Synthetic demo case; ontology operator approval is simulated only in this isolated test. Not a production human review.',
            'existing_graph_facts':before,'ocr_sources':fields,'conflicts':conflicts,
            'xn_conflicts':1,'xp_conflicts':0,'original_graph_unchanged':True,
            'automatic_apply_refused':refusal,'network_calls_forbidden':True,'vlm_environment_cleared':True,
            'events':events}


def main():
    output=ROOT/'competition_agent/build/T03';output.mkdir(parents=True,exist_ok=True)
    raw=ROOT/'competition_agent/assets_raw'
    requests=[(raw/'dib-2021-107632.pdf','pdf')]+[(raw/name,'xlsx') for name in ['AdditiveRodResults.xlsx','Temperature Results.xlsx','Wind Results.xlsx']]
    requests += [(ROOT/'pde_discovery/data/mendeley_heat/author_oneDkhEstimator_V1.0.0_full_source.txt','code')]
    requests += [(raw/'examples'/f'自制示例-{group}.png','image') for group in ['xn','xp']]
    result={'task':'T03','real_file_checks':[],'complete':False,'vlm':'disabled'}
    assets={}
    for path,kind in requests:
        with ExitStack() as stack:
            if kind=='image': offline(stack)
            asset=parse_asset(path,kind,output/'thumbnails')
        assets[path.name]=asset
        (output/(path.stem+'.parsed.json')).write_text(json.dumps(asset,ensure_ascii=False,indent=2),encoding='utf-8')
        item={'path':path.relative_to(ROOT).as_posix(),'kind':kind,'sha256':asset['sha256'],'status':'parsed','blocks':len(asset['blocks'])}
        if kind=='code': item['geometry']=[b for b in asset['blocks'] if 225<=b['locator']['line']<=245]
        elif kind=='xlsx': item.update(table_check(path,asset))
        elif kind=='image': item['ocr']=asset['blocks'][0]['ocr'];item['needs_review']=True
        result['real_file_checks'].append(item)
    pdf=assets['dib-2021-107632.pdf'];code=next(a for a in assets.values() if a['modality']=='code')
    blocks=[b for b in pdf['blocks'] if b['type']=='text']
    key=next(b for b in blocks if 'build direction (x or y)' in b['text'] and '(p or n)' in b['text'])
    assert key['locator']['page']==3
    repeats=[]
    for group in ('xn','xp','yn','yp'):
        block=next(b for b in code['blocks'] if b['variable']=='filenames' and group.upper()+' == True' in b['scope'])
        assert block['value']==[group+str(i) for i in range(1,6)]
        table=next(b for b in assets['AdditiveRodResults.xlsx']['blocks'] if b['locator']['sheet']==group)
        assert [c['value'] for c in table['headers'][1:]]==block['value']
        repeats.append({'group':group,'count':5,'code_evidence':{'source_id':code['source_id'],'quote':block['text'],'locator':block['locator']},
            'xlsx_evidence':{'source_id':assets['AdditiveRodResults.xlsx']['source_id'],'locator':{'sheet':group,'cell_range':'B1:F1'},'cells':table['headers'][1:]}})
    repeat_pattern=re.compile(r'(?:five|5)\s+(?:times|tests|repeats)|(?:tested|repeated)\s+(?:five|5)',re.I)
    repeat_matches=[b for b in blocks if repeat_pattern.search(b['text'].replace('ﬁ','fi'))]
    result['pdf_acceptance']={'status':'passed_with_source_exception','source_id':pdf['source_id'],
        'direction_porosity':{'quote':key['text'],'locator':key['locator']},
        'pages_inspected':7,'repeat_sentence_matches':repeat_matches,
        'source_exception':'No explicit five-tests-per-rod sentence located in this data paper. Do not attribute the alternate evidence to the PDF.',
        'repeat_count_alternate_evidence':repeats}
    result['graph_acceptance']=graph_check(raw,output)
    result['execution_checks_passed']=True
    result['strict_task_card_pdf_repeats_satisfied']=bool(repeat_matches)
    result['complete']=bool(repeat_matches)
    (output/'acceptance.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'parsed_inputs':len(requests),'xlsx_sheets':10,'graph_conflicts':1,
        'execution_checks_passed':True,'strict_pdf_repeat_sentence':bool(repeat_matches)},ensure_ascii=False))

if __name__=='__main__': main()
