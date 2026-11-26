"""Bounded local multimodal parsing. Source files are never executed."""
from __future__ import annotations
import ast
import hashlib
import importlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEPENDENCIES = {'pdf': [('fitz', 'PyMuPDF'), ('pdfplumber', 'pdfplumber')],
                'xlsx': [('openpyxl', 'openpyxl')],
                'image': [('rapidocr_onnxruntime', 'rapidocr-onnxruntime')], 'code': []}

class MissingDependency(RuntimeError):
    pass


def dependency_modules(kind):
    modules, missing = {}, []
    for module, package in DEPENDENCIES[kind]:
        try:
            modules[module] = importlib.import_module(module)
        except ImportError:
            missing.append(package)
    if missing:
        raise MissingDependency('Install manually in build/t02-env: ' + ', '.join(missing))
    return modules


def literal(node):
    """Literal containers and numeric unit conversion only; no names/calls/eval."""
    try:
        value = ast.literal_eval(node)
    except (ValueError, TypeError):
        if not isinstance(node, ast.BinOp) or not isinstance(node.op, (ast.Div, ast.Mult, ast.Add, ast.Sub)):
            raise ValueError('Not a literal')
        a, b = literal(node.left), literal(node.right)
        if type(a) not in (int, float) or type(b) not in (int, float):
            raise ValueError('Not numeric literal arithmetic')
        if isinstance(node.op, ast.Div): value = a / b
        elif isinstance(node.op, ast.Mult): value = a * b
        elif isinstance(node.op, ast.Add): value = a + b
        else: value = a - b
    json.dumps(value, allow_nan=False)
    return value


def parse_code(path):
    text = path.read_text(encoding='utf-8-sig')
    tree = ast.parse(text)
    blocks = []
    def walk(node, scope):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            scope = scope + [node.name]
        if isinstance(node, ast.If):
            condition = ast.get_source_segment(text, node.test)
            for child in node.body: walk(child, scope + [condition])
            for child in node.orelse: walk(child, scope + ['not (' + condition + ')'])
            return
        if isinstance(node, (ast.Assign, ast.AnnAssign)) and node.value is not None:
            try: value = literal(node.value)
            except (ValueError, TypeError, ZeroDivisionError, OverflowError): pass
            else:
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                for target in targets:
                    if isinstance(target, ast.Name):
                        quote = ast.get_source_segment(text, node)
                        start = sum(len(s) for s in text.splitlines(keepends=True)[:node.lineno-1])
                        # AST columns are UTF-8 bytes, public offsets are Unicode characters.
                        line = text.splitlines(keepends=True)[node.lineno-1]
                        start += len(line.encode('utf-8')[:node.col_offset].decode('utf-8'))
                        blocks.append({'type':'code_literal', 'text':quote, 'variable':target.id,
                            'value':value, 'expression':ast.get_source_segment(text,node.value), 'scope':scope,
                            'locator':{'file':str(path),'line':node.lineno,'end_line':node.end_lineno,
                                       'start_char':start,'end_char':start+len(quote)}})
        for child in ast.iter_child_nodes(node): walk(child, scope)
    walk(tree, [])
    return blocks


def parse_pdf(path, modules, output):
    blocks = []
    with modules['fitz'].open(str(path)) as doc:
        if len(doc)>100: raise ValueError('PDF page limit 100')
        for page_number, page in enumerate(doc,1):
            page_text = []
            for item in page.get_text('dict')['blocks']:
                loc = {'page':page_number,'bbox':list(item['bbox']), 'coordinate_system':'PDF points, top-left'}
                if item['type']==0:
                    text = '\n'.join(''.join(s['text'] for s in line['spans']) for line in item['lines'])
                    blocks.append({'type':'text','text':text,'locator':dict(loc,line=1,end_line=len(text.splitlines()),start_char=0,end_char=len(text),offset_scope='block text')})
                    page_text.append((loc['bbox'],text))
                elif item['type']==1:
                    raw = item['image']; sha = hashlib.sha256(raw).hexdigest()
                    thumb = output / (sha + '.png')
                    pix = modules['fitz'].Pixmap(raw)
                    while max(pix.width,pix.height)>512: pix.shrink(1)
                    pix.save(str(thumb))
                    blocks.append({'type':'figure','text':'','locator':loc,'image_sha256':sha,
                                   'thumbnail_path':str(thumb),'caption':'','ocr':[],
                                   'description':None,'needs_review':True})
            for block in blocks:
                if block['type']=='figure' and block['locator']['page']==page_number:
                    captions=[t for box,t in page_text if t.lstrip().lower().startswith(('fig.', 'figure'))
                              and 0 <= box[1]-block['locator']['bbox'][3] < 100]
                    block['caption']='\n'.join(captions)
    with modules['pdfplumber'].open(str(path)) as doc:
        for number,page in enumerate(doc.pages,1):
            for table in page.find_tables():
                rows=table.extract()
                blocks.append({'type':'table','text':json.dumps(rows,ensure_ascii=False),
                    'locator':{'page':number,'bbox':list(table.bbox)},
                    'headers':rows[0] if rows else [],'rows':rows[1:], 'cell_range':None})
    return blocks


def parse_xlsx(path, modules):
    api=modules['openpyxl']; blocks=[]
    formulas=api.load_workbook(path, data_only=False, read_only=True, keep_links=False)
    cached=None
    try:
        cached=api.load_workbook(path, data_only=True, read_only=True, keep_links=False)
        for ws in formulas:
            if ws.max_row is None or ws.max_column is None:
                ws.calculate_dimension(force=True)
            if ws.max_row>5000 or ws.max_column>100: raise ValueError('XLSX sheet size limit')
            rows=[]
            for row in ws.iter_rows():
                cells=[]
                for cell in row:
                    if cell.value is None: continue
                    value=cell.value
                    formula=value if cell.data_type=='f' else None
                    cached_value=cached[ws.title][cell.coordinate].value if formula else value
                    cells.append({'cell':cell.coordinate,'formula':formula,
                                  'value':cached_value,'formula_cache_missing':formula is not None and cached_value is None})
                if cells: rows.append(cells)
            blocks.append({'type':'table','text':json.dumps(rows,ensure_ascii=False,default=str),
                'locator':{'sheet':ws.title,'cell_range':ws.calculate_dimension()},
                'headers':rows[0] if rows else [],'rows':rows[1:]})
    finally:
        formulas.close()
        if cached is not None: cached.close()
    # Dates become ISO-like strings; formulas are never executed or recomputed.
    return json.loads(json.dumps(blocks,default=str))


def parse_image(path, modules, output):
    result,_=modules['rapidocr_onnxruntime'].RapidOCR()(str(path))
    ocr=[{'text':text,'bbox':box,'confidence':float(score)} for box,text,score in (result or [])]
    import cv2
    import numpy as np
    pixels=cv2.imdecode(np.frombuffer(path.read_bytes(),dtype=np.uint8),cv2.IMREAD_COLOR)
    if pixels is None: raise ValueError('Invalid image')
    scale=min(1.0,512/max(pixels.shape[:2]))
    thumbnail=output/(hashlib.sha256(path.read_bytes()).hexdigest()+'.png')
    small=cv2.resize(pixels,(max(1,round(pixels.shape[1]*scale)),max(1,round(pixels.shape[0]*scale))))
    ok,encoded=cv2.imencode('.png',small)
    if not ok: raise ValueError('Thumbnail encoding failed')
    thumbnail.write_bytes(encoded.tobytes())
    # No network fallback: VLM is deliberately optional and disabled in this version.
    return [{'type':'figure','text':'\n'.join(x['text'] for x in ocr),
        'locator':{'file':str(path),'coordinate_system':'image pixels, top-left'},
        'image_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
        'thumbnail_path':str(thumbnail),'caption':'','ocr':ocr,'description':None,
        'vlm_status':'disabled','needs_review':True}]


def parse_asset(path, kind, output_dir=None):
    if kind not in DEPENDENCIES: raise ValueError('Unsupported modality')
    path=Path(path).resolve()
    if not path.is_file() or path.stat().st_size>20*1024*1024: raise ValueError('Missing file or size limit 20 MiB')
    modules=dependency_modules(kind)
    output=Path(output_dir or ROOT/'competition_agent/build/T03/thumbnails').resolve()
    if kind in ('pdf','image'): output.mkdir(parents=True,exist_ok=True)
    blocks=(parse_code(path) if kind=='code' else parse_pdf(path,modules,output) if kind=='pdf'
            else parse_xlsx(path,modules) if kind=='xlsx' else parse_image(path,modules,output))
    sha=hashlib.sha256(path.read_bytes()).hexdigest()
    for i,block in enumerate(blocks):
        block['block_id']=f'{sha}:{i}'
        block.setdefault('needs_review',False)
    asset={'asset_id':sha,'source_id':sha,'modality':kind,'path':str(path),'sha256':sha,
           'blocks':blocks,'semantic_truth_verified':False}
    if len(json.dumps(asset,ensure_ascii=False,allow_nan=False).encode())>4*1024*1024:
        raise ValueError('Parsed asset exceeds 4 MiB limit')
    return asset


def project(asset):
    """Keep legacy text documents intact; map projected character spans back to blocks."""
    texts=[]; spans=[]; offset=0
    for block in asset['blocks']:
        text=block.get('text','')
        if not text: continue
        if asset.get('modality') == 'pdf':
            # PDF font/control placeholders are layout artifacts. Keep raw blocks
            # and replace one code point with one space only in the search view.
            text = ''.join(' ' if ord(c)<32 and c not in '\r\n\t' else c for c in text)
        spans.append({'start_char':offset,'end_char':offset+len(text),
                      'block_id':block['block_id'],'locator':block['locator'],
                      'needs_review':block['needs_review']})
        if block.get('ocr'):
            local_offset=offset
            for line in block['ocr']:
                spans.append({'start_char':local_offset,'end_char':local_offset+len(line['text']),
                    'block_id':block['block_id'],'locator':dict(block['locator'],bbox=line['bbox']),
                    'needs_review':True})
                local_offset+=len(line['text'])+1
        texts.append(text); offset+=len(text)+1
    text='\n'.join(texts)
    if not text: text='[No extracted text; inspect asset blocks]'
    if len(text.encode())>131072: raise ValueError('Projected evidence exceeds 128 KiB')
    return {'title':f"{Path(asset['path']).name[:80]} [{asset['asset_id'][:16]}]",'kind':'text','text':text},spans


def locate(asset_record, quote, block_id=None):
    text=asset_record['document']['text']; matches={}
    for span in asset_record['spans']:
        if block_id is not None and span['block_id']!=block_id: continue
        if quote not in text[span['start_char']:span['end_char']]: continue
        rank=(span['end_char']-span['start_char'], 'bbox' not in span['locator'])
        previous=matches.get(span['block_id'])
        if previous is None or rank<previous[0]: matches[span['block_id']]=(rank,span)
    return [entry[1] for entry in matches.values()]


def summaries(records, role, offset, limit):
    items=[]
    for source,record in records.items():
        a=record['asset']
        for block in a['blocks']:
            item={'source_id':source,'asset_id':a['asset_id'],'modality':a['modality'],
                  'block_id':block['block_id'],'type':block['type'],'locator':block['locator'],
                  'needs_review':block['needs_review']}
            if block['type']=='figure':
                item.update(description=block.get('description'),caption=block.get('caption'),
                            ocr=block.get('ocr',[])[:8],thumbnail_path=block.get('thumbnail_path'))
            elif role in ('asset_agent','graph_agent','pde_coordinator'):
                item['quote']=block.get('text','')[:1000]
                if block['type']=='code_literal': item.update(variable=block['variable'],value=block['value'],scope=block['scope'])
            items.append(item)
    selected=[];size=0
    for item in items[offset:offset+limit]:
        encoded=json.dumps(item,ensure_ascii=False)
        if len(encoded)>12000:
            item={k:v for k,v in item.items() if k not in ('value','ocr')}
            item['details_omitted']=True
        length=len(json.dumps(item,ensure_ascii=False).encode())
        if selected and size+length>24576: break
        selected.append(item);size+=length
    return {'blocks':selected,'total_blocks':len(items),
            'next_offset':offset+len(selected) if offset+len(selected)<len(items) else None}


def conflicts(records, facts):
    """Compare declared OCR triples against graph facts, retaining both sources."""
    from competition_agent.preprocess import parse_document
    result=[]
    for source, record in records.items():
        parsed=parse_document(record['document'],source)
        for candidate in parsed['candidates']:
            fact=candidate['fact']
            others=[f for f in facts if f['subject']==fact['subject'] and f['predicate']==fact['predicate'] and f['object']!=fact['object']]
            if others:
                result.append({'candidate':fact,'existing_facts':others,
                               'locations':locate(record,fact['quote']),
                               'status':'requires_review_not_automatically_resolved'})
    return result

def literal_candidates(records):
    """Static author parameters with scope, never evaluated program state."""
    mapping={'D':'has_diameter_m','L':'has_length_m','rho':'has_density_kg_m3','X':'has_sensor_positions_m'}
    result=[]
    for source,record in records.items():
        for block in record['asset']['blocks']:
            if block['type']!='code_literal' or block['variable'] not in mapping: continue
            scopes=block['scope']
            if 'ADD == True' not in scopes: continue
            groups=[g for g in ('xn','xp','yn','yp') if g.upper()+' == True' in scopes]
            if block['variable']=='X': groups=['xn','xp','yn','yp']
            for group in groups:
                value=block['value']
                obj=format(value,'.12g') if isinstance(value,(int,float)) else json.dumps(value)
                result.append({'fact':{'subject':group,'predicate':mapping[block['variable']],
                    'object':obj,'source_id':source,'quote':block['text']},
                    'citation':{'multimodal_locations':locate(record,block['text'],block['block_id'])},
                    'transformations':[{'rule':'AST literal arithmetic and explicit ADD/XN/XP/YN/YP branch scope','scope':scopes,'expression':block['expression']}],
                    'needs_review':False,'verification':'static source literal, not measured or executed state'})
    return result
