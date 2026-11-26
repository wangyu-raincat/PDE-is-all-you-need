"""T03 fixtures are synthetic; dependency-gated tests are not real-file acceptance."""
import importlib.util
import json
import re
import zipfile
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from competition_agent import multimodal as mm
from competition_agent.service import Lab
from tests.test_competition_agent import response

AUTHOR=mm.ROOT/'pde_discovery/data/mendeley_heat/author_oneDkhEstimator_V1.0.0_full_source.txt'

def available(*names): return all(importlib.util.find_spec(n) for n in names)

class MultimodalTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)

    def test_real_author_geometry_and_line_numbers(self):
        a=mm.parse_asset(AUTHOR,'code')
        by_line={b['locator']['line']:b for b in a['blocks']}
        self.assertEqual(by_line[225]['value'],[0,.007,.014,.021,.028,.035])
        for group,line,diameter,length,density in [('XN',228,.006287,.15209,7812.98),('YN',233,.006230,.15211,7828.79),('XP',238,.005302,.15233,7392.47),('YP',243,.006206,.15222,7756.86)]:
            for offset,var,value in [(0,'D',diameter),(1,'L',length),(2,'rho',density)]:
                block=by_line[line+offset]
                self.assertEqual(block['variable'],var)
                self.assertAlmostEqual(block['value'],value)
                self.assertIn(group+' == True',block['scope'])
                text=AUTHOR.read_text(encoding='utf-8-sig');loc=block['locator']
                self.assertEqual(text[loc['start_char']:loc['end_char']],block['text'])

    def test_no_execution_or_dynamic_evaluation(self):
        p=self.root/'code.py';p.write_text("raise RuntimeError('never execute')\nD=6.287/1000\nx=__import__('os').getcwd()\ny=2**10000000\n",encoding='utf-8')
        a=mm.parse_asset(p,'code')
        self.assertEqual([b['variable'] for b in a['blocks']],['D'])

    def test_unicode_offsets_and_branch_scope(self):
        p=self.root/'a.py';p.write_text('标题="自制"; D=1\nif flag:\n    x=[1,2]\nelse:\n    x=[3,4]\n',encoding='utf-8')
        a=mm.parse_asset(p,'code');t=p.read_text(encoding='utf-8')
        for b in a['blocks']:
            loc=b['locator'];self.assertEqual(t[loc['start_char']:loc['end_char']],b['text'])
        self.assertEqual(a['blocks'][-1]['scope'],['not (flag)'])

    def test_repeated_literals_keep_branch_specific_location(self):
        p=self.root/'repeat.py'
        p.write_text('if ADD == True:\n    if XN == True:\n        D=1/1000\n    if XP == True:\n        D=1/1000\n',encoding='utf-8')
        a=mm.parse_asset(p,'code');d,spans=mm.project(a)
        rows=mm.literal_candidates({'src':{'asset':a,'document':d,'spans':spans}})
        self.assertEqual([r['citation']['multimodal_locations'][0]['locator']['line'] for r in rows],[3,5])

    def test_missing_dependency_is_actionable_without_download(self):
        with patch.object(mm.importlib,'import_module',side_effect=ImportError):
            with self.assertRaisesRegex(mm.MissingDependency,'PyMuPDF, pdfplumber'): mm.dependency_modules('pdf')

    def test_projection_rejects_oversized_content(self):
        a={'asset_id':'abc','path':'fixture','blocks':[{'text':'a'*131073,'block_id':'b','locator':{},'needs_review':False}]}
        with self.assertRaisesRegex(ValueError,'128 KiB'): mm.project(a)

    def test_image_fixture_conflict_and_exact_ocr_locator(self):
        # Manufactured OCR result, not a claim that RapidOCR ran.
        quote='xn has_build_direction y'
        a={'asset_id':'fixture','path':'自制示例.png','modality':'image','blocks':[{'type':'figure','text':quote,'block_id':'b','needs_review':True,'locator':{'file':'fixture'},'ocr':[{'text':quote,'bbox':[[1,2],[8,2],[8,9],[1,9]],'confidence':.9}]}]}
        d,spans=mm.project(a);r={'asset':a,'document':d,'spans':spans}
        old={'subject':'xn','predicate':'has_build_direction','object':'x','source_id':'paper','quote':'xn has_build_direction x'}
        conflicts=mm.conflicts({'ocr':r},[old])
        self.assertEqual(len(conflicts),1)
        self.assertEqual(conflicts[0]['existing_facts'],[old])
        self.assertEqual(mm.locate(r,quote)[0]['locator']['bbox'],a['blocks'][0]['ocr'][0]['bbox'])
        self.assertNotIn('base64',json.dumps(mm.summaries({'ocr':r},'graph_agent',0,20)))

    def test_lab_code_ingestion_replay_citation_and_freeze(self):
        lab=Lab(self.root)
        with patch('competition_agent.service.provider.generate_observation',side_effect=response): case=lab.create_demo()['case_id']
        request=[{'kind':'code','path':str(AUTHOR)}]
        source=lab.ingest(case,request)['sources'][0]['source_id']
        self.assertEqual(source,lab.ingest(case,request)['sources'][0]['source_id'])
        context=lab.context(case,limit=40)
        literal=next(c for c in context['fact_candidates'] if c['fact']['subject']=='xn' and c['fact']['predicate']=='has_diameter_m')
        self.assertEqual(literal['fact']['object'],'0.006287')
        lab.graph_propose(case,[literal['fact']])
        fact={'source_id':source,'subject':'D','predicate':'has_literal','object':'6.287/1000','quote':'D=6.287/1000'}
        proposal=lab.graph_propose(case,[fact])
        with lab.case(case) as state:
            f=state['graph_proposals'][0]['facts'][0]
            self.assertEqual(f['citation']['multimodal_locations'][0]['locator']['line'],228)
            self.assertEqual(len(state['multimodal_assets']),1)
            state['phase']='frozen'
        with self.assertRaisesRegex(ValueError,'frozen'):lab.ingest(case,request)
        self.assertIn('proposal_id',proposal)

    def test_lab_rejects_path_escape(self):
        lab=Lab(self.root)
        with self.assertRaisesRegex(ValueError,'approved input roots'):
            lab.ingest('unused',[{'path':str(self.root/'secret.py'),'kind':'code'}])

    def test_ocr_cannot_auto_apply_to_graph(self):
        lab=Lab(self.root)
        with patch('competition_agent.service.provider.generate_observation',side_effect=response): case=lab.create_demo()['case_id']
        uploads=self.root/'uploads';uploads.mkdir();p=uploads/'card.png';p.write_bytes(b'fixture')
        a={'asset_id':'fake','path':str(p),'modality':'image','blocks':[{'type':'figure','text':'xn uses_unit m','block_id':'b','locator':{'file':str(p)},'needs_review':True,'ocr':[]}]}
        with patch.object(mm,'parse_asset',return_value=a): source=lab.ingest(case,[{'path':str(p),'kind':'image'}])['sources'][0]['source_id']
        f={'source_id':source,'subject':'xn','predicate':'uses_unit','object':'m','quote':'xn uses_unit m'}
        proposal=lab.graph_propose(case,[f])
        with self.assertRaisesRegex(ValueError,'human review'):lab.graph_apply(case,proposal['proposal_id'])
        ctx=lab.context(case)
        self.assertTrue(ctx['fact_candidates'][0]['needs_review'])

    @unittest.skipUnless(available('fitz','pdfplumber'),'manual installation required: PyMuPDF, pdfplumber')
    def test_generated_pdf_fixture(self):
        import fitz
        p=self.root/'fixture.pdf'
        with fitz.open() as doc:
            page=doc.new_page();page.insert_text((40,70),'xn build direction x; each rod tested five times.')
            doc.save(p)
        a=mm.parse_asset(p,'pdf',self.root/'thumbs')
        block=next(b for b in a['blocks'] if b['type']=='text')
        self.assertIn('five times',block['text']);self.assertEqual(block['locator']['page'],1)
        self.assertEqual(len(block['locator']['bbox']),4)

    @unittest.skipUnless(available('openpyxl'),'manual installation required: openpyxl')
    def test_generated_xlsx_formula_and_missing_cache(self):
        import openpyxl
        p=self.root/'fixture.xlsx';book=openpyxl.Workbook();ws=book.active
        ws.append(['diameter','formula']);ws.append([.006287,'=A2*1000']);book.save(p);book.close()
        # Author workbooks can omit the optional worksheet dimension element.
        with zipfile.ZipFile(p) as z: members={n:z.read(n) for n in z.namelist()}
        members['xl/worksheets/sheet1.xml']=re.sub(rb'<dimension[^>]*/>',b'',members['xl/worksheets/sheet1.xml'])
        with zipfile.ZipFile(p,'w') as z:
            for name,data in members.items(): z.writestr(name,data)
        a=mm.parse_asset(p,'xlsx');b=a['blocks'][0]
        self.assertEqual(b['locator']['cell_range'],'A1:B2')
        formula=b['rows'][0][1];self.assertEqual(formula['formula'],'=A2*1000')
        self.assertIsNone(formula['value']);self.assertTrue(formula['formula_cache_missing'])

    @unittest.skipUnless(available('rapidocr_onnxruntime'),'manual installation required: rapidocr-onnxruntime')
    def test_real_example_ocr_without_vlm_key(self):
        p=mm.ROOT/'competition_agent/assets_raw/examples/自制示例-xn.png'
        with patch.dict('os.environ',{},clear=True): a=mm.parse_asset(p,'image',self.root/'thumbs')
        block=a['blocks'][0]
        self.assertEqual(block['vlm_status'],'disabled');self.assertTrue(block['needs_review'])
        self.assertTrue(block['ocr']);self.assertTrue(Path(block['thumbnail_path']).is_file())

if __name__=='__main__': unittest.main()
