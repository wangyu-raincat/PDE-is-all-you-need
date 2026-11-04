"""Render the technical design draft and actual result table to Word."""
import json
from pathlib import Path
from docx import Document
from docx.shared import Pt
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

ROOT=Path(__file__).resolve().parents[1]
def build(output=None):
    doc=Document();style=doc.styles['Normal'];style.font.name='Arial';style.font.size=Pt(10)
    style.element.rPr.rFonts.set(qn('w:eastAsia'),'Microsoft YaHei')
    doc.add_heading('PDE 资产决策多智能体',0)
    doc.add_paragraph('开发设计文档｜本地工具与接入包阶段｜Nexent 平台联调待完成')
    code=False;table=None
    for line in (ROOT/'NEXENT_README.md').read_text().splitlines():
        if line.startswith('```'):code=not code;table=None;continue
        if not line.strip():table=None;continue
        if code:
            p=doc.add_paragraph(line);p.paragraph_format.space_after=Pt(0)
            for r in p.runs:r.font.name='Consolas';r.font.size=Pt(8)
        elif line.startswith('#'):
            depth=len(line)-len(line.lstrip('#'));doc.add_heading(line[depth:].strip(),min(depth,3))
        elif line.startswith('|'):
            cells=[x.strip() for x in line.strip('|').split('|')]
            if all(set(c)<=set('-: ') for c in cells):continue
            if table is None:table=doc.add_table(rows=0,cols=len(cells));table.style='Light Shading Accent 1'
            row=table.add_row().cells
            for c,v in zip(row,cells):c.text=v
        else:doc.add_paragraph(line.replace('**','').replace('`',''))
    result=ROOT/'competition_agent/verification/results/summary.json'
    if result.exists():
        doc.add_heading('附录：实际本地 MCP 演示结果',1)
        data=json.loads(result.read_text());doc.add_paragraph('全部为确定性路由的合成工程演示，不是 Nexent LLM 效果实验。错误率按百分比显示。')
        table=doc.add_table(rows=1,cols=6);table.style='Light Shading Accent 1'
        headers=['案例','结构完整','多余项','最大系数误差','平均预测误差','预测失败槽位']
        for c,v in zip(table.rows[0].cells,headers):c.text=v
        for r in data['rows']:
            vals=[r['case'],str(r['exact_support']),str(len(r['spurious_terms'])),f"{100*r['max_coefficient_error']:.4f}%",'失败' if r['mean_prediction_error'] is None else f"{100*r['mean_prediction_error']:.4f}%",str(r['prediction_failures'])]
            for c,v in zip(table.add_row().cells,vals):c.text=v
    target=Path(output or ROOT/'competition_agent/build/PDE多智能体开发设计文档.docx');target.parent.mkdir(parents=True,exist_ok=True);doc.save(target);return target
if __name__=='__main__':print(build())
