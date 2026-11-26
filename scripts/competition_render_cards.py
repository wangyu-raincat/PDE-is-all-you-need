"""Render explicitly synthetic record cards using Windows System.Drawing; no downloads."""
import base64
import json
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[1]

def main():
    dest=ROOT/'competition_agent/assets_raw/examples'; dest.mkdir(parents=True,exist_ok=True)
    cards=[]
    for sample,direction in [('xn','y'),('xp','x')]:
        lines=['自制示例 / 实验记录卡',f'试样号: {sample}',f'{sample} has_build_direction {direction}',
               '测试日期: 未记录', '环境温度: 20.8 C (源码设定, 非实测)',
               '风速: 5 m/s (源码猜测, 非实测)', '仅用于解析演示，不是原始实验记录']
        path=dest/f'自制示例-{sample}.png'
        def quote(s): return "'"+s.replace("'","''")+"'"
        ps="""Add-Type -AssemblyName System.Drawing
$b=[System.Drawing.Bitmap]::new(1000,620)
$g=[System.Drawing.Graphics]::FromImage($b)
$g.Clear([System.Drawing.Color]::White)
$f=[System.Drawing.Font]::new('KaiTi',23)
$g.TextRenderingHint=[System.Drawing.Text.TextRenderingHint]::AntiAliasGridFit
"""
        for i,line in enumerate(lines):
            ps+=f'$g.DrawString({quote(line)},$f,[System.Drawing.Brushes]::Black,30,{25+i*75})\n'
        ps+=f'$b.Save({quote(str(path))},[System.Drawing.Imaging.ImageFormat]::Png)\n$f.Dispose()\n$g.Dispose()\n$b.Dispose()'
        subprocess.run(['powershell','-NoProfile','-NonInteractive','-EncodedCommand',base64.b64encode(ps.encode('utf-16le')).decode()],check=True,capture_output=True)
        cards.append({'file':path.name,'synthetic':True,'expected_text':lines,
            'expected_conflict':sample=='xn','reference_direction':'x',
            'source_references':['pde_discovery/data/additive_heat/metadata_verification.md:9',
              'pde_discovery/data/mendeley_heat/author_oneDkhEstimator_V1.0.0_full_source.txt:246',
              'pde_discovery/data/mendeley_heat/author_oneDkhEstimator_V1.0.0_full_source.txt:248'],
            'note':'xn direction y is a deliberate synthetic conflict; date unknown. Expected text is fixture truth, never OCR output.'})
    (dest/'cards.json').write_text(json.dumps(cards,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

if __name__=='__main__': main()
