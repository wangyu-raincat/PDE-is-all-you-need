"""Offline, escaped HTML report; no external resources or executable source text."""
from html import escape
from pathlib import Path
import json

def percent(value):
    return '未产生有效指标' if value is None else f'{100*value:.4f}%'

def render(result, evidence, trace):
    r=result['report'];quality=r.get('evidence_quality',{})
    def e(value):return escape(str(value),quote=True)
    metrics=[('方程结构','完整恢复' if r['structure']['exact_support'] else '未完整恢复'),
             ('最大真项系数误差',percent(r['max_true_coefficient_relative_error'])),
             ('平均预测误差',percent(r['mean_prediction_error'])),
             ('预测失败槽位',f"{r['prediction_failure_slots']} / 3")]
    cards=''.join(f'<article><span>{e(k)}</span><strong>{e(v)}</strong></article>' for k,v in metrics)
    blockers=''.join(f'<li>{e(b["code"])}: {e(b["details"])}</li>' for b in quality.get('blockers',[])) or '<li>规定的合成元数据一致性检查通过；不等于工业可用。</li>'
    facts=''.join(f'<details id="fact-{e(f["id"])}"><summary>{e(f["subject"])} → {e(f["predicate"])} → {e(f["object"])}</summary><blockquote>{e(f["quote"])}</blockquote><p>来源 {e(f["source_id"])} · 位置 {e(f.get("citation",{}))}</p></details>' for f in evidence['facts'])
    paths=''
    for path in result.get('evidence_paths',{}).get('paths',[]):
        links=' → '.join(f'<a href="#fact-{e(f["id"])}">{e(f["subject"])} / {e(f["predicate"])} / {e(f["object"])}</a>' for f in path['edges'])
        paths+=f'<p>{links}</p><p>涉及 {len(path["source_ids"])} 份资料；仅为带引文的关系路径。</p>'
    timeline=''.join(f'<details><summary>{x["step"]}. {e(x["planned_role"])} · {e(x["tool"])} · {x["duration_seconds"]:.3f}s</summary><pre>{e(json.dumps(x["result"],ensure_ascii=False,indent=2))}</pre></details>' for x in trace)
    return f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>PDE 可追溯实验报告</title>
<style>body{{margin:0;background:#f3f5f9;color:#172b44;font:16px/1.6 system-ui,sans-serif}}main{{max-width:1100px;margin:auto;padding:36px 22px}}h1{{font-size:32px;line-height:1.3}}h2{{margin-top:30px}}.badge{{color:#3c5277;font-weight:600}}.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:12px}}article,section{{background:white;border:1px solid #dbe2ed;border-radius:10px;padding:18px}}article span{{display:block;color:#526580}}strong{{display:block;font-size:23px}}section{{margin-top:18px}}details{{padding:10px 0;border-bottom:1px solid #e4e9f0}}summary{{cursor:pointer;font-weight:600}}pre{{overflow-wrap:anywhere;white-space:pre-wrap;font-size:12px}}p,li,blockquote{{overflow-wrap:anywhere}}blockquote{{border-left:3px solid #4e77bb;padding-left:12px;margin-left:0}}a{{color:#1c59a7}}.muted{{color:#526580}}code{{overflow-wrap:anywhere}}</style>
<main><div class="badge">本地 MCP 工具实测 · 合成数据 · 未使用 LLM</div><h1>PDE 可追溯实验报告</h1>
<p class="muted">案例 {e(r['case_id'])}。这是工程重演，不能作为新的盲测、平台内多智能体效果或真实行业验证。</p>
<div class="cards">{cards}</div><section><h2>识别结论</h2><p>{e(r.get('equation') or '未选出可接受的方程，不强制输出结果。')}</p><p>缺失项：{e(r['structure']['missing_terms'])}；多余项：{e(r['structure']['spurious_terms'])}。</p>
<p>发现预算 {r['charged_experiments']} 次；最终评估另生成 {r['evaluation_reference_trajectories_generated']} 条轨迹。预测使用 11 帧前缀，保持 1.0 时长。</p></section>
<section><h2>证据质量与跨文档路径</h2><ul>{blockers}</ul>{paths or '<p>无可展示的完整证据路径。</p>'}{facts}</section>
<section><h2>冻结与追溯</h2><p>冻结标识 <code>{e(r['freeze_sha256'])}</code></p><p>事件链一致：{e(evidence['chain_consistent'])}。事件链不是第三方认证；来源断言也不是物理真实性证明。</p></section>
<section><h2>实际工具调用</h2><p>{len(trace)} 次调用。角色为明确的脚本路由标签，不是已运行的大模型身份。</p>{timeline}</section></main></html>'''

def save(output,result,evidence,trace):
    Path(output).write_text(render(result,evidence,trace),encoding='utf-8')
