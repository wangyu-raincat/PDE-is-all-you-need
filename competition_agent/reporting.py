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
    alignment=''.join(f'<li>{e(m["concept"])} → {e(m["standard"])} {e(m.get("section") or "目录级")} · {e(m["relation"])} · {e(m["status"])}<br>{e(m["difference"])}</li>' for m in evidence.get("alignment",[]))
    timeline=''.join(f'<details><summary>{x["step"]}. {e(x["planned_role"])} · {e(x["tool"])} · {x["duration_seconds"]:.3f}s</summary><pre>{e(json.dumps(x["result"],ensure_ascii=False,indent=2))}</pre></details>' for x in trace)
    return f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>PDE 可追溯实验报告</title>
<style>body{{margin:0;background:#f3f5f9;color:#172b44;font:16px/1.6 system-ui,sans-serif}}main{{max-width:1100px;margin:auto;padding:36px 22px}}h1{{font-size:32px;line-height:1.3}}h2{{margin-top:30px}}.badge{{color:#3c5277;font-weight:600}}.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:12px}}article,section{{background:white;border:1px solid #dbe2ed;border-radius:10px;padding:18px}}article span{{display:block;color:#526580}}strong{{display:block;font-size:23px}}section{{margin-top:18px}}details{{padding:10px 0;border-bottom:1px solid #e4e9f0}}summary{{cursor:pointer;font-weight:600}}pre{{overflow-wrap:anywhere;white-space:pre-wrap;font-size:12px}}p,li,blockquote{{overflow-wrap:anywhere}}blockquote{{border-left:3px solid #4e77bb;padding-left:12px;margin-left:0}}a{{color:#1c59a7}}.muted{{color:#526580}}code{{overflow-wrap:anywhere}}</style>
<main><div class="badge">本地 MCP 工具实测 · 合成数据 · 未使用 LLM</div><h1>PDE 可追溯实验报告</h1>
<p class="muted">案例 {e(r['case_id'])}。这是工程重演，不能作为新的盲测、平台内多智能体效果或真实行业验证。</p>
<div class="cards">{cards}</div><section><h2>识别结论</h2><p>{e(r.get('equation') or '未选出可接受的方程，不强制输出结果。')}</p><p>缺失项：{e(r['structure']['missing_terms'])}；多余项：{e(r['structure']['spurious_terms'])}。</p>
<p>发现预算 {r['charged_experiments']} 次；最终评估另生成 {r['evaluation_reference_trajectories_generated']} 条轨迹。预测使用 11 帧前缀，保持 1.0 时长。</p></section>
<section><h2>证据质量与跨文档路径</h2><ul>{blockers}</ul>{paths or '<p>无可展示的完整证据路径。</p>'}{facts}</section>
{("<section><h2>标准方法对齐</h2><ul>"+alignment+"</ul></section>") if alignment else ""}<section><h2>冻结与追溯</h2><p>冻结标识 <code>{e(r['freeze_sha256'])}</code></p><p>事件链一致：{e(evidence['chain_consistent'])}。事件链不是第三方认证；来源断言也不是物理真实性证明。</p></section>
<section><h2>实际工具调用</h2><p>{len(trace)} 次调用。角色为明确的脚本路由标签，不是已运行的大模型身份。</p>{timeline}</section></main></html>'''

def save(output,result,evidence,trace):
    Path(output).write_text(render(result,evidence,trace),encoding='utf-8')

def render_inputs(prepared):
    """Standalone pre-agent quality report, including blocked inputs."""
    summary=prepared['summary']
    rows=''.join('<tr>'+''.join(f'<td>{escape(str(v))}</td>' for v in (
        r['title'],r['classification'],r['status'],r['duplicate_of'] or '独立内容',
        len(r['candidates'])))+'</tr>' for r in prepared['records'].values())
    issues=''.join(f'<li><b>{escape(i["priority"])}</b> {escape(i["code"])}<pre>{escape(json.dumps(i,ensure_ascii=False,indent=2))}</pre></li>' for i in summary['issues'])
    status='停止并先修复资料' if summary['blocked'] else '仍需补齐示例元数据' if not summary['ready_for_demo'] else '可进入示例协作流程'
    return f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>智能体前置信息处理</title>
<style>body{{font:16px/1.6 system-ui,sans-serif;background:#f5f7fb;color:#20334d;margin:0}}main{{max-width:1000px;margin:auto;padding:24px}}table{{width:100%;min-width:640px;border-collapse:collapse;background:white}}td,th{{padding:10px;text-align:left;border-bottom:1px solid #d8e0ed;overflow-wrap:anywhere}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;font-size:13px}}.table{{overflow-x:auto}}h1{{font-size:28px}}</style><main>
<h1>智能体前置信息处理</h1><p><b>{status}</b>。先处理 P0 格式与来源冲突，再补齐 P1 元数据，最后分派智能体。</p>
<p>原始资料 {summary['stored_sources']} 份，独立内容 {summary['unique_contents']} 份，重复 {summary['duplicate_sources']} 份，隔离 {summary['quarantined_sources']} 份。此步骤不使用 LLM，不启动实验，不把 CSV 当训练数据。</p>
<div class="table"><table><thead><tr><th>资料</th><th>分类</th><th>状态</th><th>重复来源</th><th>候选事实</th></tr></thead><tbody>{rows}</tbody></table></div>
<h2>按优先级处理的问题</h2><ul>{issues or '<li>本示例的前置格式与元数据检查通过。</li>'}</ul><p>原文保留；别名只统一标签，不自动进行物理数值单位换算。原文显式断言不是外部真实性证明。</p></main></html>'''

# T07: archived decision traces are rendered without invoking decision or model tools.
import argparse
import base64
import hashlib
import re
from collections import defaultdict

_REPORT_ROOT = Path(__file__).resolve().parents[1]
_LABELS = {'draft':'草案','queue':'待审核','discard':'丢弃审计','pending':'待确认',
           'heat_compare':'训练与开发模型比较','heat_repeatability':'同杆五次复测',
           'historical_holdouts':'历史留出评分','identifiability_evidence':'可识别性诊断',
           'downgrade_triggers':'结论降级条件','diameters':'两杆直径',
           'graph_paths':'图谱路径','acquisition_links':'测量与模型链路',
           'model_evidence':'模型来源','method_evidence':'方法来源',
           'mean_D_difference':'平均 D 之差','mean_D_ratio':'平均 D 倍数',
           'larger_within_rod_range':'较大的杆内极差','parameters':'拟合参数',
           'relative_l2':'相对 L2 误差（原始比例）','evidence':'来源证据'}


def _html(value):
    return escape(str(value), quote=True)


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)


def _citations(value):
    if isinstance(value, dict):
        if 'source_id' in value and 'quote' in value:
            yield value
        for child in value.values():
            yield from _citations(child)
    elif isinstance(value, list):
        for child in value:
            yield from _citations(child)


def _citation_key(citation):
    return _json([citation['source_id'], citation.get('locator', {}), citation['quote']])


def _locations(citation):
    locator = citation.get('locator', {})
    nested = locator.get('multimodal_locations', [])
    return [item['locator'] for item in nested if 'locator' in item] or [locator]


def _location_text(locator):
    fields = []
    if locator.get('page') is not None:
        fields.append(f"第 {locator['page']} 页")
    if locator.get('bbox'):
        fields.append('bbox ' + ', '.join(f'{v:.2f}' for v in locator['bbox']))
    if locator.get('sheet'):
        fields.append(f"工作表 {locator['sheet']} · {locator.get('cell', '')}")
    if locator.get('line') is not None:
        end = locator.get('end_line', locator['line'])
        fields.append(f"行 {locator['line']}" + (f'–{end}' if end != locator['line'] else ''))
    if 'start_char' in locator:
        fields.append(f"字符 [{locator['start_char']}, {locator.get('end_char', '?')})")
    if locator.get('id'):
        fields.append('目录条目 ' + str(locator['id']))
    return ' · '.join(fields) or '位置未登记'


class _EvidenceBook:
    """Report-local anchors. File access is limited to the supplied repository root."""
    def __init__(self, traces, root):
        self.root = Path(root).resolve()
        self.records = {}
        self.aliases = defaultdict(set)
        for trace in traces:
            for citation in _citations(trace):
                key = _citation_key(citation)
                self.records.setdefault(key, dict(citation))
                for field in ('title', 'path'):
                    if citation.get(field):
                        self.aliases[citation['source_id']].add(str(citation[field]))
        self.ids = {key: f'evidence-{i:03d}' for i, key in enumerate(self.records, 1)}
        self.preview_stats = defaultdict(int)
        self._image_cache = {}
        self._workbooks = {}

    def close(self):
        for workbook in self._workbooks.values():
            workbook.close()

    def link(self, citation, label=None):
        key = _citation_key(citation)
        target = self.ids[key]
        return f'<a class="evidence-link" href="#{target}">{_html(label or target.replace("evidence-", "E"))}<span aria-hidden="true"> ↗</span></a>'

    def resolve(self, citation, locator):
        candidates = [locator.get('file'), locator.get('path'), citation.get('path')]
        if citation['source_id'].startswith('am:'):
            candidates.append(citation['source_id'][3:])
        candidates.extend(sorted(self.aliases[citation['source_id']]))
        for name in candidates:
            if not name:
                continue
            name = re.sub(r' \[[0-9a-f]+\]$', '', str(name))
            p = Path(name)
            options = [p] if p.is_absolute() else [self.root/p]
            options += [self.root/'competition_agent/assets_raw'/p.name,
                        self.root/'pde_discovery/data/mendeley_heat'/p.name]
            for option in options:
                option = option.resolve()
                if option.is_relative_to(self.root) and option.is_file():
                    return option
        return None

    def title(self, citation):
        return citation.get('title') or citation.get('path') or next(iter(sorted(self.aliases[citation['source_id']])), citation['source_id'].removeprefix('am:'))

    def preview(self, citation, locator):
        path = self.resolve(citation, locator)
        if path is None:
            self.preview_stats['archived_text'] += 1
            return '<p class="subtle">原始文本已随 trace 内嵌；下方引文及定位为本次归档证据。</p>'
        try:
            if path.suffix.lower() == '.pdf' and locator.get('page') and locator.get('bbox'):
                import fitz
                cache_key = (str(path), locator['page'], tuple(locator['bbox']))
                if cache_key not in self._image_cache:
                    with fitz.open(path) as pdf:
                        page = pdf[int(locator['page'])-1]
                        rect = fitz.Rect(locator['bbox']) & page.rect
                        if rect.is_empty:
                            raise ValueError('PDF bbox 不在页面内')
                        pixmap = page.get_pixmap(matrix=fitz.Matrix(1.7,1.7), clip=rect)
                        self._image_cache[cache_key] = base64.b64encode(pixmap.tobytes('png')).decode('ascii')
                self.preview_stats['pdf_crop'] += 1
                return f'<figure><img loading="lazy" alt="{_html(path.name)} 第 {_html(locator["page"])} 页原文坐标裁剪" src="data:image/png;base64,{self._image_cache[cache_key]}"><figcaption>原文件第 {_html(locator["page"])} 页 · 按归档 bbox 裁剪 · PDF point，左上原点</figcaption></figure>'
            if path.suffix.lower() == '.xlsx' and locator.get('sheet') and locator.get('cell'):
                import openpyxl
                if path not in self._workbooks:
                    self._workbooks[path] = openpyxl.load_workbook(path, data_only=False, read_only=True)
                cell = self._workbooks[path][locator['sheet']][locator['cell']]
                self.preview_stats['sheet_cell'] += 1
                return f'<div class="cell-preview"><span>{_html(path.name)} / {_html(locator["sheet"])}</span><table><tr><th>单元格</th><th>原始值 / 公式</th></tr><tr><td>{_html(cell.coordinate)}</td><td><mark>{_html(cell.value)}</mark></td></tr></table></div>'
            if path.suffix.lower() in {'.txt','.py','.yaml','.json','.md','.csv'}:
                text = path.read_text(encoding='utf-8-sig')
                lines = text.splitlines()
                line = locator.get('line')
                if line is None:
                    # Curated standard entries have an identifier rather than a line number.
                    line = next((i+1 for i,s in enumerate(lines) if citation['quote'] in s), None)
                if line is not None:
                    start = max(1, int(line)-1)
                    end = min(len(lines), int(locator.get('end_line', line))+1)
                    excerpt = '\n'.join(f'{i:>5}  {lines[i-1]}' for i in range(start,end+1))
                    self.preview_stats['source_lines'] += 1
                    return f'<p class="subtle">{_html(path.relative_to(self.root).as_posix())} · 原文件行 {start}–{end}</p><pre class="source-lines">{_html(excerpt)}</pre>'
        except (OSError, ValueError, KeyError, IndexError, ImportError) as error:
            self.preview_stats['unavailable_preview'] += 1
            return f'<p class="notice">原文件预览不可用：{_html(error)}。保留归档引文与位置。</p>'
        self.preview_stats['archived_text'] += 1
        return '<p class="subtle">使用下方归档原文与定位。</p>'

    def render(self):
        output = []
        for key, citation in self.records.items():
            locs = _locations(citation)
            preview = ''.join(self.preview(citation, loc) for loc in locs)
            positions = '；'.join(_location_text(loc) for loc in locs)
            label = self.ids[key].replace('evidence-', 'E')
            output.append(f'<details class="evidence-card" id="{self.ids[key]}"><summary><b>{label}</b><span>{_html(self.title(citation))}<small>{_html(positions)}</small></span></summary><div class="detail-body"><blockquote>{_html(citation["quote"])}</blockquote>{preview}<p class="source-id">source_id · {_html(citation["source_id"])}</p><details><summary>完整归档定位</summary><pre>{_html(json.dumps(citation.get("locator",{}),ensure_ascii=False,indent=2))}</pre></details></div></details>')
        return ''.join(output)


def _render_value(value, book, label='模型记录'):
    """Keep every archived numeric value available; never recompute scientific results."""
    if isinstance(value, dict):
        if 'source_id' in value and 'quote' in value:
            return book.link(value, '查看原文位置')
        rows = ''.join(f'<div class="data-row"><dt>{_html(_LABELS.get(k,k))}</dt><dd>{_render_value(v,book,k)}</dd></div>' for k,v in value.items())
        return f'<dl class="data-fields">{rows}</dl>'
    if isinstance(value, list):
        return ''.join(f'<details class="nested"><summary>{_html(_LABELS.get(label,label))} · {i}</summary>{_render_value(v,book,label)}</details>' for i,v in enumerate(value,1)) or '<span class="subtle">无记录</span>'
    if value is None:
        return '<span class="subtle">未提供</span>'
    return _html('是' if value is True else '否' if value is False else value)


def _render_graph(trace, book, seed):
    edges = {}
    reasoning = trace['reasoning']
    for path in reasoning.get('graph_paths',[]):
        for edge in path:
            edges[edge.get('id') or _json(edge['triple'])] = edge
    for edge in reasoning.get('conflicts',[]):
        edges[edge.get('id') or _json(edge['triple'])] = edge
    rendered = []
    classes = seed.get('classes',[])
    colors = ['#147d80','#5376ac','#8a61a0','#a86b27','#62774b']
    for edge in edges.values():
        subject,predicate,obj = edge['triple']
        rel = seed.get('relations',{}).get(predicate,{})
        domain,range_ = rel.get('domain','未定义类型'),rel.get('range','未定义类型')
        def color(kind):
            return colors[classes.index(kind)%len(colors)] if kind in classes else '#68757c'
        conflict = edge.get('conflict',False)
        status = _LABELS.get(edge.get('status',edge.get('tier','')),edge.get('status','未登记'))
        cites = ' '.join(book.link(c) for c in edge.get('evidence',[]))
        rendered.append(f'<div class="graph-edge {"conflict" if conflict else ""}"><div class="graph-node" style="--node:{color(domain)}"><small>{_html(domain)}</small><b>{_html(subject)}</b></div><div class="graph-relation"><span>{_html(predicate)}</span><span aria-hidden="true">⟶</span><small>{"冲突 · " if conflict else ""}{_html(status)}</small></div><div class="graph-node" style="--node:{color(range_)}"><small>{_html(range_)}</small><b>{_html(obj)}</b></div><div class="edge-cites">{cites}</div></div>')
    return ''.join(rendered) or '<p class="subtle">此问题未经过图谱路径；请查看规则与冲突记录。</p>'


def _render_assets(book, catalog):
    by_path = {a['path']:a for a in catalog.get('assets',[])}
    grouped = defaultdict(list)
    sources = defaultdict(list)
    for citation in book.records.values():
        sources[citation['source_id']].append(citation)
    for source, citations in sources.items():
        first = citations[0]
        path = next((p for c in citations for loc in _locations(c) if (p:=book.resolve(c,loc))),None)
        relative = path.relative_to(book.root).as_posix() if path else None
        asset = by_path.get(relative,{})
        modality = {'.pdf':'PDF 图文','.xlsx':'工作簿','.csv':'测量 CSV','.py':'源码','.txt':'文本 / 源码','.yaml':'规则 / 目录','.json':'结果 JSON','.md':'实验报告'}.get(path.suffix.lower() if path else '', '归档文本')
        status = asset.get('usage_status','未登记')
        status_label = {'metadata_or_result':'元数据 / 已归档结果','used':'已使用','unused':'尚未使用'}.get(status,status)
        origin = asset.get('origin') or ('自制 / 验收样例' if '自制' in book.title(first) else '规则 / 自制目录摘要' if path and path.suffix=='.yaml' else '随 trace 归档的来源')
        links = ' '.join(book.link(c) for c in citations)
        grouped[(modality,status_label)].append(f'<article class="asset-card"><span class="eyebrow">{_html(origin)}</span><h4>{_html(path.name if path else book.title(first))}</h4><p>{_html(relative or "trace 内嵌文本")}</p><p class="subtle">{len(citations)} 个引用位置 · {_html(asset.get("license","许可见原资料记录"))}</p><div class="asset-links">{links}</div></article>')
    return ''.join(f'<details class="asset-group" open><summary>{_html(modality)} <span class="tag">使用登记：{_html(status)}</span><span class="count">{len(cards)}</span></summary><div class="asset-grid">{"".join(cards)}</div></details>' for (modality,status),cards in sorted(grouped.items()))


_REPORT_CSS = '''
:root{color-scheme:light;--ink:#17343c;--muted:#52696f;--line:#dce5e3;--teal:#147d80;--orange:#ad4d2a}*{box-sizing:border-box}html{scroll-behavior:smooth;scroll-padding-top:85px}body{margin:0;background:#f3f6f4;color:var(--ink);font:15px/1.7 "Segoe UI","Microsoft YaHei","PingFang SC","Noto Sans CJK SC",sans-serif}a{color:#14686d;text-decoration-thickness:1px;text-underline-offset:3px}a:focus-visible,button:focus-visible,summary:focus-visible{outline:3px solid #d18339;outline-offset:3px}button{font:inherit;cursor:pointer}header{background:#153b43;color:#fff;padding:34px max(24px,calc((100vw - 1260px)/2)) 32px}.brand{letter-spacing:.14em;font-size:12px;color:#b8d7d5;display:flex;justify-content:space-between;gap:20px}h1{font-size:38px;letter-spacing:-1px;line-height:1.3;margin:18px 0 12px}header p{max-width:750px;color:#d1e0df;margin:0}.header-stats{display:flex;gap:32px;margin-top:24px}.header-stats b{font-size:27px;margin-right:9px;color:#e3c39b}.header-stats span{font-size:13px;color:#d4e1e0}.topnav{position:sticky;top:0;z-index:3;background:#ffffffed;border-bottom:1px solid var(--line);display:flex;gap:24px;padding:14px max(24px,calc((100vw - 1260px)/2));flex-wrap:wrap}.topnav a{text-decoration:none;font-weight:600;font-size:14px}.layout{max-width:1308px;margin:auto;padding:28px 24px;display:grid;grid-template-columns:235px minmax(0,1fr);gap:26px}aside{align-self:start;position:sticky;top:85px}aside h2{font-size:12px;letter-spacing:.12em;color:var(--muted);margin:0 0 12px}.question-tab{display:block;width:100%;text-align:left;border:1px solid transparent;border-radius:8px;background:none;padding:14px 12px;margin-bottom:7px;color:var(--muted);text-decoration:none}.question-tab b{display:block;font-size:15px;color:var(--ink)}.question-tab small{display:block;margin-top:4px}.question-tab[aria-current=true]{background:#e0eeeb;border-color:#bad3cf;box-shadow:inset 3px 0 var(--teal)}main{min-width:0}.panel,.section-card{background:white;border:1px solid var(--line);border-radius:12px;padding:26px;margin-bottom:24px}.eyebrow{text-transform:uppercase;letter-spacing:.12em;font-size:11px;font-weight:700;color:var(--teal)}h2{font-size:23px;line-height:1.4;margin:10px 0 18px}h3{font-size:17px;margin:22px 0 12px}h4{font-size:15px;margin:8px 0}.subtle,small{color:var(--muted)}.decision-state{display:inline-block;background:#fff0df;color:#82511c;border:1px solid #ecd2ae;padding:4px 12px;border-radius:20px;font-size:13px;font-weight:700}.conclusion{white-space:pre-line;font-size:16px;line-height:1.9;border-left:3px solid #cf9157;padding:4px 0 4px 17px;margin:20px 0}.next-step{background:#edf5f2;padding:15px 18px;border-radius:8px}.next-step h3{margin:0 0 5px}.next-step ul{margin:0;padding-left:20px}.tag{font-size:11px;display:inline-block;border:1px solid #d2dfdc;padding:2px 7px;border-radius:4px;color:var(--muted);font-weight:400}.trace-meta{font-size:12px;margin:18px 0;overflow-wrap:anywhere}.trace-meta code{display:block}.actions{display:flex;gap:12px;flex-wrap:wrap;align-items:center}.actions button{background:#174b51;color:white;border:0;border-radius:6px;padding:9px 14px}.handoff{white-space:pre-wrap;font-size:12px;background:#f2f5f4;padding:12px;max-height:210px;overflow:auto}.branch{border-left:2px solid #cee0dc;padding-left:16px;margin-top:16px}.branch>summary{font-size:17px}.detail-body{padding:12px 0 4px}details>summary{cursor:pointer;overflow-wrap:anywhere}summary::marker{color:#49888a}details[open]>summary{margin-bottom:8px}.nested{border:1px solid var(--line);padding:10px 12px;border-radius:6px;margin:8px 0}.evidence-link{display:inline-block;background:#eaf3f1;padding:2px 7px;border-radius:4px;margin:3px 3px 3px 0;font-size:12px;overflow-wrap:anywhere}.data-fields{margin:0}.data-row{display:grid;grid-template-columns:minmax(90px,25%) minmax(0,1fr);border-bottom:1px solid #edf1f0;padding:7px 0;gap:12px}.data-row dt{font-size:12px;color:var(--muted)}.data-row dd{margin:0;font-size:13px;overflow-wrap:anywhere}.graph-edge{display:grid;grid-template-columns:minmax(0,1fr) minmax(95px,.8fr) minmax(0,1fr);gap:6px;padding:12px;margin:8px 0;border:1px solid var(--line);border-radius:7px}.graph-node{border-left:4px solid var(--node);padding:7px 10px;background:#f5f8f7;overflow-wrap:anywhere}.graph-node small,.graph-node b{display:block}.graph-node small{font-size:10px;color:var(--node)}.graph-node b{font-size:13px}.graph-relation{text-align:center;align-self:center;overflow-wrap:anywhere;font-size:12px}.graph-relation span,.graph-relation small{display:block}.graph-relation span:nth-child(2){font-size:22px;line-height:1}.edge-cites{grid-column:1/-1}.graph-edge.conflict{border:2px solid #c05a42;background:#fff5ef}.graph-edge.conflict .graph-relation{color:#a73f2b}.notice{background:#fff2e8;border-left:3px solid #bd6944;padding:12px;font-size:13px}.asset-group{margin:16px 0}.asset-group>summary{font-weight:700}.count{float:right;border-radius:20px;background:#e6efed;padding:0 9px}.asset-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:12px}.asset-card{border:1px solid var(--line);border-radius:8px;padding:16px;min-width:0}.asset-card p{font-size:12px;overflow-wrap:anywhere;margin:7px 0}.asset-links{max-height:120px;overflow:auto}.chart-row{display:grid;grid-template-columns:90px minmax(0,1fr) 40px;gap:12px;align-items:center;margin:12px 0;font-size:13px}.bar-track{height:22px;background:#eff3f1;border-radius:4px;overflow:hidden}.bar{height:100%;background:#317f80}.chart-row:nth-child(2) .bar{background:#b98542}.chart-row:nth-child(3) .bar{background:#819397}.evidence-card{border:1px solid var(--line);padding:14px 16px;border-radius:8px;margin:10px 0;scroll-margin-top:85px}.evidence-card:target{border:2px solid #a87639;background:#fffdf7}.evidence-card>summary{display:flex;gap:12px;font-size:13px}.evidence-card>summary>b{color:var(--teal);flex:none}.evidence-card>summary small{display:block;margin-top:4px}blockquote{white-space:pre-wrap;overflow-wrap:anywhere;margin:8px 0 18px;padding:12px 15px;background:#f0f5f3;border-left:3px solid #77a49c;font-size:14px}figure{margin:15px 0}img{max-width:100%;height:auto;border:1px solid var(--line)}figcaption{font-size:12px;color:var(--muted)}pre{white-space:pre-wrap;overflow-wrap:anywhere;font:12px/1.7 Consolas,monospace}.source-lines{background:#173b43;color:#e4f0eb;padding:14px;border-radius:6px}.source-id{font-size:11px;overflow-wrap:anywhere;color:var(--muted)}table{width:100%;border-collapse:collapse;font-size:13px}td,th{text-align:left;padding:9px;border:1px solid var(--line);overflow-wrap:anywhere}mark{background:#f7e7c8;padding:2px 4px}.section-card p,.panel p,li{overflow-wrap:anywhere}.footer{font-size:12px;color:var(--muted);padding:0 8px 30px}.section-heading{display:flex;align-items:baseline;justify-content:space-between;gap:15px;flex-wrap:wrap}[hidden]{display:none!important}@media(max-width:700px){header{padding:25px 18px}h1{font-size:29px}.brand{font-size:10px}.header-stats{gap:15px}.header-stats b{font-size:23px}.header-stats span{font-size:11px}.topnav{gap:16px;padding:12px 18px;font-size:12px}.layout{display:block;padding:18px 14px}aside{position:static;margin-bottom:18px}.question-list{display:grid;grid-template-columns:1fr 1fr;gap:6px}.question-tab{margin:0;padding:10px}.question-tab b{font-size:13px}.question-tab small{font-size:11px}.panel,.section-card{padding:18px 15px}h2{font-size:21px}.conclusion{font-size:14px;padding-left:12px}.graph-edge{grid-template-columns:minmax(0,1fr);padding:10px}.graph-relation span:nth-child(2){transform:rotate(90deg)}.graph-node{padding:6px 9px}.edge-cites{grid-column:auto}.data-row{grid-template-columns:1fr;gap:3px}.branch{padding-left:10px}.asset-grid{grid-template-columns:1fr}.chart-row{grid-template-columns:70px minmax(0,1fr) 30px;gap:8px}.evidence-card{padding:12px}.evidence-card>summary{gap:8px}.source-lines{font-size:11px}}@media print{aside,.topnav,.actions{display:none}.layout{display:block}header{background:white;color:#17343c}.panel[hidden]{display:block!important}.section-card,.panel{break-inside:avoid}details>*{display:block!important}}
'''

_REPORT_JS = '''
function revealTarget(){let id;try{id=decodeURIComponent(location.hash.slice(1))}catch(e){return;}const target=document.getElementById(id);if(!target)return;let panel=target.closest('.decision-panel');if(panel)selectQuestion(panel.id);let node=target;while(node){if(node.tagName==='DETAILS')node.open=true;node=node.parentElement}requestAnimationFrame(()=>target.scrollIntoView({block:'start'}))}
function selectQuestion(id){document.querySelectorAll('.decision-panel').forEach(p=>p.hidden=p.id!==id);document.querySelectorAll('.question-tab').forEach(a=>a.setAttribute('aria-current',String(a.hash==='#'+id)))}
document.querySelectorAll('.question-tab').forEach(a=>a.addEventListener('click',()=>selectQuestion(a.hash.slice(1))));
document.querySelectorAll('[data-copy]').forEach(b=>b.addEventListener('click',async()=>{const text=document.getElementById(b.dataset.copy).textContent;try{await navigator.clipboard.writeText(text);b.textContent='已复制 trace 与报告位置'}catch(e){document.getElementById(b.dataset.copy).closest('details').open=true;const range=document.createRange();range.selectNodeContents(document.getElementById(b.dataset.copy));const selection=getSelection();selection.removeAllRanges();selection.addRange(range);b.textContent='已选中，请复制下方文字'}}));
window.addEventListener('hashchange',revealTarget);selectQuestion('question-1');revealTarget();
'''


def render_decisions(traces, *, root=_REPORT_ROOT, ontology_summary=None, seed=None, catalog=None, report_path='decision-report.html'):
    """Return a self-contained offline report and its generation receipt."""
    if not traces:
        raise ValueError('At least one archived trace is required')
    book = _EvidenceBook(traces, root)
    titles = ['两杆性质比较','仿真输入可用性','标准方法对齐','直径冲突处置','权限与调参边界']
    tabs, panels, handoffs = [], [], []
    try:
        for i, trace in enumerate(traces,1):
            decision,reasoning = trace['decision'],trace['reasoning']
            title = titles[i-1] if i<=len(titles) else f'决策 {i}'
            tabs.append(f'<a class="question-tab" href="#question-{i}"><b>0{i}　{_html(title)}</b><small>{_html(decision["confidence_level"])}</small></a>')
            links = ' '.join(book.link(c) for c in _citations(trace['rule']))
            numeric = _render_value(reasoning.get('numerical_tools',{}),book)
            acquisition = _render_value(reasoning.get('acquisition_links',[]),book,'测量与模型链路')
            graph = _render_graph(trace,book,seed or {})
            retrieval = ''.join(f'<li>{book.link(c)} {_html(c.get("title",c["source_id"]))} <small>检索通道：{_html(" / ".join(c.get("ranks",{})))}</small></li>' for c in trace.get('retrieval',[]))
            standard = _render_value(reasoning.get('standard_align',[]),book,'标准目录对齐')
            next_steps = ''.join(f'<li>{_html(s)}</li>' for s in decision.get('next_experiments',[])) or '<li>本条未提出新实验。</li>'
            gaps = '；'.join(decision.get('gaps',[])) or '本条未登记其他缺口'
            handoff = {'trace_id':trace['trace_id'],'report_path':str(report_path),'anchor':f'question-{i}',
                       'answer_text':f"{decision['conclusion']}\n结论等级：{decision['confidence_level']}\ntrace_id：{trace['trace_id']}\n报告：{report_path}#question-{i}"}
            handoffs.append(handoff)
            panels.append(f'''<article class="panel decision-panel" id="question-{i}"><span class="eyebrow">Decision 0{i} / 归档决策</span><h2>{_html(trace['question'])}</h2><span class="decision-state">{_html(decision['confidence_level'])}</span><div class="conclusion">{_html(decision['conclusion'])}</div><div class="next-step"><h3>下一步行动</h3><ul>{next_steps}</ul><p class="subtle">证据缺口：{_html(gaps)}</p></div><div class="trace-meta">trace_id<code>{_html(trace['trace_id'])}</code>本体 / 图谱版本 {_html(trace['graph_version'])} · 全案待审核 {reasoning.get('pending_review_count',0)} 项</div><div class="actions"><button data-copy="handoff-{i}">复制答复与溯源位置</button><a href="#evidence">打开原文证据库</a></div><details><summary>用于 Nexent 答复的文本</summary><pre class="handoff" id="handoff-{i}">{_html(handoff['answer_text'])}</pre></details><h3>从结论回到证据</h3><details class="branch" open><summary>01　决策规则 · {_html(trace['rule'].get('id',''))}</summary><div class="detail-body">{links}<p>{_html(trace['rule'].get('evidence',{}).get('quote',''))}</p></div></details><details class="branch"><summary>02　模型与测量记录</summary><p class="subtle">原样展示归档值，未重新拟合或评分；比例字段保持原单位。</p>{numeric}{acquisition}</details><details class="branch" open><summary>03　图谱路径与冲突</summary><p class="subtle">颜色对应种子本体类型；红框表示冲突，审核状态不等于已确认事实。以下是本问题引用的子图。</p>{graph}</details><details class="branch"><summary>04　检索命中与标准目录</summary><ul>{retrieval or '<li>本问题依据权限规则，未调用文档检索。</li>'}</ul>{standard}</details><details class="branch"><summary>归档事件标识</summary><p class="source-id">{_html(trace.get('event_sha256','未提供'))}</p><p class="subtle">本页核对 trace 内容摘要；未重新认证完整 SQLite 事件链。</p></details></article>''')
        assets = _render_assets(book,catalog or {})
        stats = ontology_summary or {}
        tiers = stats.get('tiers',{})
        total = stats.get('candidates',0)
        bars = ''.join(f'<div class="chart-row"><span>{_LABELS.get(k,k)}</span><div class="bar-track"><div class="bar" style="width:{100*v/max(total,1):.3f}%"></div></div><b>{v}</b></div>' for k,v in tiers.items())
        evidence = book.render()
        html = f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta http-equiv="Content-Security-Policy" content="default-src 'none'; img-src data:; style-src 'unsafe-inline'; script-src 'unsafe-inline'; base-uri 'none'; form-action 'none'"><title>增材制造 · 决策证据报告</title><style>{_REPORT_CSS}</style></head><body><header><div class="brand"><span>AM RESEARCH / EVIDENCE REVIEW</span><span>离线决策档案 · T07</span></div><h1>每一个判断，都有据可查。</h1><p>从两根增材杆的测量记录，追溯到模型、图谱关系与原文。保留证据不足、冲突和审核边界，帮助判断下一步实验。</p><div class="header-stats"><div><b>{len(traces):02}</b><span>决策问题</span></div><div><b>{len(book.records)}</b><span>证据位置</span></div><div><b>{len(book.aliases)}</b><span>来源标识</span></div></div></header><nav class="topnav" aria-label="报告章节"><a href="#question-1">决策审阅</a><a href="#assets">资产地图</a><a href="#ontology">本体统计</a><a href="#evidence">原文证据</a></nav><div class="layout"><aside><h2>五个决策问题</h2><nav class="question-list" aria-label="选择决策问题">{''.join(tabs)}</nav><p class="subtle">固定输入 decision_v2<br>第 0 问待后续合并补充</p></aside><main>{''.join(panels)}<section class="section-card" id="assets"><span class="eyebrow">Asset map</span><h2>本次决策引用了哪些资产</h2><p class="subtle">按模态与已有使用登记分组。沉睡度尚无量化定义，不生成分数；未登记不表示从未使用。不同 source_id 也不必然代表不同文件。</p>{assets}</section><section class="section-card" id="ontology"><span class="eyebrow">Ontology / T04</span><h2>本体构建统计与评测状态</h2><p>候选 {total} 条 · 关系 {len(stats.get('relations',{}))} 种 · 冲突候选 {stats.get('conflicts','未提供')} 条</p>{bars or '<p>未提供 T04 统计。</p>'}<div class="notice">人工准确率、召回率、一致性与工时收益：待评测。图表表示抽取分流数量，不是准确率。</div><h3>候选分数分布（启发式，未校准）</h3>{''.join(f'<div class="chart-row"><span>{_html(k)}–{float(k)+.1:.1f}</span><div class="bar-track"><div class="bar" style="width:{100*v/max(total,1):.3f}%"></div></div><b>{v}</b></div>' for k,v in sorted(stats.get('score_histogram',{}).items()))}<p class="source-id">来源：competition_agent/verification/ontology_induction/v2/summary.json。与每条 trace 中全案待审核数是不同快照，不能相加。</p></section><section class="section-card" id="evidence"><div class="section-heading"><div><span class="eyebrow">Source evidence</span><h2>原文与精确位置</h2></div><span class="tag">全部内嵌 · 可离线打开</span></div><p class="subtle">点击 E 编号展开对应原文。PDF 为归档坐标裁剪，表格保留工作表与单元格，源码显示原始行号。引文始终来自固定 trace。</p>{evidence}</section><section class="section-card"><h2>适用范围与限制</h2><p>本页展示五条已归档决策，不重新调用模型、拟合或打开新的留出评分。Q1 的参数差异不等于材料性质差异。Q4 中 7.9 mm 资料是明确标注的自制验收样例。</p><p>论文与测量资料来自 Lauren Tomanek、Daniel Stutts 的 Mendeley 数据集，DOI 10.17632/sbcf36976g.1，CC BY 4.0；模型结果、规则与目录摘要为项目派生产物。未复制标准正文。</p><p>Nexent 平台内跳转与答复仍待 T08 实测。复制按钮及 handoffs.json 提供 trace_id、报告路径与锚点；本页不更改平台运行器。</p></section><footer class="footer">PDE 资产决策多智能体 · 证据审阅报告 / 来源、结论与待办均可回溯</footer></main></div><script>{_REPORT_JS}</script></body></html>'''
        return html, {'trace_count':len(traces),'evidence_count':len(book.records),'source_id_count':len({c['source_id'] for c in book.records.values()}), 'previews':dict(book.preview_stats),'handoffs':handoffs}
    finally:
        book.close()


def save_decisions(output, *, trace_dir=None, root=_REPORT_ROOT):
    root = Path(root).resolve()
    trace_dir = Path(trace_dir or root/'competition_agent/verification/decision_v2')
    # This release intentionally consumes the five archived questions only.
    paths = [trace_dir/f'trace-{i}.json' for i in range(1,6)]
    traces = [json.loads(p.read_text(encoding='utf-8')) for p in paths]
    for trace in traces:
        payload = {k:v for k,v in trace.items() if k not in ('trace_id','event_sha256')}
        expected = trace['case_id'] + '.' + hashlib.sha256(_json(payload).encode('utf-8')).hexdigest()
        if trace['trace_id'] != expected:
            raise ValueError('Archived trace digest mismatch')
    def read(relative):
        return json.loads((root/relative).read_text(encoding='utf-8'))
    output = Path(output).resolve()
    html, receipt = render_decisions(traces,root=root,
        ontology_summary=read('competition_agent/verification/ontology_induction/v2/summary.json'),
        seed=read('competition_agent/ontology/seed.yaml'),
        catalog=read('competition_agent/scenario_am/asset_catalog.json'), report_path=str(output))
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(html,encoding='utf-8')
    receipt['inputs'] = {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    receipt['html_sha256'] = hashlib.sha256(output.read_bytes()).hexdigest()
    (output.parent/'handoffs.json').write_text(json.dumps(receipt['handoffs'],ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (output.parent/'generation.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='生成五问离线决策溯源报告，不运行模型或实验')
    parser.add_argument('--output',default=str(_REPORT_ROOT/'competition_agent/deliverables/T07/decision-report.html'))
    parser.add_argument('--trace-dir')
    args = parser.parse_args()
    print(json.dumps(save_decisions(args.output,trace_dir=args.trace_dir),ensure_ascii=False,indent=2))
