"""Render T05d archived diagnostics only; never run an optimizer."""
import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from competition_agent.scenario_am.joint_fit import DIRECTORY, read_joint_evidence


def render():
    result=read_joint_evidence()
    if not result['available']:
        raise ValueError('尚无归档结果')
    comparison=result['comparison']
    lines=['# 同杆五次记录联合拟合：预注册可识别性分析','',
        '结论：'+('xp 与 xn 满足预注册的有效 D 可辨别标准。' if comparison['distinguishable'] else 'xp 与 xn 不满足预注册的有效 D 可辨别标准，维持证据不足。'),
        '这是同一装置、同一模型下的有效参数结论，不等于材料性质差异或材料证书。','',
        '## 冻结与来源','',
        f"协议先行提交：`{result['protocol_commit']}`。",
        f"协议 SHA-256（LF 规范化）：`{result['protocol_sha256']}`。",
        f"计算开始 {result['started_at']}；结束 {result['finished_at']}。",
        '参数界限、三个起点、九点倍率、等权逐记录损失、五折 LORO 以及门槛均按 protocol.json 执行，未事后改动。',
        '输入：四杆各五条完整 CSV，清单及原始字节摘要见各杆 JSON 的 inputs；数据来源 Lauren Tomanek、Daniel Stutts，Mendeley DOI 10.17632/sbcf36976g.1，CC BY 4.0。',
        '传播器直接调用 pde_discovery.real_heat_refined 的六测点离散实现；只读原始时间、实测边界、首帧四内点，损失不计首帧。','',
        '## 四杆完整结果','',
        '| 杆 | 联合 D* (m²/s) | 联合 RMSE (K) | 剖面近优倍数 | LORO 倍数 | 预注册 D 可识别 | 单记录近优倍数（1–5） |',
        '|---|---:|---:|---:|---:|---|---|']
    def number(x):return '未获得' if x is None else f'{x:.12g}'
    for group,rod in result['rods'].items():
        d=rod['decision']
        lines.append('| '+' | '.join([group,number(rod['joint']['D']),number(rod['joint']['rmse_K']),
            number(d['profile_ratio']),number(d['loro_ratio']),'是' if d['identifiable'] else '否',
            ', '.join(number(s['profile']['ratio']) for s in rod['single_records'])])+' |')
    lines+=['',f"xp/xn 较大 D* / 较小 D* = {number(comparison['D_ratio'])}；两杆 LORO 倍数之积 = {number(comparison['loro_ratio_product'])}。",
        '判定：两杆均须剖面倍数 ≤1.5 且 LORO 倍数 ≤1.3，跨杆比值须严格超过倍数之积。其他杆负结果也完整保留。','',
        '与单记录剖面对比：xp、yp 的五条单记录本来就只有一个采样近优点，联合拟合没有进一步降低其离散倍数；yn 的联合倍数与五条单记录相同，仍未达到 1.5 门槛。xn 联合倍数 1.25，小于第二条记录的 1.5625，但不小于第一、第五条的 1。因此结果不支持“联合拟合普遍解决参数补偿”的强说法。','']
    total_starts=failed_starts=0
    for group,rod in result['rods'].items():
        lines += [f'## {group} 详细参数与诊断','',f"全精度记录：[{group}.json]({group}.json)。",'',
                  '| 记录 | h_k (s⁻¹) | c_k (K/s) | 留去该记录后的 D (m²/s) |',
                  '|---|---:|---:|---:|']
        for fold in rod['loro']:
            pair=rod['joint'].get('parameters',{}).get(fold['omitted'],{})
            lines.append(f"| {fold['omitted']} | {number(pair.get('h'))} | {number(pair.get('c'))} | {number(fold.get('D'))} |")
        lines += ['', '| 固定 D (m²/s) | 重拟合 RMSE (K) | 近优点 | 成功 |','|---|---:|---|---|']
        for point in rod['profile']['points']:
            lines.append(f"| {number(point['fixed_D'])} | {number(point['rmse_K'])} | {'是' if point['fixed_D'] in rod['profile']['near_D'] else '否'} | {point['success']} |")
        lines += ['', '完整近优集合：'+', '.join(number(v) for v in rod['profile']['near_D'])+'。',
                  f"近优集合接触网格端点：{rod['profile'].get('touches_grid_edge')}；接触参数界限：{rod['profile'].get('touches_parameter_bound')}。",'']
        fits=[rod['joint']]+rod['profile']['points']+rod['loro']
        for single in rod['single_records']:fits += [single['fit']]+single['profile']['points']
        failed=[i for i,f in enumerate(fits) if not f['success']]
        hits=[dict(fit_index=i,bounds_hit=f['bounds_hit']) for i,f in enumerate(fits) if f.get('bounds_hit')]
        for f in fits:
            total_starts+=len(f['starts']);failed_starts+=sum(not x['success'] for x in f['starts'])
        lines += [f'拟合结果数 {len(fits)}；失败结果索引 {failed}；触界结果：`{json.dumps(hits,ensure_ascii=False)}`。',
                  '每条单记录完整剖面、近优集合、裁剪映射及所有起点状态均保存在对应 JSON，不以汇总倍数代替原始曲线。','']
    lines += [f'总起点 {total_starts}；失败起点 {failed_starts}。未删除失败起点，未追加起点。','',
        '## 适用范围与限制','',
        '编号 3–5 已被查看，本实验为回顾性可识别性分析，不是盲测；LORO 只记录四条拟合得到的 D，不评分被留去记录，不产生新的留出评分。',
        '5% 近优范围属于稀疏离散网格上的损失敏感性，不是置信区间；一个近优点不意味着精确识别。共享 D 是模型假设，缩窄范围不能单独证明材料物性唯一。',
        '几何、风速与加热工况仍可能混杂。按 GB/T 22588-2008 闪光法对独立试样复测，并记录工况。',
        '既有研究代码、历史模型、旧评分与训练/开发/留出划分均未修改。Q1 只读取此归档，不在答复时重新拟合。','']
    (DIRECTORY/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')


if __name__=='__main__':render()
