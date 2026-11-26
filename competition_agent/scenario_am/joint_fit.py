"""Preregistered retrospective shared-D diagnostics; no new holdout scores."""
import hashlib
import json
import subprocess
from datetime import datetime, timezone

import numpy as np
from scipy.optimize import least_squares

from pde_discovery.real_experiment import load_measurements
from pde_discovery.real_heat_refined import _prepare, _advance
from .catalog import ROOT, evidence

DIRECTORY = ROOT / 'competition_agent/verification/joint_fit'
PROTOCOL = DIRECTORY / 'protocol.json'
PROTOCOL_COMMIT = '7a5b8e122d65f643d8b3b3fae8bf87e32ad9dda2'


def frozen_protocol():
    relative = PROTOCOL.relative_to(ROOT).as_posix()
    committed = subprocess.check_output(['git', 'show', f'{PROTOCOL_COMMIT}:{relative}'], cwd=ROOT,
        stdin=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=30)
    current = PROTOCOL.read_bytes().replace(b'\r\n', b'\n')
    if current != committed.replace(b'\r\n', b'\n'):
        raise ValueError('预注册协议已改变，停止计算')
    subprocess.run(['git', 'merge-base', '--is-ancestor', PROTOCOL_COMMIT, 'HEAD'], cwd=ROOT, check=True,
        stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30)
    return json.loads(current), hashlib.sha256(current).hexdigest()


def fit_records(records, protocol, fixed_D=None):
    """Equal record MSE, unchanged original four-interior forward propagator."""
    if not records:
        raise ValueError('至少需要一条记录')
    n = len(records)
    prepared, targets = [], []
    for r in records:
        t, u = np.asarray(r['t']), np.asarray(r['u'])
        if len(t) < 8 or u.shape != (len(t), 6) or not np.isfinite(u).all():
            raise ValueError('需要至少八帧有限六测点温升')
        prepared.append(_prepare(t, u[:, [0,5]], u[0,1:5], r['dx']))
        targets.append(u[1:,1:5])
    bounds = protocol['bounds']
    offset = int(fixed_D is None)
    if fixed_D is not None and not bounds['D'][0] <= fixed_D <= bounds['D'][1]:
        raise ValueError('固定 D 越界')
    lower = ([np.log(bounds['D'][0])] if offset else []) + [bounds['h'][0]/.01, bounds['c'][0]/.1]*n
    upper = ([np.log(bounds['D'][1])] if offset else []) + [bounds['h'][1]/.01, bounds['c'][1]/.1]*n
    lower, upper = np.array(lower), np.array(upper)
    def decode(x):
        D = float(np.exp(x[0])) if offset else float(fixed_D)
        hc = x[offset:].reshape(n,2)*[.01,.1]
        return D, hc
    def residual(x):
        D, hc = decode(x)
        return np.concatenate([(_advance(p,D,*pair)[1:]-target).ravel()/np.sqrt(n*target.size)
                               for p,pair,target in zip(prepared,hc,targets)])
    runs, solutions = [], []
    opt = protocol['optimizer']
    for i,(D,h,c) in enumerate(opt['starts_D_h_c']):
        initial = ([np.log(D)] if offset else []) + [h/.01,c/.1]*n
        try:
            result = least_squares(residual,initial,bounds=(lower,upper),
                **{k:opt[k] for k in ('method','jac','loss','max_nfev','ftol','xtol','gtol')})
            rmse = float(np.linalg.norm(result.fun))
            fitted_D,hc = decode(result.x)
            finite = bool(np.isfinite(rmse) and np.isfinite(result.x).all())
            row = dict(start_index=i,initial_D_h_c=[fixed_D if fixed_D is not None else D,h,c],
                       success=bool(result.success and finite),status=int(result.status),message=str(result.message),
                       nfev=int(result.nfev),rmse_K=rmse if finite else None,
                       D=fitted_D if finite else None,h_c=hc.tolist() if finite else None)
            solutions.append(result)
        except (ValueError,RuntimeError,FloatingPointError,OverflowError,np.linalg.LinAlgError) as error:
            row = dict(start_index=i,success=False,message=str(error),rmse_K=None)
            solutions.append(None)
        runs.append(row)
    good = [i for i,r in enumerate(runs) if r['success']]
    if not good:
        return dict(success=False,starts=runs,D=None,rmse_K=None,reason='全部固定起点失败')
    selected = min(good,key=lambda i:(runs[i]['rmse_K'],i))
    best = solutions[selected]
    D,hc = decode(best.x)
    names = (['D'] if offset else []) + [f'{r["id"]}.{k}' for r in records for k in ('h','c')]
    hits = [dict(parameter=name,bound='lower' if abs(v-lo)<=1e-5*(hi-lo) else 'upper')
            for name,v,lo,hi in zip(names,best.x,lower,upper)
            if min(abs(v-lo),abs(v-hi))<=1e-5*(hi-lo)]
    return dict(success=True,D=D,parameters={r['id']:dict(h=float(pair[0]),c=float(pair[1]))
                for r,pair in zip(records,hc)},rmse_K=runs[selected]['rmse_K'],starts=runs,
                selected_start=selected,bounds_hit=hits,record_ids=[r['id'] for r in records])


def profile_records(records, fit, protocol):
    if not fit['success']:
        return dict(complete=False,points=[],near_D=[],ratio=None)
    grid, mapping = {}, []
    for multiplier in protocol['profile']['multipliers']:
        requested = fit['D']*multiplier
        D = float(np.clip(requested,*protocol['bounds']['D']))
        mapping.append(dict(multiplier=multiplier,requested_D=requested,D=D,clipped=D!=requested))
        if D not in grid:
            grid[D] = fit_records(records,protocol,fixed_D=D)
    points = [dict(fixed_D=D,**row) for D,row in grid.items()]
    good = [p for p in points if p['success']]
    minimum = min((p['rmse_K'] for p in good),default=None)
    near = [p['fixed_D'] for p in good if p['rmse_K']<=minimum*protocol['profile']['near_optimal_factor']]
    return dict(complete=all(p['success'] for p in points),mapping=mapping,points=points,
                min_rmse_K=minimum,near_D=near,ratio=max(near)/min(near) if near else None,
                touches_grid_edge=bool(near and (min(grid) in near or max(grid) in near)),
                touches_parameter_bound=any(d in protocol['bounds']['D'] for d in near))


def rod_decision(fit, profile, loro, protocol):
    complete = fit['success'] and profile['complete'] and len(loro)==5 and all(r['success'] for r in loro)
    values = [r['D'] for r in loro if r['success']]
    ratio = max(values)/min(values) if values else None
    limits = protocol['decision']
    return dict(complete=bool(complete),profile_ratio=profile['ratio'],loro_ratio=ratio,
                identifiable=bool(complete and profile['ratio'] is not None
                    and profile['ratio']<=limits['profile_ratio_max'] and ratio<=limits['loro_ratio_max']))


def compare_rods(rods):
    a,b = rods['xp'],rods['xn']
    values = [a['joint']['D'],b['joint']['D']]
    ratio = max(values)/min(values) if all(v is not None and v>0 for v in values) else None
    spreads = [a['decision']['loro_ratio'],b['decision']['loro_ratio']]
    product = spreads[0]*spreads[1] if all(v is not None for v in spreads) else None
    return dict(D_ratio=ratio,loro_ratio_product=product,distinguishable=bool(
        a['decision']['identifiable'] and b['decision']['identifiable'] and ratio is not None and ratio>product))


def heat_joint_fit(group):
    protocol,digest = frozen_protocol()  # Must precede temperature reads and optimization.
    if group not in protocol['groups']:
        raise ValueError('group 必须为 xn/xp/yn/yp')
    manifest_path = ROOT / protocol['data']['root'] / group / 'manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    entries = list(manifest['splits'].values())+manifest['holdouts']
    entries.sort(key=lambda r:r['repeat'])
    if [r['repeat'] for r in entries] != protocol['records_per_group'] or manifest['spacing_m']!=protocol['data']['spacing_m']:
        raise ValueError('记录集合或测点间距不符合协议')
    records, inputs = [], []
    for entry in entries:
        path = (manifest_path.parent/entry['file']).resolve()
        if hashlib.sha256(path.read_bytes()).hexdigest()!=entry['sha256']:
            raise ValueError('原始输入摘要不匹配')
        t,u = load_measurements(manifest_path.parent,entry,manifest['csv_format'])
        records.append(dict(id=group+str(entry['repeat']),t=t,u=u,dx=manifest['spacing_m']))
        inputs.append(dict(id=records[-1]['id'],path=path.relative_to(ROOT).as_posix(),sha256=entry['sha256'],frames=len(t)))
    joint = fit_records(records,protocol)
    profile = profile_records(records,joint,protocol)
    loro = [dict(omitted=r['id'],**fit_records([x for x in records if x['id']!=r['id']],protocol)) for r in records]
    singles = []
    for record in records:
        fit = fit_records([record],protocol)
        singles.append(dict(id=record['id'],fit=fit,profile=profile_records([record],fit,protocol)))
    return dict(group=group,protocol_commit=PROTOCOL_COMMIT,protocol_sha256=digest,
                scope=protocol['scope'],inputs=inputs,joint=joint,profile=profile,loro=loro,
                single_records=singles,decision=rod_decision(joint,profile,loro,protocol),
                evidence_paths=[PROTOCOL.relative_to(ROOT).as_posix(),manifest_path.relative_to(ROOT).as_posix()])


def read_joint_evidence():
    """Decision service reads archived diagnostics, never starts fitting."""
    _,digest = frozen_protocol()
    path = DIRECTORY/'results.json'
    if not path.exists():
        return dict(available=False,reason='联合拟合尚未归档',distinguishable=False)
    result = json.loads(path.read_text(encoding='utf-8'))
    if result['protocol_sha256']!=digest or result['protocol_commit']!=PROTOCOL_COMMIT:
        raise ValueError('联合拟合结果协议不匹配')
    protocol,_ = frozen_protocol()
    for rod in result['rods'].values():
        points = rod['profile']['points']
        good = [p for p in points if p['success']]
        minimum = min((p['rmse_K'] for p in good),default=None)
        near = [p['fixed_D'] for p in good if p['rmse_K']<=minimum*protocol['profile']['near_optimal_factor']]
        ratio = max(near)/min(near) if near else None
        if (near!=rod['profile']['near_D'] or ratio!=rod['profile']['ratio']
                or rod['profile']['complete']!=all(p['success'] for p in points)):
            raise ValueError('剖面汇总与原始曲线不一致')
        if rod_decision(rod['joint'],rod['profile'],rod['loro'],protocol)!=rod['decision']:
            raise ValueError('联合拟合判定与冻结协议不一致')
    comparison = compare_rods(result['rods'])
    if comparison!=result['comparison']:
        raise ValueError('跨杆判定不一致')
    result['available'] = True
    result['evidence'] = evidence(path, '"comparison": '+json.dumps(result['comparison'],ensure_ascii=False))
    result['rod_evidence'] = {}
    for group,rod in result['rods'].items():
        source = DIRECTORY/f'{group}.json'
        fields = {'D':rod['joint']['D'],**rod['decision']}
        result['rod_evidence'][group] = [evidence(source, json.dumps(k)+': '+json.dumps(v))
                                         for k,v in fields.items()]
    return result


def run():
    protocol,digest = frozen_protocol()
    result = dict(protocol_commit=PROTOCOL_COMMIT,protocol_sha256=digest,
                  started_at=datetime.now(timezone.utc).isoformat(),scope=protocol['scope'],rods={})
    for group in protocol['groups']:
        print('开始 '+group,flush=True)
        rod = heat_joint_fit(group)
        result['rods'][group]=rod
        (DIRECTORY/f'{group}.json').write_text(json.dumps(rod,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
        print(group+' '+json.dumps(rod['decision']),flush=True)
    result['comparison']=compare_rods(result['rods'])
    result['finished_at']=datetime.now(timezone.utc).isoformat()
    # Single-line top-level values give citations a compact, exact original line.
    text='{\n'+',\n'.join('  '+json.dumps(k)+': '+json.dumps(v,ensure_ascii=False,allow_nan=False) for k,v in result.items())+'\n}\n'
    (DIRECTORY/'results.json').write_text(text,encoding='utf-8')
    print(json.dumps(result['comparison']),flush=True)


if __name__=='__main__':
    run()
