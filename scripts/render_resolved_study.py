"""Saved-result tables and figures; no observations, fitting or scoring."""
from pathlib import Path
import csv,json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'resolved_study/results/first_run'
METHODS=('adaptive_original','weak_original','adaptive_resolved','weak_resolved')
LABELS=('Adaptive / original','Weak / original','Adaptive / refined','Weak / refined')
COLORS=('#8c969e','#bd8932','#509ab5','#007a72')
FAMILIES=('allen_cahn','low_viscosity_burgers','ks_unstable')
FAMILY_LABELS=('Allen–Cahn','Low-viscosity Burgers','KS')


def worst(r):
    return max(c['relative_error'] for c in r['recovery']['coefficient_comparison'] if c['reference']!=0)


def main():
    records=json.loads((OUT/'comparison.json').read_text())['records']
    fields=['id','family','sampling','condition','method','selection_predictor','equation',
            'exact_support','missing_terms','extra_terms','coefficient_vector_error_percent',
            'worst_true_term_error_percent','mean_final_prediction_l2_percent','forecast_failures']
    with (OUT/'summary.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for r in records:
            writer.writerow({**{k:r.get(k) for k in fields[:7]},'exact_support':r['recovery']['exact_support'],
                'missing_terms':';'.join(r['recovery']['missing_terms']),
                'extra_terms':';'.join(r['recovery']['spurious_terms']),
                'coefficient_vector_error_percent':100*r['recovery']['coefficient_relative_l2'],
                'worst_true_term_error_percent':100*worst(r),
                'mean_final_prediction_l2_percent':None if r['mean_clean_relative_l2'] is None else 100*r['mean_clean_relative_l2'],
                'forecast_failures':r['forecast_failures']})
    summaries={}
    for method in METHODS:
        group=[r for r in records if r['method']==method]
        prediction=[r['mean_clean_relative_l2'] for r in group if r['mean_clean_relative_l2'] is not None]
        summaries[method]={'exact':sum(r['recovery']['exact_support'] for r in group),
            'extra':sum(len(r['recovery']['spurious_terms']) for r in group),
            'missing':sum(len(r['recovery']['missing_terms']) for r in group),
            'median_vector_error':float(np.median([r['recovery']['coefficient_relative_l2'] for r in group])),
            'worst_true_term_error':max(worst(r) for r in group),
            'median_prediction_l2':float(np.median(prediction)) if prediction else None,
            'max_prediction_l2':max(prediction) if prediction else None,
            'configs_above10pct':sum(v>.1 for v in prediction),
            'finite_prediction_configs':len(prediction),
            'forecast_failures':sum(r['forecast_failures'] for r in group),
            'exact_by_family':{f:sum(r['recovery']['exact_support'] for r in group if r['family']==f) for f in FAMILIES}}
    (OUT/'summary.json').write_text(json.dumps(summaries,indent=2)+'\n')
    fig,axes=plt.subplots(1,3,figsize=(13,4.5),layout='constrained')
    for ax,family,label in zip(axes,FAMILIES,FAMILY_LABELS):
        counts=[summaries[m]['exact_by_family'][family] for m in METHODS]
        ax.bar(range(4),counts,color=COLORS)
        for i,n in enumerate(counts):ax.text(i,n+.15,f'{n}/10',ha='center')
        ax.set_xticks(range(4),LABELS,rotation=28,ha='right');ax.set_ylim(0,11.4);ax.set_title(label)
    axes[0].set_ylabel('Exact term-set recovery — higher is better')
    fig.suptitle('Fresh four-arm validation | 10 paired conditions per family/method')
    fig.savefig(OUT/'support_recovery.png',dpi=160,bbox_inches='tight',pad_inches=.12);plt.close(fig)
    by={r['id']:r for r in records}
    fig,ax=plt.subplots(figsize=(7,6),layout='constrained')
    for family,label,color in zip(FAMILIES,FAMILY_LABELS,('#7b4ab5','#e18c1b','#008577')):
        pairs=[]
        for r in records:
            if r['method']=='weak_resolved' and r['family']==family:
                b=by[r['id'].removesuffix('weak_resolved')+'adaptive_resolved']
                pairs.append((max(100*worst(b),1e-12),max(100*worst(r),1e-12)))
        ax.scatter(*zip(*pairs),label=label,color=color,s=52,alpha=.8,edgecolors='white')
    limits=(1e-7,300);ax.plot(limits,limits,'--',color='#888',label='Equal error')
    ax.set(xscale='log',yscale='log',xlim=limits,ylim=limits,
        xlabel='Adaptive / refined: worst true-coefficient error (%)',
        ylabel='Weak / refined: worst true-coefficient error (%)',
        title='Coefficient comparison with the same refined predictor\nPoints below diagonal favor weak regression')
    ax.grid(which='major',alpha=.2);ax.legend(loc='upper left')
    fig.savefig(OUT/'coefficient_comparison.png',dpi=160,bbox_inches='tight',pad_inches=.12);plt.close(fig)
    rows=[r for r in records if r['family']=='ks_unstable' and r['sampling']=='coarse']
    conditions=list(dict.fromkeys(r['condition'] for r in rows))
    fig,ax=plt.subplots(figsize=(12,4.5),layout='constrained');largest=0
    for j,method in enumerate(METHODS):
        raw=[next(r['mean_clean_relative_l2'] for r in rows if r['condition']==c and r['method']==method) for c in conditions]
        values=[np.nan if v is None else 100*v for v in raw]
        largest=max([largest,1.,*[v for v in values if np.isfinite(v)]])
        bars=ax.bar(np.arange(5)+(j-1.5)*.19,values,.19,color=COLORS[j],label=LABELS[j])
        ax.bar_label(bars,labels=['failed' if not np.isfinite(v) else f'{v:.1f}' for v in values],padding=2,fontsize=8)
    ax.set_xticks(range(5),['Clean','0.1% / seed53','0.1% / seed89','1% / seed53','1% / seed89'])
    ax.set(ylabel='Final forecast relative L2 (%) — lower is better',ylim=(0,1.22*largest),
           title='Coarse KS: four-arm forecast comparison\nMean of 3 fresh final initial conditions; noisy initial observations only')
    ax.legend(loc='upper left',ncols=2)
    fig.savefig(OUT/'ks_forecast_comparison.png',dpi=160,bbox_inches='tight',pad_inches=.12);plt.close(fig)


if __name__=='__main__':main()
