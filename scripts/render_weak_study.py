"""Render saved results only; never fit or read observation arrays."""
from pathlib import Path
import csv,json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
METHODS=('plain','adaptive','weak_adaptive')
LABELS=('Plain L1','Adaptive L1','Weak adaptive L1')
COLORS=('#718096','#b7791f','#087f8c')
FAMILIES=('allen_cahn','low_viscosity_burgers','ks_unstable')
FAMILY_LABELS=('Allen–Cahn','Low-viscosity Burgers','KS')


def worst(row):
    return max(v['relative_error'] for v in row['recovery']['coefficient_comparison'] if v['reference']!=0)


def render(directory,title):
    records=json.loads((directory/'comparison.json').read_text())['records']
    fields=['id','family','sampling','condition','method','equation','exact_support','missing_terms',
            'extra_terms','vector_error_percent','worst_true_term_error_percent','prediction_l2_percent']
    with (directory/'summary.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for r in records:
            writer.writerow({**{k:r.get(k) for k in fields[:6]},'exact_support':r['recovery']['exact_support'],
                'missing_terms':';'.join(r['recovery']['missing_terms']),'extra_terms':';'.join(r['recovery']['spurious_terms']),
                'vector_error_percent':100*r['recovery']['coefficient_relative_l2'],
                'worst_true_term_error_percent':100*worst(r),
                'prediction_l2_percent':None if r['mean_clean_relative_l2'] is None else 100*r['mean_clean_relative_l2']})
    fig,axes=plt.subplots(1,3,figsize=(12,4),layout='constrained')
    for ax,family,label in zip(axes,FAMILIES,FAMILY_LABELS):
        counts=[sum(r['recovery']['exact_support'] for r in records if r['family']==family and r['method']==m) for m in METHODS]
        ax.bar(range(3),counts,color=COLORS)
        for i,n in enumerate(counts):ax.text(i,n+.18,f'{n}/10',ha='center')
        ax.set_xticks(range(3),LABELS,rotation=16);ax.set_ylim(0,11.4);ax.set_title(label)
    axes[0].set_ylabel('Exact term-set recovery (higher is better)')
    fig.suptitle(title+' | 10 paired conditions per family/method')
    fig.savefig(directory/'support_recovery.png',dpi=160);plt.close(fig)
    by={r['id']:r for r in records}
    fig,ax=plt.subplots(figsize=(7,6),layout='constrained')
    for family,label,color in zip(FAMILIES,FAMILY_LABELS,('#7b4ab5','#e18c1b','#008577')):
        pairs=[]
        for r in records:
            if r['method']=='weak_adaptive' and r['family']==family:
                b=by[r['id'].removesuffix('weak_adaptive')+'adaptive']
                pairs.append((max(100*worst(b),1e-12),max(100*worst(r),1e-12)))
        xs,ys=zip(*pairs);ax.scatter(xs,ys,label=label,color=color,s=52,alpha=.8,edgecolors='white')
    limits=(1e-7,300)
    ax.plot(limits,limits,'--',color='#888888',linewidth=1,label='Equal error')
    ax.set(xscale='log',yscale='log',xlim=limits,ylim=limits,
           xlabel='Adaptive L1: worst true-coefficient error (%)',
           ylabel='Weak adaptive L1: worst true-coefficient error (%)',
           title=title+'\nPoints below diagonal favor weak regression')
    ax.legend(loc='upper left');ax.grid(which='major',alpha=.2)
    fig.savefig(directory/'per_term_comparison.png',dpi=160);plt.close(fig)
    if directory.name=='first_run':
        rows=[r for r in records if r['family']=='ks_unstable' and r['sampling']=='coarse']
        conditions=list(dict.fromkeys(r['condition'] for r in rows))
        fig,ax=plt.subplots(figsize=(10,4),layout='constrained')
        for j,method in enumerate(('adaptive','weak_adaptive')):
            values=[100*next(r['mean_clean_relative_l2'] for r in rows if r['condition']==c and r['method']==method) for c in conditions]
            bars=ax.bar(np.arange(len(conditions))+(j-.5)*.35,values,.35,color=COLORS[j+1],label=LABELS[j+1])
            ax.bar_label(bars,fmt='%.1f',padding=3,fontsize=9)
        ax.set_xticks(range(5),['Clean','0.1% / seed43','0.1% / seed71','1% / seed43','1% / seed71'])
        ax.set(ylabel='Final forecast relative L2 (%) — lower is better',ylim=(0,125),
               title='Coarse KS: coefficients can improve while the same forecast solver worsens\nMean over 3 final initial conditions; each starts from noisy observations')
        ax.legend(loc='upper left')
        fig.savefig(directory/'ks_forecast_regression.png',dpi=160);plt.close(fig)


if __name__=='__main__':
    for name,title in [('retrospective','Retrospective / previously viewed data'),('first_run','Fresh initial-condition validation')]:
        render(ROOT/'weak_study/results'/name,title)
