"""Read frozen derived metrics only; no fitting, selection or field loading."""
from pathlib import Path
import json,csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(__file__).resolve().parents[1];out=root/'challenge_study/results/first_run'
report=json.loads((out/'comparison.json').read_text());rows=report['records']
methods=['plain','adaptive','irl1'];labels=['Plain L1','Adaptive L1','5-step IRL1'];colors=['#62758a','#138772','#cf9833']
families=['allen_cahn','low_viscosity_burgers','ks_unstable'];names=['Allen–Cahn','Low-viscosity Burgers','KS nonlinear instability']
conditions=['clean','noise0p1_s17','noise0p1_s29','noise1_s17','noise1_s29']
fields=['id','family','sampling','condition','method','exact_support','missing_terms','extra_terms','coefficient_error_percent','clean_prediction_error_percent','noisy_prediction_error_percent','forecast_failures','equation']
with (out/'summary.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
 for r in rows:
  m=r['recovery'];d={k:r[k] for k in fields if k in r};d.update(exact_support=m['exact_support'],missing_terms=';'.join(m['missing_terms']),extra_terms=';'.join(m['spurious_terms']),coefficient_error_percent=100*m['coefficient_relative_l2'],clean_prediction_error_percent=100*r['mean_clean_relative_l2'] if r['mean_clean_relative_l2'] is not None else None,noisy_prediction_error_percent=100*r['mean_noisy_relative_l2'] if r['mean_noisy_relative_l2'] is not None else None);w.writerow(d)
fig,axes=plt.subplots(1,3,figsize=(12,4.1),layout='constrained')
for ax,family,title in zip(axes,families,names):
 for j,method in enumerate(methods):
  subset=[r for r in rows if r['family']==family and r['method']==method];value=sum(r['recovery']['exact_support'] for r in subset)
  ax.bar(j,value,color=colors[j]);ax.text(j,value+.15,f'{value}/10',ha='center')
 ax.set_ylim(0,11);ax.set_yticks(range(0,11,2));ax.set_xticks(range(3),labels,rotation=15);ax.set_title(title);ax.spines[['top','right']].set_visible(False)
axes[0].set_ylabel('Exact term-set recovery (higher is better)');fig.suptitle('90 fixed configurations | 10 paired conditions per family and method')
fig.savefig(out/'recovery_summary.png',dpi=160);plt.close(fig)
fig,axes=plt.subplots(1,3,figsize=(13,5.8),layout='constrained')
for ax,family,title in zip(axes,families,names):
 z=np.zeros((10,3));texts={}
 for i,(s,c) in enumerate((s,c) for s in ['full','coarse'] for c in conditions):
  for j,m in enumerate(methods):
   r=next(r for r in rows if r['family']==family and r['sampling']==s and r['condition']==c and r['method']==m);a=r['recovery'];z[i,j]=a['exact_support'];texts[i,j]='OK' if a['exact_support'] else f"-{len(a['missing_terms'])}/+{len(a['spurious_terms'])}"
 ax.imshow(z,vmin=0,vmax=1,cmap='YlGn',aspect='auto')
 for (i,j),v in texts.items():ax.text(j,i,v,ha='center',va='center',color='white' if z[i,j] else '#263640',fontsize=9)
 ax.set_xticks(range(3),['L1','Adaptive','IRL1']);ax.set_yticks(range(10),[s+' / '+c.replace('noise0p1','0.1%').replace('noise1','1%').replace('_s',' seed') for s in ['full','coarse'] for c in conditions],fontsize=8);ax.set_title(title)
fig.suptitle('Exact support and failure anatomy | -missing / +extra terms; no post-hoc thresholding')
fig.savefig(out/'condition_matrix.png',dpi=160);plt.close(fig)
fig,axes=plt.subplots(1,2,figsize=(12,4.7),layout='constrained')
for ax,s in zip(axes,['full','coarse']):
 for j,m in enumerate(methods):
  values=[]
  for c in conditions:
   r=next(r for r in rows if r['family']=='allen_cahn' and r['sampling']==s and r['condition']==c and r['method']==m)
   term=next(t for t in r['recovery']['coefficient_comparison'] if t['name']=='u_xx');values.append(term['relative_error']*100)
  ax.plot(range(5),values,'o-',color=colors[j],label=labels[j])
 ax.set_xticks(range(5),['clean','0.1%\nseed17','0.1%\nseed29','1%\nseed17','1%\nseed29']);ax.set_title(s+' sampling');ax.set_ylabel('Diffusion coefficient relative error (%)');ax.axhline(100,color='#ad3d39',ls='--',lw=1,label='100% error (e.g. missing term)');ax.legend(fontsize=8)
fig.suptitle('Allen–Cahn small diffusion term: reference = 0.0001 | vector error can conceal this error')
fig.savefig(out/'ac_diffusion_error.png',dpi=160);plt.close(fig)
print('Rendered all90rows CSV and 3 scientific figures from saved metrics only.')
