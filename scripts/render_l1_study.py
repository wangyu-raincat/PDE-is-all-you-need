"""Render frozen derived metrics only; never fit, select or read field arrays."""
from pathlib import Path
import csv,json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
root=Path(__file__).resolve().parents[1]
out=root/'l1_study/results/first_run'
data=json.loads((out/'comparison.json').read_text()); rows=data['records']
fields=['id','suite','case','derivatives','training_trajectories','method','retrospective','exact_support','missing_count','extra_count','coefficient_error_percent','prediction_error_percent','forecast_failures','equation']
with (out/'summary.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
 for r in rows:
  d={k:r[k] for k in fields if k in r};m=r['recovery']
  d.update(exact_support=m['exact_support'],missing_count=len(m['missing_terms']),extra_count=len(m['spurious_terms']),coefficient_error_percent=100*m['coefficient_relative_l2'],prediction_error_percent=100*r['mean_final_relative_l2'] if r['mean_final_relative_l2'] is not None else None);w.writerow(d)
methods=['plain','adaptive','irl1'];colors=['#60768a','#168574','#d19b34'];labels=['Plain L1','Adaptive L1','5-step IRL1']
fig,ax=plt.subplots(1,3,figsize=(14,4.2),layout='constrained')
for i,method in enumerate(methods):
 group=[r for r in rows if r['suite']=='public' and r['method']==method]
 val=sum(r['recovery']['exact_support'] for r in group);ax[0].bar(i,val,color=colors[i]);ax[0].text(i,val+.07,f'{val}/3',ha='center')
 val=sum(len(r['recovery']['spurious_terms']) for r in group);ax[1].bar(i,val,color=colors[i]);ax[1].text(i,val+.09,str(val),ha='center')
 for j,r in enumerate(group):
  value=100*r['recovery']['coefficient_relative_l2'];x=j+(i-1)*.24
  ax[2].bar(x,value,width=.22,color=colors[i],label=labels[i] if j==0 else None)
for a,title in zip(ax[:2],['Exact support recovery (higher is better)','Total extra terms (lower is better)']):
 a.set_title(title);a.set_xticks(range(3),labels,rotation=12);a.spines[['top','right']].set_visible(False)
ax[0].set_ylim(0,3.5);ax[0].set_yticks(range(4));ax[1].set_ylim(0,4.8);ax[1].set_yticks(range(5))
ax[2].set_yscale('log');ax[2].set_xticks(range(3),['Burgers','KdV','KS']);ax[2].set_title('Coefficient vector relative error (%)');ax[2].legend(fontsize=8)
fig.suptitle('Previously viewed public numerical datasets | retrospective comparison',fontsize=13)
fig.savefig(out/'public_comparison.png',dpi=160);plt.close(fig)
fig,ax=plt.subplots(figsize=(8.2,5.4),layout='constrained')
fig.get_layout_engine().set(rect=(0,0.07,1,0.93))
configs=[(mode,n) for mode in ['integrated','pointwise_estimated','pointwise_exact'] for n in [1,3]]
values=np.zeros((6,3));extra=np.zeros((6,3),int)
for i,(mode,n) in enumerate(configs):
 for j,method in enumerate(methods):
  g=[r for r in rows if r['suite']=='analytic' and r['derivatives']==mode and r['training_trajectories']==n and r['method']==method]
  values[i,j]=sum(r['recovery']['exact_support'] for r in g);extra[i,j]=sum(len(r['recovery']['spurious_terms']) for r in g)
im=ax.imshow(values,vmin=0,vmax=3,cmap='YlGn',aspect='auto')
for i in range(6):
 for j in range(3):ax.text(j,i,f'{int(values[i,j])}/3 exact\n{extra[i,j]} extra terms',ha='center',va='center',color='white' if values[i,j]==3 else '#263640',fontsize=10)
ax.set_xticks(range(3),labels);ax.set_yticks(range(6),['Integral / 1 trajectory','Integral / 3 trajectories','Estimated pointwise / 1','Estimated pointwise / 3','Analytic pointwise / 1','Analytic pointwise / 3'])
ax.set_title('New analytic initial-condition suite\nEqual total regression rows within each input backend')
fig.text(.5,.005,'Analytic derivatives are privileged controls; all fields are synthetic, not measurements.',ha='center',fontsize=9)
fig.savefig(out/'analytic_comparison.png',dpi=160,bbox_inches='tight');plt.close(fig)
print('Rendered two figures and all 63 CSV rows from frozen scores only.')
