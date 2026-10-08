#!/usr/bin/env python3
"""Archive compact, traceable placement evidence; never substitute previews for execution."""
import argparse
import collections
import json
from pathlib import Path
import shutil
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from hierarchy_common import ROOT,write

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--evaluation',type=Path,required=True)
p.add_argument('--qt',type=Path,required=True)
p.add_argument('--output',type=Path,default=ROOT/'docs/presentation/placement_return')
a=p.parse_args()
summary=json.loads((a.evaluation/'summary.json').read_text())
manifest=json.loads((a.evaluation/'manifest.json').read_text())
assert summary['total']==20 and len(summary['cases'])==20
a.output.mkdir(parents=True,exist_ok=False)
write(a.output/'summary.json',summary);write(a.output/'manifest.json',manifest)
quality=json.loads((a.qt/'quality.json').read_text());write(a.output/'qt-quality.json',quality)
successful=[r for r in summary['cases'] if r['passed']]
if successful:
    example=successful[0]
    source=a.evaluation/('seed-'+str(example['seed']))/'execution'
    receipt=json.loads((source/'placement.json').read_text())
    assert receipt['passed'] and receipt['placement_passed'] and receipt['return_home']['passed']
    write(a.output/'example-placement.json',receipt)
    for name in ('place-supported.png','place-released.png','place-retreated.png'):
        shutil.copy2(source/name,a.output/name)
    trace=json.loads((source/'placement-trace.json').read_text())
    t0=trace[0]['time_s'];times=[r['time_s']-t0 for r in trace]
    fig,axes=plt.subplots(3,1,figsize=(10,7),sharex=True,constrained_layout=True)
    for ax,key,label,color in zip(axes,('bottom_gap_m','table_force_n','hand_cup_force_n'),
                                ('Cup bottom gap (mm)','Table support (N)','Hand-cup normal force (N)'),
                                ('#167b76','#b26328','#3d536b')):
        values=[r[key]*(1000 if key.endswith('_m') else 1) for r in trace]
        ax.plot(times,values,color=color,linewidth=1.4);ax.set_ylabel(label);ax.grid(alpha=.2)
    previous=None
    for row in trace:
        if row['phase']!=previous:
            x=row['time_s']-t0
            for ax in axes:ax.axvline(x,color='#777777',alpha=.25,linewidth=.7)
            axes[0].text(x,.98,row['phase'],rotation=90,va='top',fontsize=8,
                         transform=axes[0].get_xaxis_transform())
            previous=row['phase']
    axes[-1].set_xlabel('Simulation time since placement started (s)')
    fig.suptitle('Actual execution, seed '+str(example['seed'])+'; cup remains a free body')
    fig.savefig(str(a.output/'placement-forces.png'),dpi=150);plt.close(fig)
for case in ('first','second'):
    path=a.qt/case/'summary.json'
    if path.exists() and json.loads(path.read_text()).get('task_passed'):
        shutil.copy2(a.qt/case/'qt-result.png',a.output/'qt-result.png')
        break
selected=collections.Counter(r.get('selected') for r in successful)
failures=collections.Counter(r['reason'] for r in summary['cases'] if not r['passed'])
statistics=dict(complete_successes=summary['passed'],total=20,
    selected_grasps=dict(selected),failures=dict(failures),source_unchanged=summary['source_unchanged'],
    total_wall_s=sum(r.get('total_wall_s',0) for r in summary['cases']),
    max_final_xy_error_m=max((r['placement']['after_return']['xy_error_m'] for r in successful),default=None),
    max_placement_penetration_m=max((r['placement']['max_penetration_m'] for r in successful),default=None),
    max_return_error_m=max((r['placement']['return_home']['endpoint_error_m'] for r in successful),default=None),
    evaluation_source=str(a.evaluation),qt_source=str(a.qt))
write(a.output/'statistics.json',statistics)
print(json.dumps(statistics,indent=2))
