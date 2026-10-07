#!/usr/bin/env python3
"""Archive continuous navigation/grasp/carry development evidence and figures."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from hierarchy_common import ROOT,write

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--runs',type=Path,default=Path('/media/smgbro/shared/visual_grasp/adroit-navigation-v2'))
p.add_argument('--output',type=Path,default=ROOT/'docs/presentation/adroit_navigation_v2')
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
read=lambda path:json.loads(path.read_text())
def compact(value):
    if isinstance(value,dict):
        return {k:compact(v) for k,v in value.items() if k not in ('trace','hand_envelope_relative_m')}
    if isinstance(value,list):return [compact(v) for v in value]
    return value

names=['first-continuous-b','second-continuous-final','first-wall-final','second-wall-final']
cases=[]
for name in names:
    directory=a.runs/name
    summary=read(directory/'summary.json');adapter=read(directory/'adapter.json')
    carry=read(directory/'carry.json');local=read(directory/'local/report.json')
    local_trace=read(directory/'local/trace.json')
    passed=all([summary['passed'],adapter['navigation']['passed'],adapter['approach']['passed'],
        carry['passed'],carry['hold_supported_fraction']==1.,local['completed']==['reach','grasp','lift']])
    cases.append(dict(name=name,passed=passed,summary=summary,adapter=compact(adapter),carry=compact(carry),
        local_max_penetration_m=local['max_penetration_m'],
        local_table_contact_control_frames=sum(r['table_contacts']>0 for r in local_trace),
        local_non_target_contact_control_frames=sum(r['non_target_contacts']>0 for r in local_trace),
        strict_motion_passed=all(r['strict_passed'] for r in (adapter['navigation'],adapter['approach'],carry))))
history=[dict(name=p.parent.name,**read(p),has_carry_receipt=(p.parent/'carry.json').is_file())
         for p in sorted(a.runs.glob('*/summary.json'))]
result=dict(scope='four development cases; not a generalization success-rate estimate',
    acceptance='engineering v2 with separate original strict motion gates',
    passed=all(c['passed'] for c in cases),passed_count=sum(c['passed'] for c in cases),total=len(cases),
    cases=cases,development_history=history,training_started=False,qt_default_changed=False,
    legacy_regression=compact(read(a.runs/'legacy-regression/report.json')))
sources=[ROOT/'scripts/181_check_free_hand_local.py',Path(__file__).resolve(),
    ROOT/'configs/adroit-navigation-v1.json',ROOT/'configs/adroit-navigation-v2.json',
    ROOT/'src/fromrealhand/tabletop/control_scene.py',ROOT/'src/fromrealhand/tabletop/random_task.py']
sources+=list((ROOT/'src/fromrealhand/whole_table').glob('*.py'))
result['source_sha256']={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
write(a.output/'results.json',result)
for video in ('first','second'):
    target=a.output/(video+'-loaded-detour.png')
    shutil.copyfile(a.runs/(video+'-wall-final')/'carry-over-wall.png',target);target.chmod(0o644)
fig,axes=plt.subplots(2,1,figsize=(9,6),sharex=False)
for ax,video,color in zip(axes,('first','second'),('#197a73','#bf493e')):
    r=read(a.runs/(video+'-wall-final')/'carry.json');trace=r['trace']
    time=np.array([q['time_s'] for q in trace]);time-=time[0]-.001
    z=np.abs([q['acceleration_m_s2'][2] for q in trace])
    ax.plot(time,z,color=color,label='20 ms samples')
    ax.scatter([r['acceleration_peak_time_s'][2]],[r['max_acceleration_m_s2'][2]],color=color,
        marker='x',s=60,label='Exact substep peak')
    ax.axhline(.07,color='#555555',linestyle='--',label='Original strict Z gate: 0.07')
    ax.axhline(.5,color='#777777',linestyle=':',label='Engineering gate: 0.5')
    ax.set_ylim(-.01,.55);ax.set_title(video.title()+' video, loaded obstacle traversal')
    ax.set_ylabel('|Z acceleration| (m/s^2)');ax.set_xlabel('Carry time (s)')
    ax.grid(alpha=.15);ax.legend(fontsize=7,loc='upper right')
fig.tight_layout();fig.savefig(str(a.output/'acceleration-gates.png'),dpi=150);plt.close(fig)
print(json.dumps(dict(passed=result['passed'],cases=result['passed_count'],total=len(cases),
    strict_motion_passed=sum(c['strict_motion_passed'] for c in cases),history=len(history)),indent=2))
if not result['passed']:raise SystemExit(1)
