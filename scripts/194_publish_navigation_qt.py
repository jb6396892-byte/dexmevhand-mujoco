#!/usr/bin/env python3
"""Archive the complete evaluation denominator and a small set of actual Qt images."""
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
from fromrealhand.whole_table.task import brief


def read(path):
    return json.loads(path.read_text())


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaluation',type=Path,required=True)
    parser.add_argument('--qt',type=Path,required=True)
    parser.add_argument('--development',type=Path,nargs='*',default=[])
    parser.add_argument('--previous-evaluation',type=Path)
    parser.add_argument('--output',type=Path,default=ROOT/'docs/presentation/navigation_qt')
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    manifest=read(args.evaluation/'manifest.json');result=read(args.evaluation/'summary.json')
    if len(result['cases'])!=manifest['denominator']:
        raise ValueError('Incomplete evaluation: do not publish a reduced denominator')
    write(args.output/'heldout-manifest.json',manifest)
    write(args.output/'heldout-results.json',result)
    if args.previous_evaluation:
        for file in ('manifest','summary'):
            write(args.output/('previous-heldout-'+file+'.json'),read(args.previous_evaluation/(file+'.json')))
    write(args.output/'qt-quality.json',read(args.qt/'quality.json'))
    for name in ('first','second','clutter','stop','locked','rejected'):
        write(args.output/('qt-'+name+'.json'),brief(read(args.qt/name/'summary.json')))
    for name in ('clutter','second'):
        shutil.copyfile(args.qt/name/'qt-result.png',args.output/('qt-'+name+'.png'))
    development=[]
    for root in args.development:
        development.append(dict(path=str(root),result=read(root/'summary.json')))
    write(args.output/'development.json',development)
    successful=[r for r in result['cases'] if r['passed']]
    n=result['total'];rate=result['passed']/n;z=1.959963984540054
    center=(rate+z*z/(2*n))/(1+z*z/n)
    half=z*np.sqrt(rate*(1-rate)/n+z*z/(4*n*n))/(1+z*z/n)
    statistics=dict(
        success=result['passed'],total=result['total'],source_unchanged=result['source_unchanged'],
        wilson_95_interval=[float(center-half),float(center+half)],
        strict_success=sum(r.get('strict_passed',False) and r['passed'] for r in result['cases']),
        chosen_video_successes={v:sum(r.get('selected')==v for r in successful) for v in ('first','second')},
        near_entry_successes=sum(bool(r.get('selected_entry_frame')) for r in successful),
        rotated_successes=sum(bool(r.get('selected_yaw_deg')) for r in successful),
        preview_attempts=sum(len(r.get('attempts',[])) for r in result['cases']),
        actual_executions=sum(r.get('actual_executions',0) for r in result['cases']),
        median_planning_s=float(np.median([r['planning_seconds'] for r in result['cases']])),
        max_cup_error_mm=max((r['cup_error_m']*1000 for r in successful),default=None),
        max_carry_penetration_mm=max((r['carry_penetration_m']*1000 for r in successful),default=None),
        failures=[dict(seed=r['seed'],reason=r['reason'],attempts=r.get('attempts')) for r in result['cases'] if not r['passed']],
        scope='Known initial poses, same-model physics previews, one live execution. Not raw policy or hardware success.')
    write(args.output/'statistics.json',statistics)
    fig,axes=plt.subplots(1,2,figsize=(10,4.5))
    for case,row in zip(manifest['cases'],result['cases']):
        assert case['seed']==row['seed']
        xy=case['layout']['objects'][0]['xy'];goal=case['layout']['goal_world_m']
        color='#16846b' if row['passed'] else '#c34449'
        for ax,point in zip(axes,(xy,goal)):
            ax.scatter(*point[:2],c=color,s=35)
            ax.annotate(str(case['seed']),point[:2],fontsize=6,xytext=(3,3),textcoords='offset points')
    for ax,title in zip(axes,('Cup initial positions','Requested carry goals')):
        ax.set(xlabel='X (m)',ylabel='Y (m)',xlim=(-.425,.425),ylim=(-.4,.4),title=title)
        ax.set_aspect('equal');ax.grid(alpha=.2)
    fig.suptitle('%d/%d complete tasks; green=pass, red=fail'%(result['passed'],result['total']))
    fig.tight_layout();fig.savefig(str(args.output/'coverage.png'),dpi=150);plt.close(fig)
    write(args.output/'files.json',{p.name:hashlib.sha256(p.read_bytes()).hexdigest()
        for p in args.output.iterdir() if p.is_file() and p.name not in ('files.json','README.md')})
    for p in args.output.iterdir():
        if p.is_file():p.chmod(0o644)
    print(json.dumps(statistics,indent=2))


if __name__=='__main__':main()
