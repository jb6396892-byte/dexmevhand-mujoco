#!/usr/bin/env python3
"""Compact all outcomes and a few physical-rendering figures for presentation."""
import argparse
import json
import shutil
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from hierarchy_common import ROOT,write


def read(path):return json.loads(path.read_text())


def compact(value):
    if isinstance(value,dict):
        return {k:compact(v) for k,v in value.items() if k not in ('trace','hand_envelope_relative_m','obstacles')}
    if isinstance(value,list):return [compact(v) for v in value]
    return value


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs',type=Path,nargs='+',required=True)
    parser.add_argument('--output',type=Path,default=ROOT/'docs/presentation/navigation_grasp')
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    rounds=[]
    for root in args.runs:
        result=read(root/'summary.json')
        for case in result['cases']:
            directory=root/case['name']
            for field,file in [('adapter','adapter.json'),('carry','carry.json')]:
                if (directory/file).exists():case[field]=compact(read(directory/file))
            local=read(directory/'local/report.json')
            case['failure_contacts']=local.get('failure_contacts',[])
            case['checkpoint_sha256']=read(directory/'local/input.json').get('checkpoint_sha256') if (directory/'local/input.json').exists() else None
        rounds.append(dict(directory=str(root),**result))
        shutil.copyfile(root/'manifest.json',args.output/(root.name+'-manifest.json'))
    write(args.output/'results.json',dict(rounds=rounds,
        warning='Same layouts reused for engineering comparison, not independent held-out trials',
        planning_can_pass_above_obstacles=True,training_started=False))
    latest=args.runs[-1]
    for video,seed in [('first',4304),('second',4301)]:
        source=latest/('seed-{}-{}'.format(seed,video))
        if not read(source/'summary.json')['passed']:raise RuntimeError('Selected illustration is not successful')
        frame=source/'carry-over-wall.png' if video=='first' else source/'carry.png'
        shutil.copyfile(frame,args.output/(video+'-carry.png'))
    source=latest/'seed-4304-first'
    carry=read(source/'carry.json');adapter=read(source/'adapter.json')
    fig,axes=plt.subplots(1,2,figsize=(11,4.5))
    for axis,dims,title in zip(axes,[(0,1),(0,2)],['Top view: palm path','Side view: palm path']):
        for obstacle in carry['obstacles']:
            lo,hi=np.array(obstacle['bounds'])
            axis.add_patch(Rectangle(lo[list(dims)],*(hi-lo)[list(dims)],
                facecolor='#bbbbbb',edgecolor='#777777',alpha=.25))
        for name,report,color in [('Navigate',adapter['navigation'],'#1678a4'),
                                  ('Approach',adapter['approach'],'#e19b31'),('Carry',carry,'#a73c65')]:
            points=np.array([r['center_m'] for r in report['trace']])
            axis.plot(points[:,dims[0]],points[:,dims[1]],color=color,label=name)
        endpoints=np.array(carry['waypoints'])[[0,-1]]
        axis.plot(endpoints[:,dims[0]],endpoints[:,dims[1]],'--',color='#444444',label='Direct carry segment')
        axis.set_title(title);axis.set_xlabel('X (m)');axis.set_ylabel('XYZ'[dims[1]]+' (m)')
        axis.grid(alpha=.15);axis.legend(fontsize=8);axis.set_aspect('equal',adjustable='datalim')
    fig.tight_layout();fig.savefig(str(args.output/'physical-path.png'),dpi=150);plt.close(fig)
    for p in args.output.iterdir():
        if p.is_file():p.chmod(0o644)
    print(json.dumps([dict(round=r['directory'],passed=r['passed'],total=r['total']) for r in rounds]),flush=True)


if __name__=='__main__':main()
