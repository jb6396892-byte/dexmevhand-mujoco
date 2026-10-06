#!/usr/bin/env python3
"""Archive the complete denominator and paired-layout confidence interval."""
import argparse
from collections import Counter
import hashlib
import math
from pathlib import Path
import shutil
import subprocess
import sys
from hierarchy_common import ROOT,read,write


def wilson(successes,total):
    z=1.959963984540054; p=successes/total; divisor=1+z*z/total
    center=(p+z*z/(2*total))/divisor
    half=z*math.sqrt(p*(1-p)/total+z*z/(4*total*total))/divisor
    return [max(0.,center-half),min(1.,center+half)]


def summarize(rows):
    passed=sum(r['passed'] for r in rows)
    successful=[r for r in rows if r['passed']]
    return dict(passed=passed,total=len(rows),rate=passed/len(rows),
        failures=dict(Counter(r['reason'] for r in rows if not r['passed'])),
        max_penetration_mm=max(r['max_penetration_m'] for r in rows)*1000,
        successful_max_goal_error_mm=max((r['final']['target_distance_m']*1000 for r in successful),default=None))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,default=Path('/media/smgbro/shared/visual_grasp/random-v5'))
    p.add_argument('--qt',default='qt-final')
    p.add_argument('--history',nargs='*',default=['dev-initial','dev-hand-integral','dev-servo-anchor','dev-proprioception'])
    p.add_argument('--boundary',nargs='+',default=['target-min-boundary','target-max-boundary'])
    p.add_argument('--output',type=Path,default=ROOT/'docs/presentation/random_tabletop_v5/evidence')
    a=p.parse_args(); a.output.mkdir(parents=True,exist_ok=True)
    frozen=read(a.run/'freeze.json'); evaluation=read(a.run/'heldout/evaluation.json')
    if not evaluation['complete'] or evaluation['split']!='heldout': raise ValueError('Incomplete heldout')
    expected={(v,s) for v in frozen['videos'] for s in frozen['seeds']}
    rows=evaluation['records']
    if {(r['video'],r['seed']) for r in rows}!=expected or len(rows)!=len(expected):
        raise ValueError('Missing, repeated or extra heldout cases')
    for name,digest in frozen['sources'].items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=digest: raise ValueError('Frozen source changed: '+name)
    if hashlib.sha256(Path(frozen['checkpoint']).read_bytes()).hexdigest()!=frozen['checkpoint_sha256']:
        raise ValueError('Checkpoint changed')
    checks=subprocess.run([sys.executable,'-m','unittest','discover','-s','tests'],cwd=str(ROOT),
        stdout=subprocess.PIPE,stderr=subprocess.STDOUT,universal_newlines=True,timeout=120)
    (a.output/'tests.txt').write_text('\n'.join(line.rstrip() for line in checks.stdout.splitlines())+'\n')
    quality=read(a.run/a.qt/'quality.json')
    summary=dict(overall=summarize(rows),by_video={},by_distractor_count={},by_table_variant={},
        tests_passed=checks.returncode==0,qt_checks=quality,checkpoint=frozen['checkpoint'],
        no_new_training=True,object_input='known initial pose; hand kinematic prediction in second-video carry',
        scope=frozen['protocol'])
    for video in frozen['videos']:
        subset=[r for r in rows if r['video']==video]
        summary['by_video'][video]=dict(summarize(subset),wilson_95=wilson(sum(r['passed'] for r in subset),len(subset)))
    pair_success=sum(all(r['passed'] for r in rows if r['seed']==seed) for seed in frozen['seeds'])
    summary['both_videos_per_independent_layout']=dict(passed=pair_success,total=len(frozen['seeds']),
        wilson_95=wilson(pair_success,len(frozen['seeds'])),
        note='Two videos share each layout; do not treat 80 rollouts as 80 independent layouts')
    for key,field in [('by_distractor_count','distractor_count'),('by_table_variant','table_variant')]:
        for value in sorted({r['layout'][field] for r in rows}):
            summary[key][str(value)]=summarize([r for r in rows if r['layout'][field]==value])
    reports=[read(Path(r['output'])/'report.json') for r in rows]
    summary['execution_audit']=dict(
        clipped_action_calls=sum(r['clipped_action_calls'] for r in reports),
        learned_action_calls=sum(r['learned_action_calls'] for r in reports),
        live_object_state_writes=sum(r['state_writes_during_execution'] for r in reports),
        any_object_assistance=any(r['object_forces_applied'] for r in reports))
    boundary=[]
    for name in a.boundary:
        result=read(a.run/name/'evaluation.json')
        if not result['complete']: raise ValueError('Incomplete target boundary check')
        write(a.output/(name+'.json'),result)
        boundary.extend(result['records'])
    summary['target_boundary_regression']=summarize(boundary)
    summary['delivery_passed']=(checks.returncode==0 and all(quality.values())
        and all(v['rate']>.8 for v in summary['by_video'].values()) and all(r['passed'] for r in boundary))
    write(a.output/'summary.json',summary); write(a.output/'freeze.json',frozen)
    write(a.output/'heldout.json',evaluation); write(a.output/'development.json',read(a.run/'dev-full/evaluation.json'))
    history={}
    for name in a.history:
        history[name]=read(a.run/name/'evaluation.json')
    write(a.output/'failure-history.json',history)
    for case in ('first-manual','second-manual','empty-lift','stop','locked','instruction-rejected'):
        write(a.output/('qt-'+case+'.json'),read(a.run/a.qt/case/'summary.json'))
    for case in ('first-manual','second-manual'):
        shutil.copyfile(str(a.run/a.qt/case/'qt-result.png'),str(a.output/(case+'.png')))
    print(summary)
    if not summary['delivery_passed']: raise SystemExit(1)


if __name__=='__main__': main()
