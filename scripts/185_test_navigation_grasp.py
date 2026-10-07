#!/usr/bin/env python3
"""Frozen clutter layouts: physical navigation, learned grasp, lift and loaded carry."""
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from hierarchy_common import ROOT, write
from fromrealhand.whole_table.layout import sample
from fromrealhand.whole_table.scene import object_catalog


def read(path):
    return json.loads(path.read_text()) if path.exists() else {}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--seeds',type=int,nargs='+',default=[4301,4302,4303,4304])
    parser.add_argument('--config',type=Path,default=ROOT/'configs/adroit-navigation-v2.json')
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    cfg=read(ROOT/'configs/adroit-navigation-v1.json')
    catalog=object_catalog();cases=[]
    for index,seed in enumerate(args.seeds):
        for video,sign in [('first',1),('second',-1)]:
            xy=[sign*(.20 if index%2==0 else .16),-.04 if index%2==0 else .04]
            goal=[-sign*.20,.08+.02*(index%3),.20]
            layout=sample(seed,catalog,cfg,2+index%3,cup_xy=xy)
            layout.update(goal_world_m=[xy[0],xy[1],.16],table_rgba=[.88,.9,.91,1.],
                          cup_model='025_mug',object_scale=.8,version='navigation-grasp-v1')
            name='seed-{}-{}'.format(seed,video)
            write(args.output/(name+'-layout.json'),layout)
            cases.append(dict(name=name,seed=seed,video=video,shift=xy,goal=goal,layout=layout))
    sources=[ROOT/'scripts/181_check_free_hand_local.py',Path(__file__).resolve(),
             args.config.resolve(),ROOT/'src/fromrealhand/tabletop/random_task.py']
    sources+=list((ROOT/'src/fromrealhand/whole_table').glob('*.py'))
    write(args.output/'manifest.json',dict(cases=cases,created_before_execution=True,
        config=str(args.config.resolve()),acceptance='engineering dynamics; strict gates reported separately',
        source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}))
    results=[]
    for case in cases:
        out=args.output/case['name']
        command=[sys.executable,str(ROOT/'scripts/181_check_free_hand_local.py'),
            '--config',str(args.config.resolve()),'--video',case['video'],
            '--seed',str(case['seed']),'--layout',str(args.output/(case['name']+'-layout.json')),
            '--shift']+list(map(str,case['shift']))+['--transit','--carry-goal']+list(map(str,case['goal']))+['--output',str(out)]
        print('START '+case['name'],flush=True)
        with (args.output/(case['name']+'.log')).open('w') as log:
            try:
                code=subprocess.run(command,cwd=str(ROOT),stdout=log,stderr=subprocess.STDOUT,timeout=1200).returncode
            except subprocess.TimeoutExpired:
                code=124
        local=read(out/'local/report.json');adapter=read(out/'adapter.json');carry=read(out/'carry.json')
        nav=adapter.get('navigation',{});approach=adapter.get('approach',{})
        traces=read(out/'local/trace.json') or []
        stages=[nav,approach,carry]
        result=dict(name=case['name'],seed=case['seed'],video=case['video'],exit_code=code,
            passed=bool(code==0 and local.get('passed') and all(r.get('passed') for r in stages)),
            reason=local.get('reason','process_failed_before_report'),completed=local.get('completed',[]),
            navigation_passed=nav.get('passed',False),approach_passed=approach.get('passed',False),
            carry_passed=carry.get('passed',False),navigation_detour=nav.get('direct') is False,
            carry_reason=carry.get('reason'),initial_margin_conflicts=carry.get('initial_margin_conflicts'),
            egress_executed=bool(carry.get('egress')),
            carry_detour=carry.get('direct') is False,
            strict_motion_passed=all(r.get('strict_passed',False) for r in stages),
            local_penetration_m=local.get('max_penetration_m'),
            local_table_contact_frames=sum(r['table_contacts']>0 for r in traces),
            local_non_target_contact_frames=sum(r['non_target_contacts']>0 for r in traces),
            carry_error_m=carry.get('cup_error_m'),carry_penetration_m=carry.get('max_grasp_penetration_m'),
            hold_supported_fraction=carry.get('hold_supported_fraction'),
            learned_action_calls=local.get('learned_action_calls',0))
        results.append(result)
        write(args.output/'partial.json',results)
        print(json.dumps(result),flush=True)
    summary=dict(cases=results,passed=sum(r['passed'] for r in results),total=len(results),
        scope='predeclared development cases, not whole-table generalization',
        training_started=False,execution_object_pose_writes=0)
    write(args.output/'summary.json',summary)
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
