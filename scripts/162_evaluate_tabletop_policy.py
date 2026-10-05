#!/usr/bin/env python3
"""Live RGB-D physical policy evaluation, with explicit development/test split."""
import argparse
import hashlib
from pathlib import Path
import subprocess
import sys
from hierarchy_common import ROOT, read, write


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path)
    p.add_argument('--videos',nargs='+',choices=['first','second'],default=['first','second'])
    p.add_argument('--seeds',type=int,nargs='+',default=[3])
    p.add_argument('--split',choices=['development','heldout','regression'],default='development')
    p.add_argument('--freeze',type=Path)
    a=p.parse_args()
    protocol=read(ROOT/'configs/tabletop-dual-v3-protocol.json')
    if a.split=='heldout':
        if not a.freeze: raise ValueError('Heldout evaluation requires a frozen selection')
        frozen=read(a.freeze)
        for name,sha in frozen['source_sha256'].items():
            if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=sha:
                raise ValueError('Frozen evaluation source changed: '+name)
        if a.checkpoint and hashlib.sha256(a.checkpoint.read_bytes()).hexdigest()!=frozen['checkpoint_sha256']:
            raise ValueError('Candidate changed after freeze')
        if set(a.seeds)-set(frozen['test_seeds']): raise ValueError('Unregistered heldout seeds')
    elif a.split=='development' and set(a.seeds)&set(protocol['reserved_final_test_seeds']):
        raise ValueError('Reserved test seeds cannot be used in development')
    a.output.mkdir(parents=True,exist_ok=False)
    profiles=read(ROOT/'configs/tabletop-dual-v3-profiles.json')
    results=[]
    for video in a.videos:
        for seed in a.seeds:
            out=a.output/('%s-seed-%s'%(video,seed))
            cmd=[sys.executable,str(ROOT/'scripts/153_test_contact_tracking.py'),
                 '--scene',video,'--seed',str(seed),'--output',str(out),
                 '--live-vision','--camera',profiles['camera']]
            if a.checkpoint: cmd.extend(['--checkpoint',str(a.checkpoint.resolve())])
            for key,value in profiles['profiles'][video].items():
                flag='--'+key.replace('_','-')
                if isinstance(value,bool):
                    if value: cmd.append(flag)
                elif isinstance(value,list): cmd.extend([flag]+list(map(str,value)))
                else: cmd.extend([flag,str(value)])
            with (a.output/(out.name+'.log')).open('w') as log:
                process=subprocess.run(cmd,cwd=str(ROOT),stdout=log,stderr=subprocess.STDOUT)
            report=read(out/'report.json') if (out/'report.json').exists() else {}
            actual_error=None
            if report.get('ground_truth_after_stop') and (out/'goal.json').exists():
                import numpy as np
                actual_error=float(np.linalg.norm(np.asarray(report['ground_truth_after_stop'])[:3,3]
                                   -np.asarray(read(out/'goal.json')['goal_world_m'])))
            passed=bool(process.returncode==0 and report.get('passed') and actual_error is not None
                        and actual_error<=protocol['limits']['goal_distance_m'])
            reason=report.get('reason','worker_failed')
            if report.get('passed') and actual_error is not None and actual_error>protocol['limits']['goal_distance_m']:
                reason='post_stop_actual_goal_error'
            row=dict(video=video,seed=seed,passed=passed,returncode=process.returncode,
                reason=reason,completed=report.get('completed',[]),
                max_penetration_m=report.get('max_penetration_m'),actual_goal_error_m=actual_error,
                visual_goal_error_m=(report.get('final') or {}).get('target_distance_m'),
                output=str(out),command=cmd,clipped_action_calls=report.get('clipped_action_calls'))
            results.append(row)
            write(a.output/'evaluation.json',dict(split=a.split,records=results,complete=False,
                checkpoint=str(a.checkpoint) if a.checkpoint else None))
            print(row,flush=True)
    write(a.output/'evaluation.json',dict(split=a.split,records=results,complete=True,
        passed=sum(r['passed'] for r in results),total=len(results),
        checkpoint=str(a.checkpoint) if a.checkpoint else None))


if __name__=='__main__': main()
