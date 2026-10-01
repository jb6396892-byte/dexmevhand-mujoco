#!/usr/bin/env python3
"""One paired heldout evaluation after policy hashes are frozen, with no tuning."""
import argparse
import json
import pickle
from collections import Counter
from pathlib import Path
import numpy as np
from v10_common import ROOT,protocol,digest,reference_demo,setup_case,run_case,full_gate,failure_labels
from fromrealhand.multivideo import phase_diagnostics
from fromrealhand.policy_learning import lift_success


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--frozen',type=Path,default=ROOT/'data/processed/dual_video_v10/learning/frozen_policies.json')
    parser.add_argument('--output',type=Path,default=ROOT/'data/processed/dual_video_v10/heldout')
    args=parser.parse_args()
    study,sha=protocol();frozen=json.loads(args.frozen.read_text())
    assert frozen['protocol_sha256']==sha
    checkpoints={}
    for method,entry in frozen['policies'].items():
        assert digest(entry['policy'])==entry['policy_sha256']
        checkpoints[method]=pickle.loads(Path(entry['policy']).read_bytes())
    args.output.mkdir(parents=True,exist_ok=False)
    (args.output/'freeze_receipt.json').write_text(json.dumps(dict(protocol_sha256=sha,frozen_sha256=digest(args.frozen),policies=frozen['policies']),indent=2)+'\n')
    reports=[]
    for video in study['videos']:
        reference=reference_demo(video)['actions']
        for case in study['heldout_per_video']:
            folder=args.output/video['name']/case['name']
            source,g=setup_case(video,case,folder)
            for method in ['fixed']+study['learning']['methods']:
                report,demo=run_case(video,source,reference,folder/method,checkpoint=checkpoints.get(method))
                phases=phase_diagnostics(demo,reference,video,g)
                result=dict(video=video['name'],case=case['name'],method=method,full_pass=full_gate(report),lift_pass=lift_success(report),
                            report=report,phases=phases,failure_labels=failure_labels(report),rollout=str(folder/method/'diagnostic_rollout.pkl'))
                reports.append(result)
                (folder/method/'phases.json').write_text(json.dumps(phases,indent=2)+'\n')
                (args.output/'progress.json').write_text(json.dumps(reports,indent=2)+'\n')
                print(json.dumps(dict(video=video['name'],case=case['name'],method=method,full_pass=result['full_pass'],lift_pass=result['lift_pass'],goal_mm=report['final_distance_m']*1000,failures=result['failure_labels'])),flush=True)
    summary={}
    for method in ['fixed']+study['learning']['methods']:
        rows=[r for r in reports if r['method']==method]
        summary[method]=dict(full_count=sum(r['full_pass'] for r in rows),lift_count=sum(r['lift_pass'] for r in rows),count=len(rows),
                            mean_goal_mm=float(np.mean([r['report']['final_distance_m'] for r in rows])*1000),
                            failure_counts=dict(Counter(f for r in rows for f in r['failure_labels'])),
                            per_video={v['name']:dict(full_count=sum(r['full_pass'] for r in rows if r['video']==v['name']),
                                lift_count=sum(r['lift_pass'] for r in rows if r['video']==v['name']),count=sum(r['video']==v['name'] for r in rows)) for v in study['videos']})
        if method!='fixed':
            baseline={(r['video'],r['case']):r for r in reports if r['method']=='fixed'}
            summary[method]['paired_full_wins']=sum(r['full_pass'] and not baseline[r['video'],r['case']]['full_pass'] for r in rows)
            summary[method]['paired_full_losses']=sum(not r['full_pass'] and baseline[r['video'],r['case']]['full_pass'] for r in rows)
    result=dict(protocol_sha256=sha,frozen_sha256=digest(args.frozen),summary=summary,reports=reports,
                completed=True,training_after_heldout=False,long_training_started=False,
                scope='20 synthetic perturbations of two cached real sequences; not unseen videos, objects or hardware')
    (args.output/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print('SUMMARY',json.dumps(summary),flush=True)


if __name__=='__main__': main()
