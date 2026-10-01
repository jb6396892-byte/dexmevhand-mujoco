#!/usr/bin/env python3
"""Evaluate one frozen v11 candidate against old/new action priors, without tuning."""
import argparse
import json
import pickle
import subprocess
from collections import Counter
from pathlib import Path
import numpy as np
from v11_common import RUN,protocol,heldout_cases
from v10_common import ROOT,digest,setup_case,reference_demo,run_case,full_gate,failure_labels
from fromrealhand.policy_learning import lift_success
from fromrealhand.multivideo import phase_diagnostics


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=RUN/'heldout')
    parser.add_argument('--freeze',type=Path,default=RUN/'learning/frozen_policy.json')
    args=parser.parse_args();study,parent,sha=protocol()
    frozen_path=args.freeze;frozen=json.loads(frozen_path.read_text())
    selected=frozen['selected']
    if frozen['protocol_sha256']!=sha or digest(selected['policy'])!=selected['policy_sha256']:
        raise ValueError('Frozen candidate changed')
    if digest(RUN/'experts/references.pkl')!=frozen['reference_sha256']:
        raise ValueError('Frozen references changed')
    if 'amendment_sha256' in frozen and digest(ROOT/'configs/v11-residual-gain-amendment.json')!=frozen['amendment_sha256']:
        raise ValueError('Gain amendment changed')
    receipt=ROOT/'docs/presentation/v11/evidence/frozen-policy.json'
    if not receipt.exists() or digest(receipt)!=digest(frozen_path):
        raise ValueError('Export the freeze receipt before testing')
    committed=subprocess.check_output(['git','show','HEAD:docs/presentation/v11/evidence/frozen-policy.json'],cwd=str(ROOT))
    if committed!=frozen_path.read_bytes(): raise ValueError('Freeze receipt must be committed before testing')
    new=pickle.loads(Path(selected['policy']).read_bytes())
    old=pickle.loads((ROOT/'data/processed/dual_video_v10/learning/residual_bc/epoch_050.pickle').read_bytes())
    refs=pickle.loads((RUN/'experts/references.pkl').read_bytes())
    args.output.mkdir(parents=True,exist_ok=False)
    freeze_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=str(ROOT),text=True).strip()
    (args.output/'freeze_receipt.json').write_text(json.dumps(dict(protocol_sha256=sha,frozen_sha256=digest(frozen_path),git_commit=freeze_commit),indent=2)+'\n')
    rows=[]
    for video in parent['videos']:
        for case in heldout_cases(study):
            folder=args.output/video['name']/case['name'];source,g=setup_case(video,case,folder)
            for method in study['heldout']['methods']:
                actions=reference_demo(video)['actions'] if method=='fixed_v10' else refs[video['name']]['actions']
                cp=old if method=='residual_v10' else new if method=='residual_v11' else None
                report,demo=run_case(video,source,actions,folder/method,checkpoint=cp)
                phases=phase_diagnostics(demo,refs[video['name']]['actions'],video,g)
                row=dict(video=video['name'],case=case['name'],method=method,full_pass=full_gate(report),lift_pass=lift_success(report),
                         preferred=bool(full_gate(report) and report['max_hand_scene_penetration_m']<=study['safety']['preferred_penetration_m']),
                         report=report,phases=phases,failure_labels=failure_labels(report),rollout=str(folder/method/'diagnostic_rollout.pkl'))
                rows.append(row)
                (args.output/'progress.json').write_text(json.dumps(rows,indent=2)+'\n')
                print('TEST',json.dumps(dict(video=video['name'],case=case['name'],method=method,passed=row['full_pass'],
                                            goal_mm=report['final_distance_m']*1000,depth_mm=report['max_hand_scene_penetration_m']*1000,failures=row['failure_labels'])),flush=True)
    summary={}
    for method in study['heldout']['methods']:
        subset=[r for r in rows if r['method']==method]
        summary[method]=dict(full_count=sum(r['full_pass'] for r in subset),lift_count=sum(r['lift_pass'] for r in subset),
                            preferred_count=sum(r['preferred'] for r in subset),count=len(subset),
                            mean_goal_mm=float(np.mean([r['report']['final_distance_m'] for r in subset])*1000),
                            mean_peak_penetration_mm=float(np.mean([r['report']['max_hand_scene_penetration_m'] for r in subset])*1000),
                            failures=dict(Counter(f for r in subset for f in r['failure_labels'])),
                            per_video={v['name']:sum(r['full_pass'] for r in subset if r['video']==v['name']) for v in parent['videos']})
    for baseline in ['fixed_v10','residual_v10','fixed_v11']:
        other={(r['video'],r['case']):r for r in rows if r['method']==baseline}
        current=[r for r in rows if r['method']=='residual_v11']
        summary['residual_v11']['paired_vs_'+baseline]=dict(wins=sum(r['full_pass'] and not other[r['video'],r['case']]['full_pass'] for r in current),
            losses=sum(not r['full_pass'] and other[r['video'],r['case']]['full_pass'] for r in current),
            goal_wins=sum(r['report']['final_distance_m']<other[r['video'],r['case']]['report']['final_distance_m'] for r in current))
    result=dict(protocol_sha256=sha,frozen_sha256=digest(frozen_path),freeze_commit=freeze_commit,completed=True,
                summary=summary,reports=rows,training_after_test=False,long_training_started=False,scope=study['scope'])
    (args.output/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print('SUMMARY',json.dumps(summary),flush=True)


if __name__=='__main__': main()
