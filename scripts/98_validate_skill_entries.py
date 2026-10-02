#!/usr/bin/env python3
"""Stage 4.4: honest initial-state perturbation and fault-stop diagnostics."""
import argparse
import json
from pathlib import Path
from stage4_common import ROOT,digest,read_config,save_json
from stage4_pipeline_common import load_pieces,replay
from fromrealhand.skill_pipeline import GuardedSkill


def fixture_checks(config):
    row=dict(bottom_m=.06,target_distance_m=.02,scene_penetration_m=0.,joint_violation_rad=0.,finite=True,
             th_force_n=1.,ff_force_n=1.,mf_force_n=1.,rf_force_n=0.,lf_force_n=0.)
    unsafe=GuardedSkill('lift',config,dict(row,scene_penetration_m=.002))
    invalid=GuardedSkill('lift',config,dict(row,bottom_m=float('nan')))
    entry=GuardedSkill('lift',config,dict(row,th_force_n=0.))
    loss=GuardedSkill('transport',config,row)
    for _ in range(config['support_loss_stop_steps']): loss.update(dict(row,th_force_n=0.))
    short=dict(config,timeout_steps=dict(config['timeout_steps'],reach=3))
    empty=dict(row,th_force_n=0.,ff_force_n=0.,mf_force_n=0.)
    timeout=GuardedSkill('reach',short,empty)
    for _ in range(3): timeout.update(empty)
    return dict(unsafe_entry=unsafe.reason=='scene_penetration',nonfinite=invalid.reason=='nonfinite_state',
        wrong_entry=entry.reason=='entry_condition',contact_loss=loss.reason=='persistent_support_loss',
        timeout=timeout.status=='timeout')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,default=ROOT/'data/processed/stage4_pipeline_v2')
    args=parser.parse_args()
    inputs=read_config(args.run/'inputs/receipt.json')
    if not inputs['passed'] or inputs['index_sha256']!=digest(args.run/'inputs/index.json'):
        raise ValueError('Passed 4.3 input adapter is required')
    folder=args.run/'entries';folder.mkdir(exist_ok=False)
    manifest=read_config(args.run/'build/manifest.json')
    receipt=read_config(args.run/'build/receipt.json')
    if receipt['manifest_sha256']!=digest(args.run/'build/manifest.json'):
        raise ValueError('Skill manifest changed')
    config=manifest['config']
    cases=[]
    for name in config['pilot_trajectories']:
        entry=next(e for e in manifest['trajectories'] if e['trajectory']==name)
        pieces=load_pieces(args.run/'build',entry)
        for piece in pieces:
            for offset in config['entry_offsets_m']:
                result=replay(entry,[piece],config,offset=offset)
                cases.append(result)
                save_json(folder/'cases.json',cases)
                print(json.dumps(dict(trajectory=name,skill=piece['metadata']['skill'],offset=offset,
                    passed=result['passed'],stop_reason=result['contracts'][-1]['reason'])),flush=True)
    zero=[r for r in cases if not any(r['offset_m'])]
    perturb=[r for r in cases if any(r['offset_m'])]
    faults=fixture_checks(config)
    summary=dict(passed=bool(len(zero)==8 and all(r['passed'] for r in zero) and all(faults.values())),
        nominal_pass=sum(r['passed'] for r in zero),nominal_count=len(zero),
        perturbed_pass=sum(r['passed'] for r in perturb),perturbed_count=len(perturb),
        fixture_checks=faults,source_input_receipt_sha256=digest(args.run/'inputs/receipt.json'),
        cases_sha256=digest(folder/'cases.json'),state_writes_during_execution=0,
        scope='Fixed expert action entry sensitivity, not learned policy robustness; failed offsets are not demonstrations',
        robustness_gate='Nominal execution and correct rejection required; perturbation success is diagnostic, not guaranteed')
    save_json(folder/'receipt.json',summary)
    print(json.dumps(summary),flush=True)
    if not summary['passed']: raise SystemExit(1)


if __name__=='__main__': main()
