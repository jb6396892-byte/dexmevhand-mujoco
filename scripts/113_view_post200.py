#!/usr/bin/env python3
"""Native-window replay of verified actions from the final 200-update policy."""
import argparse
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('video', choices=['first','second'], nargs='?', default='second')
    parser.add_argument('--case', default='nominal')
    parser.add_argument('--dry-run', action='store_true')
    args, extra = parser.parse_known_args()
    run = ROOT/'data/processed/dual_video_v14c'
    if args.case == 'nominal':
        entries = json.loads((ROOT/'data/processed/dual_video_v11/experts/admission.json').read_text())['reports']
        entry = next(r for r in entries if r['video']==args.video and r['name']==
                     ('video_seed_0' if args.video=='first' else 'nominal'))
        geometry = Path(entry['geometry'])
        rollout = run/'long_training'/(args.video+'_rollout.pkl')
    else:
        evaluation = run/'post200_independent_v1'
        cases = json.loads((evaluation/'protocol.json').read_text())['cases']
        if args.case not in {c['name'] for c in cases}:
            parser.error('Case must be nominal or one of the frozen unseen_00 through unseen_15 cases')
        geometry = evaluation/'scenes'/args.video/args.case/'geometry.npz'
        rollout = evaluation/'rollouts'/args.video/args.case/'post200.pickle'
    if not geometry.is_file() or not rollout.is_file():
        parser.error('Requested policy rollout has not been generated')
    if args.dry_run:
        print(json.dumps(dict(geometry=str(geometry), rollout=str(rollout),
            mode='saved online-policy actions through env.step; free object; not fresh network inference'),indent=2))
        return
    print('Final post200 policy action replay, not kinematic object animation. Close with Ctrl+C.',flush=True)
    command = ['bash',str(ROOT/'scripts/31_view_video_faithful_gpu.sh'),
               '--rollout',str(rollout),'--geometry',str(geometry),'--simulation-only']+extra
    os.execvp(command[0],command)


if __name__ == '__main__':
    main()
