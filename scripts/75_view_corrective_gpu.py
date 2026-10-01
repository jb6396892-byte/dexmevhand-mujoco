#!/usr/bin/env python3
"""Open the native viewer for a v11 expert or frozen-policy action replay."""
import argparse
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT/'data/processed/dual_video_v11'


def replay_paths(video, expert=False):
    case = 'video_seed_0' if video == 'first' else 'nominal'
    admission = json.loads((RUN/'experts'/video/case/'admission.json').read_text())
    if expert:
        rollout = Path(admission['rollout'])
    else:
        frozen = json.loads((RUN/'learning/frozen_policy.json').read_text())
        rollout = RUN/'learning'/frozen['selected']['label']/video/case/'diagnostic_rollout.pkl'
    geometry = Path(admission['geometry'])
    if not rollout.exists() or not geometry.exists():
        raise FileNotFoundError('The selected local rollout or geometry is missing')
    return rollout, geometry


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('video', choices=['first', 'second'], nargs='?', default='second')
    parser.add_argument('--expert', action='store_true')
    args, viewer_args = parser.parse_known_args()
    rollout, geometry = replay_paths(args.video, args.expert)
    print('v11 %s: saved %s actions replayed by physics; not online network inference.' %
          (args.video, 'expert' if args.expert else 'frozen-policy'), flush=True)
    command = ['bash', str(ROOT/'scripts/31_view_video_faithful_gpu.sh'),
               '--rollout', str(rollout), '--geometry', str(geometry),
               '--source-sequence', str(ROOT/'data/real_data/relocate_mug'/
                                        ('seq_dexycb_001' if args.video == 'first' else 'seq_dexycb_002'))]
    os.execvp(command[0], command+viewer_args)


if __name__ == '__main__':
    main()
