#!/usr/bin/env python3
"""Open a native MuJoCo saved-action replay of the v12 selected policy."""
import argparse
import json
import os
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'data/processed/dual_video_v12'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('video',choices=['first','second'],nargs='?',default='second')
    parser.add_argument('--dapg',action='store_true',help='Show the smoke checkpoint, not the frozen BC')
    args,extra=parser.parse_known_args()
    frozen=json.loads((RUN/'frozen-policy.json').read_text())
    mode='dapg_smoke' if args.dapg else frozen['selected']['method']
    rollout=RUN/mode/(args.video+'_rollout.pkl')
    entries=json.loads((ROOT/'data/processed/dual_video_v11/experts/admission.json').read_text())['reports']
    entry=next(r for r in entries if r['video']==args.video and r['name']==('video_seed_0' if args.video=='first' else 'nominal'))
    if not rollout.exists(): raise FileNotFoundError(rollout)
    print('v12 %s %s: saved closed-loop actions, replayed through physics; no online inference.'%(args.video,mode),flush=True)
    command=['bash',str(ROOT/'scripts/31_view_video_faithful_gpu.sh'),'--rollout',str(rollout),
             '--geometry',entry['geometry'],'--simulation-only']+extra
    os.execvp(command[0],command)


if __name__=='__main__': main()
