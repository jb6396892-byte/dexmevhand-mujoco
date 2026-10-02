#!/usr/bin/env python3
"""Open native MuJoCo physics replay of saved v14c controller actions."""
import argparse
import json
import os
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('video',choices=['first','second'],nargs='?',default='second')
    parser.add_argument('--case')
    parser.add_argument('--dapg',action='store_true')
    args,extra=parser.parse_known_args()
    nominal='video_seed_0' if args.video=='first' else 'nominal'
    name=args.case or nominal
    if args.dapg and name!=nominal: parser.error('Only nominal post-DAPG rollouts are saved')
    entries=json.loads((ROOT/'data/processed/dual_video_v11/experts/admission.json').read_text())['reports']
    entry=next((r for r in entries if r['video']==args.video and r['name']==name),None)
    if entry is None: parser.error('Unknown development case for this video: '+name)
    run=ROOT/'data/processed/dual_video_v14c'
    rollout=run/('dapg_smoke/'+args.video+'_rollout.pkl' if args.dapg else
                 'development/'+args.video+'/'+name+'/rollout.pkl')
    if not rollout.exists(): raise FileNotFoundError(rollout)
    print('Saved closed-loop controller actions replayed through env.step; not online network inference.',flush=True)
    command=['bash',str(ROOT/'scripts/31_view_video_faithful_gpu.sh'),'--rollout',str(rollout),
             '--geometry',entry['geometry'],'--simulation-only']+extra
    os.execvp(command[0],command)


if __name__=='__main__': main()
