#!/usr/bin/env python3
"""Run predeclared development episodes only; never start learning or heldout tests."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from hierarchy_common import ROOT,read,write


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--videos',nargs='+',choices=['first','second'],default=['first','second'])
    p.add_argument('--seeds',nargs='+',type=int,default=[0,1,2,3])
    a=p.parse_args(); protocol=read(ROOT/'configs/tabletop-dual-v3-protocol.json')
    allowed=protocol['development_train_seeds']+protocol['development_validation_seeds']
    if any(s not in allowed for s in a.seeds): raise ValueError('Only development seeds may be collected')
    a.output.mkdir(parents=True,exist_ok=False)
    profile_path=ROOT/'configs/tabletop-dual-v3-profiles.json'; profiles=read(profile_path)
    write(a.output/'protocol.json',protocol); write(a.output/'profiles.json',profiles)
    records=[]
    for video in a.videos:
        cfg=profiles['profiles'][video]
        for seed in a.seeds:
            output=a.output/(video+'-seed-'+str(seed))
            cmd=[sys.executable,str(ROOT/'scripts/153_test_contact_tracking.py'),'--scene',video,
                '--seed',str(seed),'--output',str(output),'--live-vision','--camera',profiles['camera']]
            for key,value in cfg.items():
                flag='--'+key.replace('_','-')
                if isinstance(value,bool):
                    if value: cmd.append(flag)
                elif isinstance(value,list): cmd.extend([flag]+[str(v) for v in value])
                else: cmd.extend([flag,str(value)])
            with (a.output/(output.name+'.log')).open('w') as log:
                result=subprocess.run(cmd,cwd=str(ROOT),stdout=log,stderr=subprocess.STDOUT)
            report=read(output/'report.json') if (output/'report.json').exists() else {}
            row=dict(video=video,seed=seed,output=str(output),returncode=result.returncode,
                passed=report.get('positive_demonstration',False),reason=report.get('reason','worker failed'),
                split='train' if seed in protocol['development_train_seeds'] else 'validation',command=cmd)
            records.append(row); write(a.output/'collection.json',dict(records=records,training=False,
                final_holdout_executed=False,profile_sha256=hashlib.sha256(profile_path.read_bytes()).hexdigest()))
            print(json.dumps(row),flush=True)


if __name__=='__main__': main()
