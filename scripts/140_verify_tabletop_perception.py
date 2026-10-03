#!/usr/bin/env python3
"""Freeze perception sources, execute disjoint layouts, evaluate only after inference."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from fromrealhand.desktop.runtime import physics_environment,LEGACY_PYTHON


def read(path): return json.loads(path.read_text())
def write(path,value): path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def execute(command,log,env=None):
    result=subprocess.run(list(map(str,command)),cwd=str(ROOT),env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=180)
    log.write_text(result.stdout+'\nSTDERR\n'+result.stderr)
    if result.returncode: raise RuntimeError('Command failed: '+str(log)+'\n'+result.stderr[-2500:])


def sources():
    paths=list((ROOT/'src/fromrealhand/tabletop').glob('*.py'))+list((ROOT/'src/fromrealhand/perception').glob('*.py'))
    paths += [ROOT/'scripts'/name for name in ('136_capture_tabletop.py','138_estimate_tabletop.py','140_verify_tabletop_perception.py')]
    paths += [ROOT/'configs/tabletop-perception-v1.json']
    return {str(p.relative_to(ROOT)):digest(p) for p in sorted(paths)}


def evaluate(folder):
    result=read(folder/'estimate/summary.json'); rows=[]
    for row in result['results']:
        truth=read(folder/'evaluation_only'/(row['frame']+'-truth.json'))
        entry=dict(frame=row['frame'],accepted=row['accepted'],reason=row['reason'],latency_s=row['total_s'])
        if row['accepted'] and 'mug' in truth:
            T=np.asarray(row['T_world_object']); expected=np.asarray(truth['mug'])
            entry.update(position_error_m=float(np.linalg.norm(T[:3,3]-expected[:3,3])),
                rotation_error_deg=float(np.rad2deg(np.arccos(np.clip((np.trace(T[:3,:3].T@expected[:3,:3])-1)/2,-1,1)))))
        rows.append(entry)
    return rows


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--output',type=Path,required=True)
    p.add_argument('--mode',choices=['development','heldout'],default='heldout')
    args=p.parse_args(); args.output.mkdir(parents=True,exist_ok=False)
    cfg=read(ROOT/'configs/tabletop-perception-v1.json'); before=sources()
    visionroot=Path(os.environ['VISUAL_GRASP_ROOT'])
    write(args.output/'protocol.json',dict(created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        config=cfg,mode=args.mode,source_sha256=before,detector=read(visionroot/'models/grounding-dino-tiny/source.json'),
        perception_process_receives_evaluation_directory=False))
    seeds=cfg['development_seeds'] if args.mode=='development' else cfg['heldout_seeds']
    cases=[]; mesh=None
    all_seeds=[(s,False) for s in seeds]+([(s,True) for s in cfg['negative_seeds']] if args.mode=='heldout' else [])
    for seed,negative in all_seeds:
        folder=args.output/('seed-%d'%seed)
        command=[LEGACY_PYTHON,ROOT/'scripts/136_capture_tabletop.py','--seed',seed,'--frames','1','--output',folder]
        if negative: command.append('--no-target')
        execute(command,args.output/('capture-%d.log'%seed),physics_environment())
        if mesh is None: mesh=folder/'mug_model.npz'
        if not negative and digest(mesh)!=digest(folder/'mug_model.npz'):
            # npz container timestamps are ignored; canonical geometry must be invariant.
            with np.load(mesh) as a,np.load(folder/'mug_model.npz') as b:
                if not all(np.array_equal(a[k],b[k]) for k in a.files): raise ValueError('CAD depends on layout')
        obs=folder/'observations'
        if not negative and args.mode=='heldout':
            rng=np.random.RandomState(seed+9000); depth=np.load(obs/'0000-depth.npy')
            depth=(depth+rng.normal(0,cfg['noise_std_m'],depth.shape)).astype(np.float32)
            depth[rng.rand(*depth.shape)<cfg['depth_dropout_fraction']]=0
            np.save(obs/'0001-depth.npy',depth)
            for suffix in ('rgb.png','camera.json'): shutil.copyfile(str(obs/('0000-'+suffix)),str(obs/('0001-'+suffix)))
            shutil.copyfile(str(folder/'evaluation_only/0000-truth.json'),str(folder/'evaluation_only/0001-truth.json'))
        if args.mode=='heldout' and seed==seeds[0]:
            np.save(obs/'0002-depth.npy',np.zeros((720,960),np.float32))
            for suffix in ('rgb.png','camera.json'): shutil.copyfile(str(obs/('0000-'+suffix)),str(obs/('0002-'+suffix)))
            shutil.copyfile(str(folder/'evaluation_only/0000-truth.json'),str(folder/'evaluation_only/0002-truth.json'))
        execute(['bash','scripts/tabletop_python.sh','scripts/138_estimate_tabletop.py',
            '--observations',obs,'--mesh',mesh,'--output',folder/'estimate'],args.output/('estimate-%d.log'%seed))
        rows=evaluate(folder)
        cases.append(dict(seed=seed,negative=negative,rows=rows,capture=read(folder/'capture-report.json')))
        print('CASE',seed,json.dumps(rows),flush=True)
        write(args.output/'progress.json',cases)
    if sources()!=before: raise ValueError('Perception sources changed during evaluation')
    metrics={}
    for frame,name in [('0000','clean'),('0001','noisy')]:
        rows=[r for c in cases if not c['negative'] for r in c['rows'] if r['frame']==frame]
        accepted=[r for r in rows if r['accepted']]
        if rows:
            metrics[name]=dict(total=len(rows),accepted=len(accepted),
                position_median_m=float(np.median([r['position_error_m'] for r in accepted])) if accepted else None,
                position_p95_m=float(np.percentile([r['position_error_m'] for r in accepted],95)) if accepted else None,
                rotation_median_deg=float(np.median([r['rotation_error_deg'] for r in accepted])) if accepted else None,
                rotation_max_deg=max([r['rotation_error_deg'] for r in accepted],default=None),
                inference_median_s=float(np.median([r['latency_s'] for r in rows])))
    negatives=[r for c in cases for r in c['rows'] if c['negative'] or r['frame']=='0002']
    passed=all(m['accepted']/m['total']>=cfg['required_admission_fraction']
        and m['position_median_m']<=cfg['position_median_limit_m'] and m['position_p95_m']<=cfg['position_p95_limit_m']
        and m['rotation_median_deg']<=cfg['rotation_median_limit_deg'] for m in metrics.values())
    passed=passed and all(not r['accepted'] for r in negatives) and all(c['capture']['passed'] for c in cases)
    report=dict(passed=passed,mode=args.mode,metrics=metrics,cases=cases,
        negatives=len(negatives),negative_rejections=sum(not r['accepted'] for r in negatives),
        control_enabled=False,scope=cfg['scope'],truth_used_only_after_inference=True)
    write(args.output/'report.json',report); print('SUMMARY',json.dumps({k:v for k,v in report.items() if k!='cases'}),flush=True)
    if not passed: raise SystemExit(2)


if __name__=='__main__': main()
