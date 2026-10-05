#!/usr/bin/env python3
"""Archive development evidence without promoting a candidate or starting training."""
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from fromrealhand.desktop.runtime import LEGACY_PYTHON,physics_environment
STORE=Path('/media/smgbro/shared/visual_grasp/contact-control-v2')
DEST=ROOT/'docs/presentation/tabletop_contact_v2/evidence'


def read(p): return json.loads(p.read_text())
def write(p,data): p.write_text(json.dumps(data,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()


def check(command):
    env=physics_environment(); env['TMPDIR']='/tmp'
    r=subprocess.run(command,cwd=str(ROOT),env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,
        universal_newlines=True,timeout=180)
    if r.returncode: raise RuntimeError(r.stdout+r.stderr)
    return dict(command=command,exit_code=r.returncode,stdout=r.stdout,stderr=r.stderr)


def main():
    DEST.mkdir(parents=True,exist_ok=True)
    qt=STORE/'qt-regression'; quality=read(qt/'quality.json')
    if not all(quality.values()): raise ValueError('Desktop checks failed')
    final=read(qt/'transport/summary.json'); physics=Path(final['physics_run'])
    sources=read(physics/'source.json')
    for name,sha in sources.items():
        if digest(ROOT/name)!=sha: raise ValueError('Source changed after final test: '+name)
    tests=check([LEGACY_PYTHON,'-m','unittest','discover','-s','tests'])
    guard=check([LEGACY_PYTHON,'scripts/122_verify_guard_revision.py','verify'])
    write(DEST/'quality.json',dict(qt=quality,regression=tests,frozen_language=guard,source_hashes_match=True,
        independent_generalization_test=False,training_started=False,automatic_execution_enabled=False))
    cases={}
    for name in ('transport','stop','locked','second-rejected','instruction-rejected'):
        cases[name]=read(qt/name/'summary.json')
    write(DEST/'qt-cases.json',cases)
    for name in ('report','evaluation_only','adaptation','request','source','preflight','scene'):
        shutil.copyfile(str(physics/(name+'.json')),str(DEST/('final-'+name+'.json')))
    development={}
    for directory in sorted(STORE.iterdir()):
        if directory.is_dir() and (directory/'report.json').exists(): development[directory.name]=read(directory/'report.json')
    write(DEST/'development-cases.json',development)
    for name in ('lift-tracking-surface-regression.json','airborne-tracking-regression.json'):
        shutil.copyfile(str(STORE/name),str(DEST/name))
    shutil.copyfile(str(STORE/'diagnosis-initial/diagnosis.json'),str(DEST/'kinematic-diagnosis.json'))
    shutil.copyfile(str(qt/'transport/qt-result.png'),str(DEST/'qt-transport-success.png'))
    shutil.copyfile(str(STORE/'live-grasp-gain1/grasp.png'),str(DEST/'stable-grasp.png'))
    shutil.copyfile(str(qt/'transport/physics-final.png'),str(DEST/'transport-physics.png'))
    trace=read(physics/'trace.json'); report=read(physics/'report.json'); evaluation=read(physics/'evaluation_only.json')
    goal=read(physics/'adaptation.json')['goal_world_m']; truth=evaluation['ground_truth_after_stop']
    phases={}
    for phase in ('reach','grasp','lift','transport'):
        rows=[r for r in trace if r['skill']==phase]
        phases[phase]=dict(steps=len(rows),peak_penetration_m=max(r['scene_penetration_m'] for r in rows),
            max_joint_violation_rad=max(r['joint_violation_rad'] for r in rows),
            non_target_contact_steps=sum(r['non_target_contacts']>0 for r in rows),
            final_bottom_m=rows[-1]['bottom_m'],final_goal_distance_m=rows[-1]['target_distance_m'])
    root0=development['ablation-root0-v2']; root1=development['ablation-root1']
    summary=dict(scope='first reference, seed 0, development only',final=report,phases=phases,
        oracle_after_stop_goal_distance_m=math.sqrt(sum((truth[i][3]-goal[i])**2 for i in range(3))),
        visual_position_error_m=evaluation['visual_position_error_m'],
        root_ablation=dict(root0_error_m=root0['final']['root_error_m'],root1_error_m=root1['final']['root_error_m'],
            relative_error_reduction=1-root1['final']['root_error_m']/root0['final']['root_error_m'],
            both_grasp_passed=root0['passed'] and root1['passed'],cached_anchor_screening=True),
        tip_feedback_ablation=dict(passed=development['ablation-tip40']['passed'],
            reason=development['ablation-tip40']['reason'],selected_gain=0.),
        first_qt=read(STORE/'qt-final/summary.json'),final_qt=final,
        stop_latency_s=cases['stop']['cancellation_latency_s'],state_writes_during_execution=0,
        trained=False,independent_test=False)
    write(DEST/'summary.json',summary)
    files=[p for p in DEST.iterdir() if p.is_file() and p.name!='manifest.json']
    write(DEST/'manifest.json',dict(raw_store=str(STORE),physics_run=str(physics),
        sha256={p.name:digest(p) for p in files}))
    archive=STORE/'delivery'; archive.mkdir(exist_ok=True)
    for f in DEST.iterdir():
        if f.is_file(): shutil.copyfile(str(f),str(archive/f.name))
    print(json.dumps(dict(evidence=str(DEST),physics_run=str(physics),steps=report['steps'],
        success=report['status']=='success',root_error_reduction=summary['root_ablation']['relative_error_reduction'],
        oracle_goal_error_m=summary['oracle_after_stop_goal_distance_m'],stop_latency_s=summary['stop_latency_s']),ensure_ascii=False))


if __name__=='__main__': main()
