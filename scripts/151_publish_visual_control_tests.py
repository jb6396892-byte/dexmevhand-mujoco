#!/usr/bin/env python3
"""Publish measured development results, including failures; never mark a stopped run successful."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from fromrealhand.desktop.runtime import physics_environment,LEGACY_PYTHON
STORE=Path('/media/smgbro/shared/visual_grasp/control-tests-v1')
DEST=ROOT/'docs/presentation/tabletop_control_tests/evidence'


def read(p): return json.loads(p.read_text())
def write(p,x): p.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n')
def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()


def check(command):
    env=physics_environment(); env['TMPDIR']='/tmp'
    r=subprocess.run(command,env=env,cwd=str(ROOT),stdout=subprocess.PIPE,stderr=subprocess.PIPE,
        universal_newlines=True,timeout=180)
    if r.returncode: raise RuntimeError(r.stdout+r.stderr)
    return dict(command=command,exit_code=r.returncode,stdout=r.stdout,stderr=r.stderr)


def main():
    DEST.mkdir(parents=True,exist_ok=True)
    cases={}
    for folder in sorted(STORE.glob('qt-*')):
        if (folder/'summary.json').exists(): cases[folder.name]=read(folder/'summary.json')
    final=cases['qt-first-final']; stopped=cases['qt-user-stop']; second=cases['qt-second-baseline']
    quality=dict(regression=check([LEGACY_PYTHON,'-m','unittest','discover','-s','tests']),
        frozen_language=check([LEGACY_PYTHON,'scripts/122_verify_guard_revision.py','verify']),
        all_gui_responsive=all(c['gui_responsive'] and not c['surviving_workers'] for c in cases.values()),
        default_lock=cases['qt-lock']['report']['status']=='locked',
        illegal_instruction_rejected=cases['qt-reject-handoff']['report']['status']=='rejected',
        stop_passed=stopped['report']['reason']=='user_stop' and stopped['cancellation_latency_s']<3,
        end_to_end_grasp_passed=final['task_passed'],independent_generalization_run=False)
    write(DEST/'cases.json',cases); write(DEST/'quality.json',quality)
    for filename in ['tracking-regression-v2.json','workspace-first-v2.json']:
        shutil.copyfile(str(STORE/filename),str(DEST/filename))
    for name in ['probe-first-v2','probe-second-v1']:
        shutil.copyfile(str(STORE/name/'report.json'),str(DEST/(name+'.json')))
    physics=Path(final['physics_run'])
    for name in ['report.json','evaluation_only.json','adaptation.json','preflight.json','scene.json','source.json','request.json']:
        shutil.copyfile(str(physics/name),str(DEST/('final-'+name)))
    shutil.copyfile(str(STORE/'qt-first-final/qt-result.png'),str(DEST/'first-reach-stopped.png'))
    shutil.copyfile(str(STORE/'qt-second-baseline/qt-result.png'),str(DEST/'second-pose-rejected.png'))
    trace=read(physics/'trace.json')
    contact_steps=sum(any(r[f+'_force_n']>.01 for f in ('th','ff','mf','rf','lf')) for r in trace)
    final_estimate=read(sorted((physics/'observations').glob('*/estimate.json'))[-1])
    summary=dict(scope='development diagnostics; not successful grasp or generalization',
        final_steps=final['report']['steps'],contact_steps=contact_steps,
        final_reason=final['report']['reason'],max_penetration_m=final['report']['max_hand_scene_penetration_m'],
        max_joint_violation=max(r['joint_violation_rad'] for r in trace),
        final_visual_estimate=final_estimate,oracle_after_stop=read(physics/'evaluation_only.json'),
        second_reason=second['report']['reason'],stop_latency_s=stopped['cancellation_latency_s'],
        automatic_execution_locked=True,quality= {k:v for k,v in quality.items() if k not in ('regression','frozen_language')})
    write(DEST/'summary.json',summary)
    tested_sources=read(physics/'source.json')
    write(DEST/'publication-sources.json',dict(physics_source_receipt=tested_sources,
        current_sha256={str(p.relative_to(ROOT)):digest(p) for p in [
            ROOT/'src/fromrealhand/perception/tracking.py', ROOT/'tests/test_tabletop_control_candidate.py']},
        post_physics_change='Added explicit detection box boundary validation; reran cached perception regression and unit tests; full physics run predates this bounds-only change.'))
    files=[p for p in DEST.iterdir() if p.is_file() and p.name!='manifest.json']
    write(DEST/'manifest.json',dict(sha256={p.name:digest(p) for p in files},raw_store=str(STORE)))
    archive=STORE/'delivery'; archive.mkdir(exist_ok=True)
    for path in DEST.iterdir():
        if path.is_file(): shutil.copyfile(str(path),str(archive/path.name))
    print(json.dumps(dict(evidence=str(DEST),grasp_passed=quality['end_to_end_grasp_passed'],
                         contact_steps=contact_steps,steps=final['report']['steps']),ensure_ascii=False))


if __name__=='__main__': main()
