#!/usr/bin/env python3
"""Archive compact reproducible evidence, not a generalization-success claim."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import numpy as np
from hierarchy_common import ROOT,read,write


def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--store',type=Path,default=Path('/media/smgbro/shared/visual_grasp/dual-contact-v3'))
    a=p.parse_args(); store=a.store
    dest=ROOT/'docs/presentation/tabletop_dual_v3/evidence'; dest.mkdir(parents=True,exist_ok=True)
    readiness=read(store/'training-inputs/readiness.json'); qt=read(store/'qt-final/quality.json')
    if not readiness['ready_for_small_scale_generalization_training']: raise ValueError('Training input gate failed')
    if not all(v for k,v in qt.items() if k!='training_started'): raise ValueError('Qt gate failed')
    if qt['training_started'] or readiness['training_started']: raise ValueError('Unexpected training')
    collection=read(store/'frozen-development/collection.json'); matrix=[]
    for case in collection['records']:
        run=Path(case['output']); report=read(run/'report.json'); goal=read(run/'goal.json')['goal_world_m']
        for name,sha in read(run/'sources.json').items():
            if digest(ROOT/name)!=sha: raise ValueError('Source drift: '+name)
        initial=read(run/'observations/000000/estimate.json')['T_world_object']
        truth=np.asarray(report['ground_truth_after_stop'])
        matrix.append(dict(video=case['video'],seed=case['seed'],split=case['split'],passed=report['passed'],
            steps=report['steps'],max_penetration_m=report['max_penetration_m'],
            visual_goal_error_m=report['final']['target_distance_m'],
            actual_goal_error_after_stop_m=float(np.linalg.norm(truth[:3,3]-goal)),
            final_bottom_m=report['final']['bottom_m'],events=report['events'],
            orientation_fraction=read(run/'preflight.json')['orientation_transfer_fraction'],
            initial_visual_pose=initial,goal_world_m=goal,dynamics=read(run/'dynamics.json'),
            run=str(run),report_sha256=digest(run/'report.json')))
    for video in ('first','second'):
        q=read(store/'qt-final'/(video+'-transport')/'summary.json')
        physics=Path(q['physics_run'])
        for name,sha in read(physics/'source.json').items():
            if digest(ROOT/name)!=sha: raise ValueError('Qt source drift: '+name)
    commands=[['bash','scripts/137_tabletop_gpu.sh','-m','unittest','discover','-s','tests'],
        ['bash','scripts/137_tabletop_gpu.sh','scripts/122_verify_guard_revision.py','verify']]
    checks=[]
    for cmd in commands:
        result=subprocess.run(cmd,cwd=str(ROOT),stdout=subprocess.PIPE,stderr=subprocess.PIPE,
            universal_newlines=True,timeout=180)
        checks.append(dict(command=cmd,returncode=result.returncode,stdout=result.stdout,stderr=result.stderr))
        if result.returncode: raise ValueError('Regression failed: '+result.stderr)
    failures=[]
    for run in sorted(store.rglob('report.json')):
        if 'frozen-development' in run.parts: continue
        r=read(run)
        if r.get('passed') is False:
            failures.append(dict(run=str(run.parent),reason=r['reason'],steps=r.get('steps'),
                completed=r.get('completed'),max_penetration_m=r.get('max_penetration_m')))
    write(dest/'development-matrix.json',matrix); write(dest/'failure-history.json',failures)
    write(dest/'software-checks.json',checks)
    for name in ('readiness.json','loader_checks.json','index.json'):
        shutil.copyfile(str(store/'training-inputs'/name),str(dest/name))
    for name in ('protocol.json','profiles.json'):
        shutil.copyfile(str(store/'frozen-development'/name),str(dest/name))
    write(dest/'qt-quality.json',qt)
    summaries={name:read(store/'qt-final'/name/'summary.json') for name in
        ('first-transport','second-transport','second-stop','locked','instruction-rejected')}
    write(dest/'qt-cases.json',summaries)
    dry={name:read(store/('pretrain-dry-'+name)/'receipt.json') for name in ('shared','routed')}
    for receipt in dry.values():
        if any(receipt[k] for k in ('training_started','optimizer_created','backward_called','checkpoint_saved')):
            raise ValueError('Training boundary crossed')
    write(dest/'pretraining-dry-runs.json',dry)
    write(dest/'training-entry-sources.json',{str(f.relative_to(ROOT)):digest(f) for f in
        (ROOT/'scripts/158_package_dual_visual_inputs.py',ROOT/'scripts/161_prepare_tabletop_bc.py',
         ROOT/'configs/tabletop-dual-v3-pretraining.json')})
    second=store/'frozen-development/second-seed-0'
    for phase in ('grasp','transport'):
        shutil.copyfile(str(second/(phase+'.png')),str(dest/('second-'+phase+'.png')))
    shutil.copyfile(str(store/'qt-final/second-transport/qt-result.png'),str(dest/'second-qt.png'))
    summary=dict(scope='development demonstration preparation, not learned-policy generalization',
        passed=sum(c['passed'] for c in matrix),total=len(matrix),training_started=False,
        reserved_test_executed=False,source_hashes_verified=True,readiness=readiness,qt=qt,
        candidate_execution_default_locked=True,second_video_exact=False,
        controller='video-derived reference, constrained mapping, contact gates, visual carry correction')
    write(dest/'summary.json',summary)
    write(dest/'manifest.json',dict(raw_store=str(store),sha256={f.name:digest(f) for f in dest.iterdir()
        if f.is_file() and f.name!='manifest.json'}))
    print(json.dumps(summary,ensure_ascii=False))


if __name__=='__main__': main()
