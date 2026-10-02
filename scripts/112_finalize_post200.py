#!/usr/bin/env python3
"""Stage3-only result publication. Never writes into the language-model folder."""
import argparse
from collections import Counter
import importlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from fromrealhand.language_planner.contracts import read, write, digest


def wilson(success, total):
    z = 1.96
    p = success/total
    denominator = 1+z*z/total
    center = (p+z*z/(2*total))/denominator
    half = z*math.sqrt(p*(1-p)/total+z*z/(4*total*total))/denominator
    return [center-half, center+half]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'data/processed/dual_video_v14c/post200_independent_v1')
    args = parser.parse_args()
    run = args.output.resolve()
    verifier = importlib.import_module('109_evaluate_post200')
    verifier.verify(run)
    result = read(run/'summary.json')
    if not result['completed'] or len(result['reports']) != 64:
        raise ValueError('Incomplete independent evaluation')
    publication = ROOT/'docs/presentation/post200/evidence/v1'
    publication.mkdir(parents=True,exist_ok=False)
    failures = []
    for row in result['reports']:
        if not row['task_pass'] or not row['strict_pass']:
            failures.append(dict(video=row['video'],case=row['case'],method=row['method'],
                task_pass=row['task_pass'],strict_pass=row['strict_pass'],
                labels=row['failures_strict'],goal_mm=row['report']['final_distance_m']*1000,
                penetration_mm=row['report']['max_hand_scene_penetration_m']*1000))
    metrics = {}
    for method in ('pretrain','post200'):
        subset = [r for r in result['reports'] if r['method'] == method]
        metrics[method] = dict(result['summary'][method], task_wilson95=wilson(sum(r['task_pass'] for r in subset),len(subset)),
            failure_counts=dict(Counter(label for r in subset for label in r['failures_strict'])),
            per_video_lift={v:sum(r['lift_pass'] for r in subset if r['video']==v) for v in ('first','second')})
    changed = dict(improved=sum(not p['before'] and p['after'] for p in result['paired']),
                   regressed=sum(p['before'] and not p['after'] for p in result['paired']))
    screenshots = []
    for video in ('first','second'):
        row = next(r for r in result['reports'] if r['video']==video and r['method']=='post200' and r['case']=='unseen_00')
        geometry = run/'scenes'/video/row['case']/'geometry.npz'
        rollout = run/'rollouts'/video/row['case']/'post200.pickle'
        target = run/'renders'/video
        subprocess.run([sys.executable,str(ROOT/'scripts/31_view_video_faithful.py'),
            '--rollout',str(rollout),'--geometry',str(geometry),'--simulation-only','--output',str(target),
            '--frame-steps',str(row['report']['steps']-1)],cwd=str(ROOT),check=True)
        replay = read(target/'comparison.json')
        if replay['replay_max_observation_error'] > 1e-8:
            raise ValueError('Screenshot action replay differs from evaluated trajectory')
        import cv2
        source = target/'keyframes.jpg'
        image = cv2.imread(str(source))
        if image is None or image.std() < 5:
            raise ValueError('Blank screenshot')
        shutil.copyfile(str(source),str(publication/(video+'.jpg')))
        screenshots.append(dict(video=video,case=row['case'],task_pass=row['task_pass'],
            replay_error=replay['replay_max_observation_error'],pixel_std=float(image.std()),
            sha256=digest(source),scope='Saved actions from online post200 policy, physically replayed; not qpos animation'))
    for name in ('protocol.json','summary.json'):
        shutil.copyfile(str(run/name),str(publication/name))
    write(publication/'metrics.json',dict(metrics=metrics,paired_changes=changed,
        stage3_scoped_acceptance=result['stage3_scoped_acceptance'],
        generalized_to_new_videos=False,real_robot_tested=False,exact_human_fidelity_solved=False))
    write(publication/'failures.json',failures)
    write(publication/'screenshots.json',screenshots)
    write(run/'delivery.json',dict(source_checkpoint=str(ROOT/'data/processed/dual_video_v14c/long_training/iteration_0200.pickle'),
        source_sha256=digest(ROOT/'data/processed/dual_video_v14c/long_training/iteration_0200.pickle'),
        evaluation=str(run),documents=str(publication),automatic_replacement=False,
        namespace='stage3; separate from /media/smgbro/shared/lora'))
    print(json.dumps(dict(metrics=metrics,paired_changes=changed),indent=2))


if __name__=='__main__':
    main()
