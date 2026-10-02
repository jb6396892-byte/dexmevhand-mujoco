#!/usr/bin/env python3
"""Frozen paired independent scene evaluation and portable final policy delivery."""
import argparse
import datetime
import importlib
import json
import pickle
from pathlib import Path
import shutil
import subprocess
import numpy as np
import torch

from v10_common import ROOT, digest, run_case, setup_case, surface
from fromrealhand.residual_sampling import ResidualSampler
from fromrealhand.language_planner.contracts import read, write

old = importlib.import_module('77_run_v12_study')
CONFIG = ROOT/'configs/stage3-post200-eval-v1.json'


def signature(case):
    return tuple(np.round(case['cup_offset_m']+case['goal_offset_m']+[case['cup_yaw_deg']], 12))


def freeze(folder):
    folder.mkdir(parents=True, exist_ok=False)
    cfg = read(CONFIG)
    rng = np.random.RandomState(cfg['seed'])
    cases = [dict(name='unseen_%02d' % i,
        cup_offset_m=list(rng.uniform(-cfg['cup_xy_bound_m'], cfg['cup_xy_bound_m'], 2))+[0.],
        goal_offset_m=list(rng.uniform(-cfg['goal_xy_bound_m'], cfg['goal_xy_bound_m'], 2))+[0.],
        cup_yaw_deg=float(rng.uniform(-cfg['yaw_bound_deg'], cfg['yaw_bound_deg'])))
        for i in range(cfg['count_per_video'])]
    previous = []
    for path in sorted((ROOT/'configs').glob('v*-study.json')):
        spec = read(path).get('heldout', {})
        if all(k in spec for k in ('seed', 'count_per_video', 'cup_xy_bound_m', 'goal_xy_bound_m', 'yaw_bound_deg')):
            previous.extend(old.heldout_cases({'heldout': spec}))
    if {signature(c) for c in cases} & {signature(c) for c in previous}:
        raise ValueError('Duplicate old heldout scene')
    gates, parent, admission, _, _, _ = old.load_inputs()
    paths = [CONFIG, Path(__file__).resolve(), ROOT/'configs/v12-study.json',
        ROOT/'src/fromrealhand/residual_sampling.py', ROOT/'src/fromrealhand/predictive_contact.py',
        ROOT/'scripts/v10_common.py', ROOT/'scripts/77_run_v12_study.py']
    paths += [ROOT/p for p in cfg['methods'].values()]
    for video in parent['videos']:
        paths += [ROOT/p for p in video['paths'].values()]
    receipt = dict(created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(), config=cfg,
        cases=cases, old_cases_checked=len(previous), identical_old_cases=0,
        code_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=str(ROOT), universal_newlines=True).strip(),
        sha256={str(p.relative_to(ROOT)): digest(p) for p in paths}, evaluation_started=False)
    write(folder/'protocol.json', receipt)
    for method, source in cfg['methods'].items():
        target = folder/'models'/(method+'.pickle')
        target.parent.mkdir(exist_ok=True)
        shutil.copy2(str(ROOT/source), str(target))
    print('FROZEN', str(folder), digest(folder/'protocol.json'), flush=True)


def verify(folder):
    receipt = read(folder/'protocol.json')
    for name, sha in receipt['sha256'].items():
        if digest(ROOT/name) != sha:
            raise ValueError('Frozen evaluation input changed: '+name)
    for method, source in receipt['config']['methods'].items():
        if digest(folder/'models'/(method+'.pickle')) != receipt['sha256'][source]:
            raise ValueError('Shared policy copy changed')
    return receipt


def rollout(cp, video, source, half=False):
    box = []
    def factory(exp):
        controller = ResidualSampler(cp, exp, video, surface.video.contact_details, deterministic=True)
        box.append(controller)
        return controller
    report, demo = run_case(video, source, None, half=half, seed=0, action_factory=factory)
    logs = box[0].predictive_filter.logs if box[0].predictive_filter else []
    errors = [x['prediction_error'] for x in logs if x['prediction_error'] is not None]
    audit = dict(prediction_error_max=max(errors or [0.]),
        modified_steps=sum(max(x['offset_max_rad'], x.get('root_offset_max_m', 0.)) > 1e-9 for x in logs))
    if audit['prediction_error_max'] > 1e-7:
        raise RuntimeError('Predictive and live physics diverged')
    return report, demo, audit


def evaluate(folder):
    receipt = verify(folder)
    cfg = receipt['config']
    if (folder/'summary.json').exists():
        raise ValueError('Evaluation already completed; do not overwrite')
    torch.set_num_threads(1)
    gates, parent, admission, _, _, _ = old.load_inputs()
    checkpoints = {m: pickle.loads((folder/'models'/(m+'.pickle')).read_bytes()) for m in cfg['methods']}
    if checkpoints['pretrain']['predictive_filter'] != checkpoints['post200']['predictive_filter']:
        raise ValueError('Paired methods must have the same physical action filter')
    rows = read(folder/'partial.json') if (folder/'partial.json').exists() else []
    done = {(r['video'], r['case'], r['method']) for r in rows}
    for video in parent['videos']:
        for case in receipt['cases']:
            source, _ = setup_case(video, case, folder/'scenes'/video['name']/case['name'])
            for method, cp in checkpoints.items():
                if (video['name'], case['name'], method) in done:
                    continue
                report, demo, audit = rollout(cp, video, source)
                target = folder/'rollouts'/video['name']/case['name']/(method+'.pickle')
                target.parent.mkdir(parents=True, exist_ok=True)
                with target.open('xb') as stream:
                    pickle.dump({'video_faithful': demo}, stream)
                row = dict(video=video['name'], case=case['name'], method=method, report=report,
                           audit=audit, **old.gates(report, gates))
                rows.append(row)
                write(folder/'partial.json', rows)
                print(json.dumps(dict(completed=len(rows), total=64, video=video['name'], case=case['name'],
                    method=method, task=row['task_pass'], strict=row['strict_pass'],
                    depth_mm=report['max_hand_scene_penetration_m']*1000,
                    goal_mm=report['final_distance_m']*1000)), flush=True)
    summaries = {m: old.summarize([r for r in rows if r['method'] == m]) for m in checkpoints}
    nominal = []
    for video in parent['videos']:
        entry = next(e for e in admission['reports'] if e['video'] == video['name'] and e['name'] ==
                     ('video_seed_0' if video['name'] == 'first' else 'nominal'))
        for half in (False, True):
            report, _, audit = rollout(checkpoints['post200'], video, dict(geometry=entry['geometry']), half)
            nominal.append(dict(video=video['name'], half_step=half, report=report, audit=audit, **old.gates(report, gates)))
    minimum = all(summaries['post200']['per_video'][v]['task']/cfg['count_per_video'] >= .75 and
        sum(r['lift_pass'] for r in rows if r['method'] == 'post200' and r['video'] == v)/cfg['count_per_video'] >= .875
        for v in ('first', 'second'))
    no_regression = summaries['post200']['task_count'] >= summaries['pretrain']['task_count']
    passed = minimum and no_regression and all(r['task_pass'] for r in nominal)
    pairs = [dict(video=v, case=c['name'],
        before=next(r['task_pass'] for r in rows if (r['video'],r['case'],r['method'])==(v,c['name'],'pretrain')),
        after=next(r['task_pass'] for r in rows if (r['video'],r['case'],r['method'])==(v,c['name'],'post200')))
        for v in ('first','second') for c in receipt['cases']]
    verify(folder)
    write(folder/'summary.json', dict(completed=True, protocol_sha256=digest(folder/'protocol.json'),
        summary=summaries, nominal=nominal, paired=pairs, reports=rows,
        stage3_scoped_acceptance=bool(passed), no_task_regression=bool(no_regression),
        task_threshold_passed=bool(minimum), no_test_tuning=True, automatic_policy_replacement=False,
        scope=cfg['scope'], closed_loop_method='Residual policy plus unchanged predictive physics filter'))
    print('COMPLETE', json.dumps(summaries), 'stage3_scoped_acceptance', passed, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['freeze', 'evaluate', 'verify'])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    {'freeze': freeze, 'evaluate': evaluate, 'verify': verify}[args.command](args.output.resolve())
