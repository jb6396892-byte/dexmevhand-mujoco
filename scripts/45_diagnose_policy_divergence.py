#!/usr/bin/env python3
"""Compare matched expert/policy initial states, including sustained divergence."""
import argparse
import csv
import json
import pickle
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from fromrealhand.paths import configure_runtime_paths
from fromrealhand.video_fidelity import source_clock


def first_sustained(mask, count=10):
    if count < 1:
        raise ValueError('Sustain count must be positive')
    if len(mask) < count:
        return None
    hits = np.flatnonzero(np.convolve(np.asarray(mask, dtype=int), np.ones(count, dtype=int), 'valid') == count)
    return int(hits[0]) if len(hits) else None


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--demo', type=Path, default=ROOT/'data/demonstrations/relocate-mug-video-faithful-v3.pkl')
    p.add_argument('--evaluation', type=Path, default=ROOT/'data/processed/seq_dexycb_001/scene_fidelity_v3/policy_evaluation')
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    configure_runtime_paths()
    import torch
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    evaluation = json.loads((args.evaluation/'evaluation.json').read_text())
    with Path(evaluation['policy']).open('rb') as stream:
        policy = pickle.load(stream)
    with args.demo.open('rb') as stream:
        demos = pickle.load(stream)
    summaries = []
    fig, axes = plt.subplots(3, 2, figsize=(12, 10), sharex=True)
    for seed in range(6):
        expert = demos['video_seed_%d'%seed]
        with (args.evaluation/('seed_%d'%seed)/'diagnostic_rollout.pkl').open('rb') as stream:
            actual = pickle.load(stream)['video_faithful']
        qref = np.stack([s['qpos'] for s in expert['sim_data']])
        qactual = np.stack([s['qpos'] for s in actual['sim_data']])
        initial_error = float(np.max(np.abs(qref[0]-qactual[0])))
        if initial_error > 1e-10:
            raise ValueError('Unmatched initial states for seed %d'%seed)
        n = len(qref)
        if len(qactual) != n:
            raise ValueError('Rollout horizons differ')
        time = np.arange(n)*.01
        frame = 3+30*source_clock(time, 70/30, 5.)
        stages = np.select([frame < 25.5, frame < 40.5, frame < 55], ['approach', 'close', 'lift'], 'transport')
        with torch.no_grad():
            offline = policy.model(torch.as_tensor(expert['observations'], dtype=torch.float32)).numpy()
        errors = dict(action_rmse=np.sqrt(np.mean((actual['actions']-expert['actions'])**2, axis=1)),
                      offline_action_rmse=np.sqrt(np.mean((offline-expert['actions'])**2, axis=1)),
                      root_translation_m=np.linalg.norm(qactual[:, :3]-qref[:, :3], axis=1),
                      finger_joint_rmse_rad=np.sqrt(np.mean((qactual[:, 6:30]-qref[:, 6:30])**2, axis=1)),
                      object_position_m=np.linalg.norm(qactual[:, 30:33]-qref[:, 30:33], axis=1))
        thresholds = dict(action_rmse=.05, root_translation_m=.01, finger_joint_rmse_rad=.15, object_position_m=.005)
        events = {}
        for key, limit in thresholds.items():
            index = first_sustained(errors[key] > limit)
            events[key] = None if index is None else dict(step=index, time_s=float(time[index]),
                source_frame=float(frame[index]), phase=str(stages[index]), threshold=limit)
        stage_metrics = {stage: {key: float(value[stages == stage].mean()) for key, value in errors.items()}
                         for stage in ('approach', 'close', 'lift', 'transport')}
        summaries.append(dict(seed=seed, initial_state_error=initial_error, first_sustained_100ms=events,
                              phase_metrics=stage_metrics))
        with (args.output/('seed_%d.csv'%seed)).open('w') as stream:
            writer = csv.DictWriter(stream, fieldnames=['step', 'time_s', 'source_frame', 'phase']+list(errors))
            writer.writeheader()
            for i in range(n):
                writer.writerow(dict(step=i, time_s=time[i], source_frame=frame[i], phase=stages[i],
                                     **{key: value[i] for key, value in errors.items()}))
        ax = axes.flat[seed]
        ax.plot(time, qref[:, 32]*1000, label='expert mug z')
        ax.plot(time, qactual[:, 32]*1000, label='policy mug z')
        ax.plot(time, errors['object_position_m']*1000, label='position error')
        ax.set_title('Seed %d'%seed)
        ax.set_ylabel('mm')
        ax.grid(alpha=.2)
    axes[0, 0].legend()
    for ax in axes[-1]:
        ax.set_xlabel('simulation time (s)')
    fig.tight_layout()
    fig.savefig(args.output/'matched_trajectories.png', dpi=140)
    result = dict(reports=summaries, phase_definition='Controller reference frames: approach <25.5, close <40.5, lift <55, transport thereafter; not independently annotated video phases',
                  thresholds=thresholds, sustain_steps=10, dt_s=.01,
                  state_timing='Both saved state and action arrays are sampled before env.step')
    (args.output/'diagnosis.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(summaries, indent=2))


if __name__ == '__main__':
    main()
