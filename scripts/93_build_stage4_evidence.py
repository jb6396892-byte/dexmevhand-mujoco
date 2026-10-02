#!/usr/bin/env python3
"""Publish compact, derived skill evidence without models or source recordings."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=ROOT/'data/processed/stage4_skills_v1')
    parser.add_argument('--output', type=Path, default=ROOT/'docs/presentation/stage4/evidence')
    args = parser.parse_args()
    manifest_path = args.dataset/'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    validation = json.loads((args.dataset/'validation.json').read_text())
    if validation['manifest_sha256'] != hashlib.sha256(manifest_path.read_bytes()).hexdigest():
        raise ValueError('Validation no longer matches the manifest')
    args.output.mkdir(parents=True, exist_ok=False)
    figure, axes = plt.subplots(len(manifest['trajectories']), 2, figsize=(12, 6), squeeze=False)
    compact = dict(generated_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        manifest_sha256=validation['manifest_sha256'], source_demo_sha256=manifest['source_sha256'],
        config=manifest['config'], validation=validation, trajectories=[])
    for index, entry in enumerate(manifest['trajectories']):
        rows = json.loads((args.dataset/entry['trajectory']/'events.json').read_text())
        times = (np.arange(len(rows))+1)*entry['dt']
        left, right = axes[index]
        left.plot(times, [r['bottom_m']*1000 for r in rows], label='Cup bottom', color='#00856a')
        left.plot(times, [r['target_distance_m']*1000 for r in rows], label='Goal distance', color='#bc4d57')
        left.axhline(50, color='#00856a', linestyle=':', linewidth=1)
        left.axhline(30, color='#bc4d57', linestyle=':', linewidth=1)
        right.plot(times, [r['scene_penetration_m']*1000 for r in rows], color='#516ab1', label='Hand-scene depth')
        right.axhline(1., color='#bc4d57', linestyle='--', linewidth=1, label='Acceptance limit')
        left.set_ylabel(entry['video']+' video / mm')
        right.set_ylabel('Penetration / mm')
        right.set_ylim(0, 1.25)
        for axis in (left, right):
            for segment in entry['segments']:
                if segment['start']:
                    axis.axvline(segment['time_start_s'], color='#555555', linestyle=':', linewidth=.8)
                center = (segment['time_start_s']+segment['time_stop_s'])/2
                axis.text(center, .98, segment['skill'], transform=axis.get_xaxis_transform(),
                          ha='center', va='top', fontsize=8, rotation=90)
            axis.set_xlabel('Simulation time / s')
            axis.grid(alpha=.15)
            axis.legend(loc='lower left', fontsize=8)
        compact['trajectories'].append({key:entry[key] for key in (
            'trajectory','horizon','dt','segments','source_replay_observation_error')})
    figure.suptitle('Stage 4: physical-event segments, verified reference actions (not skill policies)')
    figure.tight_layout()
    figure.savefig(str(args.output/'skill-events.png'), dpi=150)
    plt.close(figure)
    (args.output/'skill-replay.json').write_text(json.dumps(compact, indent=2, allow_nan=False)+'\n')
    status = ROOT/'data/processed/dual_video_v14c/supervision_200/status.json'
    snapshot = json.loads(status.read_text())
    snapshot['scope'] = 'Timestamped progress snapshot, not live status and not independent evaluation'
    (args.output/'training-progress-snapshot.json').write_text(json.dumps(snapshot, indent=2)+'\n')
    print(json.dumps(dict(validation_passed=validation['passed'], training_iteration=snapshot['iterations'],
                         output=str(args.output))))


if __name__ == '__main__':
    main()
