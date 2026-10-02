#!/usr/bin/env python3
"""Export event-labelled skills from admitted, action-executed demonstrations."""
import argparse
import json
import pickle
from pathlib import Path

import numpy as np
from stage4_common import ROOT, capture, digest, read_config, save_json, save_pickle
from fromrealhand.skills import propose_segments


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT/'configs/stage4-skills.json')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    config = read_config(args.config)
    folder = args.output or ROOT/config['output']
    folder.mkdir(parents=True, exist_ok=False)
    source = ROOT/config['source_demonstrations']
    admission = read_config(ROOT/config['source_admission'])
    if digest(source) != admission['demo_sha256']:
        raise ValueError('Admission hash does not match demonstrations')
    demos = pickle.loads(source.read_bytes())  # Trusted local artifacts only.
    videos = {v['name']: v for v in read_config(ROOT/config['source_protocol'])['videos']}
    records = {r['video']+'/'+r['name']: r for r in admission['reports']}
    manifest = dict(schema_version=1, source_sha256=digest(source), config=config,
        config_sha256=digest(args.config), admission_sha256=digest(ROOT/config['source_admission']),
        scope='Two admitted development trajectories; no heldout data or skill-policy training',
        boundary_semantics='[start, stop) actions; events measured after each action',
        trajectories=[])
    for key in config['selected_trajectories']:
        record, demo = records[key], demos[key]
        if not record['admitted']:
            raise ValueError('Only admitted experts are supported')
        result = capture(videos[record['video']], record['geometry'], record['seed'], demo)
        error = float(np.max(np.abs(result['demo']['observations']-demo['observations'])))
        if error > config['replay_observation_tolerance'] or not result['report']['surface_physics_passed']:
            raise ValueError('Source expert replay failed: %s, error=%g' % (key, error))
        segments = propose_segments(result['rows'], config, result['dt'])
        entry = dict(trajectory=key, video=record['video'], seed=record['seed'], geometry=record['geometry'],
            geometry_sha256=digest(record['geometry']), control=videos[record['video']]['control'],
            horizon=len(demo['actions']), dt=result['dt'], segments=segments,
            source_replay_observation_error=error, source_report=result['report'])
        dest = folder/key
        dest.mkdir(parents=True)
        for segment in segments:
            lo, hi = segment['start'], segment['stop']
            piece = {name: result['demo'][name][lo:hi] for name in ('observations', 'actions', 'rewards', 'sim_data')}
            piece.update(model_data=demo['model_data'], physics_model=demo['physics_model'],
                initial_snapshot=result['snapshots'][lo], terminal_snapshot=result['snapshots'][hi],
                terminal_observation=(result['demo']['observations'][hi] if hi < entry['horizon']
                                      else result['terminal_observation']),
                initial_metrics=result['initial'] if lo == 0 else result['rows'][lo-1],
                expected_post_states=result['post_states'][lo:hi], metrics=result['rows'][lo:hi],
                metadata=dict(segment, trajectory=key, video=record['video'], dt=result['dt'],
                              source_demo_sha256=manifest['source_sha256'], action_semantics='normalized_once'))
            path = dest/(segment['skill']+'.pkl')
            save_pickle(path, piece)
            segment.update(artifact=str(path.relative_to(folder)), sha256=digest(path))
        save_json(dest/'events.json', result['rows'])
        manifest['trajectories'].append(entry)
        save_json(folder/'manifest.json', manifest)
        print(json.dumps(dict(trajectory=key, boundaries=[(s['skill'],s['start'],s['stop']) for s in segments],
                              source_replay_error=error)), flush=True)
    manifest['completed'] = True
    save_json(folder/'manifest.json', manifest)


if __name__ == '__main__':
    main()
