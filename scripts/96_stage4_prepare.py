#!/usr/bin/env python3
"""Sequential 4.1 pilot and 4.2 admission of physically verified skill data."""
import argparse
import datetime
import json
import pickle
from pathlib import Path

import numpy as np
from stage4_common import ROOT, capture, digest, read_config, save_json
from stage4_pipeline_common import export_pieces, load_pieces, replay
from fromrealhand.skill_pipeline import confirmed_segments, handoff_audit, geometry_group


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['pilot','build'])
    parser.add_argument('--config',type=Path,default=ROOT/'configs/stage4-pipeline-v2.json')
    args = parser.parse_args()
    args.config = args.config.resolve()
    plan = read_config(args.config)
    config = dict(read_config(ROOT/plan['base_config']),**plan)
    run = ROOT/plan['output']
    source = ROOT/config['source_demonstrations']
    admission = read_config(ROOT/config['source_admission'])
    if digest(source)!=admission['demo_sha256'] or digest(ROOT/plan['references'])!=admission['reference_sha256']:
        raise ValueError('Source admission hashes changed')
    protected_paths = [args.config,ROOT/plan['base_config'],source,ROOT/config['source_admission'],
        ROOT/config['source_protocol'],ROOT/plan['references'],Path(__file__).resolve(),
        ROOT/'scripts/stage4_pipeline_common.py',ROOT/'src/fromrealhand/skill_pipeline.py',
        ROOT/'scripts/stage4_common.py',ROOT/'src/fromrealhand/skills.py']
    protected = {str(p.relative_to(ROOT)):digest(p) for p in protected_paths}
    if args.stage=='pilot':
        run.mkdir(parents=True,exist_ok=False)
        save_json(run/'protocol.json',dict(config=config,sha256=protected,
            created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),heldout_used=False))
    else:
        protocol = read_config(run/'protocol.json')
        if protocol['sha256']!=protected or not read_config(run/'pilot/receipt.json')['passed']:
            raise ValueError('A matching, passed 4.1 pilot is required')
    folder = run/args.stage
    folder.mkdir(exist_ok=False)
    demos = pickle.loads(source.read_bytes())
    records = {r['video']+'/'+r['name']:r for r in admission['reports'] if r['admitted']}
    videos = {v['name']:v for v in read_config(ROOT/config['source_protocol'])['videos']}
    groups = {}
    for key,record in records.items():
        with np.load(record['geometry']) as geometry:
            groups[key] = geometry_group(geometry,record['video'])
    val_groups = {groups[key] for key in plan['validation_scenes']}
    selected = plan['pilot_trajectories'] if args.stage=='pilot' else sorted(records)
    manifest = dict(schema_version=2,config=config,source_sha256=digest(source),trajectories=[],
        user_scope_confirmation=plan['user_scope_confirmation'],human_framewise_fidelity_review='not_claimed',
        split_scope='Grouped internal development validation, not independent video generalization',
        geometry_groups=groups,validation_groups=sorted(val_groups),rejected=[],completed=False)
    for key in selected:
        record,demo = records[key],demos[key]
        result = capture(videos[record['video']],record['geometry'],record['seed'],demo)
        source_error = float(np.max(np.abs(result['demo']['observations']-demo['observations'])))
        failure = None
        if source_error>config['replay_observation_tolerance'] or not result['report']['surface_physics_passed']:
            failure = 'source_physics_or_replay'
        try:
            segments = confirmed_segments(result['rows'],config,result['dt'])
            handoffs = handoff_audit(result['rows'],segments,config)
        except ValueError as error:
            failure = str(error)
            segments,handoffs = [],[]
        if handoffs and not all(h['passed'] for h in handoffs):
            failure = 'handoff_window'
        entry = dict(trajectory=key,video=record['video'],geometry=record['geometry'],
            geometry_sha256=digest(record['geometry']),seed=record['seed'],horizon=len(demo['actions']),
            dt=result['dt'],segments=segments,handoffs=handoffs,source_replay_error=source_error,
            geometry_group=groups[key],split='validation' if groups[key] in val_groups else 'train',
            source_report=result['report'],control=videos[record['video']]['control'])
        if failure is None:
            export_pieces(folder,entry,result,manifest['source_sha256'])
            pieces = load_pieces(folder,entry)
            checks = [replay(entry,[piece],config) for piece in pieces]
            checks.append(replay(entry,pieces,config))
            entry['replays'] = checks
            if not all(check['passed'] for check in checks):
                failure = 'isolated_or_stitched_replay'
        if failure:
            manifest['rejected'].append(dict(entry,reason=failure))
        else:
            manifest['trajectories'].append(entry)
        save_json(folder/'manifest.json',manifest)
        print(json.dumps(dict(trajectory=key,admitted=failure is None,reason=failure,
            boundaries=[s['stop'] for s in segments[:-1]],handoff_fractions=[h['fraction'] for h in handoffs],
            admitted_count=len(manifest['trajectories']),rejected_count=len(manifest['rejected']))),flush=True)
    manifest['completed'] = True
    save_json(folder/'manifest.json',manifest)
    unchanged = all(digest(ROOT/path)==sha for path,sha in protected.items())
    passed = unchanged and (len(manifest['trajectories'])==len(selected) if args.stage=='pilot' else
        all(any(e['video']==v and e['split']==s for e in manifest['trajectories'])
            for v in videos for s in ('train','validation')))
    save_json(folder/'receipt.json',dict(passed=passed,inputs_unchanged=unchanged,
        completed=True,attempted=len(selected),admitted=len(manifest['trajectories']),
        rejected=len(manifest['rejected']),manifest_sha256=digest(folder/'manifest.json'),
        scope='Expert skill data, not trained skill policies or heldout evaluation'))
    if not passed:
        raise SystemExit(1)


if __name__=='__main__': main()
