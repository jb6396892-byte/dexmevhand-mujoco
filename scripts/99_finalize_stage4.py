#!/usr/bin/env python3
"""Freeze the qualified Stage 4 subset and publish small, honest evidence."""
import argparse
from collections import Counter
import datetime
from pathlib import Path
import shutil

import numpy as np
from stage4_common import ROOT, digest, experiment, read_config, restore_once, save_json
from stage4_pipeline_common import load_pieces
from fromrealhand.skill_delivery import entry_summary, failure_reason
from fromrealhand.skill_review import source_frame_record


def verify_artifacts(run):
    protocol = read_config(run/'protocol.json')
    for path, sha in protocol['sha256'].items():
        if digest(ROOT/path) != sha:
            raise ValueError('Frozen source changed: ' + path)
    receipts = {stage: read_config(run/stage/'receipt.json')
                for stage in ('pilot', 'build', 'inputs', 'entries')}
    if not all(r['passed'] for r in receipts.values()):
        raise ValueError('All preceding subphases must pass their declared scope')
    for stage in ('pilot', 'build'):
        if receipts[stage]['manifest_sha256'] != digest(run/stage/'manifest.json'):
            raise ValueError('Manifest changed: ' + stage)
    if receipts['inputs']['index_sha256'] != digest(run/'inputs/index.json'):
        raise ValueError('Input index changed')
    if receipts['entries']['source_input_receipt_sha256'] != digest(run/'inputs/receipt.json'):
        raise ValueError('Entry checks refer to a different adapter')
    if receipts['entries']['cases_sha256'] != digest(run/'entries/cases.json'):
        raise ValueError('Entry cases changed')
    index = read_config(run/'inputs/index.json')
    if index['source_manifest_sha256'] != digest(run/'build/manifest.json'):
        raise ValueError('Adapter refers to a different skill manifest')
    if digest(run/'inputs'/index['normalization']) != index['normalization_sha256']:
        raise ValueError('Normalization changed')
    manifest = read_config(run/'build/manifest.json')
    for entry in manifest['trajectories']:
        if digest(entry['geometry']) != entry['geometry_sha256']:
            raise ValueError('Geometry changed')
        if not all(r['passed'] for r in entry['replays']):
            raise ValueError('Unqualified trajectory in delivery')
        for segment in entry['segments']:
            if digest(run/'build'/segment['artifact']) != segment['sha256']:
                raise ValueError('Skill changed')
    for record in index['records']:
        if digest(run/'inputs'/record['artifact']) != record['sha256']:
            raise ValueError('Training input changed')
    return protocol, receipts, manifest, index


def key_frame(run, entry, output):
    import cv2
    from mujoco_py import MjRenderContextOffscreen
    pieces = load_pieces(run/'build', entry)
    stop = entry['segments'][2]['stop']
    actions = np.concatenate([p['actions'] for p in pieces])
    expected = np.concatenate([p['expected_post_states'] for p in pieces])
    row = pieces[2]['metrics'][-1]
    seq = 'seq_dexycb_001' if entry['video'] == 'first' else 'seq_dexycb_002'
    source = ROOT/'data/real_data/relocate_mug'/seq
    mapping_path = source/'frame_mapping.json'
    frame = source_frame_record(read_config(mapping_path), row['source_frame'])
    source_path = source/'rgb'/('%06d.jpg' % frame['output_frame'])
    rgb = cv2.imread(str(source_path))
    if rgb is None:
        raise ValueError('Missing source RGB for evidence: ' + str(source_path))
    exp = experiment(entry['geometry'], pieces[0])
    error = 0.
    try:
        # The renderer constructor calls sim.forward(); do this before restoring state.
        context = MjRenderContextOffscreen(exp.env.sim)
        restore_once(exp.env, pieces[0]['initial_snapshot'])
        for step, action in enumerate(actions[:stop]):
            exp.env.step(action)
            state = np.r_[exp.env.sim.data.qpos, exp.env.sim.data.qvel]
            error = max(error, float(np.max(np.abs(state-expected[step]))))
        if error > 1e-7:
            raise ValueError('Evidence replay state mismatch')
        context.cam.type = 0
        with np.load(entry['geometry']) as geometry:
            context.cam.lookat[:] = geometry['object_poses'][[0, -1], :3, 3].mean(axis=0)
        context.cam.distance, context.cam.azimuth, context.cam.elevation = .55, 135., -30.
        context.render(640, 480, camera_id=-1)
        rendered = cv2.cvtColor(context.read_pixels(640, 480, depth=False)[::-1], cv2.COLOR_RGB2BGR)
        if not np.array_equal(state, np.r_[exp.env.sim.data.qpos, exp.env.sim.data.qvel]):
            raise ValueError('Rendering changed state')
        if rendered.std() < 5.:
            raise ValueError('Blank rendering')
        rgb = cv2.resize(rgb, (640, 480))
        cv2.putText(rgb, 'SOURCE %s | frame %d' % (entry['video'], frame['source_frame']),
                    (12, 24), cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 0, 0), 2)
        cv2.putText(rendered, 'EXPERT transport entry | state %d' % stop,
                    (12, 24), cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 0, 0), 2)
        cv2.putText(rendered, 'bottom %.1f mm | green = target marker' % (1000*row['bottom_m']),
                    (12, 50), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 0), 1)
        target = output/(entry['video']+'-transport.jpg')
        if not cv2.imwrite(str(target), np.hstack([rgb, rendered])):
            raise IOError('Could not save key frame')
        return dict(trajectory=entry['trajectory'], state_index=stop, source_frame=frame['source_frame'],
                    source_path=str(source_path.relative_to(ROOT)), source_sha256=digest(source_path),
                    mapping_path=str(mapping_path.relative_to(ROOT)), mapping_sha256=digest(mapping_path),
                    image=target.name, image_sha256=digest(target), max_state_error=error,
                    initialization_count=1, state_writes_during_execution=0, render_std=float(rendered.std()),
                    scope='Physical expert replay, not the running learned policy')
    finally:
        exp.env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=ROOT/'data/processed/stage4_pipeline_v2')
    parser.add_argument('--publish', type=Path, default=ROOT/'docs/presentation/stage4/evidence/v2')
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    protocol, receipts, manifest, index = verify_artifacts(args.run)
    output = args.run/'delivery'
    if args.verify_only:
        freeze = read_config(output/'freeze.json')
        for path, sha in freeze['sha256'].items():
            if digest(ROOT/path) != sha:
                raise ValueError('Delivery changed: ' + path)
        receipt = read_config(output/'receipt.json')
        if receipt['freeze_sha256'] != digest(output/'freeze.json'):
            raise ValueError('Delivery freeze changed')
        print('PASS: source chain and %d frozen files' % len(freeze['sha256']))
        return
    if args.publish.exists():
        raise ValueError('Refusing to overwrite published evidence')
    output.mkdir(exist_ok=False)
    cases = read_config(args.run/'entries/cases.json')
    case_summary = entry_summary(cases)
    annotated_cases = [dict(c, failure_class=failure_reason(c)) for c in cases]
    save_json(output/'entry-cases.json', annotated_cases)
    frames = [key_frame(args.run, next(e for e in manifest['trajectories'] if e['trajectory'] == name), output)
              for name in manifest['config']['pilot_trajectories']]
    admitted = [dict(trajectory=e['trajectory'], video=e['video'], split=e['split'],
                     geometry_group=e['geometry_group'], boundaries=[s['stop'] for s in e['segments'][:-1]],
                     handoff_fractions=[h['fraction'] for h in e['handoffs']],
                     max_state_error=max(r['state_replay_error'] for r in e['replays']))
                for e in manifest['trajectories']]
    rejected = [dict(trajectory=e['trajectory'], split=e['split'], reason=e['reason'],
                     handoff_fractions=[h['fraction'] for h in e['handoffs']],
                     failed_replays=[dict(r, failure_class=failure_reason(r))
                                     for r in e.get('replays', []) if not r['passed']])
                for e in manifest['rejected']]
    save_json(output/'admission.json', dict(admitted=admitted, rejected=rejected))
    save_json(output/'subphase-receipts.json', receipts)
    status_path = ROOT/'data/processed/dual_video_v14c/supervision_200/status.json'
    save_json(output/'training-snapshot.json', dict(scope='Timestamped snapshot, not a live feed',
                                                  status=read_config(status_path)))
    summary = dict(created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        engineering_stage4_complete=True, ready_for_nominal_stage5_smoke=True,
        robust_composable_skills_ready=False, skill_policy_training_started=False, heldout_used=False,
        independent_real_videos=2, admitted_trajectories=len(admitted), rejected_trajectories=len(rejected),
        admitted_segments=len(index['records']),
        trajectory_splits=dict(Counter(e['split'] for e in admitted)),
        input_checks=receipts['inputs'], entries=case_summary, key_frames=frames,
        max_accepted_replay_error=max(e['max_state_error'] for e in admitted),
        user_scope_confirmation=manifest['user_scope_confirmation'],
        limitations=['Entry perturbations pass %d/%d; no robust skill policy claim' %
                     (receipts['entries']['perturbed_pass'], receipts['entries']['perturbed_count']),
                     'Two rejected snapshot replays remain unexplained; not admitted',
                     'Only two real videos; grouped validation is not independent generalization',
                     'Three-finger simulated support is not four-finger video fidelity or real force truth'],
        sources=manifest['config']['sources'] + [
            'https://github.com/google-deepmind/mujoco/blob/main/doc/computation/index.rst',
            'https://github.com/openai/mujoco-py/blob/master/mujoco_py/mjrendercontext.pyx'])
    save_json(output/'summary.json', summary)
    files = set(ROOT/p for p in protocol['sha256'])
    files.add(args.run/'protocol.json')
    for stage in receipts:
        files.add(args.run/stage/'receipt.json')
    files.update([args.run/'pilot/manifest.json', args.run/'build/manifest.json',
                  args.run/'inputs/index.json', args.run/'inputs/normalization.json', args.run/'entries/cases.json'])
    for entry in manifest['trajectories']:
        files.add(Path(entry['geometry']))
        files.update(args.run/'build'/s['artifact'] for s in entry['segments'])
    files.update(args.run/'inputs'/r['artifact'] for r in index['records'])
    for frame in frames:
        files.update([ROOT/frame['source_path'], ROOT/frame['mapping_path']])
    files.update(ROOT/path for path in (
        'src/fromrealhand/skill_inputs.py', 'src/fromrealhand/skill_delivery.py',
        'scripts/97_build_skill_inputs.py', 'scripts/98_validate_skill_entries.py',
        'scripts/99_finalize_stage4.py', 'configs/stage5-skill-smoke.template.json',
        'tests/test_skill_inputs.py', 'tests/test_skill_pipeline.py', 'tests/test_skill_delivery.py'))
    files.update(output.iterdir())
    verify_artifacts(args.run)
    save_json(output/'freeze.json', dict(created_at=summary['created_at'],
        sha256={str(p.relative_to(ROOT)): digest(p) for p in sorted(files)}))
    save_json(output/'receipt.json', dict(passed=True, scope='Stage4 engineering delivery, not skill learning',
        freeze_sha256=digest(output/'freeze.json'), frozen_files=len(files),
        long_skill_training_authorized=False, robust_skill_ready=False))
    args.publish.mkdir(parents=True, exist_ok=False)
    for path in output.iterdir():
        shutil.copy2(str(path), str(args.publish/path.name))
    print('PASS: Stage4 engineering delivery; Stage5 training not started')


if __name__ == '__main__':
    main()
