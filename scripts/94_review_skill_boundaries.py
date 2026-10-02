#!/usr/bin/env python3
"""Review frozen skills against physical events, loaded normals, and source RGB."""
import argparse
import datetime
from importlib import import_module
import json
import pickle
from pathlib import Path

import cv2
import numpy as np
from stage4_common import ROOT, digest, experiment, read_config, restore_once, save_json
from fromrealhand.skill_review import audit_boundaries, source_frame_record
from fromrealhand.skills import FINGERS


def normal_audit(exp, force_threshold, cosine_threshold):
    from mujoco_py import functions
    m, d = exp.model, exp.env.sim.data
    normals = {key.upper(): [] for key in FINGERS}
    for index, contact in enumerate(d.contact[:d.ncon]):
        a, b = [m.geom_id2name(int(i)) for i in (contact.geom1, contact.geom2)]
        if a in exp.hand_geoms and b in exp.mug_geoms:
            hand, sign = a, 1.
        elif b in exp.hand_geoms and a in exp.mug_geoms:
            hand, sign = b, -1.
        else:
            continue
        finger = hand[2:4].upper()
        force = np.zeros(6)
        functions.mj_contactForce(m,d,index,force)
        if finger in normals and force[0] > force_threshold:
            normals[finger].append(sign*np.asarray(contact.frame[:3]))
    cosine = import_module('51_audit_opposition').opposing_cosine(normals)
    return dict(cosine=cosine, opposing=bool(cosine is not None and cosine < cosine_threshold))


def physical_review(entry, pieces, rows, report, review, output, source):
    from mujoco_py import MjRenderContextOffscreen
    exp = experiment(entry['geometry'], pieces[0])
    actions = np.concatenate([p['actions'] for p in pieces])
    expected = np.concatenate([p['expected_post_states'] for p in pieces])
    selected = {b['boundary_state_index']-1: b['next_skill']+'_entry' for b in report['boundaries']}
    mapping = read_config(source/'frame_mapping.json')
    frame_paths = []
    contacts, image_rows, error = [], [], 0.
    try:
        restore_once(exp.env, pieces[0]['initial_snapshot'])
        context = MjRenderContextOffscreen(exp.env.sim)
        context.cam.type = 0
        context.cam.lookat[:] = np.load(entry['geometry'])['object_poses'][[0,-1],:3,3].mean(axis=0)
        context.cam.distance, context.cam.azimuth, context.cam.elevation = .55, 135., -30.
        for step, action in enumerate(actions):
            exp.env.step(action)
            actual = np.r_[exp.env.sim.data.qpos,exp.env.sim.data.qvel]
            error = max(error,float(np.max(np.abs(actual-expected[step]))))
            contacts.append(normal_audit(exp,review['force_threshold'],review['opposition_cosine_threshold']))
            if step not in selected:
                continue
            frame = source_frame_record(mapping,rows[step]['source_frame'])
            path = source/'rgb'/('%06d.jpg' % frame['output_frame'])
            rgb = cv2.imread(str(path))
            record = dict(step=step,source_frame=frame['source_frame'], output_frame=frame['output_frame'],
                          available=rgb is not None, path=str(path.relative_to(ROOT)))
            if rgb is None:
                rgb = np.full((480,640,3),230,dtype=np.uint8)
            else:
                record['sha256'] = digest(path)
            cv2.putText(rgb,'SOURCE %s frame %d' % (entry['video'],frame['source_frame']),
                        (12,24),cv2.FONT_HERSHEY_SIMPLEX,.65,(0,0,0),2)
            context.render(640,480,camera_id=-1)
            rendered = cv2.cvtColor(context.read_pixels(640,480,depth=False)[::-1],cv2.COLOR_RGB2BGR)
            if not np.array_equal(actual,np.r_[exp.env.sim.data.qpos,exp.env.sim.data.qvel]):
                raise RuntimeError('Rendering changed the live physical state')
            cv2.putText(rendered,'%s | state %d' % (selected[step],step+1),
                        (12,24),cv2.FONT_HERSHEY_SIMPLEX,.65,(0,0,0),2)
            cv2.putText(rendered,'bottom %.1f mm | %s' % (rows[step]['bottom_m']*1000,
                ','.join(f for f in FINGERS if rows[step][f+'_force_n']>review['force_threshold'])),
                (12,50),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,0,0),1)
            image_rows.append(np.hstack([rgb,rendered]))
            frame_paths.append(record)
        image_path = output/(entry['video']+'-boundaries.jpg')
        if not cv2.imwrite(str(image_path),np.vstack(image_rows)):
            raise IOError('Could not write boundary comparison')
        details = []
        for boundary in report['boundaries']:
            stop = boundary['boundary_state_index']
            lo, hi = max(0,stop-review['boundary_window_steps']),min(len(rows),stop+review['boundary_window_steps'])
            details.append(dict(next_skill=boundary['next_skill'], state_index=stop,
                cosine_at_boundary=contacts[stop-1]['cosine'], opposing_at_boundary=contacts[stop-1]['opposing'],
                window_opposing_fraction=float(np.mean([c['opposing'] for c in contacts[lo:hi]]))))
        return dict(max_state_replay_error=error, source_rgb=frame_paths,
            all_source_rgb_available=all(r['available'] for r in frame_paths), boundaries=details,
            lifted_opposing_fraction=float(np.mean([c['opposing'] for r,c in zip(rows,contacts) if r['bottom_m']>.015])),
            tail_opposing_fraction=float(np.mean([c['opposing'] for c in contacts[-100:]])),
            initialization_count=1, state_writes_during_execution=0,
            image=str(image_path.relative_to(ROOT)), image_sha256=digest(image_path),
            opposition_scope='Loaded thumb/other normal pair; diagnostic, not proof of force closure')
    finally:
        exp.env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=ROOT/'configs/stage4-boundary-review.json')
    args = parser.parse_args()
    args.config = args.config.resolve()
    config = read_config(args.config)
    dataset, output = ROOT/config['dataset'], ROOT/config['output']
    manifest = read_config(dataset/'manifest.json')
    validation = read_config(dataset/'validation.json')
    if not manifest.get('completed') or not validation['passed'] or validation['manifest_sha256'] != digest(dataset/'manifest.json'):
        raise ValueError('A complete, physically verified skill dataset is required')
    protected = {str(p.relative_to(ROOT)): digest(p) for p in
        (dataset/'manifest.json',dataset/'validation.json',args.config,ROOT/'configs/stage4-skills.json',
         ROOT/'src/fromrealhand/skills.py',ROOT/'scripts/stage4_common.py',
         ROOT/'src/fromrealhand/skill_review.py',Path(__file__).resolve())}
    for entry in manifest['trajectories']:
        path = Path(entry['geometry'])
        if digest(path) != entry['geometry_sha256']:
            raise ValueError('Frozen geometry changed')
        protected[str(path.relative_to(ROOT))] = digest(path)
        for path in (dataset/entry['trajectory']/'events.json',
                     ROOT/config['source_sequences'][entry['video']]/'meta.json',
                     ROOT/config['source_sequences'][entry['video']]/'frame_mapping.json'):
            protected[str(path.relative_to(ROOT))] = digest(path)
        for segment in entry['segments']:
            path = dataset/segment['artifact']
            if digest(path) != segment['sha256']:
                raise ValueError('Frozen skill changed')
            protected[str(path.relative_to(ROOT))] = digest(path)
    output.mkdir(parents=True,exist_ok=False)
    save_json(output/'freeze.json',dict(protected_sha256=protected,
        frozen_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),heldout_used=False))
    reports = []
    for entry in manifest['trajectories']:
        rows = read_config(dataset/entry['trajectory']/'events.json')
        result = audit_boundaries(rows,entry['segments'],manifest['config'],config,entry['dt'])
        pieces = [pickle.loads((dataset/s['artifact']).read_bytes()) for s in entry['segments']]
        source = ROOT/config['source_sequences'][entry['video']]
        with np.load(entry['geometry']) as geometry:
            if read_config(source/'meta.json')['fps'] != float(geometry['fps']):
                raise ValueError('Source FPS differs from geometry clock')
        result['physical_replay'] = physical_review(entry,pieces,rows,result,
            dict(config,force_threshold=manifest['config']['contact_force_n']),output,source)
        result.update(trajectory=entry['trajectory'], video=entry['video'],
            engineer_checks_passed=bool(result['mechanical_pass'] and
                result['physical_replay']['max_state_replay_error']<=manifest['config']['replay_state_tolerance']))
        reports.append(result)
        save_json(output/'review.json',dict(reports=reports,completed=False))
        print(json.dumps(dict(trajectory=entry['trajectory'],checks=result['checks'],
                             replay_error=result['physical_replay']['max_state_replay_error'])),flush=True)
    unchanged = all(digest(ROOT/path)==sha for path,sha in protected.items())
    report = dict(completed=True,inputs_unchanged=unchanged,heldout_used=False,reports=reports,
        engineer_checks_passed=bool(unchanged and all(r['engineer_checks_passed'] for r in reports)),
        human_review_status='pending', stage_4_1_fully_accepted=False,
        frozen_manifest_sha256=digest(dataset/'manifest.json'))
    save_json(output/'review.json',report)
    if not report['engineer_checks_passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
