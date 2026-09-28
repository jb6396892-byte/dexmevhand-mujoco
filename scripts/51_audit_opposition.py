#!/usr/bin/env python3
"""Replay actions and measure opposing loaded normals without teleporting objects."""
import argparse
import csv
import json
import pickle
from pathlib import Path
from importlib import import_module
import numpy as np

surface = import_module('33_optimize_surface_grasp')
from fromrealhand.video_fidelity import source_clock


def opposing_cosine(normals):
    thumb = normals.get('TH', [])
    others = [n for key, values in normals.items() if key != 'TH' for n in values]
    if not thumb or not others:
        return None
    return float(min(np.dot(a, b) for a in thumb for b in others))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rollout', type=Path, required=True)
    parser.add_argument('--geometry', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    with args.rollout.open('rb') as stream:
        demo = pickle.load(stream)['video_faithful']
    summary = json.loads((args.rollout.parent/'summary.json').read_text())
    exp = surface.SurfaceExperiment(args.geometry)
    e, m, d = exp.env, exp.model, exp.env.sim.data
    from mujoco_py import functions
    rows, error = [], 0.
    try:
        e.reset(); e.sim.reset()
        e.pack_mujoco_model(demo['model_data'][0])
        # Loading a model can replace MjSim, invalidating cached model/data views.
        m, d = e.sim.model, e.sim.data
        exp.model = m
        for key, values in demo.get('physics_model', {}).items():
            if key not in ('geom_margin', 'geom_gap'):
                raise ValueError('Unsupported model override')
            getattr(m, key)[:] = values
        e.pack(demo['sim_data'][0]); e.sim.forward()
        for step, action in enumerate(demo['actions']):
            error = max(error, float(np.max(np.abs(e._get_observations()-demo['observations'][step]))))
            e.step(action)
            normals = {key: [] for key in ('TH', 'FF', 'MF', 'RF', 'LF')}
            forces = dict.fromkeys(normals, 0.)
            for index, contact in enumerate(d.contact[:d.ncon]):
                a, b = [m.geom_id2name(int(i)) for i in (contact.geom1, contact.geom2)]
                if a in exp.hand_geoms and b in exp.mug_geoms:
                    hand, sign = a, 1.
                elif b in exp.hand_geoms and a in exp.mug_geoms:
                    hand, sign = b, -1.
                else:
                    continue
                key = hand[2:4].upper()
                if key not in normals:
                    continue
                force = np.zeros(6)
                functions.mj_contactForce(e.sim.model, d, index, force)
                forces[key] += max(0., float(force[0]))
                if force[0] > .01:
                    normals[key].append(sign*np.asarray(contact.frame[:3]))
            reference_force = sum(max(0., c['normal_force_n']) for c in surface.video.contact_details(e)
                                  if c['hand'][2:4] in ('th', 'ff', 'mf', 'rf', 'lf'))
            if not np.isclose(sum(forces.values()), reference_force, atol=1e-7):
                raise RuntimeError('Per-step finger force extraction disagrees with existing contact reader')
            cosine = opposing_cosine(normals)
            _, _, _, bottom = exp.contact_stats()
            clock = source_clock(step*e.control_timestep, exp.duration, summary['time_scale'])
            row = dict(step=step, time_s=float(d.time), source_frame=float(exp.geometry['source_frames'][0]+30*clock),
                       bottom_m=float(bottom), opposing_cosine=cosine,
                       opposing=cosine is not None and cosine < -.3)
            row.update({key+'_force_n': value for key, value in forces.items()})
            rows.append(row)
        with (args.output/'contacts.csv').open('w') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
        lifted = [r for r in rows if r['bottom_m'] > .015]
        tail = rows[-100:]
        result = dict(replay_max_observation_error=error, steps=len(rows),
                      lifted_steps=len(lifted),
                      lifted_opposition_fraction=float(np.mean([r['opposing'] for r in lifted])) if lifted else 0.,
                      tail_opposition_fraction=float(np.mean([r['opposing'] for r in tail])),
                      tail_contact_fraction={key: float(np.mean([r[key+'_force_n'] > .01 for r in tail])) for key in normals},
                      definition='Loaded thumb/other-finger normal pair cosine below -0.3; diagnostic, not force-closure proof',
                      rollout=str(args.rollout.resolve()), geometry=str(args.geometry.resolve()))
        for key, name in zip(('TH', 'FF', 'MF', 'RF', 'LF'), ('thumb', 'index', 'middle', 'ring', 'little')):
            measured = float(np.mean([r[key+'_force_n'] for r in tail]))
            if not np.isclose(measured, summary['tail_finger_force_n'][name], atol=1e-7):
                raise RuntimeError('Contact audit disagrees with original force report: '+name)
        (args.output/'audit.json').write_text(json.dumps(result, indent=2)+'\n')
        print(json.dumps(result, indent=2))
        if error > 1e-8:
            raise RuntimeError('Replay differs from saved observations')
    finally:
        e.close()


if __name__ == '__main__':
    main()
