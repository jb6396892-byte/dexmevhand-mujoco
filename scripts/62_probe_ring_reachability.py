#!/usr/bin/env python3
"""Offline constrained IK probe; never treat a positioned snapshot as a grasp."""
import argparse
import json
import pickle
from importlib import import_module
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from mujoco_py import functions

contact = import_module('60_refine_contact_targets')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--candidate', type=Path, required=True)
    p.add_argument('--rollout', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--include-little', action='store_true')
    p.add_argument('--whole-hand', action='store_true')
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    source = json.loads(args.candidate.read_text())
    b = source['best']
    e = contact.ContactExperiment(source['geometry'])
    demo = pickle.loads(args.rollout.read_bytes())['video_faithful']
    try:
        e.env.pack_mujoco_model(demo['model_data'][0])
        m, d = e.env.sim.model, e.env.sim.data
        for key, values in demo['physics_model'].items():
            getattr(m, key)[:] = values
        e.env.pack(demo['sim_data'][-1]); e.env.sim.forward()
        # Large detection margin is confined to this static distance probe.
        ring_ids = [i for i in range(m.ngeom) if (m.geom_id2name(i) or '').startswith('C_rf')]
        detection_ids = [i for i in range(m.ngeom) if m.geom_id2name(i) in e.hand_geoms] if args.whole_hand else ring_ids
        m.geom_margin[detection_ids] = .05
        indices = np.arange(6, 30) if args.whole_hand else np.arange(16, 25 if args.include_little else 20)
        original = d.qpos.copy()
        bounds = m.jnt_range[indices].copy()
        bounds[:, 0] += .04; bounds[:, 1] -= .04
        human_local = contact.object_relative(e.geometry['human_joints'][-1, contact.TIP_INDICES], e.geometry['object_poses'][-1])
        points = human_local @ d.body_xmat[e.env.obj_bid].reshape(3, 3).T+d.body_xpos[e.env.obj_bid]
        tip_ids = [m.site_name2id('S_'+name+'tip') for name in ('th','ff','mf','rf','lf')]

        def measure(q):
            d.qpos[:] = original; d.qpos[indices] = q
            # Geometry queries only: do not build forces for the expanded margin.
            functions.mj_kinematics(m, d)
            functions.mj_comPos(m, d)
            functions.mj_tendon(m, d)
            functions.mj_collision(m, d)
            distances, depths = [], []
            finger_gaps = dict.fromkeys(('th', 'ff', 'mf'), .05)
            for c in d.contact[:d.ncon]:
                a, b = (m.geom_id2name(int(i)) for i in (c.geom1, c.geom2))
                if ((int(c.geom1) in ring_ids and b in e.mug_geoms)
                        or (int(c.geom2) in ring_ids and a in e.mug_geoms)):
                    distances.append(float(c.dist))
                if a in e.hand_geoms or b in e.hand_geoms:
                    depths.append(max(0., -float(c.dist)))
                hand = a if a in e.hand_geoms and b in e.mug_geoms else b if b in e.hand_geoms and a in e.mug_geoms else ''
                if hand and hand[2:4] in finger_gaps:
                    finger_gaps[hand[2:4]] = min(finger_gaps[hand[2:4]], float(c.dist))
            tip_errors = np.linalg.norm(d.site_xpos[tip_ids]-points, axis=1)
            tendon = max(0., float(np.max(m.tendon_range[:, 0]-d.ten_length)),
                         float(np.max(d.ten_length-m.tendon_range[:, 1])))
            return dict(ring_gap_m=min(distances or [.05]), max_penetration_m=max(depths or [0.]),
                        max_tendon_violation_m=tendon, tip_errors_m=tip_errors.tolist(),
                        qpos=q.tolist(), supporting_finger_gaps_m=finger_gaps)

        def objective(q):
            r = measure(q)
            return (2000*(r['ring_gap_m']+.0001)**2 +
                    100*np.sum(np.square(np.asarray(r['tip_errors_m'])[3:]))+
                    .0001*np.sum((q-original[indices])**2))

        def constraints(q):
            r = measure(q)
            values = [.001-r['max_penetration_m'], .00002-r['max_tendon_violation_m'],
                      .025-max(r['tip_errors_m']), .015-np.mean(r['tip_errors_m'])]
            if args.whole_hand:
                values += [.0005-gap for gap in r['supporting_finger_gaps_m'].values()]
            return np.array(values)

        results = []
        for delta in (0., -.2, .2):
            start = np.clip(original[indices]+delta, bounds[:, 0], bounds[:, 1])
            solve = minimize(objective, start, method='SLSQP', bounds=bounds,
                             constraints=[dict(type='ineq', fun=constraints)],
                             options=dict(maxiter=200, ftol=1e-10, eps=1e-5))
            r = measure(solve.x)
            correction = np.asarray(b['joint_correction']).copy()
            correction[indices] = solve.x-e.geometry['qpos'][-1, indices]
            for idx in indices:
                name = m.joint_id2name(int(idx))
                if name[:2] in ('FF', 'MF', 'RF', 'LF') and name[-1] in ('0', '1', '2'):
                    correction[idx] -= b['close']
                elif name == 'THJ0':
                    correction[idx] += b['close']
            r.update(success=bool(solve.success), message=str(solve.message), joint_correction=correction.tolist())
            results.append(r)
            print(json.dumps(r), flush=True)
        args.output.write_text(json.dumps(dict(results=results, indices=indices.tolist(),
            scope='Static reachability only, altered detection margin, NOT physical rollout or admission'), indent=2)+'\n')
    finally:
        e.env.close()


if __name__ == '__main__':
    main()
