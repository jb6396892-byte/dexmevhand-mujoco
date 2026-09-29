#!/usr/bin/env python3
"""Bounded ring surface targets; retain raw human targets for all fidelity scores."""
import argparse
import hashlib
import itertools
import json
from importlib import import_module
from pathlib import Path

import numpy as np
import trimesh
from scipy.interpolate import PchipInterpolator

transport = import_module('55_optimize_transport')
search = import_module('59_refine_ring_contact')
from fromrealhand.video_fidelity import object_relative, TIP_INDICES
from check_codex_budget import latest_usage


class ContactExperiment(transport.TransportExperiment):
    target_fraction = 0.
    target_depth = 0.
    ring_weight = 1.
    isolate_ring = False
    initial_correction = None

    def __init__(self, geometry):
        super().__init__(geometry)
        m, d = self.model, self.env.sim.data
        self.env.sim.forward()
        meshes = []
        for gid in range(m.ngeom):
            if m.geom_bodyid[gid] != self.env.obj_bid or m.geom_type[gid] != 7 or m.geom_group[gid] != 1:
                continue
            mid = m.geom_dataid[gid]
            verts = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid]+m.mesh_vertnum[mid]]
            faces = m.mesh_face[m.mesh_faceadr[mid]:m.mesh_faceadr[mid]+m.mesh_facenum[mid]]
            world = verts @ d.geom_xmat[gid].reshape(3, 3).T+d.geom_xpos[gid]
            local = (world-d.body_xpos[self.env.obj_bid]) @ d.body_xmat[self.env.obj_bid].reshape(3, 3)
            meshes.append(trimesh.Trimesh(vertices=local, faces=faces, process=False))
        if not meshes:
            raise ValueError('No native mug visual mesh found')
        mesh = trimesh.util.concatenate(meshes)
        self.mug_surface = mesh
        raw = np.array([object_relative(j[TIP_INDICES], p)[3]
                        for j, p in zip(self.geometry['human_joints'], self.geometry['object_poses'])])
        closest, distances, _ = trimesh.proximity.closest_point_naive(mesh, raw)
        self.surface_delta = closest-raw
        self.surface_distances = distances

    def run_video(self, *args, **kwargs):
        delta = self.surface_delta*self.target_fraction
        delta += self.target_depth*self.surface_delta/np.maximum(self.surface_distances[:, None], 1e-10)
        delta *= np.minimum(1., .025/np.maximum(np.linalg.norm(delta, axis=1), 1e-10))[:, None]
        values = np.zeros((len(delta), 5, 3)); values[:, 3] = delta
        curve = PchipInterpolator(self.source_times, values, axis=0)

        def bounded(t):
            offset = curve(t)
            offset *= np.minimum(1., .025/np.maximum(np.linalg.norm(offset, axis=1), 1e-10))[:, None]
            return offset

        kwargs['tip_offset_curve'] = bounded
        kwargs['isolated_fingers'] = ('rf',) if self.isolate_ring else ()
        if self.initial_correction is not None:
            def correction_at(t):
                s = float(np.clip((t-self.grasp_time-.5)/.6, 0., 1.))
                s = s**3*(10.-15.*s+6.*s*s)
                return self.initial_correction+s*(self.correction-self.initial_correction)
            kwargs['joint_correction_curve'] = correction_at
        weights = np.array(self.feedback_weights, dtype=float)
        weights[3] = self.ring_weight
        kwargs['feedback_weights'] = weights
        report, actions = super().run_video(*args, **kwargs)
        d = self.env.sim.data
        tip = d.site_xpos[self.model.site_name2id('S_rftip')]
        local_tip = (tip-d.body_xpos[self.env.obj_bid]) @ d.body_xmat[self.env.obj_bid].reshape(3, 3)
        _, gap, _ = trimesh.proximity.closest_point_naive(self.mug_surface, local_tip[None])
        report.update(ring_target_fraction=self.target_fraction, ring_target_depth_m=self.target_depth,
                      isolate_ring_feedback=self.isolate_ring,
                      initial_joint_correction=None if self.initial_correction is None else self.initial_correction.tolist(),
                      final_ring_surface_distance_m=float(gap[0]),
                      ring_feedback_weight=self.ring_weight, max_ring_target_offset_m=float(np.linalg.norm(delta, axis=1).max()),
                      source_ring_mesh_distance_tail_m=float(self.surface_distances[-10:].mean()),
                      fidelity_reference='Unmodified original human labels; ring surface target is a simulation adaptation')
        return report, actions


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--candidate', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--fractions', type=float, nargs='+', default=[.5, 1.])
    p.add_argument('--depths', type=float, nargs='+', default=[0., .003])
    p.add_argument('--weights', type=float, nargs='+', default=[1., 3., 6.])
    p.add_argument('--verify', action='store_true')
    p.add_argument('--inspect', action='store_true', help='Save seed-zero physical rollout only; never admit it')
    p.add_argument('--isolate-ring', action='store_true')
    p.add_argument('--quota-stop', type=float, default=80.)
    args = p.parse_args()
    if (not np.isfinite(args.fractions+args.depths+args.weights).all()
            or min(args.fractions) < 0 or max(args.fractions) > 1
            or min(args.depths) < 0 or max(args.depths) > .005 or min(args.weights) <= 0):
        p.error('Fractions must be 0..1, depths 0..5mm and weights positive; all finite')
    source = json.loads(args.candidate.read_text())
    b = source['best']
    args.output.mkdir(parents=True, exist_ok=False)
    e = ContactExperiment(source['geometry'])
    transport.configure(e, b)
    e.isolate_ring = args.isolate_ring or b.get('isolate_ring_feedback', False)
    if b.get('initial_joint_correction') is not None:
        e.initial_correction = np.asarray(b['initial_joint_correction'])
    e.transport_kp, e.transport_ki = b['transport_kp'], b['transport_ki']
    kwargs = dict(scale=b['time_scale'], close=b['close'], gain=b['cartesian_gain'])

    def budget():
        usage = latest_usage()
        return not usage or usage['used_percent'] < args.quota_stop

    def setup(r):
        e.target_fraction = r['ring_target_fraction']
        e.target_depth = r['ring_target_depth_m']
        e.ring_weight = r['ring_feedback_weight']

    reports = []
    try:
        if not args.verify and not args.inspect:
            for fraction, depth, weight in itertools.product(args.fractions, args.depths, args.weights):
                if not budget():
                    break
                e.target_fraction, e.target_depth, e.ring_weight = fraction, depth, weight
                r, _ = e.run_surface(**kwargs)
                r['search_score'] = search.rank(r)
                reports.append(r)
                (args.output/'search.json').write_text(json.dumps(reports, indent=2)+'\n')
                best = min(reports, key=lambda r: (not r['surface_physics_passed'], not r['fidelity_passed'], r['search_score']))
                (args.output/'candidate.json').write_text(json.dumps(dict(
                    best=best, geometry=source['geometry'], training_ready=False,
                    source_candidate=str(args.candidate.resolve())), indent=2)+'\n')
                print(json.dumps({k:r[k] for k in ('ring_target_fraction', 'ring_target_depth_m', 'ring_feedback_weight',
                    'tail_finger_contact_fraction', 'tail_mean_tip_error_m', 'max_hand_scene_penetration_m',
                    'surface_physics_passed', 'fidelity_passed', 'search_score')}), flush=True)
            return
        setup(b)
        replay, error, half = None, None, []
        for seed in (0, 30, 31, 32, 33, 34):
            if args.inspect and seed:
                break
            if not budget():
                break
            r, actions = e.run_surface(**kwargs, seed=seed, output=args.output/('best' if seed == 0 else 'seed_%d'%seed))
            reports.append(r)
            if seed == 0:
                nominal = actions.copy()
                before = e.last_demo['observations'].copy()
                replay, _ = e.run_surface(**kwargs, saved_actions=nominal, output=args.output/'replay')
                error = float(np.max(np.abs(before-e.last_demo['observations'])))
            (args.output/'progress.json').write_text(json.dumps(reports, indent=2)+'\n')
            print(json.dumps({k:r[k] for k in ('seed', 'surface_physics_passed', 'fidelity_passed', 'final_distance_m')}), flush=True)
        if len(reports) == 6 and budget():
            dt = e.model.opt.timestep/2
            e.model.opt.timestep = dt; e.env.model_timestep = dt
            half = [e.run_surface(**kwargs, saved_actions=nominal)[0], e.run_surface(**kwargs)[0]]
        manifest = json.loads((Path(source['geometry']).parent/'manifest.json').read_text())
        unchanged = all(hashlib.sha256((transport.finger.surface.ROOT/name).read_bytes()).hexdigest() == digest
                        for name, digest in manifest['protected_files'].items())
        complete = len(reports) == 6 and len(half) == 2
        all_reports = reports+([replay] if replay else [])+half
        physics = bool(complete and unchanged and error < 1e-8 and all(r['surface_physics_passed'] for r in all_reports))
        ready = bool(physics and not manifest['failed_optimizer_frames'] and
                     all(r['fidelity_passed'] and r['final_distance_m'] <= .02 for r in all_reports))
        result = dict(best=reports[0] if reports else None, holdouts=reports[1:], replay=replay, half_timestep=half,
                      surface_physics_passed=physics, training_ready=ready, completed=complete,
                      geometry=source['geometry'], source_candidate=str(args.candidate.resolve()),
                      replay_observation_error=error, baseline_unchanged=unchanged, quota=latest_usage(),
                      training_started=False, human_contact_labels_verified=False)
        (args.output/'admission.json').write_text(json.dumps(result, indent=2)+'\n')
        print('COMPLETE', json.dumps(dict(training_ready=ready, surface_physics_passed=physics, completed=complete)), flush=True)
    finally:
        e.env.close()


if __name__ == '__main__':
    main()
