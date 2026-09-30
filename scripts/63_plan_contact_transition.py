#!/usr/bin/env python3
"""Plan supporting-contact continuation, then test native-physics tracking."""
import argparse
import hashlib
import json
import pickle
from importlib import import_module
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import transforms3d
from scipy.interpolate import PchipInterpolator
from scipy.optimize import minimize, LinearConstraint
from scipy.spatial import cKDTree

surface = import_module('33_optimize_surface_grasp')
from fromrealhand.video_fidelity import TIP_INDICES, object_relative, source_clock
from check_codex_budget import latest_usage
from mujoco_py import functions


def budget_available():
    usage = latest_usage()
    return not usage or usage['used_percent'] < 80


def contact_push(gap, dt, integral, gain, limit):
    integral = float(np.clip(integral+gain*gap*dt, 0., limit))
    return min(limit, .2+300*max(0., gap)+integral), integral


def plan(args, source, demo):
    e = import_module('60_refine_contact_targets').ContactExperiment(source['geometry'])
    e.env.pack_mujoco_model(demo['model_data'][0])
    m, d = e.env.sim.model, e.env.sim.data
    e.model = m
    for key, values in demo['physics_model'].items():
        getattr(m, key)[:] = values
    original_margins = m.geom_margin.copy()
    original_gaps = m.geom_gap.copy()
    hand = {i for i in range(m.ngeom) if m.geom_id2name(i) in e.hand_geoms}
    mug = {i for i in range(m.ngeom) if m.geom_id2name(i) in e.mug_geoms}
    tree = cKDTree(e.mug_surface.vertices)
    capsules = [i for i in hand if m.geom_type[i] == 3 and (m.geom_id2name(i) or '')[2:4] in ('th','ff','mf','rf')]
    tips = [m.site_name2id('S_'+f+'tip') for f in ('th', 'ff', 'mf', 'rf', 'lf')]
    indices = np.arange(0 if args.free_root else 6, 30)
    limits = m.jnt_range[indices].copy()
    interior = np.where(indices < 3, .001, .04)
    limits[:, 0] += interior; limits[:, 1] -= interior
    if args.effort_margin:
        limits[np.flatnonzero(indices == 16)[0], 1] = min(limits[np.flatnonzero(indices == 16)[0], 1], .3)
    states = demo['sim_data']
    qnom = np.array([s['qpos'][:30] for s in states])
    anchors = list(range(args.start, len(states), args.stride))
    if anchors[-1] != len(states)-1:
        anchors.append(len(states)-1)
    previous_delta = np.zeros(len(indices))
    rows, solutions = [], []
    initial_gap = None
    if args.resume:
        previous = json.loads((args.resume/'plan_admission.json').read_text())
        if previous['rollout_sha256'] != hashlib.sha256(args.rollout.read_bytes()).hexdigest():
            raise ValueError('Resume rollout mismatch')
        old = np.load(args.resume/'plan.npz')
        if not np.array_equal(old['joint_indices'] if 'joint_indices' in old else np.arange(6,30), indices):
            raise ValueError('Resume joint indices mismatch')
        old_rows = json.loads((args.resume/'planning.json').read_text())
        for i, r in enumerate(old_rows):
            if not r['feasible']:
                break
            if r['step'] != anchors[i] or int(old['steps'][i]) != anchors[i]:
                raise ValueError('Resume anchor mismatch')
            rows.append(r); solutions.append(old['deltas'][i].copy())
        if rows:
            initial_gap = rows[0]['ring_target']
            previous_delta = solutions[-1]
    try:
        for step in anchors:
            if rows and step <= rows[-1]['step']:
                continue
            if not budget_available():
                break
            state = states[step]
            e.env.pack(state)
            functions.mj_kinematics(m, d); functions.mj_comPos(m, d)
            functions.mj_tendon(m, d)
            limited = np.flatnonzero(m.tendon_limited)
            tendon_jac = d.ten_J[limited][:, indices].copy()
            tendon_offset = d.ten_length[limited]-tendon_jac @ d.qpos[indices]
            linear_tendon = LinearConstraint(1000*tendon_jac,
                1000*(m.tendon_range[limited,0]-tendon_offset-.00001),
                1000*(m.tendon_range[limited,1]-tendon_offset+.00001))
            t = step*e.env.control_timestep
            clock = float(source_clock(t, e.duration, source['best']['time_scale']))
            pose = np.eye(4); pose[:3, 3] = e.pcurve(clock); pose[:3, :3] = e.rcurve(clock).as_matrix()
            human = object_relative(e.jcurve(clock)[TIP_INDICES], pose)
            targets = human @ d.body_xmat[e.env.obj_bid].reshape(3, 3).T+d.body_xpos[e.env.obj_bid]
            center = qnom[step, indices]+previous_delta
            radius = np.where(indices < 3, .005, .12)
            bounds = np.column_stack([np.maximum(limits[:, 0], center-radius), np.minimum(limits[:, 1], center+radius)])
            if args.effort_margin:
                conversion = m.actuator_gainprm[indices, 0]*e.env.act_rng[indices]/(-m.actuator_biasprm[indices,1])
                safe = 1-args.effort_margin
                bounds[:,0] = np.maximum(bounds[:,0], qnom[step,indices]+(-safe-demo['actions'][step,indices])*conversion)
                bounds[:,1] = np.minimum(bounds[:,1], qnom[step,indices]+(safe-demo['actions'][step,indices])*conversion)
            if (bounds[:,0] > bounds[:,1]).any():
                raise ValueError('No joint/actuator interior intersection at step %d'%step)
            raw_center = center.copy()
            projected = minimize(lambda q:np.sum((q-raw_center)**2), np.clip(center, bounds[:,0], bounds[:,1]),
                jac=lambda q:2*(q-raw_center), method='SLSQP', bounds=bounds, constraints=[linear_tendon],
                options=dict(maxiter=100, ftol=1e-14))
            center = projected.x
            cache = [None, None]

            def measure(q):
                if cache[0] is not None and np.array_equal(q, cache[0]):
                    return cache[1]
                d.qpos[:] = state['qpos']; d.qpos[indices] = q
                functions.mj_kinematics(m, d); functions.mj_comPos(m, d)
                functions.mj_tendon(m, d); functions.mj_collision(m, d)
                gaps = dict.fromkeys(('th', 'ff', 'mf', 'rf'), .04)
                for gid in capsules:
                    axis = d.geom_xmat[gid].reshape(3, 3)[:, 2]
                    points = d.geom_xpos[gid]+np.linspace(-m.geom_size[gid,1], m.geom_size[gid,1], 11)[:,None]*axis
                    local = (points-d.body_xpos[e.env.obj_bid]) @ d.body_xmat[e.env.obj_bid].reshape(3, 3)
                    gap = float(np.min(tree.query(local)[0])-m.geom_size[gid,0])
                    finger = m.geom_id2name(gid)[2:4]
                    gaps[finger] = min(gaps[finger], gap)
                native = {}
                depth = 0.
                for c in d.contact[:d.ncon]:
                    a, b = int(c.geom1), int(c.geom2)
                    if a in hand or b in hand:
                        depth = max(depth, -float(c.dist))
                    gid = a if a in hand and b in mug else b if b in hand and a in mug else None
                    if gid is not None:
                        finger = m.geom_id2name(gid)[2:4]
                        if finger in gaps:
                            native[finger] = min(native.get(finger, .04), float(c.dist))
                gaps.update(native)
                error = np.linalg.norm(d.site_xpos[tips]-targets, axis=1)
                tendon = max(0., float(np.max(m.tendon_range[:, 0]-d.ten_length)), float(np.max(d.ten_length-m.tendon_range[:, 1])))
                result = dict(gaps=gaps, native_gaps=native, penetration=depth, tendon=tendon, tip_error=error.tolist())
                cache[:] = [q.copy(), result]
                return result

            if initial_gap is None:
                initial_gap = measure(qnom[step, indices])['gaps']['rf']
            phase = np.clip((step-args.start)/args.transition_steps, 0., 1.)
            ring_target = (1-phase)*initial_gap-phase*.0003

            def objective(q):
                r = measure(q)
                return (4000*(r['gaps']['rf']-ring_target)**2+
                        30*np.sum(np.square(r['tip_error']))+.003*np.sum((q-center)**2)
                        +10000*max(0., r['penetration']-.0005)**2)

            def constraints(q):
                r = measure(q)
                return np.array([.001-r['penetration'], .00002-r['tendon'],
                                 .025-max(r['tip_error']), .015-np.mean(r['tip_error'])]+
                                [.0001-r['gaps'][f] for f in ('th','ff','mf')])

            if step == args.start:
                solve = SimpleNamespace(x=qnom[step, indices].copy(), success=True, message='Unchanged transition anchor')
            elif (phase == 1 and np.min(constraints(center)) >= -1e-8
                  and measure(center)['native_gaps'].get('rf', float('inf')) <= .0001):
                solve = SimpleNamespace(x=center.copy(), success=True, message='Feasible held-contact continuation')
            else:
                solve = minimize(objective, np.clip(center, bounds[:, 0], bounds[:, 1]), method='SLSQP',
                                 bounds=bounds, constraints=[linear_tendon, dict(type='ineq', fun=constraints)],
                                 options=dict(maxiter=160, ftol=1e-10, eps=1e-5))
                if np.min(constraints(solve.x)) < -1e-8:
                    raw_solution = solve.x.copy()
                    repaired = minimize(lambda q:np.sum((q-raw_solution)**2), raw_solution,
                        jac=lambda q:2*(q-raw_solution), method='SLSQP', bounds=bounds, constraints=[linear_tendon],
                        options=dict(maxiter=100, ftol=1e-14))
                    if np.min(constraints(repaired.x)) >= -1e-8:
                        solve = SimpleNamespace(x=repaired.x, success=False, message='Feasible after exact tendon projection')
            report = measure(solve.x).copy()
            feasible = bool(np.isfinite(solve.x).all() and np.min(constraints(solve.x)) >= -1e-8)
            report.update(step=step, converged=bool(solve.success), feasible=feasible, message=str(solve.message), ring_target=ring_target)
            rows.append(report)
            solutions.append(solve.x-qnom[step, indices])
            previous_delta = solutions[-1]
            (args.output/'planning.json').write_text(json.dumps(rows, indent=2)+'\n')
            print(json.dumps(report), flush=True)
            if not feasible:
                break
        completed = len(rows) == len(anchors)
        unchanged = bool(np.array_equal(m.geom_margin, original_margins) and np.array_equal(m.geom_gap, original_gaps))
        ready = bool(completed and all(r['feasible'] for r in rows)
                     and rows[-1]['native_gaps'].get('rf', float('inf')) <= .0001 and unchanged)
        np.savez(args.output/'plan.npz', steps=np.asarray(anchors[:len(rows)]), deltas=np.asarray(solutions), nominal_qpos=qnom, joint_indices=indices)
        (args.output/'plan_admission.json').write_text(json.dumps(dict(
            planning_ready=ready, completed=completed, native_collision_parameters_unchanged=unchanged,
            source_candidate=str(args.candidate.resolve()),
            geometry=source['geometry'], rollout=str(args.rollout.resolve()), rollout_sha256=hashlib.sha256(args.rollout.read_bytes()).hexdigest(),
            scope='Geometry-only continuation, not physical grasp admission', quota=latest_usage()), indent=2)+'\n')
    finally:
        e.env.close()


class TrackingActions:
    def __init__(self, experiment, demo, plan_data, gain, ring_only=False, object_relative_root=False, ring_force=0., root_follow=0., ring_integral=0.):
        self.e, self.demo, self.gain = experiment, demo, gain
        self.start = int(plan_data['steps'][0])
        self.ring_only, self.object_relative_root = ring_only, object_relative_root
        self.integral = np.zeros(3)
        self.ring_force = ring_force
        self.root_follow = root_follow
        self.ring_integral_gain = ring_integral
        self.ring_integral = 0.
        self.indices = plan_data['joint_indices'] if 'joint_indices' in plan_data else np.arange(6,30)
        if ring_force:
            m, d = experiment.model, experiment.env.sim.data
            vertices = []
            for gid in range(m.ngeom):
                if m.geom_bodyid[gid] != experiment.env.obj_bid or m.geom_type[gid] != 7 or m.geom_group[gid] != 1:
                    continue
                mid = m.geom_dataid[gid]
                v = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid]+m.mesh_vertnum[mid]]
                world = v @ d.geom_xmat[gid].reshape(3,3).T+d.geom_xpos[gid]
                vertices.append((world-d.body_xpos[experiment.env.obj_bid]) @ d.body_xmat[experiment.env.obj_bid].reshape(3,3))
            self.tree = cKDTree(np.vstack(vertices))
            self.ring_geoms = [i for i in range(m.ngeom) if (m.geom_id2name(i) or '').startswith('C_rf') and m.geom_type[i] == 3]
        steps = np.r_[0, self.start-1, plan_data['steps']]
        offsets = np.vstack([np.zeros((2, len(self.indices))), plan_data['deltas']])
        self.curve = PchipInterpolator(steps, offsets, axis=0)

    def __len__(self):
        return len(self.demo['actions'])

    def __getitem__(self, step):
        action = self.demo['actions'][step].copy()
        if step < self.start:
            return action
        e, m, d = self.e.env, self.e.model, self.e.env.sim.data
        state = self.demo['sim_data'][step]
        delta = np.zeros(30); delta[self.indices] = self.curve(step)
        velocity_delta = np.zeros(30); velocity_delta[self.indices] = self.curve.derivative()(step)/e.control_timestep
        target = state['qpos'][:30]+delta
        if self.object_relative_root or self.root_follow:
            body = m.body_name2id('forearm')
            base_rotation = transforms3d.quaternions.quat2mat(m.body_quat[body])
            old_rotation = transforms3d.quaternions.quat2mat(state['qpos'][33:37])
            rotation = d.body_xmat[e.obj_bid].reshape(3, 3) @ old_rotation.T
            old_position = m.body_pos[body]+base_rotation @ target[:3]
            relative = old_position-state['qpos'][30:33]
            position = d.body_xpos[e.obj_bid]+rotation @ relative
            self.integral += .3*(state['qpos'][30:33]-d.body_xpos[e.obj_bid])*e.control_timestep
            self.integral = np.clip(self.integral, -.03, .03)
            position += self.integral
            hand_rotation = rotation @ base_rotation @ transforms3d.euler.euler2mat(*target[3:6], axes='rxyz')
            target[:3] = base_rotation.T @ (position-m.body_pos[body])
            target[3:6] = transforms3d.euler.mat2euler(base_rotation.T @ hand_rotation, axes='rxyz')
            if self.root_follow:
                correction = target[:6]-state['qpos'][:6]
                correction[3:] = (correction[3:]+np.pi)%(2*np.pi)-np.pi
                limits = np.array([.01]*3+[.12]*3)
                correction = np.clip(self.root_follow*correction, -limits, limits)
                target[:6] = state['qpos'][:6]+correction
            target[:6] = np.clip(target[:6], m.jnt_range[:6,0], m.jnt_range[:6,1])
            delta[:6] = target[:6]-state['qpos'][:6]
        desired_velocity = state['qvel'][:30]+velocity_delta
        kp = -m.actuator_biasprm[:, 1]
        gain = np.full(30, self.gain); gain[:6] = min(self.gain, 2.)
        if self.ring_only:
            gain[:] = 0.; gain[16:20] = self.gain
        blend = np.clip((step-self.start)/50., 0., 1.)
        force = kp*delta+blend*(gain*kp*(target-d.qpos[:30])+.08*np.sqrt(gain*kp)*(desired_velocity-d.qvel[:30]))
        if self.ring_force:
            loaded = {c['hand'][2:4] for c in surface.video.contact_details(e) if c['normal_force_n'] > .01}
            if {'th','ff','mf'} <= loaded and self.e.contact_stats()[3] > .015:
                choices = []
                for gid in self.ring_geoms:
                    points = d.geom_xpos[gid]+np.linspace(-m.geom_size[gid,1], m.geom_size[gid,1], 11)[:,None]*d.geom_xmat[gid].reshape(3,3)[:,2]
                    local = (points-d.body_xpos[e.obj_bid]) @ d.body_xmat[e.obj_bid].reshape(3,3)
                    dist, idx = self.tree.query(local)
                    i = int(np.argmin(dist))
                    nearest = self.tree.data[idx[i]] @ d.body_xmat[e.obj_bid].reshape(3,3).T+d.body_xpos[e.obj_bid]
                    choices.append((float(dist[i]-m.geom_size[gid,0]), gid, points[i], nearest))
                gap, gid, point, nearest = min(choices, key=lambda x:x[0])
                direction = (nearest-point)/max(float(np.linalg.norm(nearest-point)), 1e-10)
                body = m.geom_bodyid[gid]; name = m.body_id2name(body)
                jp = d.get_body_jacp(name).reshape(3,m.nv)
                jr = d.get_body_jacr(name).reshape(3,m.nv)
                jac = jp+np.cross(jr.T, point-d.body_xpos[body]).T
                ring_contacts = [c['distance_m'] for c in surface.video.contact_details(e) if c['hand'].startswith('C_rf')]
                if ring_contacts:
                    gap = min(ring_contacts)
                magnitude, self.ring_integral = contact_push(gap, e.control_timestep, self.ring_integral,
                                                              self.ring_integral_gain, self.ring_force)
                push = magnitude*direction
                force[16:20] += blend*(jac[:,16:20].T @ push)
        return np.clip(action+force/(m.actuator_gainprm[:, 0]*e.act_rng), -1., 1.)


def replay(args, source, demo):
    plan_root = args.plan
    manifest = json.loads((plan_root/'plan_admission.json').read_text())
    if not manifest['planning_ready']:
        raise ValueError('Infeasible/incomplete plan cannot enter dynamic verification')
    if hashlib.sha256(args.rollout.read_bytes()).hexdigest() != manifest['rollout_sha256']:
        raise ValueError('Plan rollout provenance mismatch')
    data = np.load(plan_root/'plan.npz')
    e = surface.SurfaceExperiment(source['geometry'])
    e.env.pack_mujoco_model(demo['model_data'][0])
    b = source['best']
    kwargs = dict(scale=b['time_scale'], close=b['close'], gain=b['cartesian_gain'])
    reports = []
    try:
        for gain in args.gains:
            if not budget_available():
                break
            r, actions = e.run_surface(**kwargs, saved_actions=TrackingActions(e, demo, data, gain, args.ring_only, args.object_relative_root, args.ring_force, args.root_follow, args.ring_integral), output=args.output/('gain_%g'%gain))
            r['tracking_gain'] = gain
            r.update(ring_only=args.ring_only, object_relative_root=args.object_relative_root, ring_force_limit_n=args.ring_force, root_follow=args.root_follow, ring_integral_gain=args.ring_integral)
            (args.output/('gain_%g'%gain)/'summary.json').write_text(json.dumps(r, indent=2)+'\n')
            reports.append(r)
            (args.output/'search.json').write_text(json.dumps(reports, indent=2)+'\n')
            print(json.dumps({k:r[k] for k in ('tracking_gain','surface_physics_passed','fidelity_passed','tail_finger_contact_fraction','final_distance_m','max_hand_scene_penetration_m')}), flush=True)
        (args.output/'status.json').write_text(json.dumps(dict(training_started=False, generalization_started=False,
            training_ready=False, trials=len(reports), scope='Nominal-scene development only; saved-action and timestep checks still required'), indent=2)+'\n')
    finally:
        e.env.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--candidate', type=Path, required=True)
    p.add_argument('--rollout', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--plan', type=Path)
    p.add_argument('--resume', type=Path)
    p.add_argument('--start', type=int, default=850)
    p.add_argument('--stride', type=int, default=50)
    p.add_argument('--transition-steps', type=int, default=300)
    p.add_argument('--gains', type=float, nargs='+', default=[0., 2., 8.])
    p.add_argument('--ring-only', action='store_true')
    p.add_argument('--object-relative-root', action='store_true')
    p.add_argument('--ring-force', type=float, default=0.)
    p.add_argument('--root-follow', type=float, default=0.)
    p.add_argument('--ring-integral', type=float, default=0.)
    p.add_argument('--free-root', action='store_true')
    p.add_argument('--effort-margin', type=float, default=0.)
    args = p.parse_args()
    source = json.loads(args.candidate.read_text())
    demo = pickle.loads(args.rollout.read_bytes())['video_faithful']
    if not np.isfinite(args.gains).all() or min(args.gains) < 0:
        p.error('Tracking gains must be finite and nonnegative')
    if not np.isfinite(args.ring_force) or not 0 <= args.ring_force <= 8:
        p.error('Ring force must be finite and between 0 and 8 N')
    if not np.isfinite(args.ring_integral) or not 0 <= args.ring_integral <= 1000:
        p.error('Ring integral gain must be finite and between 0 and 1000')
    if not np.isfinite(args.root_follow) or not 0 <= args.root_follow <= 1:
        p.error('Root following fraction must be between 0 and 1')
    if not np.isfinite(args.effort_margin) or not 0 <= args.effort_margin < .2:
        p.error('Effort margin must be finite, nonnegative and below .2')
    if args.start < 2 or args.start >= len(demo['actions']) or args.stride <= 0 or args.transition_steps <= 0:
        p.error('Invalid planning steps')
    args.output.mkdir(parents=True, exist_ok=False)
    config = {key:str(value) if isinstance(value, Path) else value for key,value in vars(args).items()}
    config.update(script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  candidate_sha256=hashlib.sha256(args.candidate.read_bytes()).hexdigest())
    (args.output/'configuration.json').write_text(json.dumps(config, indent=2)+'\n')
    (replay if args.plan else plan)(args, source, demo)


if __name__ == '__main__':
    main()
