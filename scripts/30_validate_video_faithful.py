#!/usr/bin/env python3
"""Execute the entire video trajectory and audit physics plus per-finger fidelity."""
import argparse
import csv
import hashlib
import json
import pickle
import sys
from importlib import import_module
from pathlib import Path

import numpy as np
import transforms3d
from scipy.interpolate import PchipInterpolator
from scipy.spatial.transform import Rotation, Slerp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from fromrealhand.paths import configure_runtime_paths
from fromrealhand.action_recovery import force_to_action
from fromrealhand.grasp_gate import physical_gate
from fromrealhand.video_fidelity import FINGERS, FINGER_NAMES, TIP_INDICES, HandLandmarks, fidelity_metrics, object_relative, source_clock

configure_runtime_paths()
base = import_module('24_search_physical_grasp')
contact_details = import_module('21_diagnose_geometry').contacts


class VideoExperiment(base.GraspExperiment):
    def __init__(self, geometry):
        super().__init__(geometry)
        g = self.geometry
        self.source_times = (g['source_frames']-g['source_frames'][0])/float(g['fps'])
        self.duration = float(self.source_times[-1])
        self.qcurve = PchipInterpolator(self.source_times, g['qpos'], axis=0)
        self.jcurve = PchipInterpolator(self.source_times, g['human_joints'], axis=0)
        self.pcurve = PchipInterpolator(self.source_times, g['object_poses'][:, :3, 3], axis=0)
        self.rcurve = Slerp(self.source_times, Rotation.from_matrix(g['object_poses'][:, :3, :3]))
        self.landmarks = HandLandmarks(self.model)
        self.grasp_time = self.source_times[np.argmin(np.abs(g['source_frames']-30))]

    def run_video(self, time_scale, close=0., seed=0, output=None, saved_actions=None, cartesian_gain=0., joint_correction=None,
                  closure_lead=0., feedback_weights=None, approach_gain=0., approach_root_gain=None,
                  action_selector=None):
        e, m, d = self.env, self.model, self.env.sim.data
        g = self.geometry
        e.reset()
        e.sim.reset()
        d.qpos[:30] = g['qpos'][0]
        d.qpos[30:33] = g['object_poses'][0, :3, 3]
        if seed:
            d.qpos[30:32] += np.random.RandomState(seed).uniform(-.002, .002, 2)
        d.qpos[33:37] = transforms3d.quaternions.mat2quat(g['object_poses'][0, :3, :3])
        d.qvel[:] = 0.
        d.qacc[:] = 0.
        m.body_pos[e.target_object_bid] = g['object_poses'][-1, :3, 3]
        m.body_quat[e.target_object_bid] = transforms3d.quaternions.mat2quat(g['object_poses'][-1, :3, :3])
        e.sim.forward()
        dt = e.control_timestep
        count = int(np.ceil((.5+self.duration*time_scale+2.)/dt))
        if saved_actions is not None and len(saved_actions) != count:
            raise ValueError('Saved action horizon differs from source retiming')
        if saved_actions is not None and action_selector is not None:
            raise ValueError('Cannot mix saved actions and an action selector')
        observations, actions, states, rewards, rows = [], [], [], [], []
        previous = g['qpos'][0].copy()
        kp = -m.actuator_biasprm[:, 1]
        weights = np.ones(5) if feedback_weights is None else np.asarray(feedback_weights, dtype=float)
        root_gain = approach_gain if approach_root_gain is None else float(approach_root_gain)
        if weights.shape != (5,) or not np.isfinite(weights).all() or np.any(weights <= 0):
            raise ValueError('feedback_weights must contain five finite positive values')
        streak = best_streak = 0
        for step in range(count):
            clock = float(source_clock(step*dt, self.duration, time_scale))
            desired = self.qcurve(clock)
            closing = float(np.clip((clock-self.grasp_time+closure_lead)/.2, 0., 1.)) * close
            for finger in ('FF', 'MF', 'RF', 'LF'):
                for joint in (2, 1, 0):
                    desired[m.jnt_qposadr[m.joint_name2id(finger+'J'+str(joint))]] += closing
            desired[m.jnt_qposadr[m.joint_name2id('THJ0')]] -= closing
            if joint_correction is not None:
                desired += float(np.clip((clock-self.grasp_time+closure_lead)/.35, 0., 1.))*np.asarray(joint_correction)
            desired = np.clip(desired, m.jnt_range[:30, 0], m.jnt_range[:30, 1])
            velocity = (desired-previous)/dt
            previous = desired.copy()
            if saved_actions is None:
                force = d.qfrc_bias[:30] + kp*(desired-d.qpos[:30]) + m.dof_damping[:30]*velocity
                if cartesian_gain or approach_gain or root_gain:
                    reference = np.eye(4)
                    reference[:3, 3] = self.pcurve(clock)
                    reference[:3, :3] = self.rcurve(clock).as_matrix()
                    local = object_relative(self.jcurve(clock)[TIP_INDICES], reference)
                    targets = local @ d.body_xmat[e.obj_bid].reshape(3, 3).T + d.body_xpos[e.obj_bid]
                    tips = self.landmarks.read(d)[TIP_INDICES]
                    body_name = m.body_id2name(e.obj_bid)
                    object_velocity = d.get_body_jacp(body_name).reshape(3, m.nv) @ d.qvel
                    object_omega = d.get_body_jacr(body_name).reshape(3, m.nv) @ d.qvel
                    blend = float(np.clip((clock-self.grasp_time)/.35, 0., 1.))
                    for i, finger in enumerate(FINGERS):
                        jac_full = d.get_site_jacp('S_'+finger+'tip').reshape(3, m.nv)
                        jac = jac_full[:, 6:30]
                        error = np.clip(targets[i]-tips[i], -.02, .02)
                        target_velocity = object_velocity + np.cross(object_omega, targets[i]-d.body_xpos[e.obj_bid])
                        relative_velocity = jac_full @ d.qvel-target_velocity
                        gain = cartesian_gain*weights[i]
                        feedback = gain*error - .4*np.sqrt(gain)*relative_velocity
                        force[6:] += blend*(jac.T @ feedback)
                        if approach_gain:
                            early = approach_gain*error - .4*np.sqrt(approach_gain)*relative_velocity
                            force[6:] += (1.-blend)*(jac.T @ early)
                        if root_gain:
                            early_root = root_gain*error - .4*np.sqrt(root_gain)*relative_velocity
                            force[:6] += (1.-blend)*(jac_full[:, :6].T @ early_root)
                action = force_to_action(force, d.qpos[:30], d.qvel[:30], m.actuator_gainprm[:, 0],
                                         m.actuator_biasprm, e.act_mid, e.act_rng)
            else:
                action = saved_actions[step]
            if action_selector is not None:
                action = np.asarray(action_selector(step, e._get_observations().copy(), action.copy()))
                if action.shape != (30,) or not np.isfinite(action).all():
                    raise ValueError('Action selector returned invalid control')
            observations.append(e._get_observations().copy())
            states.append(e.dump())
            actions.append(action.copy())
            _, reward, _, _ = e.step(action)
            rewards.append(reward)
            force, penetration, fingers, bottom = self.contact_stats()
            penetration = max(penetration, e.substep_max_penetration)
            per_finger = dict.fromkeys(FINGERS, 0.)
            for contact in contact_details(e):
                key = contact['hand'][2:4]
                if key in per_finger:
                    per_finger[key] += max(0., contact['normal_force_n'])
            pose = np.eye(4)
            pose[:3, 3] = d.body_xpos[e.obj_bid]
            pose[:3, :3] = d.body_xmat[e.obj_bid].reshape(3, 3)
            reference_pose = np.eye(4)
            reference_pose[:3, 3] = self.pcurve(clock)
            reference_pose[:3, :3] = self.rcurve(clock).as_matrix()
            metrics = fidelity_metrics(self.landmarks.read(d), pose, self.jcurve(clock), reference_pose)
            lifted = bottom > .015 and fingers >= 2 and force > .05
            streak = streak+1 if lifted else 0
            best_streak = max(best_streak, streak)
            row = dict(step=step, time_s=step*dt, source_frame=float(g['source_frames'][0]+clock*30),
                       bottom_m=bottom, force_n=force, fingers=fingers, penetration_m=penetration,
                       target_distance_m=float(np.linalg.norm(pose[:3, 3]-g['object_poses'][-1, :3, 3])),
                       mean_tip_error_m=metrics['mean_tip_error_m'])
            for i, finger in enumerate(FINGERS):
                row[finger+'_force_n'] = per_finger[finger]
                row[finger+'_tip_error_m'] = metrics['tip_error_m'][i]
            rows.append(row)
        tail = rows[-100:]
        report = dict(seed=seed, time_scale=time_scale, close=close, cartesian_gain=cartesian_gain, steps=count, hold_s=best_streak*dt,
                      max_bottom_m=max(r['bottom_m'] for r in rows), final_bottom_m=rows[-1]['bottom_m'],
                      final_distance_m=rows[-1]['target_distance_m'],
                      tail_min_bottom_m=min(r['bottom_m'] for r in tail),
                      tail_min_force_n=min(r['force_n'] for r in tail),
                      tail_min_fingers=min(r['fingers'] for r in tail),
                      max_penetration_m=max(r['penetration_m'] for r in rows),
                      saturation=float(np.mean(np.abs(actions)>=.999)),
                      mean_tip_error_m=float(np.mean([r['mean_tip_error_m'] for r in rows])),
                      tail_mean_tip_error_m=float(np.mean([r['mean_tip_error_m'] for r in tail])),
                      tail_finger_contact_fraction={name: float(np.mean([r[finger+'_force_n']>.01 for r in tail]))
                                                   for name, finger in zip(FINGER_NAMES, FINGERS)},
                      tail_finger_force_n={name: float(np.mean([r[finger+'_force_n'] for r in tail]))
                                          for name, finger in zip(FINGER_NAMES, FINGERS)},
                      tail_finger_tip_error_m={name: float(np.mean([r[finger+'_tip_error_m'] for r in tail]))
                                              for name, finger in zip(FINGER_NAMES, FINGERS)})
        report['physics_passed'] = physical_gate(report)
        report['actuator_saturated_steps'] = {
            m.actuator_id2name(i): int(np.sum(np.abs(np.asarray(actions)[:, i]) >= .999))
            for i in range(m.nu)}
        report['joint_correction'] = np.asarray(joint_correction if joint_correction is not None else np.zeros(30)).tolist()
        report['closure_lead'] = float(closure_lead)
        report['feedback_weights'] = weights.tolist()
        report['approach_gain'] = float(approach_gain)
        report['approach_root_gain'] = root_gain
        report['phase_mean_tip_error_m'] = {
            name: float(np.mean([r['mean_tip_error_m'] for r in rows if lo <= r['source_frame'] < hi]))
            for name, lo, hi in [('initial', 0, 12), ('approach', 12, 30), ('grasp', 30, 50), ('lift_hold', 50, float('inf'))]
            if any(lo <= r['source_frame'] < hi for r in rows)}
        fractions = report['tail_finger_contact_fraction']
        report['fidelity_passed'] = (report['tail_mean_tip_error_m'] < .015
                                     and report['mean_tip_error_m'] < .02
                                     and max(report['tail_finger_tip_error_m'].values()) < .025
                                     and fractions['thumb'] >= .8 and fractions['index'] >= .8
                                     and sum(v >= .8 for v in fractions.values()) >= 4)
        report['passed'] = bool(report['physics_passed'] and report['fidelity_passed'])
        self.last_demo = dict(observations=np.asarray(observations), actions=np.asarray(actions),
                              rewards=np.asarray(rewards), sim_data=states, model_data=[e.dump_mujoco_model()])
        if output is not None:
            output = Path(output)
            output.mkdir(parents=True, exist_ok=False)
            with (output/'steps.csv').open('w') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            (output/'summary.json').write_text(json.dumps(report, indent=2)+'\n')
            with (output/'diagnostic_rollout.pkl').open('xb') as stream:
                pickle.dump({'video_faithful': self.last_demo}, stream)
        return report, np.asarray(actions)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--geometry', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--scales', type=float, nargs='+', default=[2., 3., 4.])
    parser.add_argument('--closures', type=float, nargs='+', default=[0., .05, .1])
    parser.add_argument('--gains', type=float, nargs='+', default=[0.])
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    experiment = VideoExperiment(args.geometry)
    results = []
    for scale in args.scales:
        for close in args.closures:
            for gain in args.gains:
                report, _ = experiment.run_video(scale, close, cartesian_gain=gain)
                results.append(report)
                (args.output/'search.json').write_text(json.dumps(results, indent=2)+'\n')
                print(json.dumps(report), flush=True)
    def rank(report):
        fractions = report['tail_finger_contact_fraction']
        missing = max(0, 4-sum(v >= .8 for v in fractions.values()))
        return (not report['passed'], not report['physics_passed'],
                report['tail_mean_tip_error_m']+.01*missing, report['mean_tip_error_m'])
    best = min(results, key=rank)
    report, actions = experiment.run_video(best['time_scale'], best['close'], output=args.output/'best', cartesian_gain=best['cartesian_gain'])
    original = experiment.last_demo['observations'].copy()
    replay, _ = experiment.run_video(best['time_scale'], best['close'], saved_actions=actions, output=args.output/'replay', cartesian_gain=best['cartesian_gain'])
    replay_error = float(np.max(np.abs(original-experiment.last_demo['observations'])))
    holdouts = [experiment.run_video(best['time_scale'], best['close'], seed=seed, cartesian_gain=best['cartesian_gain'])[0] for seed in (1, 2, 3)]
    manifest = json.loads((args.geometry.parent/'manifest.json').read_text())
    unchanged = all(hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == digest
                    for name, digest in manifest['protected_files'].items())
    if not unchanged:
        raise RuntimeError('Protected baseline artifact changed')
    ready = (report['passed'] and replay['passed'] and replay_error < 1e-8
             and all(r['passed'] for r in holdouts) and not manifest['failed_optimizer_frames'])
    result = dict(training_ready=bool(ready), best=report, replay=replay, replay_observation_error=replay_error,
                  holdouts=holdouts, baseline_unchanged=unchanged,
                  root_wrist_tracking_gain=1., feedback='Object-relative fingertip impedance, wrist and fingers, damped and phase-ramped',
                  geometry=str(args.geometry.resolve()), source_manifest=str((args.geometry.parent/'manifest.json').resolve()),
                  fidelity_gate='tail mean tip error <15mm, every finger <25mm; thumb/index and >=4 fingers contact >=80% of final 1s',
                  contact_scope='Source video shows wrap grasp; human contact forces are unavailable')
    (args.output/'admission.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2), flush=True)
    experiment.env.close()


if __name__ == '__main__':
    main()
