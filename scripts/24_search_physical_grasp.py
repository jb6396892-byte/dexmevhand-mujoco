#!/usr/bin/env python3
"""Search contact-preserving, smooth grasp controllers using free-object dynamics."""
import argparse
import csv
import json
import pickle
import sys
from importlib import import_module
from pathlib import Path

import numpy as np
import transforms3d

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from fromrealhand.paths import configure_runtime_paths
from fromrealhand.action_recovery import force_to_action
from fromrealhand.grasp_gate import physical_gate
from fromrealhand.trajectory_control import smooth_blend as smooth, world_to_root_delta

configure_runtime_paths()
from hand_imitation.env.environments.ycb_relocate_env import YCBRelocate
from mujoco_py import functions




class AuditedRelocate(YCBRelocate):
    def _pre_action(self, action, policy_step=False):
        if policy_step:
            self.substep_max_penetration = 0.
        hands, mugs = set(self.robot_geom_names), set(self.body_geom_names)
        for c in self.sim.data.contact[:self.sim.data.ncon]:
            a, b = [self.sim.model.geom_id2name(int(g)) for g in (c.geom1, c.geom2)]
            if (a in hands and b in mugs) or (b in hands and a in mugs):
                self.substep_max_penetration = max(self.substep_max_penetration, -float(c.dist))
        return super()._pre_action(action, policy_step)


class GraspExperiment:
    def __init__(self, geometry):
        self.geometry = np.load(geometry)
        self.env = AuditedRelocate(has_renderer=False, object_name='mug', object_scale=.8,
                              friction=(1, .5, .01), solref='-6000 -300', randomness_scale=.25)
        self.model = self.env.sim.model
        self.hand_geoms = set(self.env.robot_geom_names)
        self.mug_geoms = set(self.env.body_geom_names)
        self.meshes = []
        for g in range(self.model.ngeom):
            if self.model.geom_bodyid[g] == self.env.obj_bid and self.model.geom_type[g] == 7 and self.model.geom_contype[g]:
                mid = self.model.geom_dataid[g]
                start = self.model.mesh_vertadr[mid]
                self.meshes.append((g, self.model.mesh_vert[start:start+self.model.mesh_vertnum[mid]].copy()))

    def contact_stats(self):
        d = self.env.sim.data
        fingers = set()
        force_sum = 0.
        penetration = 0.
        for i, c in enumerate(d.contact[:d.ncon]):
            a, b = [self.model.geom_id2name(int(x)) for x in (c.geom1, c.geom2)]
            if not ((a in self.hand_geoms and b in self.mug_geoms) or (b in self.hand_geoms and a in self.mug_geoms)):
                continue
            penetration = max(penetration, -float(c.dist))
            f = np.zeros(6)
            functions.mj_contactForce(self.model, d, i, f)
            force_sum += max(0., f[0])
            if f[0] > .01:
                name = a if a in self.hand_geoms else b
                fingers.add(name[2:4] if name.startswith('C_') else name)
        bottom = min(float((v @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g])[:, 2].min()) for g, v in self.meshes)
        return force_sum, penetration, len(fingers), bottom

    def run(self, params, output=None, seed=0, saved_actions=None):
        e, m, d = self.env, self.model, self.env.sim.data
        e.reset()
        e.sim.reset()
        rng = np.random.RandomState(seed)
        q0 = self.geometry['qpos'][0].copy()
        pose = self.geometry['object_poses'][0]
        d.qpos[:30] = q0
        d.qpos[30:33] = pose[:3, 3]
        if seed:
            d.qpos[30:32] += rng.uniform(-.002, .002, 2)
        d.qpos[33:37] = transforms3d.quaternions.mat2quat(pose[:3, :3])
        d.qvel[:] = 0
        d.qacc[:] = 0
        e.sim.forward()
        initial = e.dump()
        translation_axes = d.get_body_jacp('palm').reshape(3, m.nv)[:, :3].copy()
        target = pose[:3, 3].copy()
        target[2] += .10
        if params.get('relocate', False):
            target = self.geometry['object_poses'][-1, :3, 3].copy()
        lift_delta = world_to_root_delta(translation_axes, target-pose[:3, 3])
        frame = int(params.get('frame', 25))
        qgrip = self.geometry['qpos'][frame].copy()
        world_delta = pose[:3, 3] - self.geometry['object_poses'][frame, :3, 3]
        world_delta += np.asarray(params.get('offset', [0., 0., 0.]))
        qgrip[:3] += world_to_root_delta(translation_axes, world_delta)
        for i in [9, 10, 11, 13, 14, 15, 17, 18, 19, 22, 23, 24]:
            qgrip[i] += params.get('close', 0.)
        qgrip[29] -= params.get('thumb', 0.)
        qgrip[26] += params.get('opposition', 0.)
        qgrip = np.clip(qgrip, m.jnt_range[:30, 0]+.005, m.jnt_range[:30, 1]-.005)
        m.body_pos[e.target_object_bid] = target
        m.body_quat[e.target_object_bid] = d.qpos[33:37]
        e.sim.forward()
        kp = -m.actuator_biasprm[:, 1]
        dt = e.control_timestep
        steps = int(params.get('steps', 650)) if saved_actions is None else len(saved_actions)
        rows, actions, states, observations, rewards = [], [], [], [], []
        video = context = None
        if output is not None:
            import cv2
            from mujoco_py import MjRenderContextOffscreen
            output = Path(output)
            output.mkdir(parents=True, exist_ok=False)
            context = MjRenderContextOffscreen(e.sim)
            video = cv2.VideoWriter(str(output/'rollout.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), 25, (640, 480))
            if not video.isOpened():
                raise RuntimeError('Cannot open video writer')
        streak = best_streak = 0
        previous = q0.copy()
        for step in range(steps):
            t = step * dt
            desired = q0 + smooth((t-.5)/2.) * (qgrip-q0)
            desired[:3] += lift_delta * smooth((t-3.)/1.5)
            velocity = (desired-previous)/dt
            previous = desired.copy()
            if saved_actions is None:
                # Preserve native position-actuator stiffness and implicit joint damping.
                force = d.qfrc_bias[:30] + kp*(desired-d.qpos[:30]) + m.dof_damping[:30]*velocity
                action = force_to_action(force, d.qpos[:30], d.qvel[:30], m.actuator_gainprm[:, 0],
                                         m.actuator_biasprm, e.act_mid, e.act_rng)
            else:
                action = saved_actions[step]
            observations.append(e._get_observations().copy())
            states.append(e.dump())
            actions.append(action.copy())
            _, reward, _, _ = e.step(action)
            rewards.append(reward)
            force, penetration, fingers, bottom = self.contact_stats()
            penetration = max(penetration, e.substep_max_penetration)
            lifted = bottom > .015 and fingers >= 2 and force > .05
            streak = streak+1 if lifted else 0
            best_streak = max(best_streak, streak)
            rows.append(dict(step=step, time_s=t, z=float(d.body_xpos[e.obj_bid, 2]), bottom_m=bottom,
                             force_n=force, fingers=fingers, penetration_m=penetration,
                             root_error_m=float(np.linalg.norm(d.qpos[:3]-desired[:3])),
                             finger_error_rad=float(np.sqrt(np.mean((d.qpos[8:30]-desired[8:30])**2))),
                             target_distance_m=float(np.linalg.norm(d.body_xpos[e.obj_bid]-target))))
            if video is not None and step % 4 == 0:
                context.render(640, 480, camera_id=m.camera_name2id('frontview'))
                video.write(cv2.cvtColor(context.read_pixels(640, 480, depth=False)[::-1], cv2.COLOR_RGB2BGR))
        if video is not None:
            video.release()
        actions = np.asarray(actions)
        tail = rows[-100:]
        report = dict(params=params, seed=seed, hold_s=best_streak*dt,
                      max_bottom_m=max(r['bottom_m'] for r in rows),
                      final_bottom_m=rows[-1]['bottom_m'],
                      final_distance_m=rows[-1]['target_distance_m'],
                      max_penetration_m=max(r['penetration_m'] for r in rows),
                      final_force_n=rows[-1]['force_n'], final_fingers=rows[-1]['fingers'],
                      mean_tail_bottom_m=float(np.mean([r['bottom_m'] for r in tail])),
                      tail_min_bottom_m=min(r['bottom_m'] for r in tail),
                      tail_min_force_n=min(r['force_n'] for r in tail),
                      tail_min_fingers=min(r['fingers'] for r in tail),
                      saturation=float(np.mean(np.abs(actions) >= .999)),
                      final_root_error_m=rows[-1]['root_error_m'])
        report['passed'] = physical_gate(report)
        self.last_demo = dict(observations=np.asarray(observations), actions=actions,
                              rewards=np.asarray(rewards), sim_data=states,
                              model_data=[e.dump_mujoco_model()])
        if output is not None:
            with (output/'steps.csv').open('w') as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            (output/'summary.json').write_text(json.dumps(report, indent=2)+'\n')
            np.save(output/'actions.npy', actions)
            with (output/'diagnostic_rollout.pkl').open('wb') as f:
                pickle.dump({'physical': dict(observations=np.asarray(observations), actions=actions,
                            rewards=np.asarray(rewards), sim_data=states, model_data=[e.dump_mujoco_model()])}, f)
        return report, actions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--geometry', default='data/processed/seq_dexycb_001/repair_v2/geometry.npz')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--trials', type=int, default=12)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    experiment = GraspExperiment(args.geometry)
    results = []
    rng = np.random.RandomState(24)
    for i in range(args.trials):
        if i < 6:
            params = dict(frame=[10, 20, 30, 40, 50, 30][i], close=0. if i<5 else .2, thumb=0.)
        else:
            params = dict(frame=int(rng.randint(10, 54)), close=float(rng.uniform(-.1, .5)),
                          thumb=float(rng.uniform(-.1, .6)), opposition=float(rng.uniform(-.2, .2)),
                          offset=rng.uniform(-.008, .008, 3).tolist())
        report, _ = experiment.run(params)
        results.append(report)
        print(json.dumps(report), flush=True)
        (args.output/'search.json').write_text(json.dumps(results, indent=2)+'\n')
    best = max(results, key=lambda r: (r['passed'], r['hold_s'], r['mean_tail_bottom_m']))
    report, actions = experiment.run(best['params'], args.output/'best')
    replay, _ = experiment.run(best['params'], args.output/'replay', saved_actions=actions)
    seeds = [experiment.run(best['params'], seed=i)[0] for i in range(1, 6)]
    (args.output/'verification.json').write_text(json.dumps(dict(best=report, replay=replay, seeds=seeds), indent=2)+'\n')
    print(json.dumps(dict(best=report, replay=replay, seeds=seeds), indent=2), flush=True)


if __name__ == '__main__':
    main()
