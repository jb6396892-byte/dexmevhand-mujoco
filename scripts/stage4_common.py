"""Local adapters for physical skill export, snapshots, and read-only checks."""
import copy
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
import v10_common as common
from fromrealhand.skills import FINGERS

ROOT = common.ROOT
ARRAYS = ('ctrl', 'qfrc_applied', 'xfrc_applied', 'mocap_pos', 'mocap_quat',
          'userdata', 'qacc', 'qacc_warmstart')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_config(path):
    return json.loads(Path(path).read_text())


def snapshot(env):
    return dict(state=copy.deepcopy(env.sim.get_state()),
        arrays={key: None if getattr(env.sim.data, key) is None else getattr(env.sim.data, key).copy()
                for key in ARRAYS},
        timestep=env.timestep, cur_time=env.cur_time, done=env.done)


def restore_once(env, saved):
    """Only for a new episode/independent segment, never for a skill transition."""
    env.sim.set_state(saved['state'])
    for key, value in saved['arrays'].items():
        if value is not None:
            getattr(env.sim.data, key)[:] = value
    env.sim.forward()
    # forward overwrites these solver inputs; preserve the integration history.
    for key in ('qacc', 'qacc_warmstart'):
        getattr(env.sim.data, key)[:] = saved['arrays'][key]
    for key in ('timestep', 'cur_time', 'done'):
        setattr(env, key, saved[key])


def measure(exp, audit_start=0):
    e, m, d = exp.env, exp.model, exp.env.sim.data
    exp.sample()
    audit = exp.audit[audit_start:]
    _, _, _, bottom = exp.contact_stats()
    forces = dict.fromkeys(FINGERS, 0.)
    for contact in common.surface.video.contact_details(e):
        finger = contact['hand'][2:4]
        if finger in forces:
            forces[finger] += max(0., contact['normal_force_n'])
    row = dict(bottom_m=float(bottom), target_distance_m=float(np.linalg.norm(
        d.body_xpos[e.obj_bid]-d.body_xpos[e.target_object_bid])),
        scene_penetration_m=max(x['scene_penetration'] for x in audit),
        joint_violation_rad=max(x['joint'] for x in audit), finite=all(x['finite'] for x in audit))
    row.update({finger+'_force_n': float(value) for finger, value in forces.items()})
    return row


def experiment(geometry, demo):
    exp = common.surface.SurfaceExperiment(geometry)
    exp.env.pack_mujoco_model(demo['model_data'][0])
    for name, value in demo['physics_model'].items():
        getattr(exp.model, name)[:] = value
    return exp


def capture(video, geometry, seed, demo):
    exp = experiment(geometry, demo)
    snapshots, measurements, states = [], [], []
    original = exp.env.step
    initial = []

    def step(action):
        if not initial:
            initial.append(measure(exp))
        snapshots.append(snapshot(exp.env))
        index = len(exp.audit)
        result = original(action)
        measurements.append(measure(exp, index))
        states.append(np.r_[exp.env.sim.data.qpos.copy(), exp.env.sim.data.qvel.copy()])
        return result

    exp.env.step = step
    b = video['control']
    try:
        report, _ = exp.run_surface(scale=b['time_scale'], close=b['close'], gain=b['cartesian_gain'],
                                    saved_actions=demo['actions'], seed=seed)
        snapshots.append(snapshot(exp.env))
        for row, source in zip(measurements, exp.last_step_metrics):
            row.update(source_frame=float(source['source_frame']), step=source['step'])
        return dict(report=report, snapshots=snapshots, rows=measurements, initial=initial[0],
            post_states=np.asarray(states), demo=exp.last_demo, dt=exp.env.control_timestep,
            terminal_observation=exp.env._get_observations().copy())
    finally:
        exp.env.close()


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def save_pickle(path, value):
    with Path(path).open('xb') as stream:
        pickle.dump(value, stream, protocol=4)
