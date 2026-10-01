"""Frozen protocol, synthetic scene construction, and native-physics checks."""
import hashlib
import json
import pickle
from importlib import import_module
from pathlib import Path
import numpy as np
import transforms3d

surface = import_module('33_optimize_surface_grasp')
dynamic = import_module('64_optimize_dynamic_contact')
ROOT = surface.ROOT
from fromrealhand.video_fidelity import source_clock


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def protocol(path=ROOT/'configs/v10-study.json'):
    p = json.loads(Path(path).read_text())
    for video in p['videos']:
        for key,name in video['paths'].items():
            if digest(ROOT/name) != video['sha256'][key]:
                raise ValueError('Frozen input changed: '+name)
    return p,digest(path)


def reference_demo(video):
    return pickle.loads((ROOT/video['paths']['rollout']).read_bytes())['video_faithful']


def smooth(x):
    x = np.clip(x,0.,1.)
    return x**3*(10-15*x+6*x*x)


def scene_geometry(video,case):
    g = {k:v.copy() for k,v in np.load(ROOT/video['paths']['geometry']).items()}
    origin = g['object_poses'][0,:3,3].copy()
    rotation = transforms3d.euler.euler2mat(0,0,np.deg2rad(case['cup_yaw_deg']))
    phase = smooth((g['source_frames']-40)/(g['source_frames'][-1]-40))
    delta = np.asarray(case['cup_offset_m'])[None]+phase[:,None]*np.asarray(case['goal_offset_m'])[None]
    g['human_joints'] = (g['human_joints']-origin) @ rotation.T+origin+delta[:,None]
    g['object_poses'][:,:3,3] = (g['object_poses'][:,:3,3]-origin) @ rotation.T+origin+delta
    g['object_poses'][:,:3,:3] = rotation[None] @ g['object_poses'][:,:3,:3]
    # Deliberately retain the hand initial state: these are not whole-scene translations.
    return g


def setup_case(video,case,folder):
    folder.mkdir(parents=True,exist_ok=True)
    path = folder/'geometry.npz'
    g = scene_geometry(video,case)
    if not path.exists():
        np.savez(path,**g)
    else:
        old = np.load(path)
        if any(not np.array_equal(old[k],v) for k,v in g.items()):
            raise ValueError('Existing case geometry differs from frozen protocol')
    return dict(geometry=str(path.resolve()),best=video['control']),g


def run_case(video,source,actions,folder=None,half=False,seed=0,checkpoint=None,action_factory=None):
    exp = surface.SurfaceExperiment(source['geometry'])
    exp.env.pack_mujoco_model(reference_demo(video)['model_data'][0])
    if half:
        exp.model.opt.timestep /= 2
        exp.env.model_timestep = exp.model.opt.timestep
    b = video['control']
    try:
        if checkpoint is not None and action_factory is not None:
            raise ValueError('Choose a checkpoint or an action factory, not both')
        if action_factory is not None:
            actions = action_factory(exp)
        elif checkpoint is not None:
            from fromrealhand.multivideo import MultiVideoActions
            actions = MultiVideoActions(checkpoint,exp,video)
        report,_ = exp.run_surface(scale=b['time_scale'],close=b['close'],gain=b['cartesian_gain'],saved_actions=actions,output=folder,seed=seed)
        report['phase_contacts'] = {}
        for name,lo,hi in [('approach',0,30),('closure',30,40),('lift',40,55),('transport_hold',55,np.inf)]:
            rows = [r for r in exp.last_step_metrics if lo <= r['source_frame'] < hi]
            if rows:
                report['phase_contacts'][name] = dict(
                    loaded_fraction={finger:float(np.mean([r[finger+'_force_n']>.01 for r in rows])) for finger in surface.video.FINGERS},
                    max_penetration_m=max(r['penetration_m'] for r in rows),
                    max_bottom_m=max(r['bottom_m'] for r in rows),
                    final_goal_distance_m=rows[-1]['target_distance_m'])
        return report,exp.last_demo
    finally:
        exp.env.close()


def failure_labels(report):
    from fromrealhand.policy_learning import lift_success
    labels = []
    if not lift_success(report): labels.append('no_stable_lift')
    if report['final_distance_m'] > .02: labels.append('goal_miss')
    if report['max_hand_scene_penetration_m'] > .001: labels.append('scene_penetration')
    if report['initial_hand_scene_penetration_m'] > .0005: labels.append('initial_penetration')
    if report['max_loaded_gap_m'] > .0005: labels.append('loaded_gap')
    if report['tail_slip_m'] > .005: labels.append('slip')
    if report['saturation'] >= .01: labels.append('saturation')
    fractions = report['tail_finger_contact_fraction']
    if fractions['thumb'] < .8 or fractions['index'] < .8 or sum(x >= .8 for x in fractions.values()) < 4:
        labels.append('contact_pattern')
    if report['tail_mean_tip_error_m'] >= .015 or report['mean_tip_error_m'] >= .02 or max(report['tail_finger_tip_error_m'].values()) >= .025:
        labels.append('tip_error')
    if not report['finite']: labels.append('nonfinite')
    if report['max_joint_violation_rad'] > .02: labels.append('joint_limit')
    if not full_gate(report) and not labels: labels.append('other_physics')
    return labels


def full_gate(report):
    return bool(report['surface_physics_passed'] and report['fidelity_passed'] and report['final_distance_m'] <= .02)


def seeded_actions(video,case,source):
    demo = reference_demo(video)
    actions = demo['actions'].copy()
    exp = surface.SurfaceExperiment(source['geometry'])
    e,m = exp.env,exp.model
    e.pack_mujoco_model(demo['model_data'][0])
    body = m.body_name2id('forearm')
    base = transforms3d.quaternions.quat2mat(m.body_quat[body])
    original = np.load(ROOT/video['paths']['geometry'])
    origin = original['object_poses'][0,:3,3]
    duration = float((original['source_frames'][-1]-original['source_frames'][0])/original['fps'])
    factor = -m.actuator_biasprm[:6,1]/(m.actuator_gainprm[:6,0]*e.act_rng[:6])
    try:
        for step,state in enumerate(demo['sim_data']):
            entry = smooth(step*.01/2.)
            clock = float(source_clock(step*.01,duration,video['control']['time_scale']))
            frame = original['source_frames'][0]+clock*original['fps']
            carry = smooth((frame-40)/(original['source_frames'][-1]-40))
            rotation = transforms3d.euler.euler2mat(0,0,np.deg2rad(case['cup_yaw_deg'])*entry)
            delta = entry*np.asarray(case['cup_offset_m'])+carry*np.asarray(case['goal_offset_m'])
            q = state['qpos'][:6]
            pos = base @ q[:3]+m.body_pos[body]
            new_q = q.copy()
            new_q[:3] = base.T @ (rotation @ (pos-origin)+origin+delta-m.body_pos[body])
            new_q[3:] = transforms3d.euler.mat2euler(base.T @ rotation @ base @ transforms3d.euler.euler2mat(*q[3:],axes='rxyz'),axes='rxyz')
            actions[step,:6] += factor*(new_q-q)
        return np.clip(actions,-1.,1.)
    finally:
        e.close()


def shooting_demo(video,source,actions):
    demo = reference_demo(video)
    g = np.load(source['geometry'])
    exp = surface.SurfaceExperiment(source['geometry'])
    try:
        demo['sim_data'][0]['qpos'][30:33] = g['object_poses'][0,:3,3]
        demo['sim_data'][0]['qpos'][33:37] = transforms3d.quaternions.mat2quat(g['object_poses'][0,:3,:3])
        bid = exp.env.target_object_bid
        demo['model_data'][0]['body_pos'][bid] = g['object_poses'][-1,:3,3]
        demo['model_data'][0]['body_quat'][bid] = transforms3d.quaternions.mat2quat(g['object_poses'][-1,:3,:3])
        demo['actions'] = actions.copy()
        return demo
    finally:
        exp.env.close()
