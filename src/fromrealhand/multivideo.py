"""Sequence-aware features and phase diagnostics for independently timed grasps."""
import numpy as np
import torch
from .policy_learning import policy_features
from .video_fidelity import source_clock


def conditioned_features(observation, qpos, qvel, step, dt, duration, time_scale, video_id):
    if video_id not in (0, 1):
        raise ValueError('Unknown grasp identity')
    identity = np.zeros(2); identity[video_id] = 1.
    return np.r_[policy_features(observation,qpos,qvel,step,dt,duration,time_scale,'phase'),identity]


def trajectory_arrays(demo, reference, video, geometry):
    actions = np.asarray(demo['actions'])
    if actions.shape != np.shape(reference) or actions.shape != (video['horizon'],30):
        raise ValueError('Each trajectory must use its own video reference and horizon')
    duration = float((geometry['source_frames'][-1]-geometry['source_frames'][0])/geometry['fps'])
    x = np.stack([conditioned_features(obs,state['qpos'],state['qvel'],i,.01,duration,
                    video['control']['time_scale'],video['id'])
                  for i,(obs,state) in enumerate(zip(demo['observations'],demo['sim_data']))])
    if len(x) != len(actions) or not np.isfinite(actions).all() or not np.isfinite(reference).all():
        raise ValueError('Invalid trajectory data')
    return x,actions,actions-reference


def phase_diagnostics(demo, reference, video, geometry, expert=None):
    frames = geometry['source_frames']
    duration = float((frames[-1]-frames[0])/geometry['fps'])
    clock = source_clock(np.arange(len(demo['actions']))*.01,duration,video['control']['time_scale'])
    source_frames = frames[0]+clock*geometry['fps']
    positions = np.array([s['qpos'][30:33] for s in demo['sim_data']])
    goal = geometry['object_poses'][-1,:3,3]
    result = {}
    if expert is not None:
        target = np.array([s['qpos'][30:33] for s in expert['sim_data']])
        error = np.linalg.norm(positions-target,axis=1)
        bad = np.flatnonzero(error > .01)
        result['divergence'] = dict(cup_threshold_m=.01,
                                    first_cup_error_step=int(bad[0]) if len(bad) else None,
                                    first_cup_error_time_s=float(bad[0]*.01) if len(bad) else None)
    for name,lo,hi in [('approach',0,30),('closure',30,40),('lift',40,55),('transport_hold',55,np.inf)]:
        ids = (source_frames>=lo)&(source_frames<hi)
        if not ids.any():
            continue
        values = dict(steps=int(ids.sum()),mean_goal_distance_m=float(np.mean(np.linalg.norm(positions[ids]-goal,axis=1))),
                      max_cup_center_height_m=float(positions[ids,2].max()),
                      action_rmse_vs_frozen_reference=float(np.sqrt(np.mean((demo['actions'][ids]-reference[ids])**2))))
        if expert is not None:
            values.update(action_rmse_vs_expert=float(np.sqrt(np.mean((demo['actions'][ids]-expert['actions'][ids])**2))),
                          cup_position_rmse_vs_expert_m=float(np.sqrt(np.mean(np.sum((positions[ids]-target[ids])**2,axis=1)))))
        result[name] = values
    return result


class MultiVideoActions:
    """A state-feedback student with an explicitly selected, independently timed skill."""
    def __init__(self, checkpoint, experiment, video):
        self.checkpoint, self.experiment, self.video = checkpoint, experiment, video
        self.reference = checkpoint['references'][video['name']]
        if np.shape(self.reference) != (video['horizon'], 30):
            raise ValueError('Wrong skill reference horizon')
        expected = checkpoint['clocks'][video['name']]
        actual = [experiment.env.control_timestep, experiment.duration, video['control']['time_scale']]
        if not np.allclose(expected, actual, rtol=0., atol=1e-10):
            raise ValueError('Wrong skill control clock')

    def __len__(self):
        return self.video['horizon']

    def __getitem__(self, step):
        e = self.experiment.env
        x = conditioned_features(e._get_observations(), e.sim.data.qpos, e.sim.data.qvel,
                                 step, e.control_timestep, self.experiment.duration,
                                 self.video['control']['time_scale'], self.video['id'])
        with torch.no_grad():
            action = self.checkpoint['policy'].model(torch.as_tensor(x, dtype=torch.float32)[None]).numpy()[0]
        if self.checkpoint['method'] == 'residual_bc':
            action = action + self.reference[step]
        if action.shape != (30,) or not np.isfinite(action).all():
            raise ValueError('Invalid student action')
        # YCBRelocate clips and scales normalized controls exactly once.
        return action.copy()
