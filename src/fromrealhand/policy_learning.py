"""State features and device transfer for the existing MJRL policy network."""
import numpy as np
import torch
import transforms3d

from .video_fidelity import source_clock


def policy_features(observation, qpos, qvel, step, dt, duration, time_scale, mode):
    observation = np.asarray(observation)
    if observation.shape != (39,) or not np.isfinite(observation).all():
        raise ValueError('Expected 39 finite native observations')
    if mode == 'original':
        return observation.copy()
    if np.shape(qpos) != (37,) or np.shape(qvel) != (36,):
        raise ValueError('Expected Adroit qpos=37 and qvel=36')
    if not np.isfinite(qpos).all() or not np.isfinite(qvel).all():
        raise ValueError('Nonfinite simulation state')
    if min(dt, duration, time_scale) <= 0 or not np.isfinite([dt, duration, time_scale]).all():
        raise ValueError('Control clock parameters must be finite and positive')
    if np.linalg.norm(np.asarray(qpos)[33:37]) < 1e-12:
        raise ValueError('Object quaternion must be nonzero')
    rotation = transforms3d.quaternions.quat2mat(np.asarray(qpos)[33:37])[:, :2].ravel()
    parts = [observation, np.asarray(qvel), rotation]
    if mode == 'phase':
        parts.append(np.atleast_1d(source_clock(step*dt, duration, time_scale)/duration))
    elif mode != 'state':
        raise ValueError('Unknown policy feature mode: '+mode)
    return np.concatenate(parts)


def model_to_device(model, device):
    model.to(device)
    # Legacy MuNet stores these tensors outside the registered buffers.
    for name in ('in_shift', 'in_scale', 'out_shift', 'out_scale'):
        setattr(model, name, getattr(model, name).to(device))


class StudentActions:
    def __init__(self, checkpoint, experiment, horizon):
        self.checkpoint = checkpoint
        self.experiment = experiment
        self.horizon = horizon
        if (not np.isclose(checkpoint['dt'], experiment.env.control_timestep)
                or not np.isclose(checkpoint['duration'], experiment.duration)):
            raise ValueError('Policy and environment control clocks differ')
        reference = checkpoint.get('action_reference')
        if reference is not None and (np.shape(reference) != (horizon, 30) or not np.isfinite(reference).all()):
            raise ValueError('Residual action reference must match the rollout horizon and action dimensions')

    def __len__(self):
        return self.horizon

    def features(self, step):
        e = self.experiment.env
        return policy_features(e._get_observations(), e.sim.data.qpos, e.sim.data.qvel,
                               step, e.control_timestep, self.experiment.duration,
                               self.checkpoint['time_scale'], self.checkpoint['feature_mode'])

    def __getitem__(self, step):
        features = self.features(step)
        with torch.no_grad():
            action = self.checkpoint['policy'].model(torch.as_tensor(features, dtype=torch.float32)[None]).numpy()[0]
        if self.checkpoint.get('action_reference') is not None:
            action = action+self.checkpoint['action_reference'][step]
        if action.shape != (30,) or not np.isfinite(action).all():
            raise ValueError('Nonfinite or invalid student action')
        return action.copy()


def lift_success(report):
    return bool(report['hold_s'] >= 1. and report['tail_min_bottom_m'] > .015
                and report['tail_min_fingers'] >= 2 and report['tail_min_force_n'] > .05)
