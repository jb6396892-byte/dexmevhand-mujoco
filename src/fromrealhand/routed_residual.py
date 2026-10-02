"""Explicit skill/phase routing without changing observations or reference actions."""
import copy
import numpy as np
import torch
from .corrective_learning import aligned_phase_indices, tracking_delta
from .multivideo import MultiVideoActions, conditioned_features
from .policy_learning import model_to_device


def phase_routes(length, geometry, time_scale, half_width=10):
    if not isinstance(half_width, int) or half_width < 0:
        raise ValueError('Blend width must be a nonnegative integer')
    phase = aligned_phase_indices(length, geometry, time_scale)
    routes = np.eye(5)[phase]
    boundaries = np.flatnonzero(np.diff(phase)) + 1
    if half_width and len(boundaries) > 1 and np.min(np.diff(boundaries)) < 2*half_width:
        raise ValueError('Phase transitions overlap')
    for boundary in boundaries:
        if not half_width:
            continue
        ids = np.arange(max(0, boundary-half_width), min(length, boundary+half_width+1))
        t = np.clip((ids-boundary+half_width)/(2.*half_width), 0., 1.)
        blend = t*t*(3.-2.*t)
        routes[ids] = 0.
        routes[ids, phase[boundary-1]] = 1.-blend
        routes[ids, phase[boundary]] = blend
    return routes


class RoutedNetwork(torch.nn.Module):
    def __init__(self, model, count):
        super().__init__()
        self.heads = torch.nn.ModuleList([copy.deepcopy(model) for _ in range(count)])

    def forward(self, features, routes):
        predictions = torch.stack([head(features) for head in self.heads], dim=1)
        return (predictions*routes[:, :, None]).sum(dim=1)

    def to_runtime(self, device):
        for head in self.heads:
            model_to_device(head, device)
        return self


class RoutedActions(MultiVideoActions):
    def __getitem__(self, step):
        e = self.experiment.env
        x = conditioned_features(e._get_observations(), e.sim.data.qpos, e.sim.data.qvel,
                                 step, e.control_timestep, self.experiment.duration,
                                 self.video['control']['time_scale'], self.video['id'])
        route = self.checkpoint['routes'][self.video['name']][step]
        with torch.no_grad():
            residual = self.checkpoint['network'](torch.as_tensor(x, dtype=torch.float32)[None],
                                                   torch.as_tensor(route, dtype=torch.float32)[None]).numpy()[0]
        action = self.reference[step] + np.clip(residual, -self.limits, self.limits)
        if action.shape != (30,) or not np.isfinite(action).all():
            raise ValueError('Invalid routed action')
        return action.copy()


def student_actions(checkpoint, experiment, video):
    cls = RoutedActions if 'routing_mode' in checkpoint else MultiVideoActions
    return cls(checkpoint, experiment, video)


def takeover_weight(step, start, transition):
    if transition <= 0:
        raise ValueError('A positive takeover transition is required')
    t = np.clip((step-start)/float(transition), 0., 1.)
    return float(t*t*(3.-2.*t))


class TakeoverActions:
    """Student prefix followed by an expert; labels are actually executed controls."""
    def __init__(self, checkpoint, experiment, video, expert, start, transition, gain, label_end):
        self.student = student_actions(checkpoint, experiment, video)
        self.e, self.video, self.expert = experiment, video, expert
        self.start, self.transition, self.gain, self.label_end = start, transition, gain, label_end
        m = experiment.model
        self.conversion = -m.actuator_biasprm[:, 1]/(m.actuator_gainprm[:, 0]*experiment.env.act_rng)
        self.features, self.labels, self.steps, self.feedback = [], [], [], []

    def __len__(self):
        return self.video['horizon']

    def __getitem__(self, step):
        beta = takeover_weight(step, self.start, self.transition)
        if beta == 0.:
            return self.student[step]
        e, d = self.e.env, self.e.env.sim.data
        state = self.expert['sim_data'][step]
        delta = tracking_delta(d.qpos[:30], state['qpos'][:30], d.qvel[:30], state['qvel'][:30],
                               self.conversion, self.gain)
        label = np.clip(self.expert['actions'][step]+delta-self.student.reference[step],
                        -self.student.limits, self.student.limits)
        teacher = self.student.reference[step] + label
        if beta == 1. and step < self.label_end:
            self.features.append(conditioned_features(e._get_observations(), d.qpos, d.qvel, step,
                e.control_timestep, self.e.duration, self.video['control']['time_scale'], self.video['id']))
            self.labels.append(label.copy())
            self.steps.append(step)
            self.feedback.append(delta.copy())
        # Blend controls only; after takeover, labels exactly match executed actions.
        return teacher if beta == 1. else beta*teacher+(1.-beta)*self.student[step]
