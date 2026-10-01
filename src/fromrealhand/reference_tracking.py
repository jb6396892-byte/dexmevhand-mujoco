"""Reference-conditioned, contact-aware residual control for the v12 study."""
import numpy as np
import torch
import transforms3d

from .multivideo import conditioned_features
from .policy_learning import lift_success
from .video_fidelity import source_clock

FINGER_PREFIXES = ('th', 'ff', 'mf', 'rf', 'lf')


def task_gate(report, limits):
    """Task-level acceptance; does not silently replace the strict fidelity gate."""
    fractions = report['tail_finger_contact_fraction']
    values = [report[k] for k in ('final_distance_m', 'max_hand_scene_penetration_m',
              'initial_hand_scene_penetration_m', 'max_loaded_gap_m', 'tail_slip_m',
              'max_joint_violation_rad', 'saturation')]
    return bool(report['finite'] and np.isfinite(values).all() and lift_success(report)
                and report['final_distance_m'] <= limits['goal_m']
                and report['max_hand_scene_penetration_m'] <= limits['scene_penetration_m']
                and report['max_penetration_m'] <= limits['scene_penetration_m']
                and report['initial_hand_scene_penetration_m'] <= limits['initial_penetration_m']
                and report['max_loaded_gap_m'] <= limits['loaded_gap_m']
                and report['tail_slip_m'] <= limits['slip_m']
                and report['max_joint_violation_rad'] <= limits['joint_violation_rad']
                and report['saturation'] < limits['saturation_fraction']
                and fractions['thumb'] >= limits['thumb_contact_fraction']
                and sum(v >= limits['finger_contact_fraction'] for v in fractions.values())
                >= limits['supporting_fingers'])


def smooth(x):
    x = np.clip(x, 0., 1.)
    return x**3 * (10 - 15*x + 6*x*x)


def contact_features(contacts):
    """Actual contact distances, not nearest-surface distances when separated."""
    forces = np.zeros(5)
    distances = np.full(5, .005)
    for contact in contacts:
        finger = contact['hand'][2:4]
        if finger not in FINGER_PREFIXES:
            continue
        i = FINGER_PREFIXES.index(finger)
        forces[i] += max(0., float(contact['normal_force_n']))
        distances[i] = min(distances[i], float(contact['distance_m']))
    result = np.r_[np.clip(forces / 10., 0., 5.), np.clip(distances / .005, -1., 1.)]
    if not np.isfinite(result).all():
        raise ValueError('Nonfinite contact features')
    return result


def reference_features(base, qpos, reference_q, cup_reference, goal, contacts):
    result = np.r_[base, np.clip(np.asarray(reference_q)-qpos[:30], -.5, .5),
                   np.clip(cup_reference-qpos[30:33], -.2, .2),
                   np.clip(goal-qpos[30:33], -.2, .2), contact_features(contacts)]
    if result.shape != (130,) or not np.isfinite(result).all():
        raise ValueError('Expected 130 finite reference/contact features')
    return result


class ReferenceContext:
    """Adapt a nominal reference from observed reset cup pose and task target.

    Only desired hand targets/actions are transformed. The live cup is never moved.
    Inputs are the nominal demonstration, known goal, and observed reset state;
    no optimized per-case expert or future realized object states are consulted.
    """
    def __init__(self, experiment, video, reference_demo, geometry, contact_reader):
        self.e, self.video, self.contact_reader = experiment, video, contact_reader
        self.demo, self.geometry = reference_demo, geometry
        self.actions = None

    def initialize(self):
        e, m, d = self.e.env, self.e.model, self.e.env.sim.data
        states = self.demo['sim_data']
        g = self.geometry
        origin = g['object_poses'][0, :3, 3]
        nominal_rotation = g['object_poses'][0, :3, :3]
        rotation = transforms3d.quaternions.quat2mat(d.qpos[33:37]) @ nominal_rotation.T
        offset = d.qpos[30:33].copy()-origin
        self.goal = m.body_pos[e.target_object_bid].copy()
        nominal_goal = g['object_poses'][-1, :3, 3]
        goal_offset = self.goal-(rotation @ (nominal_goal-origin)+origin+offset)
        body = m.body_name2id('forearm')
        base = transforms3d.quaternions.quat2mat(m.body_quat[body])
        conversion = -m.actuator_biasprm[:6, 1]/(m.actuator_gainprm[:6, 0]*e.act_rng[:6])
        self.actions = np.asarray(self.demo['actions']).copy()
        self.qref = np.array([s['qpos'][:30] for s in states])
        self.cupref = np.array([s['qpos'][30:33] for s in states])
        yaw = transforms3d.euler.mat2euler(rotation)[2]
        for step, state in enumerate(states):
            entry = smooth(step*e.control_timestep/2.)
            clock = source_clock(step*e.control_timestep, self.e.duration, self.video['control']['time_scale'])
            frame = float(g['source_frames'][0]+clock*g['fps'])
            carry = smooth((frame-40)/(g['source_frames'][-1]-40))
            r = transforms3d.euler.euler2mat(0., 0., yaw*entry)
            delta = entry*offset+carry*goal_offset
            q = state['qpos'][:6]
            pos = base @ q[:3]+m.body_pos[body]
            target = q.copy()
            target[:3] = base.T @ (r @ (pos-origin)+origin+delta-m.body_pos[body])
            target[3:] = transforms3d.euler.mat2euler(
                base.T @ r @ base @ transforms3d.euler.euler2mat(*q[3:], axes='rxyz'), axes='rxyz')
            self.actions[step, :6] += conversion*(target-q)
            self.qref[step, :6] = target
            self.cupref[step] = rotation @ (state['qpos'][30:33]-origin)+origin+offset+carry*goal_offset

    def features(self, step):
        if self.actions is None:
            self.initialize()
        e, d = self.e.env, self.e.env.sim.data
        base = conditioned_features(e._get_observations(), d.qpos, d.qvel, step,
                                    e.control_timestep, self.e.duration,
                                    self.video['control']['time_scale'], self.video['id'])
        return reference_features(base, d.qpos, self.qref[step], self.cupref[step],
                                  self.goal, self.contact_reader(e))


class ReferenceActions:
    def __init__(self, checkpoint, experiment, video, contact_reader):
        self.checkpoint, self.video = checkpoint, video
        expected = checkpoint['clocks'][video['name']]
        actual = [experiment.env.control_timestep, experiment.duration, video['control']['time_scale']]
        if not np.allclose(expected, actual, rtol=0., atol=1e-10):
            raise ValueError('Reference clock mismatch')
        self.context = ReferenceContext(experiment, video, checkpoint['reference_demos'][video['name']],
                                        checkpoint['nominal_geometry'][video['name']], contact_reader)
        self.features_seen, self.raw_actions = [], []

    def __len__(self):
        return self.video['horizon']

    def __getitem__(self, step):
        x = self.context.features(step)
        with torch.no_grad():
            residual = self.checkpoint['policy'].model(torch.as_tensor(x, dtype=torch.float32)[None]).numpy()[0]
        self.features_seen.append(x)
        self.raw_actions.append(residual.copy())
        return residual_action(self.context.actions[step], residual,
                               self.checkpoint['residual_limits'][self.video['name']])


def residual_action(reference, residual, limits):
    reference, residual, limits = [np.asarray(x) for x in (reference, residual, limits)]
    if any(x.shape != (30,) or not np.isfinite(x).all() for x in (reference, residual, limits)) or np.any(limits <= 0):
        raise ValueError('Expected finite actions and positive residual limits')
    # Do not apply actuator ranges here: env.step owns normalized-action scaling.
    return reference+np.clip(residual, -limits, limits)
