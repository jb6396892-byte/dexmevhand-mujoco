"""Candidate object-relative reference adaptation. No simulator object access."""
import numpy as np


def perception_interval(config, phase):
    period=float(config.get('perception_period_by_phase_s',{}).get(phase,config['perception_period_sim_s']))
    if not np.isfinite(period) or not 0<period<=config['max_pose_sim_age_s']:
        raise ValueError('Invalid phase perception interval')
    return period


def rigid(value):
    t = np.asarray(value, dtype=float)
    if t.shape != (4, 4) or not np.isfinite(t).all():
        raise ValueError('Expected finite 4x4 pose')
    r = t[:3, :3]
    if (not np.allclose(t[3], [0, 0, 0, 1], atol=1e-6)
            or not np.allclose(r.T @ r, np.eye(3), atol=1e-4)
            or abs(np.linalg.det(r)-1) > 1e-4):
        raise ValueError('Invalid rigid transform')
    return t.copy()


def pose_from_estimate(row, sim_time, config):
    if row.get('accepted') is not True or row.get('reason') != 'pose_accepted':
        raise ValueError('Perception rejected: '+str(row.get('reason')))
    age = float(sim_time)-float(row['camera_time_s'])
    if not np.isfinite(age) or not -1e-6 <= age <= config['max_pose_sim_age_s']:
        raise ValueError('Expired or future visual pose')
    t = rigid(row['T_world_object'])
    if np.any(t[:3, 3] < config['workspace_min_m']) or np.any(t[:3, 3] > config['workspace_max_m']):
        raise ValueError('Visual object outside workspace')
    quality = np.asarray([row.get('fitness', 0), row.get('rmse_m', float('inf'))], dtype=float)
    if not np.isfinite(quality).all() or quality[0] <= .85 or quality[1] >= .003:
        raise ValueError('Registration quality below threshold')
    return t


def check_pose_jump(previous, current, config):
    a, b = rigid(previous), rigid(current)
    distance = np.linalg.norm(a[:3, 3]-b[:3, 3])
    angle = np.rad2deg(np.arccos(np.clip((np.trace(a[:3, :3].T @ b[:3, :3])-1)/2, -1, 1)))
    if distance > config['max_pose_jump_m'] or angle > config['max_pose_jump_deg']:
        raise ValueError('Visual pose discontinuity; reacquisition required')


class VisualReference:
    """One initial visual anchor, nominal finger actions, bounded root feedback.

    Later visual poses monitor execution, never silently move the initial anchor.
    All action corrections are in normalized actuator units, not motor ctrl units.
    """
    def __init__(self, actions, pre_qpos, object_reference, estimate, base, model, config, dt):
        self.actions = np.asarray(actions, dtype=float)
        self.qpos = np.asarray(pre_qpos, dtype=float)
        if (self.actions.ndim != 2 or self.actions.shape[1] != 30
                or self.qpos.shape != self.actions.shape or not np.isfinite(self.actions).all()
                or not np.isfinite(self.qpos).all() or np.max(np.abs(self.actions)) > 1+1e-8):
            raise ValueError('Invalid reference actions or pre-action states')
        self.reference, self.estimate = rigid(object_reference), rigid(estimate)
        self.delta = self.estimate @ np.linalg.inv(self.reference)
        translation = np.linalg.norm(self.estimate[:3, 3]-self.reference[:3, 3])
        self.yaw = float(np.arctan2(self.delta[1, 0], self.delta[0, 0]))
        tilt = np.rad2deg(np.arccos(np.clip(self.delta[2, 2], -1, 1)))
        if (translation > config['max_translation_m'] or abs(np.rad2deg(self.yaw)) > config['max_yaw_deg']
                or tilt > config['max_anchor_tilt_deg']):
            raise ValueError('Visual anchor outside candidate adaptation envelope')
        # Small rigid-pose changes are retained, including the source mug's tilt.
        from scipy.spatial.transform import Rotation
        fraction=float(config.get('orientation_transfer_fraction',1.))
        if not np.isfinite(fraction) or not 0<=fraction<=1: raise ValueError('Invalid orientation transfer fraction')
        self.rotation_vector = Rotation.from_matrix(self.delta[:3, :3]).as_rotvec()*fraction
        self.delta[:3,:3]=Rotation.from_rotvec(self.rotation_vector).as_matrix()
        self.delta[:3,3]=self.estimate[:3,3]-self.delta[:3,:3] @ self.reference[:3,3]
        self.base, self.config, self.dt = rigid(base), config, float(dt)
        self.reference_base = rigid(model.get('reference_base', base))
        self.ranges = np.asarray(model['joint_range'], dtype=float)
        gain, bias, rng = (np.asarray(model[k], dtype=float) for k in ('gain', 'bias', 'action_range'))
        if np.any(gain[:6] <= 0) or np.any(rng[:6] <= 0):
            raise ValueError('Unsupported root actuator')
        self.factor = -bias[:6, 1]/(gain[:6]*rng[:6])

    def goal(self, original_goal):
        return self.delta[:3, :3] @ np.asarray(original_goal)+self.delta[:3, 3]

    def preflight(self, stop):
        targets = np.asarray([self.desired_qpos(i) for i in range(stop)])
        if np.any(targets[:,:6] < self.ranges[:6,0]) or np.any(targets[:,:6] > self.ranges[:6,1]):
            invalid=(targets[:,:6]<self.ranges[:6,0]) | (targets[:,:6]>self.ranges[:6,1])
            frame,axis=np.argwhere(invalid)[0]
            raise ValueError('Full transformed reference is outside joint workspace: frame=%d axis=%d value=%.6f range=%s'%
                (frame,axis,targets[frame,axis],self.ranges[axis].tolist()))
        # Zero feedback gives a feedforward-envelope check, not a dynamics guarantee.
        for i in range(stop): self.action(i, targets[i])
        return dict(steps=stop,root_min=targets[:,:6].min(0).tolist(),root_max=targets[:,:6].max(0).tolist(),
                    physics_success_not_implied=True)

    def desired_qpos(self, index):
        import transforms3d.euler as euler
        from scipy.spatial.transform import Rotation
        q = self.qpos[index].copy()
        fraction = np.clip(index*self.dt/self.config['entry_blend_s'], 0., 1.)
        blend = fraction**3*(10-15*fraction+6*fraction*fraction)
        rotation = Rotation.from_rotvec(self.rotation_vector*blend).as_matrix()
        br, bp = self.base[:3, :3], self.base[:3, 3]
        rr, rp = self.reference_base[:3,:3], self.reference_base[:3,3]
        point = rr @ q[:3]+rp
        desired = q.copy()
        anchor = self.reference[:3, 3]
        offset = (self.estimate[:3, 3]-anchor)*blend
        desired[:3] = br.T @ (rotation @ (point-anchor)+anchor+offset-bp)
        desired[3:6] = euler.mat2euler(br.T @ rotation @ rr @ euler.euler2mat(*q[3:6], axes='rxyz'), axes='rxyz')
        # Choose the equivalent Euler representation nearest to the recorded pose.
        desired[3:6] += 2*np.pi*np.round((q[3:6]-desired[3:6])/(2*np.pi))
        return desired

    def action(self, index, hand_qpos):
        q = self.qpos[index]
        desired = self.desired_qpos(index)
        br, bp = self.base[:3,:3], self.base[:3,3]
        rp = self.reference_base[:3,3]
        if np.any(desired[:6] < self.ranges[:6, 0]) or np.any(desired[:6] > self.ranges[:6, 1]):
            raise ValueError('Transformed wrist reference exceeds joint limits at %d: %s'%(index,desired[:6].tolist()))
        error = desired[:6]-np.asarray(hand_qpos)[:6]
        correction = self.factor*(desired[:6]-q[:6]+self.config['root_feedback_gain']*error)
        installation = np.zeros(6); installation[:3] = br.T @ (rp-bp)
        if np.max(np.abs(correction-self.factor*installation)) > self.config['max_action_correction']:
            raise ValueError('Excessive normalized action correction at %d: %.6f'%(index,float(np.max(np.abs(correction-self.factor*installation)))))
        action = self.actions[index].copy(); action[:6] += correction
        if not np.isfinite(action).all() or np.max(np.abs(action)) > 1+1e-8:
            raise ValueError('Adapted action saturated; stop rather than hide clipping')
        return action
