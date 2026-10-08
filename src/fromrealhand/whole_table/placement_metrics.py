"""Read-only placement diagnostics; contact forces are expressed in world axes."""
import numpy as np
from .hand_scene import geom_bounds
from ..tabletop.scene import mesh_local


def world_contact_force(frame, force, body_is_second):
    return np.asarray(frame).reshape(3, 3).T.dot(force[:3]) * (1 if body_is_second else -1)


class PlacementMetrics:
    def __init__(self, env):
        import transforms3d
        from hand_imitation.env.models.objects import YCB_ORIENTATION
        self.env, self.sim = env, env.sim
        m = self.sim.model
        self.cup = m.body_name2id('mug_0')
        self.table = m.geom_name2id('table_collision')
        self.vertices = np.concatenate([mesh_local(m, g)[0] for g in range(m.ngeom)
            if m.geom_bodyid[g] == self.cup and m.geom_type[g] == 7
            and (m.geom_contype[g] or m.geom_conaffinity[g])])
        nominal_rotation=transforms3d.quaternions.quat2mat(YCB_ORIENTATION['mug'])
        self.up_local = nominal_rotation.T.dot([0., 0., 1.])
        self.table_bounds = geom_bounds(self.sim, self.table)
        self.table_z = float(self.table_bounds[1, 2])
        self.rest_origin_z = float(self.table_z-(self.vertices.dot(nominal_rotation.T))[:,2].min())
        self.weight = float(m.body_mass[self.cup] * -m.opt.gravity[2])
        self.hand = {m.body_name2id('forearm')}
        for bid in range(m.nbody):
            if m.body_parentid[bid] in self.hand: self.hand.add(bid)

    def read(self):
        import mujoco_py
        m, d = self.sim.model, self.sim.data
        R = d.body_xmat[self.cup].reshape(3, 3)
        pos = d.body_xpos[self.cup].copy()
        points = self.vertices.dot(R.T) + pos
        velocity = d.get_joint_qvel('mug_joint_0')
        table_force = np.zeros(3); hand_force = 0.; cup_contacts = 0; hand_contacts = 0
        illegal = []; peak = 0.
        for i in range(d.ncon):
            c = d.contact[i]; a, b = int(c.geom1), int(c.geom2)
            ba, bb = int(m.geom_bodyid[a]), int(m.geom_bodyid[b])
            cup = ba == self.cup or bb == self.cup
            hand = ba in self.hand or bb in self.hand
            if not (cup or hand): continue
            peak = max(peak, -float(c.dist))
            wrench = np.zeros(6); mujoco_py.functions.mj_contactForce(m, d, i, wrench)
            if cup and self.table in (a, b):
                table_force += world_contact_force(c.frame, wrench, bb == self.cup)
                cup_contacts += int(wrench[0] > .001)
            elif cup and hand:
                hand_force += max(0., float(wrench[0])); hand_contacts += int(c.dist <= .0005)
            elif ba in self.hand and bb in self.hand:
                pass
            elif c.dist < 0:
                illegal.append([m.geom_id2name(a), m.geom_id2name(b), float(c.dist)])
        return dict(cup_position_m=pos.tolist(), bottom_gap_m=float(points[:, 2].min()-self.table_z),
            rest_origin_z_m=self.rest_origin_z,
            tilt_deg=float(np.rad2deg(np.arccos(np.clip((R.dot(self.up_local))[2], -1, 1)))),
            linear_speed_m_s=float(np.linalg.norm(velocity[:3])), angular_speed_rad_s=float(np.linalg.norm(velocity[3:])),
            table_force_n=float(table_force[2]), table_weight_ratio=float(table_force[2]/self.weight),
            table_contacts=cup_contacts, hand_cup_contacts=hand_contacts, hand_cup_force_n=hand_force,
            penetration_m=max(0., peak), illegal_contacts=illegal,
            cup_bounds_m=[points.min(0).tolist(), points.max(0).tolist()])


def stable_placement(rows, goal_xy):
    if not rows: return dict(passed=False, reason='empty_stability_window')
    positions = np.array([r['cup_position_m'][:2] for r in rows])
    checks = dict(
        goal=bool(np.linalg.norm(positions[-1]-goal_xy) <= .02),
        upright=all(r['tilt_deg'] <= 10 for r in rows),
        grounded=all(abs(r['bottom_gap_m']) <= .002 for r in rows),
        slow=all(r['linear_speed_m_s'] <= .01 and r['angular_speed_rad_s'] <= .1 for r in rows),
        drift=bool(np.max(np.linalg.norm(positions-positions[0], axis=1)) <= .005),
        support=bool(np.mean([r['table_contacts'] > 0 for r in rows]) >= .95
                     and .8 <= np.mean([r['table_weight_ratio'] for r in rows]) <= 1.2),
        released=all(r['hand_cup_contacts'] == 0 for r in rows),
        safe=all(not r['illegal_contacts'] and r['penetration_m'] <= .001 for r in rows))
    return dict(passed=all(checks.values()), checks=checks, final=rows[-1],
                xy_error_m=float(np.linalg.norm(positions[-1]-goal_xy)), samples=len(rows))
