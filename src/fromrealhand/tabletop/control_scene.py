"""Unvalidated control scene, separate from the frozen perception benchmark."""
import xml.etree.ElementTree as ET
import numpy as np
from .scene import build


def create(reference_env, initial_snapshot, seed, dt, installation_slide_offset=None, initial_hand_world_shift=None, layout=None):
    import mujoco_py
    import transforms3d
    # Build/settle only at episode initialization. No object reset after vision.
    parked, xml, mesh, report = build(seed, layout=layout)
    root = ET.fromstring(xml)
    equality = root.find('equality')
    for node in list(equality):
        if node.tag == 'joint': equality.remove(node)
    ET.SubElement(root.find('worldbody'), 'site', name='visual_goal', type='sphere',
                  pos='0 0 .2', size='.006', rgba='0 .8 .35 .7', group='5')
    position=np.array([0., .65, .50]); look=np.array([0.,0.,.08])
    z=position-look; z/=np.linalg.norm(z)
    x=np.cross([0.,0.,1.],z); x/=np.linalg.norm(x); y=np.cross(z,x)
    ET.SubElement(root.find('worldbody'),'camera',name='rgbd_side',mode='fixed',
        pos=' '.join(map(str,position)),xyaxes=' '.join(map(str,np.r_[x,y])),fovy='47')
    model = mujoco_py.load_model_from_xml(ET.tostring(root, encoding='unicode'))
    sim = mujoco_py.MjSim(model)
    old = reference_env.sim.model
    # Copy named physical parameters, never assume indices survive scene changes.
    groups = {
        'body': ['body_pos', 'body_quat', 'body_mass', 'body_inertia', 'body_ipos', 'body_iquat'],
        'geom': ['geom_pos', 'geom_quat', 'geom_friction', 'geom_margin', 'geom_gap', 'geom_solref', 'geom_solimp', 'geom_contype', 'geom_conaffinity'],
        'joint': ['jnt_range', 'jnt_stiffness'],
        'actuator': ['actuator_gainprm', 'actuator_biasprm', 'actuator_ctrlrange', 'actuator_forcerange', 'actuator_gear']}
    for kind, fields in groups.items():
        for name in getattr(old, kind+'_names'):
            if not name or name not in getattr(model, kind+'_names'): continue
            # Keep the new table's dimensions and position, and new camera layout.
            if kind == 'body' and (name == 'world' or name.startswith('table') or name == 'floor'): continue
            if kind == 'geom' and (name.startswith('table') or name.startswith('floor')): continue
            a = getattr(old, kind+'_name2id')(name); b = getattr(model, kind+'_name2id')(name)
            for field in fields: getattr(model, field)[b] = getattr(old, field)[a]
    if model.joint_names[:30] != old.joint_names[:30] or model.actuator_names != old.actuator_names:
        raise ValueError('Hand joint/actuator order mismatch')
    if not np.array_equal(model.actuator_trnid[:, 0], np.arange(30)):
        raise ValueError('Expected one actuator per scalar hand joint')
    model.dof_damping[:30] = old.dof_damping[:30]
    model.dof_armature[:30] = old.dof_armature[:30]
    model.opt.timestep = old.opt.timestep
    # Fixed table installation, independent of seed/estimated object pose.
    # Recenter slide coordinates without moving the initial hand in world space.
    slide_offset = np.array([-.01, .095, 0.] if installation_slide_offset is None else installation_slide_offset)
    if slide_offset.shape != (3,) or not np.isfinite(slide_offset).all():
        raise ValueError('Invalid fixed installation offset')
    forearm = model.body_name2id('forearm')
    reference_base = np.eye(4)
    reference_base[:3,:3] = transforms3d.quaternions.quat2mat(model.body_quat[forearm])
    reference_base[:3,3] = model.body_pos[forearm].copy()
    model.body_pos[forearm] -= reference_base[:3,:3] @ slide_offset
    # Recompute model constants after inertial parameters were transferred.
    mujoco_py.functions.mj_setConst(model, sim.data)
    for name in report['objects']:
        joint = name+'_joint_0'
        sim.data.set_joint_qpos(joint, parked.data.get_joint_qpos(joint).copy())
        sim.data.set_joint_qvel(joint, np.zeros(6))
    state = initial_snapshot['state']
    sim.data.qpos[:30] = state.qpos[:30]
    sim.data.qpos[:3] += slide_offset
    if initial_hand_world_shift is not None:
        shift=np.asarray(initial_hand_world_shift,dtype=float)
        if shift.shape!=(3,) or not np.isfinite(shift).all(): raise ValueError('Invalid initial hand shift')
        sim.data.qpos[:3] += reference_base[:3,:3].T @ shift
    sim.data.qvel[:30] = state.qvel[:30]
    sim.data.ctrl[:] = initial_snapshot['arrays']['ctrl']
    sim.forward()
    report.update(control_enabled=True, candidate_not_validated=True, hand_mode='actuated; parking equalities removed',
                  initial_hand_source='frozen reference, independent of current object truth',
                  installation_slide_offset_m=slide_offset.tolist(),reference_base=reference_base.tolist())
    env=ControlEnv(sim, dt); env.reference_base=reference_base
    return env, mesh, report


class ControlEnv:
    def __init__(self, sim, dt):
        self.sim, self.dt = sim, float(dt)
        self.mid = np.mean(sim.model.actuator_ctrlrange, axis=1)
        self.rng = np.diff(sim.model.actuator_ctrlrange, axis=1).ravel()/2
        self.substeps = int(round(dt/sim.model.opt.timestep))
        if self.substeps < 1 or abs(self.substeps*sim.model.opt.timestep-dt) > 1e-9:
            raise ValueError('Non-integral physics/control timestep ratio')

    def step(self, action, audit):
        a = np.asarray(action, dtype=float)
        if a.shape != (30,) or not np.isfinite(a).all() or np.max(np.abs(a)) > 1+1e-8:
            raise ValueError('Invalid normalized control')
        self.sim.data.ctrl[:] = self.mid+self.rng*a
        for _ in range(self.substeps):
            self.sim.step()
            audit()

    def contacts(self):
        import mujoco_py
        m, d = self.sim.model, self.sim.data
        forces = dict.fromkeys(('th', 'ff', 'mf', 'rf', 'lf'), 0.)
        penetration, non_target, table_contacts = 0., 0, 0
        for i in range(d.ncon):
            c = d.contact[i]
            names = [m.geom_id2name(int(g)) or '' for g in (c.geom1, c.geom2)]
            hand = [n for n in names if n.startswith('C_')]
            if not hand: continue
            penetration = max(penetration, max(0., -float(c.dist)))
            mug = any(m.body_id2name(int(m.geom_bodyid[g])) == 'mug_0' for g in (c.geom1, c.geom2))
            if mug:
                wrench = np.zeros(6); mujoco_py.functions.mj_contactForce(m, d, i, wrench)
                key = hand[0][2:4].lower()
                if key in forces: forces[key] += max(0., float(wrench[0]))
            elif len(hand) == 1 and c.dist <= 0:
                if 'table_collision' in names: table_contacts += 1
                else: non_target += 1
        violation = np.maximum(m.jnt_range[:30, 0]-d.qpos[:30], d.qpos[:30]-m.jnt_range[:30, 1])
        return dict(scene_penetration_m=penetration, joint_violation_rad=max(0., float(violation.max())),
            finite=bool(np.isfinite(d.qpos).all() and np.isfinite(d.qvel).all()),
            max_hand_speed=float(np.max(np.abs(d.qvel[:30]))), non_target_contacts=non_target,
            table_contacts=table_contacts,
            **{k+'_force_n': float(v) for k, v in forces.items()})
