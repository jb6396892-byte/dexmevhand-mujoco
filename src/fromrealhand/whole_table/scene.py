"""Physical XYZ gantry with a constrained local hand for the F1 milestone."""
import xml.etree.ElementTree as ET
import numpy as np
from ..tabletop.scene import ASSETS, fmt, mesh_local
from .layout import OBJECTS

PLATFORM_JOINTS = ('gantry_x', 'gantry_y', 'gantry_z')


def object_catalog():
    import mujoco_py
    from hand_imitation.env.models import TableArena
    from hand_imitation.env.models.objects import YCB_ORIENTATION
    arena = TableArena(table_full_size=(.85, .80, .05), table_offset=(0, 0, .75), bottom_pos=(0, 0, -.75))
    for index, name in enumerate(OBJECTS):
        arena.add_ycb_object(name, pos=[2.+index, 0., .3], quat=YCB_ORIENTATION[name],
                             free=True, scale=.8 if name == 'mug' else 1.)
    sim = mujoco_py.MjSim(mujoco_py.load_model_from_xml(arena.get_xml()))
    sim.forward()
    result = {}
    for name in OBJECTS:
        bid = sim.model.body_name2id(name+'_0')
        points = [mesh_local(sim.model, g)[0] for g in range(sim.model.ngeom)
                  if sim.model.geom_bodyid[g] == bid and sim.model.geom_type[g] == 7]
        result[name] = np.concatenate(points) @ sim.data.body_xmat[bid].reshape(3, 3).T
    return result


def box(parent, name, pos, size, color, mass=None):
    attributes = dict(name=name, type='box', pos=fmt(pos), size=fmt(size), rgba=fmt(color),
                      contype='1', conaffinity='1', friction='1 .5 .01')
    if mass is not None: attributes['mass'] = str(mass)
    return ET.SubElement(parent, 'geom', **attributes)


def make_xml(layout, config):
    import mujoco_py
    import transforms3d
    from hand_imitation.env.models import TableArena
    from hand_imitation.env.models.base import MujocoXML
    from hand_imitation.env.models.objects import YCB_ORIENTATION
    robot = MujocoXML(str(ASSETS/'adroit/adroit_relocate.xml'))
    root, world = robot.root, robot.worldbody
    root.set('model', 'Whole-table F1: physical gantry, parked local hand')
    root.find('option').set('timestep', str(config['timestep_s']))
    root.find('option').set('iterations', '80')
    root.find('option').set('solver', 'Newton')
    ET.SubElement(ET.SubElement(root, 'visual'), 'global', offwidth='1280', offheight='960')
    forearm = world.find("body[@name='forearm']")
    local_joints = [j.get('name') for j in forearm.iter('joint')]
    local_actuators = [a.get('name') for a in root.find('actuator')]
    # These relative constraints park only the existing hand, never the gantry or objects.
    for name in local_joints:
        ET.SubElement(robot.equality, 'joint', name='f1_park_'+name, joint1=name,
                      polycoef='0 0 0 0 0', solref='.004 1')
    world.remove(forearm)
    frame = ET.SubElement(world, 'body', name='gantry_frame')
    for x in (-.74, .74):
        for y in (-1.35, 1.35):
            box(frame, 'post_%s_%s' % (x, y), [x, y, .20], [.025, .025, .90], [.35, .38, .40, 1])
    for y in (-1.35, 1.35):
        box(frame, 'rail_'+str(y), [0, y, 1.12], [.79, .03, .025], [.40, .43, .45, 1])
    axis = ET.SubElement(world, 'body', name='gantry_x_body', pos='0 0 1.06')
    box(axis, 'gantry_bridge', [0, 0, 0], [.04, 1.4, .025], [.25, .55, .57, 1], 6)
    carriage = ET.SubElement(axis, 'body', name='gantry_y_body', pos='0 0 -.08')
    box(carriage, 'gantry_carriage', [0, 0, 0], [.07, .07, .07], [.65, .68, .70, 1], 2)
    tool = ET.SubElement(carriage, 'body', name='gantry_z_body', pos='0 0 -.88')
    box(tool, 'gantry_stem', [0, -.50, .38], [.02, .02, .36], [.58, .60, .62, 1], 1)
    box(tool, 'gantry_bracket', [0, -.25, .74], [.025, .26, .025], [.25, .55, .57, 1], .5)
    box(tool, 'gantry_forearm_mount', [0, -.365, .05], [.025, .145, .025], [.25, .55, .57, 1], .3)
    box(tool, 'gantry_z_slide', [0, 0, .70], [.02, .02, .35], [.58, .60, .62, 1], .4)
    forearm.set('pos', '0 -.5 0')
    tool.append(forearm)
    contact = root.find('contact')
    if contact is None: contact = ET.SubElement(root, 'contact')
    # Ideal slide interfaces use joint damping, not rubbing collision geometry.
    for first, second in (('gantry_frame', 'gantry_x_body'), ('gantry_x_body', 'gantry_z_body'),
                          ('gantry_y_body', 'gantry_z_body'), ('gantry_z_body', 'forearm')):
        ET.SubElement(contact, 'exclude', body1=first, body2=second)
    for i, body in enumerate((axis, carriage, tool)):
        name = PLATFORM_JOINTS[i]
        vector = np.eye(3)[i]
        ET.SubElement(body, 'joint', name=name, type='slide', axis=fmt(vector), limited='true',
                      range=fmt([config['joint_min_m'][i], config['joint_max_m'][i]]), damping='5', armature='.05')
        gain, damping = config['position_gain'], config['velocity_gain']
        ET.SubElement(root.find('actuator'), 'general', name=name+'_servo', joint=name,
            gainprm=str(gain), biasprm=fmt([0, -gain, -damping]), biastype='affine',
            ctrllimited='true', ctrlrange=fmt([config['joint_min_m'][i]-.1, config['joint_max_m'][i]+.1]),
            forcelimited='true', forcerange=fmt([-config['force_limit_n'], config['force_limit_n']]))
    arena = TableArena(table_full_size=tuple(config['table_size_xy_m'])+(.05,),
                       table_offset=(0, 0, .75), bottom_pos=(0, 0, -.75))
    for item in layout['objects']:
        name = item['name']
        quat = transforms3d.quaternions.qmult(transforms3d.quaternions.axangle2quat([0, 0, 1],
            np.deg2rad(item['yaw_deg'])), YCB_ORIENTATION[name])
        arena.add_ycb_object(name, pos=item['xy']+[.3], quat=quat, free=True,
                            scale=.8 if name == 'mug' else 1., margin='.0005', condim='4', friction='1 .5 .01')
    robot.merge(arena)
    ET.SubElement(world, 'light', pos='0 0 2', dir='0 0 -1', diffuse='.7 .7 .7')
    # Calibrate once from fixed geometry so S_grasp=(0,0,.1)+gantry_q, independent of layout.
    calibration = mujoco_py.MjSim(mujoco_py.load_model_from_xml(robot.get_xml()))
    calibration.forward()
    offset = np.array([0., 0., .1])-calibration.data.get_site_xpos('S_grasp')
    forearm.set('pos', fmt(np.fromstring(forearm.get('pos'), sep=' ')+offset))
    return robot.get_xml(), local_joints, local_actuators


class PlatformScene:
    def __init__(self, layout, config):
        import mujoco_py
        self.config, self.layout = config, layout
        self.xml, hand_names, actuator_names = make_xml(layout, config)
        self.sim = mujoco_py.MjSim(mujoco_py.load_model_from_xml(self.xml))
        m, d = self.sim.model, self.sim.data
        self.joint_ids = np.array([m.joint_name2id(n) for n in PLATFORM_JOINTS])
        self.qids, self.vids = m.jnt_qposadr[self.joint_ids], m.jnt_dofadr[self.joint_ids]
        self.aids = np.array([m.actuator_name2id(n+'_servo') for n in PLATFORM_JOINTS])
        self.hand_qids = np.array([m.jnt_qposadr[m.joint_name2id(n)] for n in hand_names])
        self.hand_aids = np.array([m.actuator_name2id(n) for n in actuator_names])
        if len(self.hand_qids) != 30 or len(self.hand_aids) != 30 or m.nu != 33:
            raise ValueError('Expected named 3+30 scalar controls')
        self.mapping = dict(platform_qpos=self.qids.tolist(), platform_actuators=self.aids.tolist(),
                            local_qpos=self.hand_qids.tolist(), local_actuators=self.hand_aids.tolist())
        d.qpos[self.qids] = config['home_m']
        self.target = np.array(config['home_m'], dtype=float)
        self.sim.forward()
        for item in layout['objects']:
            name = item['name']; bid = m.body_name2id(name+'_0')
            points = np.concatenate([mesh_local(m, g)[0] for g in range(m.ngeom)
                                     if m.geom_bodyid[g] == bid and m.geom_type[g] == 7])
            bottom = (points @ d.body_xmat[bid].reshape(3, 3).T)[:, 2].min()
            q = d.get_joint_qpos(name+'_joint_0').copy(); q[2] = .001-bottom
            d.set_joint_qpos(name+'_joint_0', q)
        self.sim.forward()
        self.steps = 0
        self.peak_penetration = 0.
        for _ in range(int(np.ceil(config['settling_s']/config['timestep_s']))): self.step(self.target)
        self.initial_site = d.get_site_xpos('S_grasp').copy()

    def step(self, target):
        target = np.asarray(target, dtype=float)
        if (target.shape != (3,) or not np.isfinite(target).all()
                or np.any(target < self.config['joint_min_m']) or np.any(target > self.config['joint_max_m'])):
            raise ValueError('Platform target outside joint limits')
        self.target = target.copy()
        d = self.sim.data
        # Gravity compensation is commanded through the force-limited motor, not external forces.
        control = target+d.qfrc_bias[self.vids]/self.config['position_gain']
        ranges = self.sim.model.actuator_ctrlrange[self.aids]
        if np.any(control < ranges[:, 0]) or np.any(control > ranges[:, 1]):
            raise ValueError('Motor control authority exceeded')
        d.ctrl[self.aids] = control
        self.sim.step(); self.steps += 1
        peak = max([max(0., -float(d.contact[i].dist)) for i in range(d.ncon)]+[0.])
        self.peak_penetration = max(self.peak_penetration, peak)
        if not np.isfinite(d.qpos).all() or not np.isfinite(d.qvel).all():
            raise RuntimeError('Nonfinite physics state')

    def position(self): return self.sim.data.qpos[self.qids].copy()

    def object_report(self):
        m, d = self.sim.model, self.sim.data
        half = np.array(self.config['table_size_xy_m'])/2
        rows = []
        for item in self.layout['objects']:
            name = item['name']; bid = m.body_name2id(name+'_0')
            points = np.concatenate([mesh_local(m, g)[0] for g in range(m.ngeom)
                                     if m.geom_bodyid[g] == bid and m.geom_type[g] == 7])
            world = points @ d.body_xmat[bid].reshape(3, 3).T+d.body_xpos[bid]
            lo, hi = world.min(axis=0), world.max(axis=0)
            velocity = d.get_joint_qvel(name+'_joint_0')
            linear_speed = float(np.linalg.norm(velocity[:3]))
            angular_speed = float(np.linalg.norm(velocity[3:]))
            rows.append(dict(name=name, world_position_m=d.body_xpos[bid].tolist(),
                min_m=lo.tolist(), max_m=hi.tolist(),
                linear_speed_m_s=linear_speed, angular_speed_rad_s=angular_speed,
                low_velocity=bool(linear_speed < .01 and angular_speed < .05),
                supported=bool(np.all(lo[:2] >= -half) and np.all(hi[:2] <= half)
                    and -.001 <= lo[2] <= .002)))
        return rows
