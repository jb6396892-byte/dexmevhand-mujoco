"""Actuated floating Adroit. No gantry, mocap weld, or execution pose writes."""
import xml.etree.ElementTree as ET
import numpy as np
from ..tabletop.scene import ASSETS, fmt, mesh_local

TRANSLATION_JOINTS = ('ARTx', 'ARTy', 'ARTz')


def make_hand_xml(layout, config, fixtures=()):
    import mujoco_py
    import transforms3d
    from hand_imitation.env.models import TableArena
    from hand_imitation.env.models.base import MujocoXML
    from hand_imitation.env.models.objects import YCB_ORIENTATION
    robot = MujocoXML(str(ASSETS/'adroit/adroit_relocate.xml'))
    root = robot.root
    root.set('model', 'Adroit reference-center navigation')
    option = root.find('option')
    option.set('timestep', str(config['timestep_s']))
    option.set('solver', 'Newton'); option.set('iterations', '80')
    ET.SubElement(ET.SubElement(root, 'visual'), 'global', offwidth='1280', offheight='960')
    forearm = robot.worldbody.find("body[@name='forearm']")
    palm = forearm.find(".//body[@name='palm']")
    ET.SubElement(palm, 'site', name='S_nav', pos=palm.find("geom[@name='C_palm0']").get('pos'),
                  type='sphere', size='.006', rgba='0 .85 .85 1', group='3')
    names = [j.get('name') for j in forearm.iter('joint')]
    for joint in forearm.iter('joint'):
        if joint.get('name') in TRANSLATION_JOINTS:
            joint.set('range', '-.85 .85'); joint.set('margin', '0')
    for motor in root.find('actuator'):
        name = motor.get('joint')
        if name in TRANSLATION_JOINTS:
            kp, kd, force = config['translation_gain'], config['translation_damping'], config['translation_force_n']
            limits = [-.95, .95]
        else:
            kp, kd, force = (150., 15., 50.) if name.startswith('ARR') else (30., 1., 8.)
            limits = np.fromstring(motor.get('ctrlrange'), sep=' ')+[-.10, .10]
        motor.set('gainprm', str(kp)); motor.set('biasprm', fmt([0, -kp, -kd]))
        motor.set('ctrlrange', fmt(limits))
        motor.set('forcelimited', 'true'); motor.set('forcerange', fmt([-force, force]))
    arena = TableArena(table_full_size=tuple(config['table_size_xy_m'])+(.05,),
                       table_offset=(0, 0, .75), bottom_pos=(0, 0, -.75))
    for item in layout['objects']:
        name = item['name']
        quat = transforms3d.quaternions.qmult(transforms3d.quaternions.axangle2quat([0,0,1],
                np.deg2rad(item['yaw_deg'])), YCB_ORIENTATION[name])
        arena.add_ycb_object(name, pos=item['xy']+[.3], quat=quat, free=True,
                            scale=.8 if name == 'mug' else 1., margin='.0005', condim='4', friction='1 .5 .01')
    robot.merge(arena)
    for fixture in fixtures:
        ET.SubElement(robot.worldbody, 'geom', name=fixture['name'], type='box',
            pos=fmt(fixture['pos']), size=fmt(fixture['size']), rgba='.35 .4 .45 1', contype='1', conaffinity='1')
    ET.SubElement(robot.worldbody, 'light', pos='0 0 2', dir='0 0 -1', diffuse='.7 .7 .7')
    ET.SubElement(robot.worldbody, 'site', name='navigation_goal', type='sphere',
                  pos=fmt(config['home_m']), size='.008', rgba='.1 .9 .2 .7', group='3')
    calibration = mujoco_py.MjSim(mujoco_py.load_model_from_xml(robot.get_xml()))
    calibration.forward()
    offset = np.asarray(config['reference_origin_m'])-calibration.data.get_site_xpos('S_nav')
    forearm.set('pos', fmt(np.fromstring(forearm.get('pos'), sep=' ')+offset))
    return robot.get_xml(), names


def geom_bounds(sim, gid):
    """World AABB, including visual meshes and collision primitives."""
    m, d = sim.model, sim.data
    kind = m.geom_type[gid]
    if kind == 7:
        local = mesh_local(m, gid)[0]
        bid = m.geom_bodyid[gid]
        points = local @ d.body_xmat[bid].reshape(3,3).T+d.body_xpos[bid]
        return np.array([points.min(0), points.max(0)])
    R, size = d.geom_xmat[gid].reshape(3,3), m.geom_size[gid]
    if kind == 6: ext = np.abs(R) @ size
    elif kind == 3: ext = size[0]+np.abs(R[:,2])*size[1]
    elif kind == 2: ext = np.full(3, size[0])
    elif kind == 5: ext = size[0]*np.sqrt(R[:,0]**2+R[:,1]**2)+size[1]*np.abs(R[:,2])
    elif kind == 4: ext = np.sqrt((R**2) @ (size**2))
    else: raise ValueError('Unsupported bounded geom type: '+str(kind))
    return np.array([d.geom_xpos[gid]-ext, d.geom_xpos[gid]+ext])


class HandScene:
    def __init__(self, layout, config, fixtures=()):
        import mujoco_py
        self.config, self.layout = config, layout
        self.xml, names = make_hand_xml(layout, config, fixtures)
        self.sim = mujoco_py.MjSim(mujoco_py.load_model_from_xml(self.xml))
        m, d = self.sim.model, self.sim.data
        joints = [m.joint_name2id(n) for n in names]
        self.qids = m.jnt_qposadr[joints]; self.vids = m.jnt_dofadr[joints]
        self.aids = np.array([m.actuator_name2id('A_'+n) for n in names])
        self.tcols = np.array([names.index(n) for n in TRANSLATION_JOINTS])
        self.posture_cols = np.array([i for i in range(30) if i not in self.tcols])
        if m.nu != 30 or any(n.startswith('gantry') for n in m.body_names):
            raise ValueError('Expected original 30-channel hand without a platform')
        self.sim.forward()
        self.origin = d.get_site_xpos('S_nav').copy()
        self.basis = d.get_site_jacp('S_nav').reshape(3,m.nv)[:,self.vids[self.tcols]].copy()
        if not np.allclose(self.basis.T @ self.basis, np.eye(3), atol=1e-8):
            raise ValueError('Invalid translation calibration')
        self.inverse_basis = np.linalg.inv(self.basis)
        d.qpos[self.qids[self.tcols]] = self.to_joints(config['home_m'])
        self.sim.forward()
        self.object_vertices = {}
        for item in layout['objects']:
            name = item['name']; bid = m.body_name2id(name+'_0')
            points = np.concatenate([mesh_local(m,g)[0] for g in range(m.ngeom)
                                     if m.geom_bodyid[g] == bid and m.geom_type[g] == 7])
            self.object_vertices[name] = points
            bottom = (points @ d.body_xmat[bid].reshape(3,3).T)[:,2].min()
            q = d.get_joint_qpos(name+'_joint_0').copy(); q[2] = .001-bottom
            d.set_joint_qpos(name+'_joint_0',q)
        self.sim.forward()
        self.hand_bodies = {m.body_name2id('forearm')}
        for bid in range(m.nbody):
            if m.body_parentid[bid] in self.hand_bodies: self.hand_bodies.add(bid)
        self.hand_geoms = [g for g in range(m.ngeom) if int(m.geom_bodyid[g]) in self.hand_bodies]
        self.fixture_gids = [m.geom_name2id(f['name']) for f in fixtures]+[m.geom_name2id('table_collision')]
        self.target = np.array(config['home_m'], dtype=float)
        self.steps = 0
        self.kp = m.actuator_gainprm[self.aids,0]
        for _ in range(int(round(config['settling_s']/config['timestep_s']))): self.step(self.target)
        if self.contacts()['hand_environment_contacts']:
            raise ValueError('Initial hand intersects environment')
        self.hand_envelope = self.hand_shapes()-self.position()
        self.calibration = dict(site='S_nav', local_palm_center_m=[-.008,0,.038],
            origin_m=self.origin.tolist(), translation_basis=self.basis.tolist(),
            translation_joint_names=list(TRANSLATION_JOINTS),
            translation_qpos=self.qids[self.tcols].tolist(),
            translation_actuators=self.aids[self.tcols].tolist(),
            hand_envelope_relative_m=self.hand_envelope.tolist(),
            calibrated_home_error_m=float(np.linalg.norm(self.position()-config['home_m'])),
            actuator_count=int(m.nu), platform_body_count=0)

    def to_joints(self, center): return self.inverse_basis @ (np.asarray(center)-self.origin)

    def position(self): return self.sim.data.get_site_xpos('S_nav').copy()

    def velocity(self): return self.basis @ self.sim.data.qvel[self.vids[self.tcols]]

    def acceleration(self): return self.basis @ self.sim.data.qacc[self.vids[self.tcols]]

    def step(self, center):
        center = np.asarray(center, dtype=float)
        if (center.shape != (3,) or not np.isfinite(center).all()
                or np.any(center < self.config['workspace_min_m']) or np.any(center > self.config['workspace_max_m'])):
            raise ValueError('Navigation center outside workspace')
        m,d = self.sim.model,self.sim.data
        desired = np.zeros(30); desired[self.tcols] = self.to_joints(center)
        control = desired+d.qfrc_bias[self.vids]/self.kp
        bounds = m.actuator_ctrlrange[self.aids]
        if np.any(control < bounds[:,0]) or np.any(control > bounds[:,1]):
            raise ValueError('Actuator command exceeds authority')
        self.target = center.copy(); d.ctrl[self.aids] = control
        self.sim.step(); self.steps += 1
        if not np.isfinite(d.qpos).all() or not np.isfinite(d.qvel).all():
            raise RuntimeError('Nonfinite physics state')

    def hand_bounds(self):
        boxes = np.array([geom_bounds(self.sim,g) for g in self.hand_geoms])
        return np.array([boxes[:,0].min(0),boxes[:,1].max(0)])

    def hand_shapes(self):
        m=self.sim.model;groups={}
        for g in self.hand_geoms:
            if self.config.get('collision_geometry_only'):
                if not (m.geom_contype[g] or m.geom_conaffinity[g]):continue
                groups[g]=[geom_bounds(self.sim,g)]
                continue
            body=m.body_id2name(int(m.geom_bodyid[g]))
            key=body if body in ('forearm','wrist','palm','mug_0') else body[:2]
            groups.setdefault(key,[]).append(geom_bounds(self.sim,g))
        return np.array([[np.array(v)[:,0].min(0),np.array(v)[:,1].max(0)] for v in groups.values()])

    def obstacles(self):
        m,d = self.sim.model,self.sim.data
        rows = [dict(name=m.geom_id2name(g),bounds=geom_bounds(self.sim,g).tolist()) for g in self.fixture_gids]
        for name,points in self.object_vertices.items():
            bid=m.body_name2id(name+'_0')
            world=points @ d.body_xmat[bid].reshape(3,3).T+d.body_xpos[bid]
            rows.append(dict(name=name,bounds=[world.min(0).tolist(),world.max(0).tolist()]))
        return rows

    def contacts(self):
        m,d=self.sim.model,self.sim.data
        count,peak=0,0.
        for c in d.contact[:d.ncon]:
            a=int(m.geom_bodyid[c.geom1]) in self.hand_bodies
            b=int(m.geom_bodyid[c.geom2]) in self.hand_bodies
            if a != b and c.dist <= 0:
                count+=1; peak=max(peak,-float(c.dist))
        posture=float(np.max(np.abs(d.qpos[self.qids[self.posture_cols]])))
        return dict(hand_environment_contacts=count, hand_penetration_m=peak, posture_error_rad=posture)
