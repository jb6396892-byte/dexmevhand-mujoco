"""Reuse DexMV's robosuite-derived arena and exact YCB visual/collision assets."""
import hashlib
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np

SIM_ROOT=Path('/home/smgbro/dexmv-sim')
ASSETS=SIM_ROOT/'hand_imitation/env/models/assets'


def fmt(values): return ' '.join(str(float(x)) for x in values)


def mesh_local(model,gid):
    import transforms3d
    mid=int(model.geom_dataid[gid]); va=model.mesh_vertadr[mid]; vn=model.mesh_vertnum[mid]
    fa=model.mesh_faceadr[mid]; fn=model.mesh_facenum[mid]
    vertices=model.mesh_vert[va:va+vn].copy()
    vertices=vertices @ transforms3d.quaternions.quat2mat(model.geom_quat[gid]).T+model.geom_pos[gid]
    return vertices,model.mesh_face[fa:fa+fn].copy()


def build(seed=0, target=True):
    import mujoco_py
    import transforms3d
    from hand_imitation.env.models import TableArena
    from hand_imitation.env.models.base import MujocoXML
    from hand_imitation.env.models.objects import YCB_ORIENTATION
    rng=np.random.RandomState(seed)
    arena=TableArena(table_full_size=(.85,.80,.05),table_offset=(0,0,.75),bottom_pos=(0,0,-.75))
    robot=MujocoXML(str(ASSETS/'adroit/adroit_relocate.xml'))
    robot.root.set('model','RGBD tabletop perception sandbox')
    robot.root.find('option').set('timestep','.002')
    visual=ET.SubElement(robot.root,'visual')
    ET.SubElement(visual,'global',offwidth='960',offheight='720')
    # MSAA resolves depth from a subpixel sample, biasing metric backprojection.
    ET.SubElement(visual,'quality',offsamples='0')
    robot.worldbody.find("body[@name='forearm']").set('pos','-.20 -.75 .25')
    # Only this sensing scene parks the hand. No learned controller is claimed.
    for joint in robot.worldbody.iter('joint'):
        ET.SubElement(robot.equality,'joint',joint1=joint.get('name'),polycoef='0 0 0 0 0',solref='.004 1')
    positions=[('banana',[-.23,.21]),('sugar_box',[.20,.22]),
               ('mustard_bottle',[.24,-.07]),('tomato_soup_can',[-.24,-.02])]
    if target: positions.insert(0,('mug',rng.uniform(-.02,.02,2)))
    for name,xy in positions:
        scale=.8 if name=='mug' else 1.
        yaw=rng.uniform(-15,15) if name=='mug' else rng.uniform(-10,10)
        quat=transforms3d.quaternions.qmult(transforms3d.quaternions.axangle2quat([0,0,1],np.deg2rad(yaw)),YCB_ORIENTATION[name])
        arena.add_ycb_object(name,pos=[float(xy[0]),float(xy[1]),.3],quat=quat,
            scale=scale,free=True,margin='.0005',condim='4',friction='1 .5 .01')
    position=np.array([.60,0.,.58]); look=np.array([0.,.04,.025])
    z=(position-look); z/=np.linalg.norm(z)
    x=np.cross([0,0,1],z); x/=np.linalg.norm(x); y=np.cross(z,x)
    ET.SubElement(arena.worldbody,'camera',name='rgbd',mode='fixed',pos=fmt(position),
        xyaxes=fmt(np.r_[x,y]),fovy='47')
    robot.merge(arena)
    xml=robot.get_xml(); model=mujoco_py.load_model_from_xml(xml); sim=mujoco_py.MjSim(model)
    sim.forward()
    exported=None
    for name,xy in positions:
        bid=model.body_name2id(name+'_0')
        gids=np.where(model.geom_bodyid==bid)[0]
        minimum=float('inf')
        for gid in gids:
            if model.geom_type[gid]!=7: continue
            vertices,_=mesh_local(model,gid)
            rotated=vertices @ sim.data.body_xmat[bid].reshape(3,3).T
            minimum=min(minimum,float(rotated[:,2].min()))
            if name=='mug' and model.geom_contype[gid]==0:
                vertices,faces=mesh_local(model,gid); exported=dict(vertices=vertices,faces=faces,scale=np.array(.8))
        q=sim.data.get_joint_qpos(name+'_joint_0').copy(); q[2]=.001-minimum
        sim.data.set_joint_qpos(name+'_joint_0',q)
    sim.forward()
    initial=max([max(0.,-float(sim.data.contact[i].dist)) for i in range(sim.data.ncon)]+[0.])
    maxpen=initial
    for _ in range(600):
        sim.step()
        maxpen=max(maxpen,max([max(0.,-float(sim.data.contact[i].dist)) for i in range(sim.data.ncon)]+[0.]))
    report=dict(seed=seed,objects=[name for name,_ in positions],initial_penetration_m=initial,
        settling_max_penetration_m=maxpen,settling_steps=600,finite=bool(np.isfinite(sim.data.qpos).all()),
        hand_mode='parked by joint equality only in this perception sandbox',control_enabled=False,
        object_scale=.8,object_source='existing DexMV YCB 025_mug visual mesh and v0 collision parts',
        all_objects_free=True)
    return sim,xml,exported,report


def ground_truth(sim):
    result={}
    for name in ('mug','banana','sugar_box','mustard_bottle','tomato_soup_can'):
        if name+'_0' not in sim.model.body_names: continue
        bid=sim.model.body_name2id(name+'_0'); T=np.eye(4)
        T[:3,:3]=sim.data.body_xmat[bid].reshape(3,3); T[:3,3]=sim.data.body_xpos[bid]
        result[name]=T.tolist()
    return result


def asset_manifest(xml):
    files={}
    for node in ET.fromstring(xml).find('asset'):
        if node.get('file'):
            p=Path(node.get('file')); files[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
    return dict(upstream_arena='https://github.com/ARISE-Initiative/robosuite',
        local_source=str(SIM_ROOT),files_sha256=files)
