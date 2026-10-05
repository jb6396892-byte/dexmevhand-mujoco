#!/usr/bin/env python3
"""Kinematic comparison on separate diagnostic data; never evidence of a grasp."""
import argparse
from pathlib import Path
import numpy as np
import transforms3d
from hierarchy_common import ROOT, SkillRegistry, verify_delivery, read, write
from stage4_pipeline_common import load_pieces
from stage4_common import experiment
from fromrealhand.tabletop.control import VisualReference
from fromrealhand.tabletop.control_scene import create

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--estimate',type=Path,default=Path('/media/smgbro/shared/visual_grasp/control_runs/20261003T064548706107Z/observations/000000/estimate.json'))
a=p.parse_args(); a.output.mkdir(parents=True,exist_ok=False)
r=SkillRegistry.load(ROOT/'configs/skill_registry.yaml'); run=verify_delivery(r)
entry=next(e for e in read(run/'build/manifest.json')['trajectories'] if e['trajectory']==r.config['scenes']['first'])
pieces=load_pieces(run/'build',entry); exp=experiment(entry['geometry'],pieces[0])
try:
    env,mesh,scene=create(exp.env,pieces[0]['initial_snapshot'],0,entry['dt'])
    m,d=env.sim.model,env.sim.data; old=exp.env.sim
    actions=np.concatenate([s['actions'] for s in pieces])
    states=[s for p in pieces for s in p['sim_data']]
    qpos=np.array([s['qpos'][:30] for s in states])
    base=np.eye(4); b=m.body_name2id('forearm'); base[:3,3]=m.body_pos[b]
    base[:3,:3]=transforms3d.quaternions.quat2mat(m.body_quat[b])
    with np.load(entry['geometry']) as g: ref=g['object_poses'][0].copy()
    cfg=read(ROOT/'configs/tabletop-control-candidate.json')
    adapter=VisualReference(actions,qpos,ref,read(a.estimate)['T_world_object'],base,
        dict(joint_range=m.jnt_range[:30],gain=m.actuator_gainprm[:,0],bias=m.actuator_biasprm,
             action_range=env.rng,reference_base=env.reference_base),cfg,entry['dt'])
    sites=['S_grasp','S_thtip','S_fftip','S_mftip','S_rftip','S_lftip']
    rows=[]
    for idx in (0,200,400,511,542,623,1000,len(qpos)-1):
        old.data.qpos[:]=states[idx]['qpos']; old.forward()
        d.qpos[:30]=adapter.desired_qpos(idx); env.sim.forward()
        source=np.array([old.data.get_site_xpos(n).copy() for n in sites])
        desired=source @ adapter.delta[:3,:3].T+adapter.delta[:3,3]
        actual=np.array([d.get_site_xpos(n).copy() for n in sites])
        oldforces={f:pieces[0]['initial_metrics'].get(f+'_force_n') for f in ('th','ff','mf','rf','lf')}
        contacts=[]
        for c in old.data.contact[:old.data.ncon]:
            names=[old.model.geom_id2name(int(g)) for g in (c.geom1,c.geom2)]
            if any(n and n.startswith('C_') for n in names):
                contacts.append(dict(geoms=names,distance=float(c.dist)))
        rows.append(dict(index=idx,sites=sites,source_sites=source.tolist(),transformed_sites=desired.tolist(),
            adapter_sites=actual.tolist(),transfer_error_m=np.linalg.norm(actual-desired,axis=1).tolist(),
            source_object_position=old.data.body_xpos[exp.env.obj_bid].tolist(),source_contacts=contacts))
    model_diff={}
    for field in ('geom_size','geom_pos','geom_quat','geom_friction','geom_margin','geom_solref','geom_solimp'):
        delta=[]
        for n in old.model.geom_names:
            if n and n in m.geom_names:
                v=float(np.max(np.abs(getattr(old.model,field)[old.model.geom_name2id(n)]-getattr(m,field)[m.geom_name2id(n)])))
                if v>1e-8: delta.append(dict(name=n,error=v))
        model_diff[field]=delta
    write(a.output/'diagnosis.json',dict(kinematic_only=True,rows=rows,model_diff=model_diff,
        delta=adapter.delta.tolist(),source_base=env.reference_base.tolist(),joint_names=list(m.joint_names[:30]),
        joint_axes=m.jnt_axis[:30].tolist(),joint_pos=m.jnt_pos[:30].tolist()))
    print([(r['index'],r['transfer_error_m']) for r in rows]); print(model_diff)
finally: exp.env.close()
