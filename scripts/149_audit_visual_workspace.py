#!/usr/bin/env python3
"""Audit the entire reference's transformed joint envelope before motion."""
import argparse
from pathlib import Path
import numpy as np
import transforms3d
from hierarchy_common import ROOT,SkillRegistry,verify_delivery,read,write
from stage4_pipeline_common import load_pieces
from stage4_common import experiment
from fromrealhand.tabletop.control import VisualReference

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--estimate',type=Path,required=True); p.add_argument('--output',type=Path,required=True)
a=p.parse_args(); r=SkillRegistry.load(ROOT/'configs/skill_registry.yaml'); run=verify_delivery(r)
entry=next(e for e in read(run/'build/manifest.json')['trajectories'] if e['trajectory']==r.config['scenes']['first'])
pieces=load_pieces(run/'build',entry); exp=experiment(entry['geometry'],pieces[0])
try:
    m=exp.env.sim.model; b=m.body_name2id('forearm'); base=np.eye(4)
    base[:3,:3]=transforms3d.quaternions.quat2mat(m.body_quat[b]); base[:3,3]=m.body_pos[b]
    cfg=read(ROOT/'configs/tabletop-control-candidate.json')
    with np.load(entry['geometry']) as g: ref=g['object_poses'][0]
    adapter=VisualReference(np.concatenate([p['actions'] for p in pieces]),
        np.asarray([s['qpos'][:30] for p in pieces for s in p['sim_data']]),ref,
        read(a.estimate)['T_world_object'],base,
        dict(joint_range=m.jnt_range[:30],gain=m.actuator_gainprm[:,0],bias=m.actuator_biasprm,action_range=exp.env.act_rng),cfg,entry['dt'])
    targets=np.asarray([adapter.desired_qpos(i) for i in range(len(adapter.actions))])
    lo=targets[:,:6].min(0); hi=targets[:,:6].max(0)
    report=dict(minimum=lo.tolist(),maximum=hi.tolist(),range=m.jnt_range[:6].tolist(),
        feasible_installation_slide_offset_min=(m.jnt_range[:3,0]-lo[:3]).tolist(),
        feasible_installation_slide_offset_max=(m.jnt_range[:3,1]-hi[:3]).tolist(),motion_steps=0)
    correction=adapter.factor*(targets[:,:6]-adapter.qpos[:,:6])
    report['required_visual_correction_max']=float(np.abs(correction).max())
    report['required_visual_correction_per_joint']=np.abs(correction).max(0).tolist()
    shift=np.array([-.01,.095,0.,0.,0.,0.])
    controls=adapter.actions.copy(); controls[:,:6]+=correction+adapter.factor*shift
    report['feedforward_action_abs_max_with_installation']=float(np.abs(controls).max())
    write(a.output,report); print(report)
finally: exp.env.close()
