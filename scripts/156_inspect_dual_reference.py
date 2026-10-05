#!/usr/bin/env python3
"""Inspect both frozen references in separate FK states; never a motor acceptance."""
import argparse
from pathlib import Path
import numpy as np
from hierarchy_common import ROOT,SkillRegistry,verify_delivery,read,write
from stage4_pipeline_common import load_pieces
from stage4_common import experiment
from fromrealhand.tabletop.contact_control import object_pose
from fromrealhand.tabletop.scene import mesh_local

p=argparse.ArgumentParser(description=__doc__); p.add_argument('--output',type=Path,required=True)
a=p.parse_args(); a.output.mkdir(parents=True,exist_ok=False)
r=SkillRegistry.load(ROOT/'configs/skill_registry.yaml'); run=verify_delivery(r); out={}
target=np.asarray(read(ROOT/'docs/presentation/tabletop_contact_v2/evidence/final-adaptation.json')['T_world_object_visual'])
for name in ('first','second'):
    e=next(x for x in read(run/'build/manifest.json')['trajectories'] if x['trajectory']==r.config['scenes'][name])
    pieces=load_pieces(run/'build',e); exp=experiment(e['geometry'],pieces[0]); sim=exp.env.sim
    try:
        states=[s for p in pieces for s in p['sim_data']]
        frames=[0]+[s['stop']-1 for s in e['segments']]
        rows=[]; sites=['S_grasp','S_thtip','S_fftip','S_mftip','S_rftip','S_lftip']
        for idx in frames:
            sim.data.qpos[:]=states[idx]['qpos']; sim.forward()
            pose=object_pose(states[idx]['qpos']); delta=target @ np.linalg.inv(pose)
            tips=np.array([sim.data.get_site_xpos(s).copy() for s in sites])
            transformed=tips @ delta[:3,:3].T+delta[:3,3]
            bid=sim.model.body_name2id('mug_0')
            gid=next(i for i in range(sim.model.ngeom) if sim.model.geom_bodyid[i]==bid and sim.model.geom_contype[i]==0)
            vertices,_=mesh_local(sim.model,gid)
            rows.append(dict(index=idx,pose=pose.tolist(),source_sites=tips.tolist(),
                transformed_sites=transformed.tolist(),local_mesh_bounds=[vertices.min(0).tolist(),vertices.max(0).tolist()],
                visual_geom_quat=sim.model.geom_quat[gid].tolist(),root_qpos=states[idx]['qpos'][:6].tolist()))
        out[name]=dict(segments=e['segments'],geometry=e['geometry'],rows=rows,kinematic_only=True)
    finally: exp.env.close()
write(a.output/'references.json',out)
for name,report in out.items():
    print(name,[(s['skill'],s['start'],s['stop']) for s in report['segments']])
    for row in report['rows']:
        print(row['index'],'Rzz',row['pose'][2][2],'palm',row['source_sites'][0],
            'mapped',row['transformed_sites'][0],'mapped finger z',[round(p[2],3) for p in row['transformed_sites'][1:]])
