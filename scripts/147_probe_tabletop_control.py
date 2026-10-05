#!/usr/bin/env python3
"""Independent scene/vision probe. Never executes a grasp action."""
import argparse
import json
from pathlib import Path
import threading
import traceback
import numpy as np
from hierarchy_common import SkillRegistry, verify_delivery, read, write, ROOT
from stage4_pipeline_common import load_pieces
from stage4_common import experiment,restore_once
from fromrealhand.tabletop.control_scene import create
from fromrealhand.tabletop.scene import ground_truth
from fromrealhand.tabletop.camera import read_rgbd
from fromrealhand.desktop.rendering import stream_context


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scene',choices=['first','second'],default='second')
    p.add_argument('--seed',type=int,default=0)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--vision',action='store_true')
    a=p.parse_args(); a.output.mkdir(parents=True,exist_ok=False)
    r=SkillRegistry.load(ROOT/'configs/skill_registry.yaml'); run=verify_delivery(r)
    m=read(run/'build/manifest.json')
    e=next(x for x in m['trajectories'] if x['trajectory']==r.config['scenes'][a.scene])
    pieces=load_pieces(run/'build',e); exp=experiment(e['geometry'],pieces[0])
    result=dict(scene=a.scene,seed=a.seed,grasp_actions=0)
    try:
        restore_once(exp.env,pieces[0]['initial_snapshot'])
        sites=['S_grasp','S_thtip','S_fftip','S_mftip','S_rftip','S_lftip']
        before=np.asarray([exp.env.sim.data.site_xpos[exp.env.sim.model.site_name2id(n)].copy() for n in sites])
        env,mesh,report=create(exp.env,pieces[0]['initial_snapshot'],a.seed,e['dt'])
        sim=env.sim; context=stream_context(sim)
        after=np.asarray([sim.data.site_xpos[sim.model.site_name2id(n)].copy() for n in sites])
        result['installation_hand_world_error_m']=float(np.abs(before-after).max())
        context.vopt.geomgroup[2]=0; context.vopt.geomgroup[4]=0; context.vopt.sitegroup[:]=0
        rgb,depth,calibration=read_rgbd(sim,context,'rgbd')
        import cv2
        cv2.imwrite(str(a.output/'initial.png'),cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR))
        np.save(str(a.output/'depth.npy'),depth)
        truth=ground_truth(sim)
        with np.load(e['geometry']) as g:
            result['reference_object']=g['object_poses'][0].tolist()
        result.update(scene_report=report,initial_contacts=env.contacts(),calibration=calibration,
            initial_truth_evaluation_only=truth,neq=sim.model.neq,nu=sim.model.nu,
            root_joint_limits=sim.model.jnt_range[:6].tolist(),root_qpos=sim.data.qpos[:6].tolist(),
            all_contact_pairs=[dict(a=sim.model.geom_id2name(int(c.geom1)),b=sim.model.geom_id2name(int(c.geom2)),distance=float(c.dist))
                               for c in sim.data.contact[:sim.data.ncon]],pixel_std=float(rgb.std()))
        if a.vision:
            from fromrealhand.tabletop.vision_client import VisionClient
            client=VisionClient('/media/smgbro/shared/visual_grasp',a.output,mesh,threading.Event(),90)
            try: estimate=client.capture(sim,context)
            finally: client.close()
            result['estimate']=estimate
            if estimate.get('accepted'):
                t=np.asarray(estimate['T_world_object']); gt=np.asarray(truth['mug'])
                result['position_error_m']=float(np.linalg.norm(t[:3,3]-gt[:3,3]))
                result['rotation_error_deg']=float(np.rad2deg(np.arccos(np.clip((np.trace(t[:3,:3].T@gt[:3,:3])-1)/2,-1,1))))
        result['passed']=bool(sim.model.neq==0 and sim.model.nu==30 and rgb.std()>5
            and result['installation_hand_world_error_m']<1e-8
            and result['initial_contacts']['finite'] and result['initial_contacts']['scene_penetration_m']<=.001
            and not result['initial_contacts']['non_target_contacts']
            and (not a.vision or (result['estimate']['accepted'] and result.get('position_error_m',1)<.01)))
    except Exception as error:
        result.update(passed=False,error=str(error),traceback=traceback.format_exc())
    finally:
        exp.env.close(); write(a.output/'report.json',result)
        print(json.dumps(result,ensure_ascii=False),flush=True)
        import glfw
        glfw.terminate()
    return 0 if result['passed'] else 1


if __name__=='__main__': raise SystemExit(main())
