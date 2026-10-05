#!/usr/bin/env python3
"""Development motor rollouts; no learning, no object forces or state resets."""
import argparse
import hashlib
import json
import threading
from pathlib import Path
import numpy as np
import transforms3d
from hierarchy_common import ROOT, SkillRegistry, verify_delivery, read, write
from stage4_pipeline_common import load_pieces
from stage4_common import experiment
from fromrealhand.tabletop.control import VisualReference, pose_from_estimate, check_pose_jump, perception_interval
from fromrealhand.tabletop.control_scene import create
from fromrealhand.tabletop.contact_control import reference_contacts, ContactTracker, FINGERS, opposing_contacts
from fromrealhand.tabletop.scene import ground_truth
from fromrealhand.tabletop.training_inputs import features,episode_arrays,SKILLS,FEATURE_VERSION


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--scene',choices=['first','second'],default='first')
    p.add_argument('--upright-correspondence',action='store_true')
    p.add_argument('--clearance',type=float,default=0.)
    p.add_argument('--stable-carry',action='store_true')
    p.add_argument('--bounded-orientation',action='store_true')
    p.add_argument('--installation-y',type=float,default=.06)
    p.add_argument('--goal-offset',type=float,nargs=3,default=[0.,0.,0.])
    p.add_argument('--kp',type=float,default=160.)
    p.add_argument('--root-gain',type=float,default=1.)
    p.add_argument('--finger-gain',type=float,default=0.)
    p.add_argument('--anchor',type=int,default=400)
    p.add_argument('--live-vision',action='store_true')
    p.add_argument('--camera',choices=['rgbd','rgbd_side'],default='rgbd')
    p.add_argument('--seed',type=int,default=0)
    p.add_argument('--hold-steps',type=int,default=100)
    p.add_argument('--goal',choices=['reach','grasp','lift','transport'],default='transport')
    p.add_argument('--checkpoint',type=Path)
    a=p.parse_args(); a.output.mkdir(parents=True,exist_ok=False)
    source_files=[Path(__file__).resolve(),ROOT/'src/fromrealhand/tabletop/contact_control.py',
        ROOT/'src/fromrealhand/tabletop/control.py',ROOT/'src/fromrealhand/tabletop/control_scene.py',
        ROOT/'src/fromrealhand/perception/tracking.py',ROOT/'src/fromrealhand/tabletop/vision_client.py',
        ROOT/'src/fromrealhand/tabletop/functional_reference.py',ROOT/'src/fromrealhand/perception/association.py',
        ROOT/'scripts/142_tabletop_vision_worker.py',ROOT/'configs/tabletop-dual-v3-protocol.json']
    source_files.extend([ROOT/'src/fromrealhand/tabletop/training_inputs.py',ROOT/'configs/tabletop-dual-v3-profiles.json'])
    write(a.output/'sources.json',{str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in source_files})
    if not a.live_vision and a.seed!=0: raise ValueError('Cached anchor valid only for its identical seed-0 scene')
    r=SkillRegistry.load(ROOT/'configs/skill_registry.yaml'); run=verify_delivery(r)
    entry=next(e for e in read(run/'build/manifest.json')['trajectories'] if e['trajectory']==r.config['scenes'][a.scene])
    pieces=load_pieces(run/'build',entry); exp=experiment(entry['geometry'],pieces[0])
    env,mesh,scene=create(exp.env,pieces[0]['initial_snapshot'],a.seed,entry['dt'],[-.01,a.installation_y,0.],
        [0.,0.,a.clearance] if a.clearance else None)
    sim=env.sim; m,d=sim.model,sim.data; vision=None; context=None
    mug=m.body_name2id('mug_0')
    write(a.output/'scene.json',scene)
    write(a.output/'dynamics.json',dict(mug_mass_kg=float(m.body_mass[mug]),
        mug_inertia=m.body_inertia[mug].tolist(),object_scale=.8,
        mug_geom_friction={m.geom_id2name(i):m.geom_friction[i].tolist() for i in range(m.ngeom)
            if m.geom_bodyid[i]==mug},physical_parameters_changed_for_this_run=False))
    learner=None
    if a.checkpoint:
        from fromrealhand.tabletop.learned_control import LearnedControl
        learner=LearnedControl(a.checkpoint)
    rows=[]; commands=[]; samples=[]; references=[]; clocks=[]; phases=[]; visual_frames=[]
    phase='initial'; cursor=0; peak=0.; completed=[]; report={}; events=[]
    try:
        cfg=read(ROOT/'configs/tabletop-control-candidate.json')
        states=[s for part in pieces for s in part['sim_data']]
        local,source_poses=reference_contacts(exp.env.sim,states)
        if a.upright_correspondence:
            from fromrealhand.tabletop.functional_reference import upright_source_correspondence
            local,source_poses,correspondence=upright_source_correspondence(local,source_poses,a.anchor)
            write(a.output/'correspondence.json',correspondence)
        if a.live_vision:
            from fromrealhand.desktop.rendering import stream_context
            from fromrealhand.tabletop.vision_client import VisionClient
            context=stream_context(sim)
            context.vopt.geomgroup[2]=0; context.vopt.geomgroup[4]=0; context.vopt.sitegroup[:]=0
            vision=VisionClient('/media/smgbro/shared/visual_grasp',a.output,mesh,threading.Event(),90)
            latest=vision.capture(sim,context,a.camera); pose=pose_from_estimate(latest,d.time,cfg)
        else:
            latest=read(Path('/media/smgbro/shared/visual_grasp/control_runs/20261003T064548706107Z/observations/000000/estimate.json'))
            pose=np.asarray(latest['T_world_object']); latest['camera_time_s']=0.
        base=np.eye(4); b=m.body_name2id('forearm'); base[:3,3]=m.body_pos[b]
        base[:3,:3]=transforms3d.quaternions.quat2mat(m.body_quat[b])
        from fromrealhand.tabletop.functional_reference import prepare_reference
        stop=next(s['stop'] for s in entry['segments'] if s['skill']==a.goal)
        adapter,goal,preflight=prepare_reference(np.concatenate([s['actions'] for s in pieces]),
            np.array([s['qpos'][:30] for s in states]),source_poses,pose,base,
            dict(joint_range=m.jnt_range[:30],gain=m.actuator_gainprm[:,0],bias=m.actuator_biasprm,
                action_range=env.rng,reference_base=env.reference_base),cfg,entry['dt'],vars(a),entry['segments'],stop)
        write(a.output/'goal.json',dict(goal_world_m=goal.tolist(),offset_from_source_goal_m=a.goal_offset,
            live_object_state_modified=False))
        if a.stable_carry:
            grasp_stop=next(s['stop'] for s in entry['segments'] if s['skill']=='grasp')
            anchor=grasp_stop-1
            local[anchor:]=local[anchor].copy()
        controller=ContactTracker(adapter,local,kp=a.kp,root_gain=a.root_gain,finger_gain=a.finger_gain)
        targets=np.array([adapter.desired_qpos(i) for i in range(stop)])
        write(a.output/'workspace.json',dict(root_min=targets[:,:6].min(0).tolist(),root_max=targets[:,:6].max(0).tolist(),
            limits=m.jnt_range[:6].tolist(),delta=adapter.delta.tolist()))
        write(a.output/'preflight.json',preflight)
        write(a.output/'config.json',dict(vars(a),output=str(a.output),checkpoint=str(a.checkpoint) if a.checkpoint else None,independent_test=False,training=False,
            cached_initial_anchor=not a.live_vision,source_anchor=a.anchor))
        def audit():
            nonlocal peak
            q=env.contacts(); peak=max(peak,q['scene_penetration_m'])
            if not q['finite'] or q['scene_penetration_m']>.001 or q['joint_violation_rad']>.02 or q['non_target_contacts']:
                raise RuntimeError('physics_safety_stop:'+json.dumps(q))
            if q['max_hand_speed']>20: raise RuntimeError('speed_stop')
        def supported(q):
            return q['th_force_n']>.01 and sum(q[f+'_force_n']>.01 for f in FINGERS)>=3 and q['opposition']
        def event(q,skill):
            if skill=='reach': return any(q[f+'_force_n']>.01 for f in FINGERS)
            if skill=='grasp': return supported(q)
            if skill=='lift': return supported(q) and q['bottom_m']>=.05
            return supported(q) and q['bottom_m']>=.05 and q['target_distance_m']<=.02
        for segment in entry['segments']:
            phase=segment['skill']; streak=0; support_lost=0; visual_confirmations=set()
            if a.stable_carry and phase=='lift':
                if vision:
                    latest=vision.capture(sim,context,a.camera); new=pose_from_estimate(latest,d.time,cfg)
                    check_pose_jump(pose,new,cfg); pose=new
                    controller.start_carry(sim,pose,goal,segment['start']-1,segment['stop']-1,len(states))
                else:
                    # Cached screening never uses object truth to select actions.
                    controller.start_carry(sim,pose,goal,segment['start']-1,segment['stop']-1,len(states))
            indices=list(range(segment['start'],segment['stop']))+[segment['stop']-1]*a.hold_steps
            for j,index in enumerate(indices):
                cursor=index
                if vision and d.time-latest['camera_time_s']>=perception_interval(cfg,phase):
                    latest=vision.capture(sim,context,a.camera); new=pose_from_estimate(latest,d.time,cfg)
                    check_pose_jump(pose,new,cfg); pose=new
                action=controller.action(index,sim,pose,latest['frame'] if vision else None)
                desired,ref=controller.reference(index,sim)
                sample=features(d.qpos[:30],d.qvel[:30],d.get_site_xpos('S_grasp'),pose,goal,a.scene,phase,
                    index,(segment['start'],segment['stop']),ref,desired)
                if learner:
                    expert_action=action.copy()
                    action=learner.action(sample,ref,a.scene,phase)
                    controller.last_action=action.copy()
                commands.append(action.tolist()); env.step(action,audit)
                samples.append(sample); references.append(ref.copy()); clocks.append(index)
                phases.append(SKILLS.index(phase)); visual_frames.append(int(latest.get('frame',0)))
                q=env.contacts(); contact=opposing_contacts(sim); q.update(opposition=contact['opposition'],normal_dots=contact['normal_dots'])
                # In cached-anchor screening, truth is ONLY a labelled evaluation metric.
                measured=pose if vision else np.asarray(ground_truth(sim)['mug'])
                q.update(step=len(rows)+1,source_index=index,phase=phase,
                    bottom_m=float((mesh['vertices'] @ measured[:3,:3].T+measured[:3,3])[:,2].min()),
                    target_distance_m=float(np.linalg.norm(measured[:3,3]-goal)),**controller.last)
                rows.append(q); streak=streak+1 if event(q,phase) else 0
                if learner:
                    q['expert_action_error_max']=float(np.max(np.abs(action-expert_action)))
                    q['expert_action_error_rms']=float(np.sqrt(np.mean((action-expert_action)**2)))
                if event(q,phase) and vision: visual_confirmations.add(latest['frame'])
                elif not event(q,phase): visual_confirmations.clear()
                if phase in ('lift','transport'):
                    support_lost=0 if supported(q) else support_lost+1
                    if support_lost>=15: raise RuntimeError('persistent_support_loss')
                required=max(segment['confirmation_steps'],50 if phase=='grasp' else 0)
                visual_ok=not vision or phase not in ('lift','transport') or len(visual_confirmations)>=2
                if j>=segment['stop']-segment['start']-1 and streak>=required and visual_ok:
                    break
            if streak<required or not visual_ok: raise RuntimeError('stage_not_confirmed:'+phase)
            completed.append(phase)
            events.append(dict(phase=phase,step=len(rows),reference_step=cursor,
                consecutive_steps=streak,visual_confirmations=len(visual_confirmations)))
            if context:
                import cv2
                context.render(960,720,camera_id=m.camera_name2id('rgbd'))
                rgb=context.read_pixels(960,720,depth=False)[::-1].copy()
                cv2.imwrite(str(a.output/(phase+'.png')),cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR))
            if phase==a.goal: break
        report.update(passed=True,reason='requested_stages_complete')
    except Exception as error:
        report.update(passed=False,reason=str(error))
    finally:
        if vision: vision.close()
        report.update(steps=len(rows),phase=phase,source_index=cursor,completed=completed,max_penetration_m=peak,
            final=rows[-1] if rows else None,ground_truth_after_stop=ground_truth(sim)['mug'],
            live_vision=a.live_vision,grasp_generalization_claim=False,state_writes_during_execution=0,
            object_forces_applied=False,training=False,final_contact_geometry=opposing_contacts(sim),events=events,
            controller='learned_'+learner.mode if learner else 'expert',
            checkpoint=str(a.checkpoint) if a.checkpoint else None,
            learned_action_calls=learner.calls if learner else 0,
            clipped_action_calls=learner.clipped_calls if learner else 0,
            feature_version=FEATURE_VERSION,pre_action_collection=True,positive_demonstration=False)
        if samples:
            arrays=episode_arrays(samples,commands[:len(samples)],references,clocks,phases,visual_frames)
            positive=bool(report.get('passed') and a.live_vision and a.goal=='transport' and not learner)
            np.savez_compressed(str(a.output/('demonstration.npz' if positive else 'diagnostic-only.npz')),**arrays)
            report['positive_demonstration']=positive
        write(a.output/'report.json',report); write(a.output/'trace.json',rows)
        np.save(str(a.output/'actions.npy'),np.asarray(commands)); print(json.dumps(report),flush=True)
        exp.env.close()
        if context:
            import glfw
            glfw.terminate()


if __name__=='__main__': main()
