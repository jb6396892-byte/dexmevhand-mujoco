#!/usr/bin/env python3
"""Explicitly opt-in candidate: visual pose -> reference adapter -> motor actions."""
import argparse
import base64
import contextlib
import ctypes
import importlib
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from fromrealhand.desktop.runtime import emit


def execute(args):
    if not args.allow_unvalidated_tabletop:
        emit('result', report=dict(status='locked', reason='candidate_not_tested', steps=0, simulation_created=False))
        return
    from fromrealhand.desktop.planning import validated_plan
    plan = validated_plan(args.root, args.output)
    if plan['goal'] == 'stop':
        emit('result', report=dict(status='stopped', reason='user_stop', steps=0, simulation_created=False)); return
    import numpy as np
    import cv2
    import transforms3d
    from hierarchy_common import SkillRegistry, verify_delivery, read, write
    from stage4_pipeline_common import load_pieces
    from stage4_common import experiment
    from fromrealhand.desktop.rendering import stream_context
    from fromrealhand.tabletop.control_scene import create
    from fromrealhand.tabletop.control import VisualReference, pose_from_estimate, check_pose_jump, perception_interval
    from fromrealhand.tabletop.contact_control import reference_contacts, ContactTracker, opposing_contacts
    from fromrealhand.tabletop.vision_client import VisionClient
    from fromrealhand.tabletop.camera import transform
    from fromrealhand.tabletop.training_inputs import features
    learner=None
    if args.checkpoint:
        from fromrealhand.tabletop.learned_control import LearnedControl
        learner=LearnedControl(args.checkpoint)
    controls = importlib.import_module('126_stream_simulation').Controls()
    cfg = read(ROOT/'configs/tabletop-control-candidate.json')
    contact_cfg = read(ROOT/'configs/tabletop-contact-v2.json')
    dual_cfg=read(ROOT/'configs/tabletop-dual-v3-profiles.json')
    profile=dual_cfg['profiles'].get(plan['scene'])
    contact_mode = profile is not None
    if contact_mode:
        contact_cfg=dict(contact_cfg,stable_reference_anchor=profile['anchor'],camera=dual_cfg['camera'],
            installation_slide_offset_m=[-.01,profile['installation_y'],0.],
            additional_root_feedback_gain=profile['root_gain'],fingertip_cartesian_gain=profile['kp'],
            phase_extension_steps=profile['hold_steps'],profile_version='dual-v3')
    camera_name = contact_cfg['camera'] if contact_mode else 'rgbd'
    registry = SkillRegistry.load(ROOT/'configs/skill_registry.yaml'); registry.validate_plan(plan)
    run = verify_delivery(registry); manifest = read(run/'build/manifest.json')
    entry = next(e for e in manifest['trajectories'] if e['trajectory'] == registry.config['scenes'][plan['scene']])
    pieces = load_pieces(run/'build', entry)
    output = args.visual_root.resolve()/'control_runs'/args.output.name; output.mkdir(parents=True, exist_ok=False)
    write(output/'request.json', dict(plan=plan, seed=args.seed, candidate_not_validated=True,
        language_run=str(args.output), config=cfg, contact_config=contact_cfg if contact_mode else None,
            sensor_camera=camera_name,acceptance_inherited=False,checkpoint=str(args.checkpoint) if args.checkpoint else None))
    sources=[Path(__file__),ROOT/'src/fromrealhand/tabletop/control.py',ROOT/'src/fromrealhand/tabletop/control_scene.py',
        ROOT/'src/fromrealhand/perception/tracking.py',ROOT/'configs/tabletop-control-candidate.json',
        ROOT/'src/fromrealhand/tabletop/contact_control.py',ROOT/'configs/tabletop-contact-v2.json',
        ROOT/'src/fromrealhand/tabletop/vision_client.py',ROOT/'scripts/142_tabletop_vision_worker.py',
        ROOT/'configs/tabletop-dual-v3-profiles.json',ROOT/'src/fromrealhand/tabletop/functional_reference.py',
        ROOT/'src/fromrealhand/perception/association.py']
    write(output/'source.json',{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
    vision, reference, env, adapter, contact_controller = None, None, None, None, None
    rows, actions, events = [], [], []
    cursor = 0; phase = None; started = time.monotonic(); latest = None; pose = None; goal = None
    last_frame, peak, support_lost = 0., 0., 0
    report = dict(status='stopped', reason='initialization_failed', candidate_not_validated=True)
    try:
        with contextlib.redirect_stdout(sys.stderr):
            reference = experiment(entry['geometry'], pieces[0])
            env, mesh, scene = create(reference.env, pieces[0]['initial_snapshot'], args.seed, entry['dt'],
                contact_cfg['installation_slide_offset_m'] if contact_mode else None,
                [0.,0.,profile['clearance']] if contact_mode and profile['clearance'] else None)
            context = stream_context(env.sim)
            context.vopt.geomgroup[2] = 0; context.vopt.geomgroup[4] = 0
            context.vopt.sitegroup[:] = 0
        sim, model = env.sim, env.sim.model
        write(output/'scene.json', scene)
        bounds = {s['skill']: (s['start'], s['stop']) for s in entry['segments']}
        gl = ctypes.CDLL('libGL.so.1'); gl.glGetString.restype = ctypes.c_char_p
        emit('ready', bounds=bounds, total_steps=bounds[plan['goal']][1], dt=entry['dt'], backend='visual_reference_candidate',
            policy_mode=learner.mode if learner else 'expert',
            gl={k: (gl.glGetString(v) or b'unknown').decode() for k,v in [('renderer',0x1F01),('version',0x1F02)]})
        emit('status', message='加载视觉模型，仿真暂停', state='perception')
        vision = VisionClient(args.visual_root, output, mesh, controls.stop, cfg['max_inference_wall_s'])

        def observe():
            nonlocal latest, pose
            emit('status', message='RGB-D 定位中，仿真暂停', state='perception')
            context.vopt.sitegroup[5] = 0  # Never put the task marker in sensor RGB-D.
            row = vision.capture(sim, context, camera_name)
            row['goal_world_m'] = None if goal is None else goal.tolist()
            images = {}
            for name, file in [('rgb','rgb.png'),('depth','depth-preview.png'),('overlay','overlay.png')]:
                im = cv2.imread(str(Path(row['observation_directory'])/file))
                if im is None: raise ValueError('Missing perception image: '+file)
                ok, buf = cv2.imencode('.jpg', im, [cv2.IMWRITE_JPEG_QUALITY,75])
                if not ok: raise ValueError('JPEG encoding failed')
                images[name] = base64.b64encode(buf).decode('ascii')
            emit('perception', estimate=row, images=images, output=str(output))
            candidate = pose_from_estimate(row, sim.data.time, cfg)
            if pose is not None: check_pose_jump(pose, candidate, cfg)
            pose, latest = candidate, row
            emit('status', message='视觉候选控制中（未验收）', state='simulation')

        observe()
        with np.load(entry['geometry'], allow_pickle=False) as g:
            source_object = g['object_poses'][0].copy(); old_goal = g['object_poses'][-1,:3,3].copy()
        nominal_actions = np.concatenate([p['actions'] for p in pieces])
        pre_qpos = np.asarray([s['qpos'][:30] for p in pieces for s in p['sim_data']])
        if contact_mode:
            local_tips, source_poses = reference_contacts(reference.env.sim,
                [s for p in pieces for s in p['sim_data']])
            if profile['upright_correspondence']:
                from fromrealhand.tabletop.functional_reference import upright_source_correspondence
                local_tips,source_poses,correspondence=upright_source_correspondence(
                    local_tips,source_poses,profile['anchor'])
                write(output/'correspondence.json',correspondence)
            source_object = source_poses[contact_cfg['stable_reference_anchor']]
            old_goal = source_poses[-1,:3,3].copy()
        base = np.eye(4); bid = model.body_name2id('forearm')
        base[:3,:3] = transforms3d.quaternions.quat2mat(model.body_quat[bid]); base[:3,3] = model.body_pos[bid]
        model_parameters=dict(joint_range=model.jnt_range[:30],gain=model.actuator_gainprm[:,0],
            bias=model.actuator_biasprm,action_range=env.rng,reference_base=env.reference_base)
        if contact_mode:
            from fromrealhand.tabletop.functional_reference import prepare_reference
            adapter,goal,preflight=prepare_reference(nominal_actions,pre_qpos,source_poses,pose,base,
                model_parameters,cfg,entry['dt'],profile,entry['segments'],bounds[plan['goal']][1])
            if profile['stable_carry']:
                anchor=bounds['grasp'][1]-1
                local_tips[anchor:]=local_tips[anchor].copy()
            contact_controller=ContactTracker(adapter,local_tips,kp=contact_cfg['fingertip_cartesian_gain'],
                root_gain=contact_cfg['additional_root_feedback_gain'],finger_gain=contact_cfg['finger_joint_feedback_gain'])
        else:
            adapter=VisualReference(nominal_actions,pre_qpos,source_object,pose,base,model_parameters,cfg,entry['dt'])
            goal=adapter.goal(old_goal); preflight=adapter.preflight(bounds[plan['goal']][1])
        write(output/'preflight.json',preflight)
        if np.any(goal < cfg['workspace_min_m']) or np.any(goal > cfg['workspace_max_m']):
            raise ValueError('Adapted goal outside workspace')
        model.site_pos[model.site_name2id('visual_goal')] = goal
        sim.forward()  # Visual marker only, no object/hand state assignment.
        write(output/'adaptation.json', dict(T_world_object_visual=pose.tolist(), T_delta=adapter.delta.tolist(),
            goal_world_m=goal.tolist(), object_pose_source='rgbd_only', backend='learned_'+learner.mode if learner else 'expert_reference_plus_root_feedback',
            not_dapg_policy=True, state_writes_during_execution=0,contact_tracking=contact_mode,
            goal_source='transformed reference endpoint plus declared profile offset' if contact_mode else 'geometry endpoint',
            camera=camera_name))
        write(output/'functional_profile.json',dict(profile=profile,source_video=plan['scene'],
            video_exact=False if profile and profile['stable_carry'] else None,training_started=False))

        def audit():
            nonlocal peak
            if controls.stop.is_set(): raise RuntimeError('user_stop')
            r = env.contacts(); peak = max(peak, r['scene_penetration_m'])
            if not r['finite']: raise RuntimeError('nonfinite_state')
            for key, maximum in [('scene_penetration_m',cfg['max_penetration_m']),
                ('joint_violation_rad',cfg['max_joint_violation_rad']), ('max_hand_speed',cfg['max_hand_speed'])]:
                if r[key] > maximum: raise RuntimeError('safety_stop:'+key)
            if r['non_target_contacts']: raise RuntimeError('non_target_collision')
            return r

        def metrics():
            r = audit()
            if contact_mode:
                geometry=opposing_contacts(sim,cfg['force_threshold_n'],contact_cfg['opposed_normal_dot_threshold'],
                    contact_cfg['opposed_finger_count'])
                r.update(opposition=geometry['opposition'],normal_dots=geometry['normal_dots'])
                r.update(contact_controller.last)
            pose_from_estimate(latest, sim.data.time, cfg)
            r.update(bottom_m=float(transform(mesh['vertices'], pose)[:,2].min()),
                target_distance_m=float(np.linalg.norm(pose[:3,3]-goal)),
                pose_sim_age_s=float(sim.data.time-latest['camera_time_s']), object_metrics_source='rgbd_estimate')
            return r

        def supported(r):
            return r['th_force_n'] > cfg['force_threshold_n'] and sum(r[f+'_force_n'] > cfg['force_threshold_n']
                for f in ('th','ff','mf','rf','lf')) >= cfg['supported_fingers'] and (not contact_mode or r['opposition'])

        def event(skill, r):
            if skill == 'reach': return any(r[f+'_force_n'] > cfg['force_threshold_n'] for f in ('th','ff','mf','rf','lf'))
            if skill == 'grasp': return supported(r)
            if skill == 'lift': return supported(r) and r['bottom_m'] >= cfg['lift_height_m']
            return supported(r) and r['bottom_m'] >= cfg['lift_height_m'] and r['target_distance_m'] <= cfg['goal_tolerance_m']

        def draw(row, action):
            nonlocal last_frame
            # New mode has a fixed calibrated camera, not the legacy orbit controls.
            context.vopt.sitegroup[5] = 1
            context.render(960,720,camera_id=model.camera_name2id('rgbd'))
            rgb = context.read_pixels(960,720,depth=False)[::-1].copy()
            if rgb.std() < 5: raise RuntimeError('blank_frame')
            ok, buf = cv2.imencode('.jpg', cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY,75])
            if not ok: raise RuntimeError('frame_encoding_failed')
            last_frame = time.monotonic()
            emit('frame', jpeg=base64.b64encode(buf).decode('ascii'), step=cursor, skill=phase,
                actual_control_steps=len(rows),reference_step=cursor,
                sim_time_s=float(sim.data.time), metrics=row, max_penetration_m=peak,
                action_abs_max=float(np.max(np.abs(action))), elapsed_s=last_frame-started)

        previous = None; completed = []
        for phase in plan['skills']:
            emit('stage', skill=phase, step=cursor)
            if cursor != bounds[phase][0]: raise RuntimeError('reference_clock_mismatch')
            observe(); row = metrics()
            if previous and not event(previous, row): raise RuntimeError('skill_precondition_failed:'+phase)
            if contact_mode and profile['stable_carry'] and phase=='lift':
                contact_controller.start_carry(sim,pose,goal,bounds['grasp'][1]-1,bounds['lift'][1]-1,len(nominal_actions))
            streak = 0; visual_confirmations = set()
            required_steps=next(s['confirmation_steps'] for s in entry['segments'] if s['skill']==phase)
            if contact_mode and phase=='grasp': required_steps=max(required_steps,contact_cfg['grasp_confirmation_steps'])
            extension=0
            while cursor < bounds[phase][1] or (contact_mode and
                    (streak < required_steps or (phase in ('lift','transport') and len(visual_confirmations)<cfg['minimum_visual_confirmations']))):
                if len(rows) >= cfg['max_total_steps'] or time.monotonic()-started > cfg['wall_timeout_s']:
                    raise RuntimeError('execution_budget_exceeded')
                holding=cursor>=bounds[phase][1]
                if holding:
                    if extension>=contact_cfg['phase_extension_steps']: raise RuntimeError('skill_extension_timeout:'+phase)
                    extension+=1
                if sim.data.time-latest['camera_time_s'] >= perception_interval(cfg,phase):
                    observe()
                index=min(cursor,bounds[phase][1]-1)
                action = contact_controller.action(index,sim,pose,latest['frame']) if contact_mode else adapter.action(index, sim.data.qpos[:30].copy())
                if learner:
                    if not contact_mode: raise RuntimeError('Learned policy needs a registered contact profile')
                    desired,ref=contact_controller.reference(index,sim)
                    sample=features(sim.data.qpos[:30],sim.data.qvel[:30],sim.data.get_site_xpos('S_grasp'),
                        pose,goal,plan['scene'],phase,index,bounds[phase],ref,desired)
                    action=learner.action(sample,ref,plan['scene'],phase)
                    contact_controller.last_action=action.copy()
                audit(); actions.append(action.copy()); env.step(action, audit)
                if not holding: cursor += 1
                row = metrics(); rows.append(dict(row, step=len(rows)+1, reference_step=cursor, skill=phase))
                ok = event(phase, row); streak = streak+1 if ok else 0
                if ok: visual_confirmations.add(latest['frame'])
                else: visual_confirmations.clear()
                if phase in ('lift','transport'):
                    support_lost = 0 if supported(row) else support_lost+1
                    if support_lost >= cfg['support_loss_stop_steps']: raise RuntimeError('persistent_support_loss')
                if time.monotonic()-last_frame > 1/15: draw(row, action)
                controls.stop.wait(entry['dt'])
            observe(); row = metrics()
            if event(phase, row): visual_confirmations.add(latest['frame'])
            if (not event(phase,row) or streak < required_steps
                    or (phase in ('lift','transport') and len(visual_confirmations) < cfg['minimum_visual_confirmations'])):
                raise RuntimeError('skill_success_not_confirmed:'+phase)
            completed.append(phase); previous = phase
            events.append(dict(skill=phase, step=len(rows),reference_step=cursor,extension_steps=extension,
                consecutive_success_steps=streak,visual_confirmations=len(visual_confirmations)))
        draw(row, actions[-1])
        report.update(status='success', reason='candidate_plan_completed', completed=completed, final_metrics=row)
    except Exception as error:
        report.update(status='stopped', reason=str(error), error_type=type(error).__name__)
    finally:
        if env is not None:
            # Evaluation after control has stopped; never used for action selection.
            from fromrealhand.tabletop.scene import ground_truth
            actual=ground_truth(env.sim)['mug']
            evaluation=dict(ground_truth_after_stop=actual,used_for_control=False,
                actual_hand_qpos=env.sim.data.qpos[:30].tolist(),final_contacts=env.contacts())
            if pose is not None:
                evaluation['visual_position_error_m']=float(np.linalg.norm(np.asarray(actual)[:3,3]-pose[:3,3]))
            if adapter is not None:
                index=min(cursor,len(adapter.actions)-1)
                desired=(contact_controller.reference(index,env.sim)[0] if contact_controller is not None
                    else adapter.desired_qpos(index))
                evaluation['desired_hand_qpos']=desired.tolist()
                evaluation['root_tracking_error']= (env.sim.data.qpos[:6]-desired[:6]).tolist()
            write(output/'evaluation_only.json',evaluation)
        if vision: vision.close()
        if reference: reference.env.close()
        report.update(steps=len(rows),reference_step=cursor, candidate_not_validated=True, output=str(output), events=events,
            wall_s=time.monotonic()-started, state_writes_during_execution=0,
            object_pose_input='rgbd_only', not_dapg_policy=True, perception_mode=cfg['perception_mode'],
            max_hand_scene_penetration_m=peak, attempted_action_count=len(actions),
            policy_mode=learner.mode if learner else 'expert',
            checkpoint=str(args.checkpoint) if args.checkpoint else None,
            learned_action_calls=learner.calls if learner else 0,
            clipped_action_calls=learner.clipped_calls if learner else 0,
            partial_final_action_possible=len(actions)>len(rows),contact_tracking=contact_mode)
        write(output/'report.json',report); write(output/'trace.json',rows)
        np.save(str(output/'actions.npy'), np.asarray(actions))
        emit('result',report=report)
        import glfw
        glfw.terminate()


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True); p.add_argument('--output',type=Path,required=True)
    p.add_argument('--visual-root',type=Path,default=Path('/media/smgbro/shared/visual_grasp'))
    p.add_argument('--seed',type=int,default=0)
    p.add_argument('--allow-unvalidated-tabletop',action='store_true')
    p.add_argument('--checkpoint',type=Path)
    args=p.parse_args()
    try: execute(args)
    except Exception as error:
        emit('error',message=str(error),error_type=type(error).__name__); raise SystemExit(1)
