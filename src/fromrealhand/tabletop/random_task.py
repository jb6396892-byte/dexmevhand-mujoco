"""Known-initial-pose tasks. Shared by the batch evaluator and Qt worker."""
import hashlib
import copy
import json
import time
from pathlib import Path
import numpy as np
from .random_scene import sample
from .control_scene import create
from .contact_control import reference_contacts, ContactTracker, opposing_contacts, FINGERS
from .functional_reference import prepare_reference, upright_source_correspondence
from .training_inputs import features
from .learned_control import LearnedControl


class RandomTask:
    def __init__(self, root, video, checkpoint, protocol=None):
        from hierarchy_common import SkillRegistry, verify_delivery, read
        from stage4_pipeline_common import load_pieces
        from stage4_common import experiment
        self.root, self.video = Path(root), video
        registry = SkillRegistry.load(self.root/'configs/skill_registry.yaml')
        run = verify_delivery(registry)
        self.entry = next(e for e in read(run/'build/manifest.json')['trajectories']
                          if e['trajectory'] == registry.config['scenes'][video])
        self.pieces = load_pieces(run/'build', self.entry)
        self.reference = experiment(self.entry['geometry'], self.pieces[0])
        self.states = [s for p in self.pieces for s in p['sim_data']]
        self.local, self.poses = reference_contacts(self.reference.env.sim, self.states)
        self.profile = dict(read(self.root/'configs/tabletop-dual-v3-profiles.json')['profiles'][video])
        if self.profile['upright_correspondence']:
            self.local, self.poses, _ = upright_source_correspondence(self.local, self.poses, self.profile['anchor'])
        self.config = read(self.root/'configs/tabletop-control-candidate.json')
        self.protocol_path=Path(protocol) if protocol else self.root/'configs/tabletop-random-v5.json'
        self.protocol = read(self.protocol_path)
        self.config['max_translation_m']=self.protocol.get('max_anchor_translation_m',self.config['max_translation_m'])
        self.actions = np.concatenate([p['actions'] for p in self.pieces])
        self.qpos = np.asarray([s['qpos'][:30] for s in self.states])
        self.learner = LearnedControl(checkpoint)
        self.checkpoint = str(Path(checkpoint).resolve())

    def close(self):
        self.reference.env.close()

    def run(self, seed, output, goal=None, count=None, stop_skill='transport', callback=None,
            cancelled=None, realtime=False, screenshots=False, cup_xy=None,
            scene_adapter=None, completion=None, scene_fixtures=(), scene_layout=None, step_callback=None,
            start_reference_step=0):
        import transforms3d
        from hierarchy_common import write
        output = Path(output); output.mkdir(parents=True, exist_ok=False)
        layout = (sample(seed, self.protocol, goal, count, cup_xy) if scene_layout is None
                  else copy.deepcopy(scene_layout))
        write(output/'layout.json', layout)
        rows, events, completed = [], [], []
        peak, phase, index = 0., 'initialization', 0
        anchor_tips, predicted_pose = None, None
        env, context, controller = None, None, None
        started = time.monotonic(); last_draw = 0.
        old_calls, old_clips = self.learner.calls, self.learner.clipped_calls
        report = dict(status='stopped', reason='initialization_failed')
        goal = np.asarray(layout['goal_world_m'])
        bounds = {s['skill']:(s['start'],s['stop']) for s in self.entry['segments']}
        if stop_skill not in bounds: raise ValueError('Unknown requested skill')
        if type(start_reference_step) is not int or not bounds['reach'][0]<=start_reference_step<bounds['reach'][1]:
            raise ValueError('Reference entry must be inside the reach phase')
        try:
            profile = dict(self.profile, goal_world_m=goal.tolist())
            if self.video=='first': profile['installation_y']=.04
            installation=self.protocol.get('installation_offsets',{}).get(self.video,[-.01,profile['installation_y'],0.])
            env, mesh, scene = create(self.reference.env, self.pieces[0]['initial_snapshot'], seed,
                self.entry['dt'], installation,
                [0.,0.,profile['clearance']] if profile['clearance'] else None, layout=layout,fixtures=scene_fixtures)
            source_poses = self.poses
            if scene_adapter is not None:
                source_poses = scene_adapter(env, mesh, scene, source_poses)
                goal = np.asarray(layout['goal_world_m'])
                profile['goal_world_m'] = goal.tolist()
                write(output/'layout.json',layout)
            sim, m, d = env.sim, env.sim.model, env.sim.data
            mug = m.body_name2id('mug_0')
            # The only object pose supplied to the motor policy is this initial measurement.
            initial = np.eye(4); initial[:3,:3] = d.body_xmat[mug].reshape(3,3)
            initial[:3,3] = d.body_xpos[mug]
            write(output/'scene.json', scene)
            base = np.eye(4); b = m.body_name2id('forearm')
            base[:3,3] = m.body_pos[b]; base[:3,:3] = transforms3d.quaternions.quat2mat(m.body_quat[b])
            adapter, _, preflight = prepare_reference(self.actions, self.qpos, source_poses, initial, base,
                dict(joint_range=m.jnt_range[:30], gain=m.actuator_gainprm[:,0], bias=m.actuator_biasprm,
                     action_range=env.rng, reference_base=env.reference_base), self.config, self.entry['dt'],
                profile, self.entry['segments'], bounds[stop_skill][1])
            write(output/'preflight.json', preflight)
            write(output/'input.json',dict(initial_known_pose=initial.tolist(),goal_world_m=goal.tolist(),
                checkpoint=self.checkpoint,checkpoint_sha256=hashlib.sha256(Path(self.checkpoint).read_bytes()).hexdigest(),
                continuous_object_pose_for_action=False,
                initial_hand_repositioned=bool(scene.get('initial_hand_repositioned',False)),
                initialization_frame_translation_m=scene.get('local_frame_translation_m',[0,0,0])))
            controller = ContactTracker(adapter, self.local, kp=0., root_gain=profile['root_gain'], finger_gain=0.)
            if callback or screenshots:
                from fromrealhand.desktop.rendering import stream_context
                context = stream_context(sim)
                context.vopt.geomgroup[2]=0; context.vopt.geomgroup[4]=0; context.vopt.sitegroup[:]=0
                context.vopt.sitegroup[5]=1
            m.site_pos[m.site_name2id('visual_goal')] = goal; sim.forward()
            if callback:
                import ctypes
                gl=ctypes.CDLL('libGL.so.1'); gl.glGetString.restype=ctypes.c_char_p
                gl_info={key:(gl.glGetString(value) or b'unknown').decode()
                         for key,value in [('renderer',0x1F01),('version',0x1F02)]}
                if gl_info['version']=='unknown': raise RuntimeError('Missing GL version')
                callback('ready',dict(bounds=bounds,total_steps=bounds[stop_skill][1],dt=self.entry['dt'],
                    gl=gl_info))
                callback('random_scene',dict(layout=layout,initial_known_pose=initial.tolist(),output=str(output),
                    checkpoint=self.checkpoint))

            def audit():
                nonlocal peak
                if cancelled is not None and cancelled(): raise RuntimeError('user_stop')
                q = env.contacts(); peak=max(peak,q['scene_penetration_m'])
                if not q['finite']: raise RuntimeError('nonfinite_state')
                for key,limit in [('scene_penetration_m',.001),('joint_violation_rad',.02),('max_hand_speed',20.)]:
                    if q[key]>limit: raise RuntimeError('physics_safety_stop:'+key)
                if q['non_target_contacts']: raise RuntimeError('non_target_collision')

            def supported(q):
                return q['th_force_n']>.01 and sum(q[f+'_force_n']>.01 for f in FINGERS)>=3 and q['opposition']

            def metrics():
                q=env.contacts(); q.update(opposing_contacts(sim))
                # Live truth is used for contact/success evaluation and display, not action features.
                rot=d.body_xmat[mug].reshape(3,3); pos=d.body_xpos[mug]
                q.update(bottom_m=float((mesh['vertices'] @ rot.T+pos)[:,2].min()),
                    target_distance_m=float(np.linalg.norm(pos-goal)),cup_position_m=pos.tolist(),
                    metric_source='simulation_evaluation',phase=phase,step=len(rows)+1,source_index=index)
                desired=controller.reference(index,sim)[0]
                q['root_tracking_world_m']=(adapter.base[:3,:3] @ (desired[:3]-d.qpos[:3])).tolist()
                if predicted_pose is not None:
                    q['proprioceptive_cup_prediction_m']=predicted_pose[:3,3].tolist()
                    q['prediction_error_m']=float(np.linalg.norm(predicted_pose[:3,3]-pos))
                q.pop('pairs',None)
                return q

            def success(q):
                if phase=='reach': return any(q[f+'_force_n']>.01 for f in FINGERS)
                if phase=='grasp': return supported(q)
                if phase=='lift': return supported(q) and q['bottom_m']>=.05
                return supported(q) and q['bottom_m']>=.05 and q['target_distance_m']<=.02

            def draw(q, action, filename=None):
                nonlocal last_draw
                if context is None: return
                import cv2
                context.render(960,720,camera_id=m.camera_name2id('rgbd'))
                rgb=context.read_pixels(960,720,depth=False)[::-1].copy()
                if rgb.std()<5: raise RuntimeError('blank_frame')
                bgr=cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR)
                if filename: cv2.imwrite(str(output/filename),bgr)
                if callback:
                    import base64
                    ok,buf=cv2.imencode('.jpg',bgr,[cv2.IMWRITE_JPEG_QUALITY,75])
                    if not ok: raise RuntimeError('frame_encoding_failed')
                    callback('frame',dict(jpeg=base64.b64encode(buf).decode('ascii'),step=index+1,skill=phase,
                        sim_time_s=float(d.time),metrics=q,max_penetration_m=peak,
                        action_abs_max=float(np.max(np.abs(action))),elapsed_s=time.monotonic()-started))
                last_draw=time.monotonic()

            for segment in self.entry['segments']:
                phase=segment['skill']; streak=0; lost=0
                if callback: callback('stage',dict(skill=phase,step=segment['start']))
                if phase=='lift' and profile['stable_carry']:
                    controller.start_carry(sim,initial,goal,bounds['grasp'][1]-1,bounds['lift'][1]-1,len(self.actions))
                    anchor_tips=np.array([d.get_site_xpos('S_'+f+'tip').copy() for f in ('th','ff','mf')])
                    # Re-anchor the root servo at the measured grasp posture instead of
                    # freezing a dynamic feedforward command with a nonzero tracking offset.
                    q=d.qpos[:6]
                    equilibrium=(d.qfrc_bias[:6]-m.actuator_biasprm[:6,0]-m.actuator_biasprm[:6,1]*q)/m.actuator_gainprm[:6,0]
                    controller.carry['action'][:6]=(equilibrium-env.mid[:6])/env.rng[:6]
                required=max(segment['confirmation_steps'],50 if phase=='grasp' else 0)
                effective_start=max(segment['start'],start_reference_step)
                indices=list(range(effective_start,segment['stop']))+[segment['stop']-1]*profile['hold_steps']
                for j,index in enumerate(indices):
                    audit()
                    predicted_pose=initial.copy()
                    if anchor_tips is not None:
                        tips=np.array([d.get_site_xpos('S_'+f+'tip').copy() for f in ('th','ff','mf')])
                        predicted_pose[:3,3]+=np.mean(tips-anchor_tips,axis=0)
                    controller.action(index,sim,predicted_pose,index//10 if anchor_tips is not None else None)
                    desired,reference=controller.reference(index,sim)
                    feature=features(d.qpos[:30],d.qvel[:30],d.get_site_xpos('S_grasp'),predicted_pose,goal,self.video,
                        phase,index,bounds[phase],reference,desired)
                    action=self.learner.action(feature,reference,self.video,phase)
                    controller.last_action=action.copy()
                    env.step(action,audit)
                    q=metrics(); rows.append(q); streak=streak+1 if success(q) else 0
                    if step_callback:step_callback(env,q,action)
                    if phase in ('lift','transport'):
                        lost=0 if supported(q) else lost+1
                        if lost>=15: raise RuntimeError('persistent_support_loss')
                    if callback and time.monotonic()-last_draw>1/15: draw(q,action)
                    if realtime: time.sleep(self.entry['dt'])
                    if j>=segment['stop']-effective_start-1 and streak>=required: break
                if streak<required: raise RuntimeError('stage_not_confirmed:'+phase)
                completed.append(phase); events.append(dict(phase=phase,steps=len(rows),confirmation_steps=streak))
                draw(q,action,phase+'.png' if screenshots else None)
                if phase==stop_skill: break
            if completion is not None:
                report['continuation'] = completion(env, controller, rows, initial, goal)
            report.update(status='success',reason='random_task_completed')
        except Exception as error:
            report.update(status='stopped',reason=str(error),error_type=type(error).__name__)
            if env is not None:
                report['failure_contacts']=[dict(geom1=env.sim.model.geom_id2name(c.geom1),
                    geom2=env.sim.model.geom_id2name(c.geom2),distance_m=float(c.dist))
                    for c in env.sim.data.contact[:env.sim.data.ncon] if c.dist<0]
        finally:
            report.update(passed=report['status']=='success',video=self.video,seed=seed,steps=len(rows),phase=phase,
                completed=completed,events=events,final=rows[-1] if rows else None,max_penetration_m=peak,
                start_reference_step=start_reference_step,
                wall_s=time.monotonic()-started,layout=layout,checkpoint=self.checkpoint,
                learned_action_calls=self.learner.calls-old_calls,clipped_action_calls=self.learner.clipped_calls-old_clips,
                state_writes_during_execution=0,object_forces_applied=False,pose_input='initial_known_pose_only',
                live_truth_usage='stage termination, safety, evaluation and display; never motor action features')
            if completion is not None:
                report['pose_input']='local policy: initial known pose; continuation: see separate receipt'
                report['live_truth_usage']='local policy safety/evaluation; continuation separately audited'
            write(output/'report.json',report); write(output/'trace.json',rows)
            if callback: callback('result',dict(report=report))
            if context:
                import glfw
                glfw.terminate()
        return report
