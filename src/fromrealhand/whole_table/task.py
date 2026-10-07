"""Full-table task planning with isolated candidate rollouts and one live execution."""
import copy
import json
import time
from pathlib import Path
import numpy as np
from .layout import sample
from .scene import object_catalog
from .local_adapter import MotionBridge,translate_initial_scene,rotate_initial_grasp
from .navigation_runner import navigate,capture
from .loaded_motion import carry,margin_conflicts
from ..tabletop.random_task import RandomTask


def load_config(path):
    path=Path(path);cfg=json.loads(path.read_text())
    base=load_config(path.parent/cfg['base_config']) if 'base_config' in cfg else {}
    base.update(cfg)
    return base


def validate_settings(config,goal=None,speed=1.,clearance=.025):
    if not np.isfinite([speed,clearance]).all() or not .5<=speed<=1. or not .025<=clearance<=.04:
        raise ValueError('Speed must be 0.5-1.0; planning clearance must be 25-40 mm')
    if goal is not None:
        g=np.asarray(goal,dtype=float)
        if g.shape!=(3,) or not np.isfinite(g).all() or np.any(g<config['goal_min_m']) or np.any(g>config['goal_max_m']):
            raise ValueError('Goal outside registered navigation workspace')


def make_layout(seed,config,goal=None,count=None,cup_xy=None,catalog=None):
    validate_settings(config,goal)
    rng=np.random.RandomState(seed)
    if count is None:count=int(rng.randint(*[config['distractor_count_range'][0],config['distractor_count_range'][1]+1]))
    result=sample(seed,object_catalog() if catalog is None else catalog,config,count,cup_xy)
    target=rng.uniform(config['goal_min_m'],config['goal_max_m']) if goal is None else np.asarray(goal)
    result.update(goal_world_m=target.tolist(),table_rgba=[.88,.90,.91,1.],
                  version=config['version'],cup_model='025_mug',object_scale=.8)
    return result


def brief(value):
    if isinstance(value,dict):
        return {k:brief(v) for k,v in value.items() if k not in ('trace','hand_envelope_relative_m','obstacles')}
    if isinstance(value,list):return [brief(v) for v in value]
    return value


class NavigationTask:
    def __init__(self,root,checkpoint,config):
        self.root,self.checkpoint=Path(root),Path(checkpoint)
        self.config=load_config(config);self.tasks={}

    def close(self):
        for task in self.tasks.values():task.close()

    def _candidate(self,video,layout,output,stop_skill,config,observer=None,cancelled=None,entry_frame=0,yaw_deg=0):
        from hierarchy_common import write
        if video not in self.tasks:self.tasks[video]=RandomTask(self.root,video,self.checkpoint,self.root/'configs/tabletop-random-v6.json')
        task=self.tasks[video];output=Path(output);output.mkdir(parents=True,exist_ok=False)
        xy=np.asarray(next(o['xy'] for o in layout['objects'] if o['name']=='mug'));delta=np.r_[xy,0.]
        local_layout=copy.deepcopy(layout);local_layout['goal_world_m']=[xy[0],xy[1],.16]
        goal=np.asarray(layout['goal_world_m']);receipt={};phase=['navigate']

        def check_stop():
            if cancelled and cancelled():raise RuntimeError('user_stop')

        def stage(name):
            check_stop();phase[0]=name
            if observer:observer.stage(name)

        def nav_frame(bridge,report,row):
            check_stop()
            if observer:observer.frame(bridge.env,phase[0])

        def setup(env,mesh,scene,poses):
            check_stop()
            poses=translate_initial_scene(env,delta,poses,scene,translate_object=False)
            if yaw_deg:poses=rotate_initial_grasp(env,poses,yaw_deg,scene)
            if entry_frame:
                import transforms3d
                from ..tabletop.functional_reference import prepare_reference
                m,d=env.sim.model,env.sim.data;bid=m.body_name2id('mug_0')
                initial=np.eye(4);initial[:3,:3]=d.body_xmat[bid].reshape(3,3);initial[:3,3]=d.body_xpos[bid]
                b=m.body_name2id('forearm');base=np.eye(4);base[:3,3]=m.body_pos[b]
                base[:3,:3]=transforms3d.quaternions.quat2mat(m.body_quat[b])
                adapter,_,_=prepare_reference(task.actions,task.qpos,poses,initial,base,
                    dict(joint_range=m.jnt_range[:30],gain=m.actuator_gainprm[:,0],bias=m.actuator_biasprm,
                         action_range=env.rng,reference_base=env.reference_base),task.config,task.entry['dt'],
                    dict(task.profile,goal_world_m=local_layout['goal_world_m']),task.entry['segments'],
                    next(s['stop'] for s in task.entry['segments'] if s['skill']=='lift'))
                # Select the initial hand preshape only; navigation reaches this pose physically.
                d.qpos[:30]=adapter.desired_qpos(entry_frame);d.qvel[:30]=0.;env.sim.forward()
                receipt['entry_reference_step']=entry_frame
            bridge=MotionBridge(env,config)
            target=bridge.position()+[0,0,config['pregrasp_hover_m'][video]]
            scene['initial_hand_repositioned']=True
            env.sim.data.qpos[bridge.tcols]=bridge.to_joints(config['home_m'])
            env.sim.data.qvel[:30]=0.;env.sim.forward();bridge.target=np.asarray(config['home_m'])
            bridge.hand_envelope=bridge.hand_shapes()-bridge.position()
            requested=target+config['approach_offset_m']
            try:
                for _ in range(int(config['settling_s']/config['timestep_s'])):
                    check_stop();bridge.step(bridge.target)
                if observer:observer.attach(env,mesh,goal,layout,video,task.entry)
                stage('navigate')
                requested=target+config['approach_offset_m']
                receipt['navigation']=navigate(bridge,requested,nav_frame)
                if not receipt['navigation']['passed']:raise RuntimeError('navigation_acceptance_failed')
                bridge.config=dict(config,clearance_m=config['approach_clearance_m'],
                    execution_clearance_m=config['approach_execution_clearance_m'],
                    max_velocity_m_s=[.025,.025,.02],max_acceleration_m_s2=[.05,.05,.04],max_jerk_m_s3=[.15,.15,.12])
                bridge.hand_envelope=bridge.hand_shapes()-bridge.position();stage('approach')
                requested=target
                receipt['approach']=navigate(bridge,target,nav_frame)
                if not receipt['approach']['passed']:raise RuntimeError('approach_acceptance_failed')
                m,d=env.sim.model,env.sim.data
                env.motor_handoff=dict(start=float(d.time),duration=1.,gain=m.actuator_gainprm[:,0].copy(),
                    bias=m.actuator_biasprm[:,:3].copy(),ctrl=d.ctrl.copy())
            except ValueError as error:
                receipt['failure']=dict(reason=str(error),phase=phase[0],goal=requested.tolist(),
                    conflicts=margin_conflicts(bridge,requested,bridge.config['clearance_m']))
                raise
            finally:
                bridge.restore();write(output/'adapter.json',receipt)
            return poses

        def local_frame(env,row,action):
            check_stop()
            if phase[0]!=row['phase']:stage(row['phase'])
            if observer:observer.frame(env,phase[0],row,action)

        def finish(env,controller,rows,initial,local_goal):
            if stop_skill!='transport':return dict(passed=True,scope='partial skill request')
            stage('transport')
            result=carry(env,controller.last_action,goal,config,nav_frame,cancelled=cancelled)
            write(output/'carry.json',result)
            if observer:
                observer.frame(env,'transport',force=True)
                if result['passed']:
                    view=MotionBridge(env,config,held_action=controller.last_action)
                    try:capture(view,output/'carry.png')
                    finally:view.restore()
            if not result['passed']:raise RuntimeError('carry_acceptance_failed:'+result.get('reason','gate_failed'))
            return result

        report=task.run(layout['seed'],output/'local',stop_skill='lift' if stop_skill=='transport' else stop_skill,
            scene_layout=local_layout,scene_adapter=setup,completion=finish,
            step_callback=local_frame,cancelled=cancelled,start_reference_step=entry_frame)
        report['layout']=copy.deepcopy(layout);report['adapter']=brief(receipt)
        report['continuation']=brief(report.get('continuation',{}))
        if (output/'carry.json').exists():report['carry']=brief(json.loads((output/'carry.json').read_text()))
        write(output/'summary.json',brief(report))
        return report

    def run(self,layout,output,preferred='first',mode='auto',stop_skill='transport',
            speed=1.,clearance=.025,observer=None,cancelled=None):
        from hierarchy_common import write
        output=Path(output);output.mkdir(parents=True,exist_ok=False)
        validate_settings(self.config,layout['goal_world_m'],speed,clearance)
        if preferred not in ('first','second') or mode not in ('auto','fixed') or stop_skill not in ('reach','grasp','lift','transport'):
            raise ValueError('Invalid grasp selection or requested skill')
        cfg=dict(self.config,clearance_m=clearance,
                 max_velocity_m_s=(np.array(self.config['max_velocity_m_s'])*speed).tolist(),
                 carry_speed_scale=speed)
        write(output/'request.json',dict(layout=layout,preferred=preferred,mode=mode,stop_skill=stop_skill,
                                       speed=speed,clearance_m=clearance,config=cfg))
        videos=[preferred]+([v for v in cfg['candidate_videos'] if v!=preferred] if mode=='auto' else [])
        attempts=[];selected=None;selected_entry=0;selected_yaw=0;began=time.monotonic()
        candidates=[(yaw,f,v) for yaw in cfg.get('grasp_yaws_deg',[0])
                    for f in (cfg.get('entry_frames',[0]) if yaw==0 else [cfg.get('entry_frames',[0])[-1]]) for v in videos]
        for yaw_deg,entry_frame,video in candidates:
            if cancelled and cancelled():break
            if observer:observer.status('候选预演：%s / 帧 %s / 方位 %s°；尚未执行'%(video,entry_frame,yaw_deg))
            name='preview-'+video+('-'+str(entry_frame) if entry_frame else '')+('-yaw'+str(yaw_deg) if yaw_deg else '')
            trial=self._candidate(video,layout,output/name,stop_skill,cfg,cancelled=cancelled,entry_frame=entry_frame,yaw_deg=yaw_deg)
            attempts.append(dict(video=video,passed=trial['passed'],reason=trial['reason'],
                                 entry_frame=entry_frame,yaw_deg=yaw_deg,wall_s=trial['wall_s'],steps=trial['steps']))
            write(output/'planning.json',dict(attempts=attempts,selected=video if trial['passed'] else None))
            if trial['passed']:selected=video;selected_entry=entry_frame;selected_yaw=yaw_deg;break
        planning_s=time.monotonic()-began
        if cancelled and cancelled():
            report=dict(status='stopped',reason='user_stop',passed=False,steps=0,actual_executions=0)
        elif selected is None:
            report=dict(status='stopped',reason='no_feasible_grasp_candidate',passed=False,steps=0,actual_executions=0)
        else:
            if observer:observer.status('预演通过，开始执行：'+selected)
            if observer:observer.settings.update(entry_reference_step=selected_entry,grasp_yaw_deg=selected_yaw)
            report=self._candidate(selected,layout,output/'execution',stop_skill,cfg,observer,cancelled,entry_frame=selected_entry,yaw_deg=selected_yaw)
            report['actual_executions']=1
        report.update(planning_attempts=attempts,selected_video=selected,preferred_video=preferred,
            selected_entry_frame=selected_entry if selected is not None else None,
            selected_yaw_deg=selected_yaw if selected is not None else None,
            planning_seconds=planning_s,layout=layout,speed_scale=speed,planning_clearance_m=clearance,
            total_wall_s=time.monotonic()-began,execution_resets=0,planning_model='same MuJoCo dynamics in isolated previews',
            success_scope='model-based candidate selection plus learned local grasp; not raw policy generalization')
        write(output/'summary.json',brief(report))
        return report
