"""Contact-gated table placement after a continuous physical carry."""
import numpy as np
from .placement_metrics import PlacementMetrics, stable_placement
from .navigation import NavigationRejected
from .navigation_runner import navigate, capture
from .local_adapter import MotionBridge
from .hand_scene import geom_bounds
from .navigation import inflated_boxes, segments_clear
from ..desktop.navigation_metrics import joint_limit_metrics


def placement_posture(bridge, metrics):
    """Choose a small wrist rotation with table clearance, using scratch FK only."""
    import mujoco_py
    from scipy.spatial.transform import Rotation
    from itertools import product
    m,d=bridge.sim.model,bridge.sim.data;cup=d.body_xpos[metrics.cup].copy()
    cup_points=metrics.vertices.dot(d.body_xmat[metrics.cup].reshape(3,3).T)
    boxes=[geom_bounds(bridge.sim,g) for g in bridge.hand_geoms if m.geom_bodyid[g]!=metrics.cup
           and (m.geom_contype[g] or m.geom_conaffinity[g])]
    points=np.array([[box[bits[0],0],box[bits[1],1],box[bits[2],2]]
                     for box in boxes for bits in product((0,1),repeat=3)])-cup
    up=d.body_xmat[metrics.cup].reshape(3,3).dot(metrics.up_local)
    options=[]
    for x in range(-20,21,2):
        for y in range(-20,21,2):
            delta=Rotation.from_euler('xy',[x,y],degrees=True).as_matrix()
            tilt=np.rad2deg(np.arccos(np.clip(delta.dot(up)[2],-1,1)))
            gap=(points.dot(delta.T))[:,2].min()-(cup_points.dot(delta.T))[:,2].min()
            if tilt<=9 and gap>=.004:options.append((x*x+y*y,-gap,delta))
    if not options:raise NavigationRejected('no_table_clearance_posture')
    options.sort(key=lambda row:row[:2]);delta=options[0][2]
    scratch=mujoco_py.MjSim(m);scratch.set_state(bridge.sim.get_state());scratch.forward()
    palm=m.body_name2id('palm');desired=delta.dot(d.body_xmat[palm].reshape(3,3))
    for _ in range(40):
        R=scratch.data.body_xmat[palm].reshape(3,3)
        error=Rotation.from_matrix(desired.dot(R.T)).as_rotvec()
        if np.linalg.norm(error)<1e-5:break
        J=scratch.data.get_body_jacr('palm').reshape(3,m.nv)[:,3:6]
        scratch.data.qpos[3:6]+=np.linalg.lstsq(J,error,rcond=None)[0]
        scratch.forward()
    target=scratch.data.qpos[3:6].copy()
    if np.any(target<m.jnt_range[3:6,0]) or np.any(target>m.jnt_range[3:6,1]):
        raise NavigationRejected('placement_wrist_range')
    return delta,target


def release_posture(bridge, metrics):
    """Separate actual contact patches along their outward normals; keep wrists fixed."""
    import mujoco_py
    from scipy.optimize import least_squares
    m,d=bridge.sim.model,bridge.sim.data
    scratch=mujoco_py.MjSim(m);scratch.set_state(bridge.sim.get_state());scratch.forward()
    anchors=[]
    for c in d.contact[:d.ncon]:
        a,b=int(m.geom_bodyid[c.geom1]),int(m.geom_bodyid[c.geom2])
        if metrics.cup not in (a,b):continue
        hand=b if a==metrics.cup else a
        if hand not in metrics.hand:continue
        normal=np.array(c.frame[:3])*(1 if a==metrics.cup else -1)
        point=np.array(c.pos)
        local=d.body_xmat[hand].reshape(3,3).T.dot(point-d.body_xpos[hand])
        anchors.append((hand,local,point+normal*.012+[0,0,.003]))
    if not anchors:raise NavigationRejected('release_has_no_contact_anchors')
    start=d.qpos[8:30].copy();bounds=m.jnt_range[8:30].copy()
    feasible=(-m.actuator_biasprm[8:30,0,None]-m.actuator_gainprm[8:30,0,None]*m.actuator_ctrlrange[8:30])/m.actuator_biasprm[8:30,1,None]
    bounds[:,0]=np.maximum(bounds[:,0],feasible.min(1)+1e-5)
    bounds[:,1]=np.minimum(bounds[:,1],feasible.max(1)-1e-5)
    def residual(q):
        scratch.data.qpos[8:30]=q;scratch.forward()
        error=[scratch.data.body_xmat[b].reshape(3,3).dot(local)+scratch.data.body_xpos[b]-target
               for b,local,target in anchors]
        return np.r_[np.asarray(error).ravel(),.002*(q-start)]
    fit=least_squares(residual,np.clip(start,bounds[:,0]+1e-7,bounds[:,1]-1e-7),
                      bounds=(bounds[:,0],bounds[:,1]),max_nfev=100,ftol=1e-6)
    target=d.qpos[:30].copy();target[8:30]=fit.x
    return target,dict(anchor_count=len(anchors),cost=float(fit.cost),converged=bool(fit.success))


def place_and_return(bridge, goal_xy, home, config, output, stage=None, on_step=None, cancelled=None):
    from hierarchy_common import write
    metrics = PlacementMetrics(bridge.env)
    initial = metrics.read()
    report = dict(passed=False, reason='placement_diagnostic_only', initial=initial,
        home_m=np.asarray(home).tolist(), object_pose_writes_during_execution=0, object_forces_applied=False)
    cfg=config['placement'];dt=config['timestep_s'];trace=[];phase=['preplace'];peak=0.;placed=False
    motion_peak=dict(speed_m_s=0.,acceleration_m_s2=0.,tracking_m=0.)
    phase_acceleration={}
    def change(name):
        phase[0]=name
        if stage:stage(name)
    def tick(active, target):
        nonlocal peak
        if cancelled and cancelled():raise NavigationRejected('user_stop')
        active.step(target);q=metrics.read();peak=max(peak,q['penetration_m'])
        motion_peak['speed_m_s']=max(motion_peak['speed_m_s'],float(np.max(np.abs(active.velocity()))))
        motion_peak['acceleration_m_s2']=max(motion_peak['acceleration_m_s2'],float(np.max(np.abs(active.acceleration()))))
        phase_acceleration[phase[0]]=max(phase_acceleration.get(phase[0],0.),float(np.max(np.abs(active.acceleration()))))
        tracking=float(np.linalg.norm(active.position()-target));motion_peak['tracking_m']=max(motion_peak['tracking_m'],tracking)
        limits=joint_limit_metrics(active.sim.data.qpos,active.sim.model.jnt_range,active.sim.model.jnt_range[:3])
        if tracking>.015 or limits['joint_violation_rad']>.02 or motion_peak['acceleration_m_s2']>.5:
            report['motion_failure']=dict(tracking_m=tracking,joint_limit=limits,
                acceleration_m_s2=active.acceleration().tolist(),
                limiter=getattr(active,'acceleration_limit_audit',None))
            raise NavigationRejected('placement_motion_limits')
        if active.steps%20==0:
            trace.append(dict(time_s=float(active.sim.data.time),phase=phase[0],**q))
            if on_step:on_step(active,{},q)
        if q['illegal_contacts'] or q['penetration_m']>.001:
            raise NavigationRejected('placement_collision_or_penetration')
        if q['tilt_deg']>(max(15.,initial['tilt_deg']+2) if phase[0]=='preplace' else 15):
            raise NavigationRejected('placement_tilt')
        return q
    try:
        if cfg.get('diagnostic_only'):return report
        change('preplace')
        if initial['tilt_deg']>30:raise NavigationRejected('preplace_tilt_outside_correction_range')
        bounds=np.array(initial['cup_bounds_m']);lo,hi=metrics.table_bounds
        if np.any(bounds[0,:2]<lo[:2]+.005) or np.any(bounds[1,:2]>hi[:2]-.005):
            raise NavigationRejected('placement_outside_table')
        # The palm may approach below the transit floor; physical contacts remain checked.
        bridge.config=dict(bridge.config,workspace_min_m=[config['workspace_min_m'][0],config['workspace_min_m'][1],-.1])
        delta,wrist=placement_posture(bridge,metrics)
        from scipy.spatial.transform import Rotation
        rotvec=Rotation.from_matrix(delta).as_rotvec();start_wrist=bridge.posture_target[3:6].copy()
        cup=bridge.sim.data.body_xpos[metrics.cup].copy();offset=bridge.position()-cup
        for i in range(int(4./dt)):
            t=min(1.,(i+1)*dt/4.);a=t*t*t*(10+t*(-15+6*t))
            bridge.posture_target[3:6]=start_wrist+(wrist-start_wrist)*a
            bridge.origin=bridge.position()-bridge.basis.dot(bridge.sim.data.qpos[bridge.tcols])
            tick(bridge,cup+Rotation.from_rotvec(rotvec*a).as_matrix().dot(offset))
        for _ in range(int(1./dt)):tick(bridge,bridge.target)
        report['posture_adjustment_rad']=rotvec.tolist();report['aligned']=metrics.read()
        bridge.hand_envelope=bridge.hand_shapes()-bridge.position()
        destination=bridge.position()+[0,0,-metrics.read()['bottom_gap_m']]
        obstacles=[o for o in bridge.obstacles() if o['name']!='table_collision']
        if not segments_clear(bridge.position(),destination,inflated_boxes(bridge.hand_envelope,obstacles,.002))[0]:
            raise NavigationRejected('placement_descent_blocked')
        change('lower');target=bridge.target.copy();streak=0.;q=initial
        lower_timeout=max(cfg['lower_timeout_s'],metrics.read()['bottom_gap_m']/cfg['lower_speed_m_s']+15.)
        report['lower_timeout_s']=lower_timeout
        for _ in range(int(lower_timeout/dt)):
            if q['bottom_gap_m']<.003 and not bridge.config.get('command_acceleration_limit_m_s2'):
                bridge.config=dict(bridge.config,command_acceleration_limit_m_s2=.30)
            speed=min(cfg['lower_speed_m_s'],.0005 if q['bottom_gap_m']<.005 else cfg['lower_speed_m_s'])
            target[2]-=speed*dt
            q=tick(bridge,target)
            streak=streak+dt if q['table_weight_ratio']>.2 else 0.
            if streak>=.1:break
        else:raise NavigationRejected('lower_timeout')
        change('settle');wrench=tuple(v.copy() for v in bridge.payload_wrench);streak=0.
        for i in range(int(cfg['settle_timeout_s']/dt)):
            alpha=min(1.,(i+1)*dt/2.)
            bridge.payload_wrench=tuple(v*(1-alpha) for v in wrench)
            target[2]-=np.clip((.95-q['table_weight_ratio'])*.001,-.0005,.0005)*dt
            q=tick(bridge,target)
            if alpha>=1 and .85<q['table_weight_ratio']<1.1 and q['linear_speed_m_s']<.01:streak+=dt
            else:streak=0.
            if streak>=.3:break
        else:raise NavigationRejected('table_support_not_confirmed')
        capture(bridge,output/'place-supported.png')
        change('release');held=bridge.held_ctrl.copy();m=bridge.sim.model
        bridge.config=dict(bridge.config,command_acceleration_limit_m_s2=.35)
        m.actuator_biasprm[6:30,2]-=2.
        opened,report['release_ik']=release_posture(bridge,metrics)
        opened=(-m.actuator_biasprm[:,0]-m.actuator_biasprm[:,1]*opened)/m.actuator_gainprm[:,0]
        opened[6:8]+=bridge.sim.data.qfrc_bias[6:8]/m.actuator_gainprm[6:8,0]
        n=int(cfg['release_seconds']/dt)
        for i in range(n):
            t=(i+1)/n;alpha=t*t*t*(10+t*(-15+6*t))
            bridge.held_ctrl[6:]=held[6:]*(1-alpha)+opened[6:]*alpha
            q=tick(bridge,target)
        for _ in range(int(.5/dt)):q=tick(bridge,target)
        capture(bridge,output/'place-released.png')
        if q['table_weight_ratio']<.5:raise NavigationRejected('release_support_lost')
        if q['hand_cup_contacts']:raise NavigationRejected('release_contact_not_cleared')
        # Empty-hand control and obstacle geometry now include the released cup.
        released_action=(bridge.held_ctrl-bridge.env.mid)/bridge.env.rng
        empty=MotionBridge(bridge.env,bridge.config,held_action=released_action)
        try:
            change('retreat')
            away=empty.position()[:2]-np.asarray(q['cup_position_m'])[:2]
            away/=max(np.linalg.norm(away),1e-6)
            displacement=np.r_[away*.06,cfg['retreat_m']]
            report['retreat_displacement_m']=displacement.tolist()
            # Preserve released finger torques while withdrawing away from the rim.
            start=empty.target.copy()
            duration=max(8.,np.linalg.norm(displacement)/.015)
            n=int(duration/dt)
            for i in range(n):
                t=(i+1)/n;alpha=t*t*t*(10+t*(-15+6*t))
                target=start+displacement*alpha
                q=tick(empty,target)
                if q['hand_cup_force_n']>.05:raise NavigationRejected('retreat_cup_contact')
            empty.hand_envelope=empty.hand_shapes()-empty.position()
            change('verify');rows=[]
            for _ in range(int(cfg['verify_seconds']/dt)):rows.append(tick(empty,empty.target))
            verified=stable_placement(rows,goal_xy);report['stability']=verified
            if not verified['passed']:raise NavigationRejected('placement_stability_failed')
            placed=True;capture(empty,output/'place-retreated.png')
            change('return_home')
            def navigation_step(active,result,row):
                if cancelled and cancelled():raise NavigationRejected('user_stop')
                if on_step:on_step(active,result,row)
            # Far from the cup, freeze the settled open posture for a fixed navigation envelope.
            returning=MotionBridge(bridge.env,empty.config)
            returning.handoff_seconds=2.
            try:
                for _ in range(int(2./dt)):tick(returning,returning.target)
                returning.hand_envelope=returning.hand_shapes()-returning.position()
                result=navigate(returning,home,navigation_step)
                write(output/'return-trace.json',result.get('trace',[]))
                report['return_home']={k:v for k,v in result.items() if k not in ('trace','obstacles','hand_envelope_relative_m')}
                if not result['passed']:raise NavigationRejected('return_home_failed')
                rows=[]
                for _ in range(int(1./dt)):rows.append(tick(returning,returning.target))
                final=stable_placement(rows,goal_xy);report['after_return']=final
                if not final['passed']:raise NavigationRejected('cup_unstable_after_return')
                if np.linalg.norm(returning.position()-home)>.003:raise NavigationRejected('home_error')
                capture(returning,output/'place-home.png')
            finally:
                if hasattr(returning,'last_navigation'):
                    result=returning.last_navigation
                    write(output/'return-trace.json',result.get('trace',[]))
                    report['return_home']={k:v for k,v in result.items() if k not in ('trace','obstacles','hand_envelope_relative_m')}
                returning.restore()
        finally:empty.restore()
        report.update(passed=True,reason='placed_and_returned')
    except (NavigationRejected,ValueError,RuntimeError) as error:
        report.update(reason=str(error),failure_phase=phase[0],error_type=type(error).__name__)
        capture(bridge,output/'place-failure.png')
    finally:
        report.update(placement_passed=placed,final=metrics.read(),max_penetration_m=peak,
            final_hand_qpos=bridge.sim.data.qpos[:30].tolist(),motion_peak=motion_peak,
            phase_acceleration_m_s2=phase_acceleration)
        write(output/'placement.json',report);write(output/'placement-trace.json',trace)
    return report
