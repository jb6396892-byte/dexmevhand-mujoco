"""Execute rest-to-rest safe segments through actuators; stop simulation on violation."""
import numpy as np
from .motion import rest_to_rest
from .navigation import plan, inflated_boxes, segments_clear, NavigationRejected


def navigate(scene, goal, on_step=None):
    cfg=scene.config
    report=plan(scene.position(),goal,scene.hand_envelope,scene.obstacles(),cfg)
    dt=cfg['timestep_s']; interval=max(1,int(round(cfg['control_period_s']/dt)))
    rows=[]; peak_v=np.zeros(3); peak_a=np.zeros(3)
    report['trace']=rows;scene.last_navigation=report
    peak_track=0.; peak_posture=0.; contacts=0; start_time=scene.sim.data.time
    for waypoint in report['waypoints'][1:]:
        duration,curve=rest_to_rest(scene.target,waypoint,cfg)
        for index in range(int(np.ceil((duration+.5)/dt))):
            target=curve(min((index+1)*dt,duration))
            scene.step(target)
            peak_v=np.maximum(peak_v,np.abs(scene.velocity()))
            peak_a=np.maximum(peak_a,np.abs(scene.acceleration()))
            track=float(np.linalg.norm(scene.position()-target)); peak_track=max(peak_track,track)
            audit=scene.contacts(); contacts+=audit['hand_environment_contacts']
            peak_posture=max(peak_posture,audit['posture_error_rad'])
            if contacts or track>cfg['tracking_tolerance_m'] or peak_posture>cfg['posture_tolerance_rad'] or not audit.get('payload_valid',True):
                report.update(passed=False,stop_audit=audit,stop_tracking_error_m=track,stop_center_m=scene.position().tolist())
                raise NavigationRejected('execution_paused: contact, tracking or posture violation')
            if index%interval==0:
                actual=scene.hand_shapes()-scene.position()
                boxes=inflated_boxes(actual,scene.obstacles(),cfg['execution_clearance_m'])
                if not segments_clear(scene.position(),scene.position(),boxes)[0]:
                    raise NavigationRejected('execution_paused: clearance violation')
                rows.append(dict(time_s=float(scene.sim.data.time),center_m=scene.position().tolist(),
                                 target_m=target.tolist(),tracking_error_m=track))
                if on_step: on_step(scene,report,rows[-1])
    error=float(np.linalg.norm(scene.position()-goal))
    checks=dict(endpoint=bool(error<cfg['position_tolerance_m']),no_environment_contact=contacts==0,
                velocity=bool(np.all(peak_v<=np.array(cfg['max_velocity_m_s'])+.005)),
                acceleration=bool(np.all(peak_a<=np.array(cfg['max_acceleration_m_s2'])+.02)))
    report.update(passed=all(checks.values()),checks=checks,
                  endpoint_error_m=error,max_tracking_error_m=peak_track,
                  max_velocity_m_s=peak_v.tolist(),max_acceleration_m_s2=peak_a.tolist(),
                  max_posture_error_rad=peak_posture,hand_environment_contacts=contacts,
                  simulation_seconds=float(scene.sim.data.time-start_time),trace=rows)
    return report


def capture(scene,path,report=None):
    from ..desktop.rendering import stream_context
    from PIL import Image
    context=stream_context(scene.sim)
    context.cam.lookat[:]=[0,0,.12]; context.cam.distance=1.65
    context.cam.azimuth=135; context.cam.elevation=-40
    context.vopt.geomgroup[2]=0; context.vopt.geomgroup[4]=0
    context.render(1200,900)
    pixels=context.read_pixels(1200,900,depth=False)[::-1]
    if np.std(pixels)<5: raise RuntimeError('Blank frame')
    Image.fromarray(pixels).save(str(path))
    return dict(pixel_std=float(np.std(pixels)),width=1200,height=900)
