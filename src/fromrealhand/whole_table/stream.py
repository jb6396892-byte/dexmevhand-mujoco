"""Qt telemetry for the live execution only; candidate previews are not displayed as execution."""
import base64
import ctypes
import time
from pathlib import Path
import numpy as np
from ..desktop.rendering import stream_context
from ..tabletop.contact_control import opposing_contacts


class Observer:
    def __init__(self,callback,output,checkpoint,settings,realtime=True):
        self.callback=callback;self.output=Path(output);self.output.mkdir(parents=True,exist_ok=True)
        self.checkpoint=str(checkpoint);self.settings=settings;self.realtime=realtime
        self.context=None;self.last_draw=0.;self.started=time.monotonic();self.peak=0.;self.last_step=0

    def status(self,text):self.callback('status',dict(message=text))
    def stage(self,name):self.callback('stage',dict(skill=name))

    def attach(self,env,mesh,goal,layout,video,entry):
        self.context=stream_context(env.sim);self.mesh=mesh;self.goal=np.asarray(goal)
        self.context.cam.lookat[:]=[0,0,.12];self.context.cam.distance=1.65
        self.context.cam.azimuth=135;self.context.cam.elevation=-40
        self.context.vopt.geomgroup[2]=0;self.context.vopt.geomgroup[4]=0
        self.context.vopt.sitegroup[:]=0;self.context.vopt.sitegroup[5]=1
        self.clock=(time.monotonic(),env.sim.data.time)
        gl=ctypes.CDLL('libGL.so.1');gl.glGetString.restype=ctypes.c_char_p
        info={k:(gl.glGetString(v) or b'unknown').decode() for k,v in [('renderer',0x1F01),('version',0x1F02)]}
        if info['version']=='unknown':raise RuntimeError('Missing GL version')
        self.callback('ready',dict(bounds={s['skill']:[s['start'],s['stop']] for s in entry['segments']},
            total_steps=entry['segments'][-1]['stop'],gl=info))
        m,d=env.sim.model,env.sim.data;bid=m.body_name2id('mug_0');pose=np.eye(4)
        pose[:3,:3]=d.body_xmat[bid].reshape(3,3);pose[:3,3]=d.body_xpos[bid]
        self.callback('random_scene',dict(layout=layout,initial_known_pose=pose.tolist(),
            output=str(self.output.parent),checkpoint=self.checkpoint))
        self.callback('navigation_settings',dict(selected_video=video,**self.settings))
        self.frame(env,'navigate',force=True)

    def frame(self,env,phase,row=None,action=None,force=False):
        if self.context is None:return
        if self.realtime:
            time.sleep(max(0.,env.sim.data.time-self.clock[1]-(time.monotonic()-self.clock[0])))
        now=time.monotonic()
        if not force and now-self.last_draw<1./15:return
        import cv2
        m,d=env.sim.model,env.sim.data;bid=m.body_name2id('mug_0')
        q=dict(row) if row is not None else env.contacts()
        q.update(opposing_contacts(env.sim));q.pop('pairs',None)
        position=d.body_xpos[bid];rotation=d.body_xmat[bid].reshape(3,3)
        q.update(bottom_m=float((self.mesh['vertices']@rotation.T+position)[:,2].min()),
            target_distance_m=float(np.linalg.norm(position-self.goal)),cup_position_m=position.tolist(),
            palm_position_m=d.geom_xpos[m.geom_name2id('C_palm0')].tolist(),phase=phase)
        self.peak=max(self.peak,q['scene_penetration_m'])
        if row is not None:self.last_step=int(row['source_index'])+1
        m.site_pos[m.site_name2id('visual_goal')]=self.goal
        env.sim.forward()
        self.context.render(960,720)
        rgb=self.context.read_pixels(960,720,depth=False)[::-1].copy()
        if rgb.std()<5:raise RuntimeError('blank_frame')
        bgr=cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR)
        ok,buffer=cv2.imencode('.jpg',bgr,[cv2.IMWRITE_JPEG_QUALITY,80])
        if not ok:raise RuntimeError('frame_encoding_failed')
        if force:cv2.imwrite(str(self.output/(phase+'.png')),bgr)
        self.callback('frame',dict(jpeg=base64.b64encode(buffer).decode('ascii'),step=self.last_step,
            skill=phase,metrics=q,sim_time_s=float(d.time),max_penetration_m=self.peak,
            elapsed_s=now-self.started,action_abs_max=float(np.max(np.abs(action))) if action is not None else 0.,
            control_kind='learned' if action is not None else 'navigation_servo'))
        self.last_draw=now
