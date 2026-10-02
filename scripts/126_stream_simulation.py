#!/usr/bin/env python3
"""Isolated legacy MuJoCo process: frames, telemetry, camera commands and bounded stop."""
import argparse
import base64
import contextlib
import copy
import ctypes
import hashlib
import json
from pathlib import Path
import queue
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from fromrealhand.desktop.planning import validated_plan
from fromrealhand.desktop.rendering import stream_context
from fromrealhand.desktop.runtime import emit


class Controls:
    def __init__(self):
        self.stop = threading.Event(); self.camera = queue.Queue(maxsize=2)
        threading.Thread(target=self.read,daemon=True).start()

    def read(self):
        for line in sys.stdin:
            try:
                value = json.loads(line)
                if value.get('command')=='stop': self.stop.set()
                elif value.get('command')=='camera':
                    if self.camera.full(): self.camera.get_nowait()
                    self.camera.put_nowait(value)
            except (ValueError,AttributeError,queue.Empty,queue.Full): pass
        self.stop.set()


def execute(args):
    # Verify intent again before importing/creating the simulator.
    plan = validated_plan(args.root,args.output)
    from hierarchy_common import ReferenceBackend, SkillExecutor, SkillRegistry, BackendStopped, verify_delivery, read, write
    import numpy as np
    import cv2
    from stage4_common import experiment,restore_once,measure
    from stage4_pipeline_common import load_pieces
    registry = SkillRegistry.load(ROOT/'configs/skill_registry.yaml')
    registry.validate_plan(plan)
    output = args.output/'physics'; output.mkdir(exist_ok=False)
    if plan['goal']=='stop':
        report = dict(status='stopped',reason='user_stop',steps=0,simulation_created=False,plan=plan)
        write(output/'report.json',report); emit('result',report=report); return
    controls = Controls()
    if controls.stop.is_set(): raise RuntimeError('Controller disconnected before initialization')

    class StreamBackend(ReferenceBackend):
        def __init__(self):
            # Match the frozen reference initializer; only context construction differs.
            run = verify_delivery(registry)
            manifest = read(run/'build/manifest.json')
            trajectory = registry.config['scenes'][plan['scene']]
            self.entry = next(e for e in manifest['trajectories'] if e['trajectory']==trajectory)
            self.pieces = load_pieces(run/'build',self.entry)
            self.contract_config = copy.deepcopy(manifest['config'])
            self.bounds = {s['skill']:(s['start'],s['stop']) for s in self.entry['segments']}
            self.actions = np.concatenate([p['actions'] for p in self.pieces])
            self.expected = np.concatenate([p['expected_post_states'] for p in self.pieces])
            self.cursor,self.audit_index,self.max_error = 0,0,0.
            self.stopped,self.fault,self.fired = False,None,False
            self.rows,self.executed_actions = [],[]
            self.render_enabled,self.context,self.last_row = False,None,None
            self.exp = experiment(self.entry['geometry'],self.pieces[0]); self.measure = measure
            try:
                self.context = stream_context(self.exp.env.sim)
                self.context.cam.type = 0
                with np.load(self.entry['geometry']) as geometry:
                    self.context.cam.lookat[:] = geometry['object_poses'][[0,-1],:3,3].mean(axis=0)
                self.context.cam.distance,self.context.cam.azimuth,self.context.cam.elevation = .55,135.,-30.
                restore_once(self.exp.env,self.pieces[0]['initial_snapshot'])
                self.initialization_count = 1
                self.started = time.monotonic(); self.metrics()
            except BaseException:
                self.exp.env.close(); raise
            self.active = None; self.last_frame = 0.; self.frame_count = 0
            self.frame_hashes = set(); self.pixel_std_min = float('inf'); self.last_image = None
            self.action_abs_max = 0.; self.camera_changes = 0
            gl = ctypes.CDLL('libGL.so.1'); gl.glGetString.restype = ctypes.c_char_p
            self.gl_info = {name:gl.glGetString(value).decode() for name,value in
                [('vendor',0x1F00),('renderer',0x1F01),('version',0x1F02)]}
            import mujoco_py
            self.gl_info['extension'] = mujoco_py.cymj.__file__

        def camera_input(self):
            while not controls.camera.empty():
                value = controls.camera.get_nowait()
                presets = {'front':(135.,-30.),'side':(215.,-20.),'top':(135.,-80.)}
                if value.get('preset') in presets:
                    self.context.cam.azimuth,self.context.cam.elevation = presets[value['preset']]
                zoom = value.get('zoom',100)
                if isinstance(zoom,(int,float)) and np.isfinite(zoom):
                    self.context.cam.distance = .55*100/max(60,min(160,zoom))
                self.camera_changes += 1

        def action(self,skill):
            if controls.stop.is_set(): raise BackendStopped('User stop')
            if skill!=self.active:
                self.active=skill; emit('stage',skill=skill,step=self.cursor)
            self.camera_input()
            return super().action(skill)

        def step(self,action):
            if controls.stop.is_set(): raise BackendStopped('User stop')
            row = super().step(action)
            self.action_abs_max = float(np.max(np.abs(action)))
            if time.monotonic()-self.last_frame>=1./15: self.draw()
            remaining = self.cursor*self.entry['dt']/args.speed-(time.monotonic()-self.started)
            if remaining>0: controls.stop.wait(remaining)
            return row

        def draw(self):
            before = np.r_[self.exp.env.sim.data.qpos,self.exp.env.sim.data.qvel].copy()
            self.context.render(960,600,camera_id=-1)
            frame = self.context.read_pixels(960,600,depth=False)[::-1].copy()
            if not np.array_equal(before,np.r_[self.exp.env.sim.data.qpos,self.exp.env.sim.data.qvel]):
                raise RuntimeError('Rendering changed simulation state')
            self.pixel_std_min = min(self.pixel_std_min,float(frame.std()))
            if frame.std()<5: raise RuntimeError('Blank simulation frame')
            self.last_image = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)
            ok,encoded = cv2.imencode('.jpg',self.last_image,[cv2.IMWRITE_JPEG_QUALITY,80])
            if not ok: raise RuntimeError('Frame encoding failed')
            self.frame_hashes.add(hashlib.sha256(encoded).hexdigest()); self.frame_count+=1
            self.last_frame = time.monotonic()
            emit('frame',jpeg=base64.b64encode(encoded).decode('ascii'),step=self.cursor,skill=self.active,
                sim_time_s=self.cursor*self.entry['dt'],metrics=self.last_row,
                max_penetration_m=max([r['scene_penetration_m'] for r in self.rows]+[self.last_row['scene_penetration_m']]),
                action_abs_max=self.action_abs_max,elapsed_s=self.last_frame-self.started)

    with contextlib.redirect_stdout(sys.stderr): backend = StreamBackend()
    # Legacy initialization output stays off the structured stdout channel.
    emit('ready',gl=backend.gl_info,bounds=backend.bounds,total_steps=backend.bounds[plan['goal']][1],
         backend='verified_reference',dt=backend.entry['dt'])
    try:
        backend.draw()
        report = SkillExecutor(registry).run(plan,backend)
        backend.draw()
        report.update(physics=backend.summary(),simulation_created=True,
            render=dict(gl=backend.gl_info,frames=backend.frame_count,unique_frames=len(backend.frame_hashes),
                pixel_std_min=backend.pixel_std_min,camera_changes=backend.camera_changes,render_state_writes=0),
            backend='verified_reference',not_dapg_policy=True)
        if backend.last_image is not None: cv2.imwrite(str(output/'final.jpg'),backend.last_image)
        np.savez_compressed(str(output/'trace.npz'),actions=np.asarray(backend.executed_actions),
            bottom_m=np.asarray([r['bottom_m'] for r in backend.rows]),
            penetration_m=np.asarray([r['scene_penetration_m'] for r in backend.rows]))
        write(output/'report.json',report); emit('result',report=report)
    finally:
        backend.close()
        import glfw
        glfw.terminate()


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--speed',type=float,default=1.)
    args = parser.parse_args()
    if not .5<=args.speed<=4.: parser.error('speed must be in [0.5, 4]')
    try: execute(args)
    except Exception as error:
        emit('error',message=str(error),error_type=type(error).__name__)
        raise SystemExit(1)
