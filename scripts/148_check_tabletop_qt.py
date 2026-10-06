#!/usr/bin/env python3
"""Run an actual Qt/model/physics case and preserve compact runtime evidence."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
from PySide6.QtCore import QTimer, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication
from fromrealhand.desktop.tabletop_window import TabletopWindow

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--scene',choices=['first','second'],default='second')
p.add_argument('--seed',type=int,default=0)
p.add_argument('--goal',choices=['reach','grasp','lift','transport'],default='transport')
p.add_argument('--instruction')
p.add_argument('--cancel-step',type=int)
p.add_argument('--locked',action='store_true')
p.add_argument('--checkpoint',type=Path)
p.add_argument('--random-mode',action='store_true')
p.add_argument('--target-world',type=float,nargs=3)
p.add_argument('--count',type=int)
a=p.parse_args(); a.output.mkdir(parents=True,exist_ok=False)
app=QApplication([]); app.setStyle('Fusion')
window_type=TabletopWindow
if a.random_mode:
    from fromrealhand.desktop.random_window import RandomTabletopWindow
    window_type=RandomTabletopWindow
w=window_type('/media/smgbro/shared/lora','/media/smgbro/shared/visual_grasp',not a.locked,a.checkpoint)
w.scene.setCurrentIndex(0 if a.scene=='first' else 1); w.seed.setValue(a.seed)
if a.random_mode:
    if a.target_world:
        w.target_mode.setCurrentIndex(1)
        for spin,value in zip(w.targets,a.target_world): spin.setValue(value*1000)
    if a.count is not None: w.count.setValue(a.count)
w.instruction.setPlainText(a.instruction or dict(reach='接近杯子',grasp='握住杯子',lift='抓起杯子',transport='把杯子搬到目标位置')[a.goal])
w.show(); state=dict(heartbeat=time.monotonic(),max_gap_s=0.,frames=set(),messages=[],cancelled_at=None,done=False)
original=w.message


def message(packet):
    state['messages'].append({k:v for k,v in packet.items() if k not in ('jpeg','images')})
    original(packet)


w.message=message


def heartbeat():
    now=time.monotonic(); state['max_gap_s']=max(state['max_gap_s'],now-state['heartbeat']); state['heartbeat']=now


def frame(packet):
    state['frames'].add(hashlib.sha256(packet['jpeg'].encode()).hexdigest())
    if a.cancel_step is not None and packet['step']>=a.cancel_step and state['cancelled_at'] is None:
        state['cancelled_at']=time.monotonic(); QTest.mouseClick(w.stop_button,Qt.LeftButton)


def finish(report):
    if state['done']: return
    state['done']=True
    def save():
        if not w.overlay.image.isNull(): w.views.setCurrentWidget(w.overlay)
        w.info_tabs.setCurrentWidget(w.parameters)
        if a.random_mode:
            w.sidebar_scroll.verticalScrollBar().setValue(0)
            w.sidebar_scroll.horizontalScrollBar().setValue(0)
        app.processEvents()
        w.grab().save(str(a.output/'qt-result.png'))
        if not w.frame.image.isNull(): w.frame.image.save(str(a.output/'physics-final.png'))
        alive=[]
        for pid in w.started_processes:
            try: os.kill(pid,0); alive.append(pid)
            except ProcessLookupError: pass
        result=dict(report=report,scene=a.scene,seed=a.seed,goal=a.goal,instruction=w.instruction.toPlainText(),
            random_mode=a.random_mode,requested_target=a.target_world,requested_count=a.count,
            frames_received=w.frames,unique_frames=len(state['frames']),max_gui_gap_s=state['max_gap_s'],
            surviving_workers=alive,language_run=str(w.output),physics_run=str(w.visual_output),
            gui_responsive=state['max_gap_s']<.5,actual_pipeline=True,pre_recorded_images=False,
            task_passed=report.get('status')=='success',
            cancellation_latency_s=None if state['cancelled_at'] is None else time.monotonic()-state['cancelled_at'])
        if a.random_mode and report.get('layout'):
            layout=report['layout']
            result['target_matches_ui']=(a.target_world is None or all(abs(x-y)<1e-8 for x,y in zip(layout['goal_world_m'],a.target_world)))
            result['count_matches_ui']=(a.count is None or layout['distractor_count']==a.count)
        (a.output/'summary.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
        (a.output/'messages.json').write_text(json.dumps(state['messages'],indent=2,ensure_ascii=False)+'\n')
        (a.output/'stderr-ui.log').write_text(w.log.toPlainText())
        print(json.dumps(result,ensure_ascii=False),flush=True)
        w.close(); app.exit(0)
    QTimer.singleShot(250,save)


def begin():
    if a.locked:
        disabled=not w.run_button.isEnabled(); w.start(True)
        finish(dict(status='locked' if disabled and w.process is None else 'error',
                    reason='default_lock_checked',steps=0,simulation_created=False)); return
    QTest.mouseClick(w.run_button,Qt.LeftButton)


def timeout():
    if state['done']: return
    w.stop()
    QTimer.singleShot(6500,lambda:finish(dict(status='error',reason='test_timeout',steps=0)))


timer=QTimer(); timer.timeout.connect(heartbeat); timer.start(50)
w.frame_received.connect(frame); w.run_finished.connect(finish)
QTimer.singleShot(200,begin); QTimer.singleShot(1000000,timeout)
raise SystemExit(app.exec())
