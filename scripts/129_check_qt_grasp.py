#!/usr/bin/env python3
"""Exercise the real Qt window, model and physics; no prerecorded frames or fake policy."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from PySide6.QtCore import Qt,QTimer
from PySide6.QtGui import QInputMethodEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication
from fromrealhand.desktop.window import GraspWindow

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--root',type=Path,default=Path('/media/smgbro/shared/lora'))
parser.add_argument('--output',type=Path,required=True)
parser.add_argument('--quick',action='store_true')
args=parser.parse_args(); args.output.mkdir(parents=True,exist_ok=False)
app=QApplication([]); app.setStyle('Fusion'); app.setQuitOnLastWindowClosed(False)
window=GraspWindow(args.root); window.show()
cases=[dict(name='second-transport',scene=1,instruction='把杯子搬到目标位置',status='success',goal='transport',capture=True)]
if not args.quick:
    cases += [dict(name='first-reach',scene=0,instruction='手先到杯子边上去',status='success',goal='reach',small=True),
        dict(name='cancel-running',scene=1,instruction='抓起杯子',status='stopped',cancel_step=200),
        dict(name='reject-handoff',scene=0,instruction='将杯子放到我手上',status='rejected',capture=True),
        dict(name='preview',scene=1,instruction='握住杯子',status='preview',preview=True),
        dict(name='explicit-stop',scene=0,instruction='停止',status='stopped',goal='stop'),
        dict(name='first-transport-repeat',scene=0,instruction='把杯子搬到目标位置',status='success',goal='transport'),
        dict(name='cancel-planning',scene=1,instruction='抓起杯子',status='stopped',cancel_planning=True),
        dict(name='close-running',scene=1,instruction='抓起杯子',status='stopped',cancel_step=200,close=True)]
state=dict(index=-1,results=[],frames=set(),heartbeat=time.monotonic(),gap=0.,cancel_time=None)
heartbeat=QTimer(); heartbeat.setInterval(50)


def tick():
    now=time.monotonic(); state['gap']=max(state['gap'],now-state['heartbeat']); state['heartbeat']=now


def begin():
    state['index']+=1
    if state['index']>=len(cases):
        report=dict(passed=all(x['passed'] for x in state['results']) and state['gap']<.5 and state.get('input_ok',False),cases=state['results'],
            max_heartbeat_gap_s=state['gap'],worker_pids=window.started_processes,chinese_input_event_passed=state.get('input_ok'),
            gui='PySide6 6.8.3 actual X11 desktop',frames_from='MuJoCo GPU worker, not video playback')
        for pid in window.started_processes:
            try: os.kill(pid,0); report['passed']=False; report.setdefault('surviving_workers',[]).append(pid)
            except ProcessLookupError: pass
        (args.output/'summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        print(json.dumps(report,ensure_ascii=False),flush=True); window.close(); app.exit(0 if report['passed'] else 1); return
    case=cases[state['index']]; state['frames']=set(); state['cancel_time']=None
    window.resize(1000,720) if case.get('small') else window.resize(1360,880)
    window.scene.setCurrentIndex(case['scene']); window.instruction.clear()
    event=QInputMethodEvent(); event.setCommitString(case['instruction']); app.sendEvent(window.instruction,event)
    state['input_ok']=window.instruction.toPlainText()==case['instruction']
    QTest.mouseClick(window.preview if case.get('preview') else window.run_button,Qt.LeftButton)
    state['disabled_during_run']=not window.run_button.isEnabled() and not window.scene.isEnabled()
    if case.get('cancel_planning'): QTimer.singleShot(200,lambda:QTest.mouseClick(window.stop_button,Qt.LeftButton))
    print('START',case['name'],flush=True)


def frame(message):
    state['frames'].add(hashlib.sha256(message['jpeg'].encode()).hexdigest())
    case=cases[state['index']]
    if window.frames==10:
        window.camera.setCurrentIndex(1); window.zoom.setValue(110)
    if window.frames==25:
        window.camera.setCurrentIndex(0); window.zoom.setValue(100)
    if case.get('cancel_step') and message['step']>=case['cancel_step'] and state['cancel_time'] is None:
        state['cancel_time']=time.monotonic()
        if case.get('close'): window.close()
        else: QTest.mouseClick(window.stop_button,Qt.LeftButton)


def finished(report):
    case=cases[state['index']]
    passed=report['status']==case['status'] and state['disabled_during_run'] and window.run_button.isEnabled()
    if case['status']=='success':
        p=report.get('physics',{}); r=report.get('render',{})
        passed=passed and report['plan']['goal']==case['goal'] and p.get('max_state_replay_error',1.)<=1e-7
        passed=passed and p.get('max_penetration_m',1.)<=.001 and p.get('state_writes_during_execution')==0
        passed=passed and window.frames>5 and len(state['frames'])>5 and r.get('pixel_std_min',0)>5
    if case['status']=='rejected':
        passed=passed and window.frames==0 and not (window.output/'physics').exists()
    if case.get('preview'): passed=passed and not (window.output/'physics').exists()
    if case.get('cancel_planning'): passed=passed and not (window.output/'physics').exists() and window.frames==0
    if case.get('cancel_step'):
        passed=passed and report.get('reason')=='cancelled' and report['steps']<window.total_steps
        passed=passed and time.monotonic()-state['cancel_time']<3
    if case.get('goal')=='stop': passed=passed and report.get('steps')==0 and not report.get('simulation_created',True)
    item=dict(case=case,passed=bool(passed),report=report,frames_received=window.frames,
        unique_received_frames=len(state['frames']),run=str(window.output),window_size=[window.width(),window.height()],
        frame_size=[window.frame.width(),window.frame.height()],image_size=[window.frame.image.width(),window.frame.image.height()])
    state['results'].append(item)
    def capture_and_continue():
        widgets=[]
        def collect(layout):
            for i in range(layout.count()):
                item=layout.itemAt(i)
                if item.widget(): widgets.append(item.widget())
                elif item.layout(): collect(item.layout())
        collect(window.sidebar.layout())
        overlap=[(a.metaObject().className(),b.metaObject().className())
            for i,a in enumerate(widgets) for b in widgets[i+1:] if a.geometry().intersects(b.geometry())]
        item['sidebar_overlaps']=overlap
        item['sidebar_scroll_max']=window.sidebar_scroll.verticalScrollBar().maximum()
        item['status_current_run']=window.output.name in window.statusBar().currentMessage()
        item['log_current_run']=case['instruction'] in window.log.toPlainText() and report.get('reason',report['status']) in window.log.toPlainText()
        item['passed']=item['passed'] and not overlap and item['status_current_run'] and item['log_current_run']
        if case.get('small'): item['passed']=item['passed'] and item['sidebar_scroll_max']>0
        if not window.frames:
            item['passed']=item['passed'] and all(bar.value()==0 for bar in window.phase_bars.values())
            item['passed']=item['passed'] and all('#53606b' in label.styleSheet() for label in window.phase_labels.values())
        if case.get('capture') or case.get('small'): window.grab().save(str(args.output/(case['name']+'.png')))
        if case.get('close') and window.isVisible(): item['passed']=False
        print('FINISH',case['name'],item['passed'],report.get('reason'),flush=True); QTimer.singleShot(200,begin)
    QTimer.singleShot(150,capture_and_continue)


def timeout():
    print('GUI acceptance timed out',flush=True); window.stop()
    QTimer.singleShot(7000,lambda:app.exit(2))

window.frame_received.connect(frame); window.run_finished.connect(finished)
heartbeat.timeout.connect(tick); heartbeat.start(); QTimer.singleShot(200,begin); QTimer.singleShot(600000,timeout)
raise SystemExit(app.exec())
