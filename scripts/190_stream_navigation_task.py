#!/usr/bin/env python3
"""Validated language plan to whole-table candidate planning and live Qt frames."""
import argparse
import contextlib
import importlib
from pathlib import Path
import sys
from hierarchy_common import ROOT
from fromrealhand.desktop.runtime import emit
from fromrealhand.desktop.planning import validated_plan
from fromrealhand.whole_table.task import NavigationTask,make_layout,brief
from fromrealhand.whole_table.stream import Observer

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
p.add_argument('--visual-root',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True)
p.add_argument('--seed',type=int,default=5301);p.add_argument('--count',type=int)
p.add_argument('--target-world',nargs=3,type=float);p.add_argument('--cup-xy',nargs=2,type=float)
p.add_argument('--protocol',type=Path,default=ROOT/'configs/tabletop-navigation-v5.json')
p.add_argument('--mode',choices=['auto','fixed'],default='auto')
p.add_argument('--speed',type=float,default=1.);p.add_argument('--clearance',type=float,default=.025)
a=p.parse_args();task=None
def callback(kind,payload):
    with contextlib.redirect_stdout(sys.__stdout__):emit(kind,**payload)
try:
    plan=validated_plan(a.root,a.output)
    if plan['goal']=='stop':emit('result',report=dict(status='stopped',reason='user_stop',steps=0))
    else:
        controls=importlib.import_module('126_stream_simulation').Controls()
        output=a.visual_root/'navigation-v4/qt-runs'/a.output.name
        output.mkdir(parents=True,exist_ok=False)
        observer=Observer(callback,output/'frames',a.checkpoint,dict(speed=a.speed,clearance_m=a.clearance,mode=a.mode))
        with contextlib.redirect_stdout(sys.stderr):
            task=NavigationTask(ROOT,a.checkpoint,a.protocol)
            layout=make_layout(a.seed,task.config,a.target_world,a.count,a.cup_xy)
            result=task.run(layout,output/'task',preferred=plan['scene'],mode=a.mode,stop_skill=plan['goal'],
                speed=a.speed,clearance=a.clearance,observer=observer,cancelled=controls.stop.is_set)
        emit('result',report=brief(result))
except Exception as error:
    emit('error',message=str(error),error_type=type(error).__name__)
    raise SystemExit(1)
finally:
    if task:task.close()
