#!/usr/bin/env python3
"""Qt worker for the same random-task executor used in independent evaluation."""
import argparse
import contextlib
import importlib
from pathlib import Path
import sys
from hierarchy_common import ROOT
from fromrealhand.desktop.runtime import emit
from fromrealhand.desktop.planning import validated_plan
from fromrealhand.tabletop.random_task import RandomTask

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--root',type=Path,required=True); p.add_argument('--output',type=Path,required=True)
p.add_argument('--visual-root',type=Path,required=True); p.add_argument('--checkpoint',type=Path,required=True)
p.add_argument('--seed',type=int,default=0); p.add_argument('--count',type=int)
p.add_argument('--target-world',nargs=3,type=float)
a=p.parse_args(); task=None
try:
    plan=validated_plan(a.root,a.output)
    if plan['goal']=='stop':
        emit('result',report=dict(status='stopped',reason='user_stop',steps=0))
    else:
        controls=importlib.import_module('126_stream_simulation').Controls()
        with contextlib.redirect_stdout(sys.stderr): task=RandomTask(ROOT,plan['scene'],a.checkpoint)
        # Libraries may write to stdout during model construction; reserve it for protocol messages.
        def callback(kind, payload):
            with contextlib.redirect_stdout(sys.__stdout__): emit(kind,**payload)
        with contextlib.redirect_stdout(sys.stderr):
            task.run(a.seed,a.visual_root/'random-v5/qt-runs'/a.output.name,
                goal=a.target_world,count=a.count,stop_skill=plan['goal'],callback=callback,
                cancelled=controls.stop.is_set,realtime=True)
except Exception as error:
    emit('error',message=str(error),error_type=type(error).__name__)
    raise SystemExit(1)
finally:
    if task: task.close()
