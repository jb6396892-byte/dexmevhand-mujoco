#!/usr/bin/env python3
"""Small reproducible window/offscreen GL probe; run each backend in a fresh process."""
import argparse
import ctypes
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--mode',choices=['window','stream'],default='window')
parser.add_argument('--output',type=Path)
args = parser.parse_args()
import mujoco_py
import numpy as np
print('EXTENSION '+mujoco_py.cymj.__file__,flush=True)
model = mujoco_py.load_model_from_xml('<mujoco><worldbody><light pos="0 0 3"/><geom type="plane" size="2 2 .1" rgba=".8 .8 .8 1"/><body pos="0 0 .3"><joint type="free"/><geom type="sphere" size=".1" rgba=".1 .7 .3 1"/></body></worldbody></mujoco>')
sim = mujoco_py.MjSim(model)
if args.mode=='window':
    context = mujoco_py.MjViewer(sim)
    for _ in range(3): sim.step(); context.render()
else:
    from fromrealhand.desktop.rendering import stream_context
    context = stream_context(sim)
    context.cam.lookat[:]=[0,0,.15]; context.cam.distance=1.2
    context.render(640,480)
gl = ctypes.CDLL('libGL.so.1'); gl.glGetString.restype=ctypes.c_char_p
pixels = context.read_pixels(640,480,depth=False)
result = dict(mode=args.mode,extension=mujoco_py.cymj.__file__,
    vendor=gl.glGetString(0x1F00).decode(),renderer=gl.glGetString(0x1F01).decode(),
    version=gl.glGetString(0x1F02).decode(),pixel_std=float(pixels.std()),shape=list(pixels.shape))
if args.output:
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result),flush=True)
if result['pixel_std']<5: raise RuntimeError('Blank GL probe')
import glfw
glfw.terminate()
