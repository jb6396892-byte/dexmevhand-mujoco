#!/usr/bin/env python3
"""Check legacy GLX context ownership across preview cleanup and live rendering."""
import gc
import json
import mujoco_py
from fromrealhand.desktop.rendering import stream_context

xml='''<mujoco><worldbody><light pos="0 0 2"/>
<geom type="plane" size="1 1 .1" rgba=".8 .8 .8 1"/>
<body pos="0 0 .2"><freejoint/><geom type="box" size=".1 .1 .1" rgba="1 0 0 1"/></body>
</worldbody></mujoco>'''
old=mujoco_py.MjSim(mujoco_py.load_model_from_xml(xml))
live=mujoco_py.MjSim(mujoco_py.load_model_from_xml(xml))
preview=stream_context(old);context=stream_context(live)
values=[]
for current in (preview,context,preview,context):
    current.opengl_context.make_context_current();current.render(320,240)
    values.append(float(current.read_pixels(320,240,depth=False).std()))
del current,preview,old
gc.collect()
context.opengl_context.make_context_current();context.render(320,240)
values.append(float(context.read_pixels(320,240,depth=False).std()))
result=dict(pixel_std=values,nonblank=all(v>5 for v in values),reused=stream_context(live) is context)
print(json.dumps(result))
raise SystemExit(0 if result['nonblank'] and result['reused'] else 1)
