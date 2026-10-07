#!/usr/bin/env python3
"""Native MuJoCo window: full-table objects and actuated platform, no grasping."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import mujoco_py
import glfw
from hierarchy_common import ROOT
from fromrealhand.whole_table.layout import sample
from fromrealhand.whole_table.scene import PlatformScene, object_catalog
from fromrealhand.whole_table.motion import rest_to_rest

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--seed', type=int, default=9)
p.add_argument('--count', type=int, choices=range(5), default=4)
p.add_argument('--cup-xy', type=float, nargs=2)
p.add_argument('--seconds', type=float, default=0, help='Zero keeps the window open')
a = p.parse_args()
config = json.loads((ROOT/'configs/whole-table-f1.json').read_text())
scene = PlatformScene(sample(a.seed, object_catalog(), config, a.count, a.cup_xy), config)
viewer = mujoco_py.MjViewer(scene.sim)
viewer.cam.lookat[:] = [0, 0, .36]; viewer.cam.distance=3.9
viewer.cam.azimuth=135; viewer.cam.elevation=-25; viewer.vopt.geomgroup[4]=0
viewer.vopt.geomgroup[2]=0
print('F1: platform motion only; local hand parked; no grasp policy.', flush=True)
targets = [config['home_m'], [-.38,-.35,.35], [.38,-.35,.35], [.38,.35,.35], [-.38,.35,.35]]
started = time.monotonic(); i = 0; frames = 0
initial_position = scene.position().copy()
simulation_start = scene.sim.data.time
render_interval = max(1, int(round(.016/config['timestep_s'])))
try:
    while not glfw.window_should_close(viewer.window):
        i = (i+1) % len(targets)
        duration, curve = rest_to_rest(scene.position(), targets[i], config)
        for step in range(int(np.ceil((duration+1)/config['timestep_s']))):
            if a.seconds and time.monotonic()-started >= a.seconds: raise KeyboardInterrupt
            scene.step(curve(min((step+1)*config['timestep_s'],duration)))
            if step % render_interval == 0:
                viewer.render(); frames += 1
                time.sleep(max(0., scene.sim.data.time-simulation_start-(time.monotonic()-started)))
            if glfw.window_should_close(viewer.window): raise KeyboardInterrupt
except KeyboardInterrupt:
    pass
finally:
    print(json.dumps(dict(rendered_frames=frames, wall_seconds=time.monotonic()-started,
                          simulation_seconds=scene.sim.data.time-simulation_start,
                          initial_position_m=initial_position.tolist(),
                          final_position_m=scene.position().tolist())), flush=True)
    glfw.destroy_window(viewer.window)
