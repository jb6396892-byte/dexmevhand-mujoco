#!/usr/bin/env python3
"""Native actuated Adroit navigation window, no XYZ platform model."""
import argparse
import json
import time
import glfw
import mujoco_py
from hierarchy_common import ROOT
from fromrealhand.whole_table.layout import sample
from fromrealhand.whole_table.scene import object_catalog
from fromrealhand.whole_table.hand_scene import HandScene
from fromrealhand.whole_table.navigation import NavigationRejected
from fromrealhand.whole_table.navigation_runner import navigate

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--seed',type=int,default=9)
p.add_argument('--count',type=int,choices=range(5),default=4)
p.add_argument('--goal',type=float,nargs=3,default=[.32,.28,.34])
p.add_argument('--seconds',type=float,default=0)
p.add_argument('--wall',action='store_true')
a=p.parse_args();cfg=json.loads((ROOT/'configs/adroit-navigation-v1.json').read_text())
fixtures=[dict(name='test_wall',pos=[0,0,.18],size=[.035,.8,.18])] if a.wall else []
scene=HandScene(sample(a.seed,object_catalog(),cfg,a.count),cfg,fixtures)
viewer=mujoco_py.MjViewer(scene.sim)
viewer.cam.lookat[:]=[0,0,.12];viewer.cam.distance=1.65
viewer.cam.azimuth=135;viewer.cam.elevation=-40
viewer.vopt.geomgroup[2]=0;viewer.vopt.geomgroup[4]=0
started=time.monotonic(); sim_start=scene.sim.data.time;frames=[0]
def render(scene,report,row):
    if glfw.window_should_close(viewer.window) or (a.seconds and time.monotonic()-started>=a.seconds):
        raise KeyboardInterrupt
    viewer.render();frames[0]+=1
    time.sleep(max(0.,scene.sim.data.time-sim_start-(time.monotonic()-started)))
try:
    scene.sim.model.site_pos[scene.sim.model.site_name2id('navigation_goal')]=a.goal
    result=navigate(scene,a.goal,render)
    print(json.dumps({k:v for k,v in result.items() if k!='trace'}),flush=True)
    while not glfw.window_should_close(viewer.window):
        render(scene,None,None);time.sleep(.016)
except NavigationRejected as error: print('STOPPED:',str(error),flush=True)
except KeyboardInterrupt: pass
finally:
    print('rendered_frames',frames[0],flush=True)
    glfw.destroy_window(viewer.window)
