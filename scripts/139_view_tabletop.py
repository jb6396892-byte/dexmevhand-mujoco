#!/usr/bin/env python3
"""Native interactive MuJoCo tabletop viewer. Perception only, no grasp controller."""
import argparse
import time
import mujoco_py
from fromrealhand.tabletop.scene import build

p=argparse.ArgumentParser(description=__doc__); p.add_argument('--seed',type=int,default=0)
p.add_argument('--verify-frames',type=int,default=0)
args=p.parse_args(); sim,_,_,report=build(args.seed)
viewer=mujoco_py.MjViewer(sim)
viewer.vopt.geomgroup[2]=0; viewer.vopt.geomgroup[4]=0; viewer.vopt.sitegroup[:]=0
viewer.cam.type=2; viewer.cam.fixedcamid=sim.model.camera_name2id('rgbd')
print('Perception sandbox. Adroit parked; no grasp policy. Ctrl+C / close window to exit.',flush=True)
count=0
try:
    while True:
        start=time.monotonic()
        for _ in range(8): sim.step()
        viewer.add_overlay(mujoco_py.const.GRID_TOPLEFT,'TABLETOP RGB-D','Perception only | no grasp control')
        viewer.render(); count+=1
        if args.verify_frames and count>=args.verify_frames: break
        time.sleep(max(0.,1/60-(time.monotonic()-start)))
except KeyboardInterrupt: pass
finally:
    import glfw
    glfw.terminate()
