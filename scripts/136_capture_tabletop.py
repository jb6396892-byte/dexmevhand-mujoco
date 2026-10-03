#!/usr/bin/env python3
"""Capture physically settled RGB-D, with truth stored outside perception inputs."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import cv2
from fromrealhand.tabletop.scene import build,ground_truth,asset_manifest
from fromrealhand.tabletop.camera import read_rgbd,backproject,transform,project
from fromrealhand.desktop.rendering import stream_context


def write(path,value): path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def capture(args):
    args.output.mkdir(parents=True,exist_ok=False)
    observations=args.output/'observations'; observations.mkdir()
    truthdir=args.output/'evaluation_only'; truthdir.mkdir()
    sim,xml,mesh,report=build(args.seed,not args.no_target)
    (args.output/'scene.xml').write_text(xml)
    write(args.output/'assets.json',asset_manifest(xml))
    if mesh is not None: np.savez_compressed(str(args.output/'mug_model.npz'),**mesh)
    context=stream_context(sim)
    context.vopt.geomgroup[:]=1; context.vopt.geomgroup[2]=0; context.vopt.geomgroup[4]=0
    context.vopt.sitegroup[:]=0
    frames=[]
    for index in range(args.frames):
        if index:
            for _ in range(17): sim.step()
        rgb,depth,calib=read_rgbd(sim,context,'rgbd')
        if rgb.std()<5: raise RuntimeError('Blank RGB frame')
        cv2.imwrite(str(observations/('%04d-rgb.png'%index)),cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR))
        np.save(str(observations/('%04d-depth.npy'%index)),depth)
        write(observations/('%04d-camera.json'%index),calib)
        write(truthdir/('%04d-truth.json'%index),ground_truth(sim))
        points=backproject(depth,np.array(calib['K']))
        world=transform(points,np.array(calib['T_world_camera']))
        # Planar table interiors should reconstruct at z=0 without using an object mask.
        roi=(abs(world[:,:,0])<.12)&(world[:,:,1]>.12)&(world[:,:,1]<.30)
        errors=np.abs(world[:,:,2][roi]); valid=errors<.01
        v,u=np.indices(depth.shape)
        uv=project(points[::16,::16],np.array(calib['K']))
        reproj=float(np.max(abs(uv-np.stack([u,v],axis=-1)[::16,::16])))
        frames.append(dict(index=index,rgb_std=float(rgb.std()),
            table_plane_median_error_m=float(np.median(errors[valid])),
            table_plane_samples=int(valid.sum()),roundtrip_max_pixel_error=reproj,
            near_m=calib['near_m'],far_m=calib['far_m'],time_s=calib['time_s']))
        if index==0:
            view=cv2.applyColorMap(np.uint8(np.clip((depth-.4)/.8,0,1)*255),cv2.COLORMAP_TURBO)
            cv2.imwrite(str(args.output/'depth-preview.png'),view)
            np.savez_compressed(str(args.output/'pointcloud.npz'),xyz=world[::4,::4],rgb=rgb[::4,::4])
    report.update(frames=frames,camera='rgbd',truth_isolated=True,render_state_writes=0)
    report['passed']=report['finite'] and report['initial_penetration_m']<1e-6 and all(
        f['table_plane_samples']>100 and f['table_plane_median_error_m']<.001 and f['roundtrip_max_pixel_error']<1e-5 for f in frames)
    write(args.output/'capture-report.json',report)
    print(json.dumps(report),flush=True)
    import glfw
    glfw.terminate()
    if not report['passed']: raise SystemExit(2)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seed',type=int,default=0); p.add_argument('--frames',type=int,default=3)
    p.add_argument('--no-target',action='store_true'); capture(p.parse_args())
