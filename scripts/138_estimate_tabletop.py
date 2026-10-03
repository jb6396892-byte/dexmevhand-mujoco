#!/usr/bin/env python3
"""Run perception from an observation-only directory and known static CAD."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import cv2
import numpy as np
from fromrealhand.perception.detector import CupDetector
from fromrealhand.perception.registration import UprightRegistration,depth_foreground
from fromrealhand.tabletop.camera import transform,project


def write(path,value): path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def run(args):
    args.output.mkdir(parents=True,exist_ok=False)
    detector=CupDetector(Path(os.environ['VISUAL_GRASP_ROOT'])/'models/grounding-dino-tiny')
    pose=UprightRegistration(args.mesh)
    reports=[]
    for path in sorted(args.observations.glob('*-rgb.png')):
        start=time.monotonic(); prefix=path.name.split('-')[0]
        rgb=cv2.cvtColor(cv2.imread(str(path)),cv2.COLOR_BGR2RGB)
        depth=np.load(args.observations/(prefix+'-depth.npy'),allow_pickle=False)
        calib=json.loads((args.observations/(prefix+'-camera.json')).read_text())
        K=np.array(calib['K']); Twc=np.array(calib['T_world_camera'])
        boxes,latency=detector.detect(rgb)
        row=dict(frame=prefix,accepted=False,boxes=boxes,detector_s=latency,
            inference_inputs='RGB + metric depth + camera calibration + static CAD; no simulation object pose or mask',
            camera_time_s=calib['time_s'],device=detector.device,source_rgb_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        display=cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR)
        try:
            if not boxes: raise ValueError('No cup detected')
            if len(boxes)>1: raise ValueError('Ambiguous multiple cup detections')
            box=boxes[0]['box_xyxy']
            mask,points,plane=depth_foreground(depth,K,Twc,box)
            result=pose.estimate(points,plane); T=result.pop('T')
            row.update(result,plane_world=plane.tolist(),T_world_object=T.tolist(),T_camera_object=(np.linalg.inv(Twc)@T).tolist(),
                mask_pixels=int(mask.sum()),reason='pose_accepted' if result['accepted'] else 'registration_quality_failed')
            cv2.imwrite(str(args.output/(prefix+'-mask.png')),np.uint8(mask)*255)
            display[mask]=(.65*display[mask]+.35*np.array([40,210,90])).astype(np.uint8)
            x1,y1,x2,y2=np.int32(box); cv2.rectangle(display,(x1,y1),(x2,y2),(0,190,240),2)
            axes=transform(np.array([[0,0,0],[.05,0,0],[0,.05,0],[0,0,.05]]),np.linalg.inv(Twc)@T)
            uv=np.int32(np.rint(project(axes,K)))
            for i,color in enumerate([(0,0,255),(0,220,0),(255,70,0)],1): cv2.arrowedLine(display,tuple(uv[0]),tuple(uv[i]),color,2)
        except ValueError as error: row['reason']=str(error)
        row['total_s']=time.monotonic()-start
        cv2.putText(display,('%s | %.2fs | RGB-D estimate'%(row['reason'],row['total_s'])),(18,30),cv2.FONT_HERSHEY_SIMPLEX,.65,(30,30,30),2)
        cv2.imwrite(str(args.output/(prefix+'-overlay.png')),display)
        reports.append(row); write(args.output/(prefix+'-estimate.json'),row)
        print(json.dumps(row),flush=True)
    write(args.output/'summary.json',dict(frames=len(reports),accepted=sum(r['accepted'] for r in reports),
        results=reports,ground_truth_used=False,control_enabled=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--observations',type=Path,required=True)
    p.add_argument('--mesh',type=Path,required=True); p.add_argument('--output',type=Path,required=True)
    run(p.parse_args())
