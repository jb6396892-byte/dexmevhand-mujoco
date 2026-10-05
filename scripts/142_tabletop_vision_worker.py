#!/usr/bin/env python3
"""Persistent isolated vision process; stdin requests contain observation paths only."""
import argparse
import contextlib
import hashlib
import json
from pathlib import Path
import sys
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--model', type=Path, required=True)
    a = p.parse_args(); root = a.run.resolve()
    from fromrealhand.desktop.runtime import emit
    with contextlib.redirect_stdout(sys.stderr):
        import cv2
        import numpy as np
        from fromrealhand.perception.detector import CupDetector
        from fromrealhand.perception.tracking import CupTracker
        from fromrealhand.perception.association import select_tracking_box
        from fromrealhand.tabletop.camera import transform, project
        detector = CupDetector(a.model); tracker = CupTracker(root/'mug_model.npz')
    emit('ready', backend='candidate_rgbd_tracker', validated=False)
    detector_misses=0
    for line in sys.stdin:
        request = json.loads(line)
        if request.get('command') == 'stop': break
        frame = str(request['frame'])
        if not frame.isdigit() or len(frame) != 6: raise ValueError('Invalid frame id')
        directory = root/'observations'/frame
        calib = json.loads((directory/'camera.json').read_text())
        image = directory/'rgb.png'
        rgb = cv2.cvtColor(cv2.imread(str(image)), cv2.COLOR_BGR2RGB)
        depth = np.load(str(directory/'depth.npy'), allow_pickle=False)
        k, twc = np.asarray(calib['K']), np.asarray(calib['T_world_camera'])
        start = time.monotonic()
        row = dict(accepted=False, frame=frame, camera_time_s=calib['time_s'],
                   source_rgb_sha256=hashlib.sha256(image.read_bytes()).hexdigest())
        display = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        try:
            with contextlib.redirect_stdout(sys.stderr):
                boxes, latency = detector.detect(rgb)
                row.update(boxes=boxes, detector_s=latency)
                box,association=select_tracking_box(boxes,tracker.previous,tracker.registration.vertices,k,twc,
                    allow_depth_fallback=detector_misses<5)
                detector_misses=detector_misses+1 if association['mode']=='depth_tracking_without_detection' else 0
                association['consecutive_detector_misses']=detector_misses
                row['association']=association
                mask, fit = tracker.estimate(depth, k, twc, box)
            t = fit.pop('T'); row.update(fit, T_world_object=t.tolist(),
                T_camera_object=(np.linalg.inv(twc) @ t).tolist(),
                reason='pose_accepted' if fit['accepted'] else 'tracking_quality_failed')
            display[mask] = (.65*display[mask]+.35*np.array([40,210,90])).astype(np.uint8)
            axes = transform(np.array([[0,0,0],[.04,0,0],[0,.04,0],[0,0,.04]]), np.linalg.inv(twc) @ t)
            uv = np.int32(np.rint(project(axes, k)))
            for i, color in enumerate(((0,0,255),(0,220,0),(255,70,0)), 1):
                cv2.arrowedLine(display, tuple(uv[0]), tuple(uv[i]), color, 2)
        except ValueError as error: row['reason'] = str(error)
        row.update(total_s=time.monotonic()-start, ground_truth_used=False, candidate_not_validated=True)
        cv2.imwrite(str(directory/'overlay.png'), display)
        (directory/'estimate.json').write_text(json.dumps(row, indent=2, allow_nan=False)+'\n')
        emit('estimate', **row)


if __name__ == '__main__': main()
