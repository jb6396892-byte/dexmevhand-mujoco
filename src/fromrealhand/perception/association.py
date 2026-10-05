"""Associate detections with an already registered object using visual history."""
import numpy as np
from ..tabletop.camera import transform,project


def select_tracking_box(boxes,previous,vertices,k,twc,allow_depth_fallback=False):
    if previous is None:
        if len(boxes)!=1: raise ValueError('Initialization requires exactly one mug')
        return boxes[0]['box_xyxy'],dict(mode='unique_initial_detection',selected=0)
    uv=project(transform(vertices,np.linalg.inv(twc) @ previous),k)
    predicted=np.r_[uv.min(0)-12,uv.max(0)+12]
    candidates=[]
    for i,row in enumerate(boxes):
        b=np.asarray(row['box_xyxy'],dtype=float)
        if b.shape!=(4,) or not np.isfinite(b).all() or np.any(b[2:]<=b[:2]): continue
        lo=np.maximum(b[:2],predicted[:2]); hi=np.minimum(b[2:],predicted[2:])
        intersection=np.prod(np.maximum(hi-lo,0.))
        union=np.prod(b[2:]-b[:2])+np.prod(predicted[2:]-predicted[:2])-intersection
        if intersection/max(union,1.)>.15: candidates.append(i)
    if not candidates and allow_depth_fallback:
        return predicted.tolist(),dict(mode='depth_tracking_without_detection',detected=len(boxes),
            predicted_box=predicted.tolist(),requires_fresh_depth_registration=True)
    if len(candidates)!=1: raise ValueError('Visual identity ambiguous or missing')
    index=candidates[0]
    return boxes[index]['box_xyxy'],dict(mode='previous_visual_projection',selected=index,
        detected=len(boxes),predicted_box=predicted.tolist())
