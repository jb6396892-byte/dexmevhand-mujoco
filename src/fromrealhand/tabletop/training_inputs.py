"""Visual-only object features collected before each physical motor action."""
import numpy as np
from .control import rigid

SKILLS=('reach','grasp','lift','transport')
VIDEOS=('first','second')
FEATURE_DIM=139
FEATURE_VERSION='tabletop_visual_reference_v1'


def features(hand_qpos,hand_qvel,palm,visual_pose,goal,video,skill,reference_step,bounds,reference_action,desired):
    q=np.asarray(hand_qpos); v=np.asarray(hand_qvel)
    reference_action=np.asarray(reference_action); desired=np.asarray(desired)
    palm=np.asarray(palm); goal=np.asarray(goal); pose=rigid(visual_pose)
    lo,hi=bounds
    if (q.shape!=(30,) or v.shape!=(30,) or reference_action.shape!=(30,) or desired.shape!=(30,)
            or palm.shape!=(3,) or goal.shape!=(3,) or video not in VIDEOS or skill not in SKILLS
            or not lo<=reference_step<hi):
        raise ValueError('Invalid visual training input or reference clock')
    phase=np.eye(4)[SKILLS.index(skill)]; identity=np.eye(2)[VIDEOS.index(video)]
    x=np.r_[q,v,pose[:3,3]-palm,pose[:3,:2].T.ravel(),goal-pose[:3,3],phase,
        (reference_step-lo)/float(max(hi-lo-1,1)),identity,reference_action,desired-q]
    if x.shape!=(FEATURE_DIM,) or not np.isfinite(x).all(): raise ValueError('Nonfinite visual features')
    return x.astype(np.float32)


def episode_arrays(inputs,actions,reference_actions,reference_steps,phases,visual_frames):
    x=np.asarray(inputs,dtype=np.float32); a=np.asarray(actions,dtype=np.float32)
    ref=np.asarray(reference_actions,dtype=np.float32); n=len(x)
    if n==0 or x.shape!=(n,FEATURE_DIM) or a.shape!=(n,30) or ref.shape!=a.shape:
        raise ValueError('Incomplete pre-action episode')
    arrays=dict(inputs=x,actions=a,reference_actions=ref,residual_actions=a-ref,
        reference_steps=np.asarray(reference_steps,dtype=np.int64),
        phases=np.asarray(phases,dtype=np.int64),visual_frames=np.asarray(visual_frames,dtype=np.int64))
    if any(len(value)!=n or not np.isfinite(value).all() for value in arrays.values()):
        raise ValueError('Misaligned or nonfinite episode')
    if np.max(np.abs(a))>1+1e-6: raise ValueError('Non-normalized expert action')
    return arrays
