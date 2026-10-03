"""Pinhole RGB-D in OpenCV axes: right, down, forward; all lengths in metres."""
import numpy as np


def intrinsics(width, height, fovy):
    f = height / (2*np.tan(np.deg2rad(fovy)/2))
    return np.array([[f,0,(width-1)/2],[0,f,(height-1)/2],[0,0,1.]])


def metric_depth(buffer, near, far):
    if not 0 < near < far: raise ValueError('Invalid clipping planes')
    b = np.asarray(buffer, dtype=np.float64)
    if np.any(~np.isfinite(b)) or np.any((b<0)|(b>1)): raise ValueError('Invalid depth buffer')
    return (near/(1-b*(1-near/far))).astype(np.float32)


def backproject(depth, K):
    v,u = np.indices(depth.shape)
    return np.stack([(u-K[0,2])*depth/K[0,0], (v-K[1,2])*depth/K[1,1], depth],axis=-1)


def transform(points, T):
    return np.asarray(points) @ np.asarray(T)[:3,:3].T + np.asarray(T)[:3,3]


def project(points, K):
    p=np.asarray(points)
    if np.any(p[...,2]<=0): raise ValueError('Points behind camera')
    return p[...,:2]/p[...,2:3]*np.array([K[0,0],K[1,1]])+K[:2,2]


def camera_to_world(sim, camera_name):
    cid=sim.model.camera_name2id(camera_name)
    T=np.eye(4)
    T[:3,:3]=sim.data.cam_xmat[cid].reshape(3,3) @ np.diag([1.,-1.,-1.])
    T[:3,3]=sim.data.cam_xpos[cid]
    return T


def read_rgbd(sim, context, name, width=960, height=720):
    cid=sim.model.camera_name2id(name)
    before=np.r_[sim.data.qpos,sim.data.qvel].copy()
    context.render(width,height,camera_id=cid)
    rgb,zbuffer=context.read_pixels(width,height,depth=True)
    near=sim.model.vis.map.znear*sim.model.stat.extent
    far=sim.model.vis.map.zfar*sim.model.stat.extent
    if not np.array_equal(before,np.r_[sim.data.qpos,sim.data.qvel]):
        raise RuntimeError('Rendering mutated physics')
    return rgb[::-1].copy(),metric_depth(zbuffer[::-1],near,far),dict(
        K=intrinsics(width,height,sim.model.cam_fovy[cid]).tolist(),
        T_world_camera=camera_to_world(sim,name).tolist(),width=width,height=height,
        near_m=float(near),far_m=float(far),time_s=float(sim.data.time),
        axes='OpenCV x-right y-down z-forward',depth_units='metres',
        depth_type='optical-axis z, not Euclidean range',camera=name,
        rgb_depth_synchronized=True,render_state_writes=0)
