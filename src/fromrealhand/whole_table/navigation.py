"""Conservative fixed-posture configuration-space planning using SciPy Dijkstra."""
import time
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra


class NavigationRejected(ValueError): pass


def inflated_boxes(hand_envelope, obstacles, clearance):
    parts = np.asarray(hand_envelope).reshape(-1,2,3)
    return np.array([[np.asarray(o['bounds'])[0]-hand[1]-clearance,
                      np.asarray(o['bounds'])[1]-hand[0]+clearance]
                     for o in obstacles for hand in parts]).reshape(-1,2,3)


def segments_clear(starts, ends, boxes):
    """Continuous segment/AABB slab test; no unchecked edges between grid nodes."""
    starts,ends=np.atleast_2d(starts),np.atleast_2d(ends)
    delta=ends-starts
    valid=np.ones(len(starts),dtype=bool)
    parallel=np.abs(delta)<1e-12
    for lo,hi in boxes:
        outside=np.any(parallel & ((starts<lo)|(starts>hi)),axis=1)
        safe_delta=np.where(parallel,1.,delta)
        a,b=(lo-starts)/safe_delta,(hi-starts)/safe_delta
        enter=np.max(np.where(parallel,-np.inf,np.minimum(a,b)),axis=1)
        leave=np.min(np.where(parallel,np.inf,np.maximum(a,b)),axis=1)
        hit=(~outside)&(np.maximum(enter,0)<=np.minimum(leave,1))
        valid &= ~hit
    return valid


def plan(start, goal, hand_envelope, obstacles, config):
    began=time.monotonic()
    start,goal=np.asarray(start,dtype=float),np.asarray(goal,dtype=float)
    lower,upper=np.array(config['workspace_min_m']),np.array(config['workspace_max_m'])
    for name,p in (('start',start),('goal',goal)):
        if p.shape!=(3,) or not np.isfinite(p).all() or np.any(p<lower) or np.any(p>upper):
            raise NavigationRejected(name+'_outside_workspace')
    boxes=inflated_boxes(hand_envelope,obstacles,config['clearance_m'])
    for name,p in (('start',start),('goal',goal)):
        if not segments_clear(p,p,boxes)[0]: raise NavigationRejected(name+'_in_collision_margin')
    if segments_clear(start,goal,boxes)[0]:
        path=np.array([start,goal]); nodes=2; direct=True
    else:
        step=config['grid_resolution_m']
        axes=[np.unique(np.r_[np.arange(lower[i],upper[i],step),upper[i],start[i],goal[i]]) for i in range(3)]
        shape=tuple(len(a) for a in axes)
        xyz=np.stack(np.meshgrid(*axes,indexing='ij'),axis=-1).reshape(-1,3)
        free=segments_clear(xyz,xyz,boxes)
        grid=np.arange(len(xyz)).reshape(shape)
        sources,targets,weights=[],[],[]
        for axis in range(3):
            a=[slice(None)]*3; b=list(a); a[axis]=slice(None,-1); b[axis]=slice(1,None)
            u,v=grid[tuple(a)].ravel(),grid[tuple(b)].ravel()
            mask=free[u]&free[v]; u,v=u[mask],v[mask]
            mask=segments_clear(xyz[u],xyz[v],boxes); u,v=u[mask],v[mask]
            sources.extend((u,v)); targets.extend((v,u))
            w=np.linalg.norm(xyz[u]-xyz[v],axis=1); weights.extend((w,w))
        graph=coo_matrix((np.concatenate(weights),(np.concatenate(sources),np.concatenate(targets))),
                         shape=(len(xyz),len(xyz))).tocsr()
        ids=[np.ravel_multi_index(tuple(int(np.flatnonzero(axes[i]==p[i])[0]) for i in range(3)),shape)
             for p in (start,goal)]
        distances,previous=dijkstra(graph,indices=ids[0],return_predecessors=True)
        if not np.isfinite(distances[ids[1]]): raise NavigationRejected('no_path_in_fixed_posture_grid')
        indices=[ids[1]]
        while indices[-1]!=ids[0]: indices.append(int(previous[indices[-1]]))
        raw=xyz[indices[::-1]]; kept=[raw[0]]; i=0
        while i<len(raw)-1:
            j=len(raw)-1
            while not segments_clear(raw[i],raw[j],boxes)[0]: j-=1
            kept.append(raw[j]); i=j
        path=np.array(kept); nodes=int(free.sum()); direct=False
    return dict(waypoints=path.tolist(), direct=direct, free_grid_nodes=nodes,
                planning_seconds=time.monotonic()-began,
                path_length_m=float(np.linalg.norm(np.diff(path,axis=0),axis=1).sum()),
                clearance_m=config['clearance_m'], obstacles=obstacles,
                hand_envelope_relative_m=np.asarray(hand_envelope).tolist(),
                scope='fixed posture per-part conservative AABBs; translational navigation only')
