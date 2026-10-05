"""Explicit source-asset correspondence, not an object state correction."""
import numpy as np
from .control import VisualReference


def prepare_reference(actions,qpos,source_poses,estimate,base,model,config,dt,profile,segments,stop):
    """Select the largest feasible orientation transfer before any motor step."""
    attempts=[]
    fractions=(1.,.75,.5,.25,0.) if profile.get('bounded_orientation') else (1.,)
    for fraction in fractions:
        try:
            adapter=ClearanceReference(actions,qpos,source_poses[profile['anchor']],estimate,base,model,
                dict(config,orientation_transfer_fraction=fraction),dt)
            adapter.set_clearance(profile['clearance'],350,500)
            goal=adapter.goal(source_poses[-1,:3,3])+np.asarray(profile['goal_offset'])
            if profile['stable_carry']:
                grasp=next(s['stop'] for s in segments if s['skill']=='grasp')-1
                lift=next(s['stop'] for s in segments if s['skill']=='lift')-1
                adapter.set_carry(grasp,lift,len(actions),goal-adapter.goal(source_poses[grasp,:3,3]))
            receipt=adapter.preflight(stop)
            receipt.update(orientation_transfer_fraction=fraction,failed_candidates=attempts,
                estimated_object_pose_unchanged=True,not_video_exact=fraction<1.)
            return adapter,goal,receipt
        except ValueError as error:
            attempts.append(dict(fraction=fraction,reason=str(error)))
    raise ValueError('No feasible orientation transfer: '+str(attempts))


class ClearanceReference(VisualReference):
    def set_carry(self, anchor, lift_stop, stop, displacement, lift_height=.08):
        displacement=np.asarray(displacement,dtype=float)
        if (not 0<=anchor<lift_stop<stop<=len(self.actions) or displacement.shape!=(3,)
                or not np.isfinite(displacement).all() or not np.isfinite(lift_height) or lift_height<=0):
            raise ValueError('Invalid carry schedule')
        self.actions=self.actions.copy(); self.qpos=self.qpos.copy()
        self.actions[anchor:]=self.actions[anchor].copy()
        self.qpos[anchor:]=self.qpos[anchor].copy()
        self.carry=(anchor,lift_stop,stop,displacement.copy(),float(lift_height))
        for index in range(anchor,stop):
            offset=self.carry_offset(index)
            dq=self.reference_base[:3,:3].T @ self.delta[:3,:3].T @ offset
            self.qpos[index,:3]+=dq
            self.actions[index,:3]+=self.factor[:3]*dq
        if not np.isfinite(self.actions).all() or np.max(np.abs(self.actions))>1+1e-8:
            raise ValueError('Planned carry feedforward exceeds actuator range')

    def carry_offset(self,index):
        anchor,lift_stop,stop,displacement,lift_height=self.carry
        if index<anchor: return np.zeros(3)
        lift=np.array([0.,0.,lift_height])
        if index<lift_stop:
            t=np.clip((index-anchor)/float(lift_stop-anchor),0.,1.)
            return lift*t**3*(10-15*t+6*t*t)
        t=np.clip((index-lift_stop)/float(stop-1-lift_stop),0.,1.)
        return lift+(displacement-lift)*t**3*(10-15*t+6*t*t)

    def set_clearance(self, height, start, stop):
        if not np.isfinite([height,start,stop]).all() or height<0 or not 0<=start<stop:
            raise ValueError('Invalid clearance schedule')
        self.clearance=(float(height),int(start),int(stop))

    def desired_qpos(self,index):
        desired=super().desired_qpos(index)
        height,start,stop=getattr(self,'clearance',(0.,0,1))
        t=np.clip((index-start)/float(stop-start),0.,1.)
        blend=t**3*(10-15*t+6*t*t)
        desired[:3] += self.base[:3,:3].T @ np.array([0.,0.,height*(1-blend)])
        return desired


def upright_source_correspondence(local_tips, poses, anchor):
    """Map the source mug's inverted up axis while preserving its handle x axis.

    This changes task correspondence, not the live object or recorded hand pose.
    It must be independently validated as a functional, not video-exact, grasp.
    """
    tips=np.asarray(local_tips,dtype=float).copy(); result=np.asarray(poses,dtype=float).copy()
    if tips.shape != (len(result),5,3) or result.shape[1:] != (4,4):
        raise ValueError('Invalid source correspondence arrays')
    if not np.isfinite(tips).all() or not np.isfinite(result).all() or not 0<=anchor<len(result):
        raise ValueError('Invalid source correspondence values')
    inverted=result[anchor,2,2] < -.8
    if inverted:
        flip=np.diag([1.,-1.,-1.,1.])
        result=result @ flip
        tips=tips @ flip[:3,:3]
    return tips,result,dict(inverted_source=bool(inverted),live_state_changed=False,
        mapping='preserve handle x, reverse source y/z axes' if inverted else 'identity',
        video_exact=False if inverted else None)
