"""Bounded state-feedback labels and phase-balanced corrective imitation."""
import numpy as np
from .multivideo import MultiVideoActions, conditioned_features


def phase_weights(length, masses, dt=.01):
    bins=np.array([0.,.5,5.5,7.1666666667,9.6666666667,np.inf])
    phase=np.searchsorted(bins[1:],np.arange(length)*dt,side='right')
    weights=np.zeros(length)
    if len(masses)!=5 or not np.isfinite(masses).all() or np.any(np.asarray(masses)<=0):
        raise ValueError('Five positive phase masses required')
    for i,mass in enumerate(masses):
        ids=phase==i
        if ids.any(): weights[ids]=mass/ids.sum()
    if not len(weights): raise ValueError('Empty trajectory')
    return weights/weights.sum()


def tracking_delta(q,qref,v,vref,conversion,gain,velocity_ratio=.02):
    arrays=[np.asarray(x,dtype=float) for x in (q,qref,v,vref,conversion)]
    if any(x.shape!=(30,) or not np.isfinite(x).all() for x in arrays):
        raise ValueError('Tracking requires finite hand states and conversion factors')
    if not np.isfinite([gain,velocity_ratio]).all() or min(gain,velocity_ratio)<0:
        raise ValueError('Tracking gains must be nonnegative')
    q,qref,v,vref,conversion=arrays
    error=np.clip(qref-q,-np.r_[np.full(3,.01),np.full(27,.1)],np.r_[np.full(3,.01),np.full(27,.1)])
    velocity=np.clip(vref-v,-1.,1.)
    return gain*conversion*(error+velocity_ratio*velocity)


class CorrectiveActions:
    def __init__(self,experiment,video,expert,reference,limits,checkpoint,gain,beta):
        self.e,self.video,self.expert=experiment,video,expert
        self.reference,self.limits=reference,np.asarray(limits)
        if not 0<=beta<=1: raise ValueError('Mixture beta must be in [0,1]')
        self.gain,self.beta=gain,beta
        self.student=MultiVideoActions(checkpoint,experiment,video) if checkpoint is not None else None
        m=experiment.model
        self.conversion=-m.actuator_biasprm[:,1]/(m.actuator_gainprm[:,0]*experiment.env.act_rng)
        self.features=[];self.labels=[];self.feedback=[];self.qerrors=[]

    def __len__(self): return self.video['horizon']

    def __getitem__(self,step):
        e=self.e.env;d=e.sim.data;state=self.expert['sim_data'][step]
        delta=tracking_delta(d.qpos[:30],state['qpos'][:30],d.qvel[:30],state['qvel'][:30],self.conversion,self.gain)
        raw=self.expert['actions'][step]+delta-self.reference[step]
        label=np.clip(raw,-self.limits,self.limits)
        expert_action=self.reference[step]+label
        x=conditioned_features(e._get_observations(),d.qpos,d.qvel,step,e.control_timestep,self.e.duration,
                               self.video['control']['time_scale'],self.video['id'])
        self.features.append(x);self.labels.append(label);self.feedback.append(delta)
        self.qerrors.append(np.linalg.norm(d.qpos[:30]-state['qpos'][:30]))
        student=self.student[step] if self.student is not None else expert_action
        # Only controls are blended. No state or object pose is written here.
        return self.beta*expert_action+(1-self.beta)*student
