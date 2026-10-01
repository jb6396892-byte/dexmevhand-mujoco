"""Bounded state-feedback labels and phase-balanced corrective imitation."""
import numpy as np
from .multivideo import MultiVideoActions, conditioned_features
from .video_fidelity import source_clock


def phase_weights(length, masses, dt=.01):
    """Historical v11 fixed control windows, retained for exact reproduction."""
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


def aligned_phase_indices(length,geometry,time_scale,dt=.01,warmup=.5,source_boundaries=(30.,40.,55.)):
    """Map each control step through the same nonlinear clock as the simulator."""
    frames=np.asarray(geometry['source_frames'],dtype=float)
    fps=float(geometry['fps']);boundaries=np.asarray(source_boundaries,dtype=float)
    if (length<=0 or frames.ndim!=1 or len(frames)<2 or not np.isfinite(frames).all()
            or np.any(np.diff(frames)<=0) or boundaries.shape!=(3,)
            or not np.isfinite(boundaries).all() or np.any(np.diff(boundaries)<=0)):
        raise ValueError('Ordered source frames and three phase boundaries required')
    if not np.isfinite([fps,dt,time_scale,warmup]).all() or min(fps,dt,time_scale)<=0 or warmup<0:
        raise ValueError('Invalid control or source clock')
    times=np.arange(length)*dt
    source=frames[0]+fps*source_clock(times,(frames[-1]-frames[0])/fps,time_scale,warmup)
    phase=np.searchsorted(boundaries,source,side='right')+1
    phase[times<warmup]=0
    return phase


def aligned_phase_weights(length,masses,geometry,time_scale,dt=.01,warmup=.5,source_boundaries=(30.,40.,55.)):
    """Corrected sampler for a FUTURE study; not used by frozen v11 checkpoints."""
    masses=np.asarray(masses,dtype=float)
    if masses.shape!=(5,) or not np.isfinite(masses).all() or np.any(masses<=0):
        raise ValueError('Five positive phase masses required')
    phase=aligned_phase_indices(length,geometry,time_scale,dt,warmup,source_boundaries)
    counts=np.bincount(phase,minlength=5)
    weights=masses[phase]/counts[phase]
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
