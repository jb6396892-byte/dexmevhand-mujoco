"""Versioned, causal boundary confirmation and guarded reference execution."""
import hashlib
import numpy as np
from .skills import SKILLS, SkillContract, event, supported, sustained_event, validate_segments


def skill_config(config, skill):
    return dict(config, event_persistence_steps=config['confirmation_steps'].get(
        skill,config['event_persistence_steps']))


def confirmed_segments(rows, config, dt):
    bounds = [0]
    for skill in SKILLS[:-1]:
        bounds.append(sustained_event(rows,skill,config,start=bounds[-1],
                                     count=config['confirmation_steps'][skill]))
    bounds.append(len(rows))
    if any(b<=a for a,b in zip(bounds,bounds[1:])):
        raise ValueError('An event leaves an empty skill')
    segments = []
    for skill,lo,hi in zip(SKILLS,bounds,bounds[1:]):
        segments.append(dict(skill=skill,start=lo,stop=hi,steps=hi-lo,
            time_start_s=lo*dt,time_stop_s=hi*dt,source_frame_first=rows[lo]['source_frame'],
            source_frame_last=rows[hi-1]['source_frame'],review_status='physics_event_scope_confirmed',
            boundary_basis='causal_persistent_event' if skill!='transport' else 'demonstration_end',
            confirmation_steps=config['confirmation_steps'].get(skill,config['transport_hold_steps'])))
    validate_segments(segments,len(rows))
    return segments


def handoff_audit(rows, segments, config):
    report = []
    for segment in segments[:-1]:
        start = segment['stop']
        window = rows[start:start+config['handoff_window_steps']]
        fraction = float(np.mean([event(segment['skill'],row,config) for row in window]))
        report.append(dict(skill=segment['skill'],state_index=start,steps=len(window),fraction=fraction,
            passed=bool(len(window)==config['handoff_window_steps'] and fraction>=config['handoff_min_fraction'])))
    return report


class GuardedSkill:
    def __init__(self,skill,config,initial):
        self.contract = SkillContract(skill,skill_config(config,skill),initial)
        self.config,self.skill,self.lost = config,skill,0

    @property
    def status(self): return self.contract.status

    @property
    def reason(self): return self.contract.reason

    @property
    def steps(self): return self.contract.steps

    def update(self,row):
        status = self.contract.update(row)
        if status in ('failed','timeout'):
            return status
        if self.skill in ('lift','transport'):
            self.lost = 0 if supported(row,self.config) else self.lost+1
            if self.lost >= self.config['support_loss_stop_steps']:
                self.contract.status,self.contract.reason = 'failed','persistent_support_loss'
        return self.status


def geometry_group(geometry,video):
    digest = hashlib.sha256(video.encode('ascii'))
    for name in sorted(geometry):
        value = np.asarray(geometry[name])
        if value.dtype.hasobject:
            raise ValueError('Object arrays cannot define scene identity')
        digest.update(name.encode('ascii'))
        digest.update(str(value.shape).encode('ascii'))
        digest.update(value.dtype.str.encode('ascii'))
        digest.update(np.ascontiguousarray(value).tobytes())
    return digest.hexdigest()
