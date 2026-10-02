"""Physical-event skill labels and contracts, independent of the DAPG trainer."""
import numpy as np

SKILLS = ('reach', 'grasp', 'lift', 'transport')
FINGERS = ('th', 'ff', 'mf', 'rf', 'lf')


def supported(row, config):
    loaded = [row[f + '_force_n'] > config['contact_force_n'] for f in FINGERS]
    return bool(loaded[0] and sum(loaded) >= config['support_fingers'])


def event(skill, row, config):
    if skill == 'reach':
        return any(row[f + '_force_n'] > config['contact_force_n'] for f in FINGERS)
    if skill == 'grasp':
        return supported(row, config)
    if skill == 'lift':
        return supported(row, config) and row['bottom_m'] >= config['lift_bottom_m']
    if skill == 'transport':
        return (supported(row, config) and row['bottom_m'] >= config['hold_bottom_m']
                and row['target_distance_m'] <= config['goal_tolerance_m'])
    raise ValueError('Unknown skill: ' + skill)


def safety_failure(row, config):
    values = [row[k] for k in ('bottom_m', 'target_distance_m', 'scene_penetration_m',
                               'joint_violation_rad')] + [row[f + '_force_n'] for f in FINGERS]
    if not row['finite'] or not np.isfinite(values).all():
        return 'nonfinite_state'
    if row['scene_penetration_m'] > config['max_penetration_m']:
        return 'scene_penetration'
    if row['joint_violation_rad'] > config['max_joint_violation_rad']:
        return 'joint_limit'
    return None


def sustained_event(rows, skill, config, start=0, count=None):
    required = count or config['event_persistence_steps']
    streak = 0
    for index in range(start, len(rows)):
        streak = streak + 1 if event(skill, rows[index], config) else 0
        if streak >= required:
            return index + 1  # Post-action events delimit half-open action slices.
    raise ValueError('No persistent %s event after step %d' % (skill, start))


def propose_segments(rows, config, dt):
    if not rows or dt <= 0:
        raise ValueError('Nonempty trace and positive control timestep required')
    bounds = [0]
    for skill in SKILLS[:-1]:
        bounds.append(sustained_event(rows, skill, config, start=bounds[-1]))
    bounds.append(len(rows))
    if any(stop <= start for start, stop in zip(bounds, bounds[1:])):
        raise ValueError('Physical boundaries leave an empty skill')
    segments = []
    for skill, start, stop in zip(SKILLS, bounds, bounds[1:]):
        segments.append(dict(skill=skill, start=start, stop=stop, steps=stop-start,
            time_start_s=start*dt, time_stop_s=stop*dt,
            source_frame_first=rows[start]['source_frame'],
            source_frame_last=rows[stop-1]['source_frame'],
            boundary_basis='persistent_physical_event' if skill != 'transport' else 'demonstration_end',
            review_status='pending'))
    validate_segments(segments, len(rows))
    return segments


def validate_segments(segments, horizon):
    if tuple(s['skill'] for s in segments) != SKILLS:
        raise ValueError('Expected reach, grasp, lift, transport in order')
    cursor = 0
    for segment in segments:
        start, stop = segment['start'], segment['stop']
        if not isinstance(start, int) or not isinstance(stop, int) or start != cursor or stop <= start:
            raise ValueError('Segments must be nonempty, contiguous half-open intervals')
        cursor = stop
    if cursor != horizon:
        raise ValueError('Segments must cover the entire action horizon')


class SkillContract:
    """Feedback-only status tracker. This object never writes simulation state."""
    def __init__(self, skill, config, initial_row):
        if skill not in SKILLS:
            raise ValueError('Unknown skill: ' + skill)
        self.skill, self.config = skill, config
        self.steps, self.streak = 0, 0
        self.status, self.reason = 'running', None
        failure = safety_failure(initial_row, config)
        prerequisite = {'reach': None, 'grasp': 'reach', 'lift': 'grasp', 'transport': 'lift'}[skill]
        if failure:
            self.status, self.reason = 'failed', failure
        elif prerequisite and not event(prerequisite, initial_row, config):
            self.status, self.reason = 'failed', 'entry_condition'

    def update(self, row):
        if self.status in ('failed', 'timeout'):
            return self.status
        self.steps += 1
        failure = safety_failure(row, self.config)
        if failure:
            self.status, self.reason = 'failed', failure
            return self.status
        self.streak = self.streak + 1 if event(self.skill, row, self.config) else 0
        required = self.config['transport_hold_steps'] if self.skill == 'transport' else self.config['event_persistence_steps']
        # A later safety failure can still invalidate a replay after success.
        if self.streak >= required:
            self.status, self.reason = 'success', 'physical_event'
        elif self.steps >= self.config['timeout_steps'][self.skill]:
            self.status, self.reason = 'timeout', 'step_budget'
        else:
            self.status, self.reason = 'running', None
        return self.status


class SegmentedReference:
    """Action-provider adapter for a no-reset, clock-preserving reference replay."""
    def __init__(self, actions, segments, action_dim=30):
        actions = np.asarray(actions)
        if actions.ndim != 2 or actions.shape[1] != action_dim or not np.isfinite(actions).all():
            raise ValueError('Finite [T, action_dim] actions required')
        if np.max(np.abs(actions)) > 1. + 1e-8:
            raise ValueError('Expected normalized actions; do not scale twice')
        validate_segments(segments, len(actions))
        self.actions, self.segments = actions.copy(), [dict(s) for s in segments]

    def __len__(self):
        return len(self.actions)

    def skill_at(self, step):
        if step < 0 or step >= len(self.actions):
            raise IndexError(step)
        return next(s['skill'] for s in self.segments if s['start'] <= step < s['stop'])

    def __getitem__(self, step):
        self.skill_at(step)
        return self.actions[step].copy()


def execute_reference(env, reference, config, measure, after_step=None):
    """Continuous reference replay, not a learned or generalized skill policy."""
    row = measure()
    summaries, active = [], None
    for step in range(len(reference)):
        skill = reference.skill_at(step)
        if active is None or skill != active.skill:
            if active is not None:
                summaries.append(dict(skill=active.skill, status=active.status, reason=active.reason,
                                      steps=active.steps))
                if active.status != 'success':
                    return summaries
            active = SkillContract(skill, config, row)
            if active.status == 'failed':
                summaries.append(dict(skill=skill, status=active.status, reason=active.reason, steps=0))
                return summaries
        env.step(reference[step])
        row = measure()
        active.update(row)
        if after_step is not None:
            after_step(step, row)
        if active.status in ('failed', 'timeout'):
            summaries.append(dict(skill=skill, status=active.status, reason=active.reason, steps=active.steps))
            return summaries
    summaries.append(dict(skill=active.skill, status=active.status, reason=active.reason, steps=active.steps))
    return summaries
