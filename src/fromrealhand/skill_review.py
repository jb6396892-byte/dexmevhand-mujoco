"""Read-only evidence for proposed skill boundaries; no automatic human approval."""
import numpy as np
from .skills import FINGERS, event, propose_segments, safety_failure, validate_segments


def longest_run(values):
    longest = streak = 0
    for value in values:
        streak = streak + 1 if value else 0
        longest = max(longest, streak)
    return longest


def first_run_end(values, length):
    if length < 1:
        raise ValueError('Positive persistence required')
    streak = 0
    for index, value in enumerate(values):
        streak = streak + 1 if value else 0
        if streak >= length:
            return index + 1
    return None


def audit_boundaries(rows, segments, config, review, dt):
    validate_segments(segments, len(rows))
    proposed = propose_segments(rows, config, dt)
    matches = all(a['start'] == b['start'] and a['stop'] == b['stop']
                  for a, b in zip(segments, proposed))
    frames = np.asarray([r['source_frame'] for r in rows])
    safety = [safety_failure(row, config) for row in rows]
    reports = []
    window = review['boundary_window_steps']
    count = config['event_persistence_steps']
    for segment in segments[:-1]:
        stop, skill = segment['stop'], segment['skill']
        confirmation = rows[stop-count:stop]
        before = rows[max(segment['start'],stop-window):stop]
        after = rows[stop:min(stop+window,len(rows))]
        fraction = float(np.mean([event(skill,r,config) for r in after]))
        row = rows[stop-1]
        reports.append(dict(skill=skill, next_skill=segments[len(reports)+1]['skill'], boundary_state_index=stop,
            confirmation_action_range=[stop-count,stop], boundary_time_s=stop*dt,
            first_confirming_post_time_s=(stop-count+1)*dt,
            source_frame_at_last_action=float(row['source_frame']),
            confirmation_pass=bool(len(confirmation)==count and all(event(skill,r,config) for r in confirmation)),
            before_event_fraction=float(np.mean([event(skill,r,config) for r in before])),
            after_event_fraction=fraction, after_window_steps=len(after),
            handoff_warning=bool(fraction < review['handoff_event_fraction_warning']),
            loaded_fingers=[f for f in FINGERS if row[f+'_force_n'] > config['contact_force_n']],
            bottom_m=row['bottom_m'], target_distance_m=row['target_distance_m'],
            window_max_penetration_m=max(r['scene_penetration_m'] for r in before+after)))
    finger_timing = {}
    for finger in FINGERS:
        mask = [r[finger+'_force_n'] > config['contact_force_n'] for r in rows]
        end = first_run_end(mask, count)
        finger_timing[finger] = dict(first_confirmed_state_index=end,
            first_confirmed_time_s=end*dt if end is not None else None,
            source_frame=float(frames[end-1]) if end is not None else None,
            tail_contact_fraction=float(np.mean(mask[-config['transport_hold_steps']:])),
            longest_contact_s=longest_run(mask)*dt)
    sensitivity = []
    for force in review['sensitivity_contact_force_n']:
        for persistence in review['sensitivity_persistence_steps']:
            variant = dict(config, contact_force_n=force, event_persistence_steps=persistence)
            try:
                probe = propose_segments(rows, variant, dt)
                shifts = [a['stop']-b['stop'] for a,b in zip(probe[:-1],segments[:-1])]
                sensitivity.append(dict(contact_force_n=force, persistence_steps=persistence,
                    boundaries=[s['stop'] for s in probe[:-1]], shift_steps=shifts,
                    max_abs_shift_s=max(abs(s) for s in shifts)*dt, events_found=True))
            except ValueError as error:
                sensitivity.append(dict(contact_force_n=force, persistence_steps=persistence,
                                        events_found=False, reason=str(error)))
    checks = dict(boundaries_reproduced=matches, finite_source_clock=bool(np.isfinite(frames).all()),
        monotonic_source_clock=bool(np.all(np.diff(frames) >= 0)), all_steps_safe=not any(safety),
        persistent_events=all(r['confirmation_pass'] for r in reports),
        terminal_hold=bool(len(rows)>=config['transport_hold_steps'] and all(
            event('transport',r,config) for r in rows[-config['transport_hold_steps']:])) )
    return dict(checks=checks, mechanical_pass=all(checks.values()), boundaries=reports,
        finger_timing=finger_timing, sensitivity=sensitivity,
        handoff_warning_count=sum(r['handoff_warning'] for r in reports),
        human_review_status='pending', boundary_change_proposed=False)


def source_frame_record(mapping, source_frame):
    valid = [row for row in mapping if row['valid']]
    if not valid or not np.isfinite(source_frame):
        raise ValueError('Valid source frame mapping required')
    return min(valid, key=lambda row: abs(row['source_frame']-source_frame))
