"""Read-only training health checks, independent of the frozen learning code."""
import json
import math
from pathlib import Path


def atomic_json(path, value):
    def finite_json(item):
        if isinstance(item, float) and not math.isfinite(item):
            return None
        if isinstance(item, dict):
            return {k: finite_json(v) for k, v in item.items()}
        if isinstance(item, (list, tuple)):
            return [finite_json(v) for v in item]
        return item
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(finite_json(value), indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def health(rows, limits, max_kl):
    errors, warnings = [], []
    if not rows:
        return dict(errors=[], warnings=[], iterations=0, sampled_steps=0)
    zero_streak = 0
    for index, row in enumerate(rows, 1):
        if row['iteration'] != index:
            errors.append('nonsequential_iteration')
        values = [row['measured_kl'], row['parameter_delta_l2'], row['contact_cost_sum']]
        values += list(row['returns']) + list(row['baseline_errors'])
        if not all(math.isfinite(float(x)) for x in values):
            errors.append('nonfinite_update')
        if row['measured_kl'] > max_kl + 1e-9:
            errors.append('kl_exceeded')
        zero_streak = zero_streak + 1 if row['parameter_delta_l2'] <= 0 else 0
        if zero_streak >= limits['max_consecutive_zero_updates']:
            errors.append('consecutive_zero_updates')
        for case in row['reports']:
            report = case['report']
            measurements = [report[k] for k in ('max_hand_scene_penetration_m', 'final_distance_m',
                                                  'max_joint_violation_rad')]
            if not report['finite'] or not all(math.isfinite(float(x)) for x in measurements):
                errors.append('nonfinite_physics')
            if report['max_hand_scene_penetration_m'] > limits['catastrophic_penetration_m']:
                errors.append('catastrophic_penetration')
    recent = [p for row in rows[-limits['warning_window_iterations']:] for p in row['reports']]
    all_cases = [p for row in rows for p in row['reports']]
    fraction = sum(p['task_pass'] for p in recent) / len(recent)
    if fraction < limits['warning_task_fraction']:
        warnings.append('low_recent_task_success')
    if any(not p['task_pass'] for p in recent):
        warnings.append('exploration_task_failures')
    return dict(errors=sorted(set(errors)), warnings=warnings, iterations=len(rows),
        sampled_steps=sum(r['sampled_steps'] for r in rows), trajectories=len(all_cases),
        task_pass=sum(p['task_pass'] for p in all_cases), strict_pass=sum(p['strict_pass'] for p in all_cases),
        recent_task_fraction=fraction, max_measured_kl=max(r['measured_kl'] for r in rows),
        latest_mean_return=rows[-1]['returns'][0], nonzero_updates=sum(r['parameter_delta_l2'] > 0 for r in rows))
